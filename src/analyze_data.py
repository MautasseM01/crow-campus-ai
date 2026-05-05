"""
analyze_data.py — قراءة detections.csv وطباعة تقرير النشاط
Read detections.csv and print an activity report.
"""

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

DATA_FILE = Path("data") / "detections.csv"

TARGET_SPECIES = {
    "Carrion Crow",
    "Hooded Crow",
    "Rook",
    "Common Raven",
    "Eurasian Jackdaw",
}


def load_detections(path: Path) -> list[dict]:
    """
    يقرأ ملف CSV ويعيد قائمة الاكتشافات.
    Read the CSV file and return a list of detection records.

    Args:
        path: مسار ملف البيانات / Path to the detections CSV.

    Returns:
        قائمة من القواميس / List of detection dicts.
    """
    if not path.exists():
        print(f"لم يُعثر على الملف: {path} / File not found: {path}")
        return []

    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row["confidence"] = float(row["confidence"])
                row["timestamp"] = datetime.fromisoformat(row["timestamp"])
                records.append(row)
            except (ValueError, KeyError):
                continue  # تخطي الصفوف التالفة / skip malformed rows

    return records


def species_summary(records: list[dict]) -> Counter:
    """
    يحسب عدد الاكتشافات لكل نوع.
    Count detections per species.
    """
    return Counter(r["species"] for r in records)


def hourly_activity(records: list[dict]) -> dict[int, int]:
    """
    يحسب عدد الاكتشافات لكل ساعة من اليوم.
    Count detections per hour of the day.
    """
    counts: dict[int, int] = defaultdict(int)
    for r in records:
        counts[r["timestamp"].hour] += 1
    return dict(sorted(counts.items()))


def confidence_stats(records: list[dict]) -> dict:
    """
    يحسب إحصائيات الثقة (min, max, mean).
    Compute confidence statistics: min, max, mean.
    """
    if not records:
        return {"min": 0, "max": 0, "mean": 0}
    confs = [r["confidence"] for r in records]
    return {
        "min": round(min(confs), 4),
        "max": round(max(confs), 4),
        "mean": round(sum(confs) / len(confs), 4),
    }


def print_report(records: list[dict]):
    """
    يطبع تقريراً شاملاً عن نشاط الغربان.
    Print a full activity report.
    """
    total = len(records)
    crow_records = [r for r in records if r["species"] in TARGET_SPECIES]

    print("══════════════════════════════════════════════════════")
    print("  🐦‍⬛ CrowCampus AI — تقرير النشاط / Activity Report")
    print("══════════════════════════════════════════════════════\n")

    print(f"إجمالي الاكتشافات / Total detections : {total}")
    print(f"اكتشافات الغربان / Crow detections  : {len(crow_records)}\n")

    # --- إحصائيات الأنواع ---
    print("── الأنواع الأكثر اكتشافاً / Top Species ───────────")
    species = species_summary(records)
    for sp, count in species.most_common(10):
        marker = "🐦" if sp in TARGET_SPECIES else "  "
        bar = "█" * min(count, 40)
        print(f"  {marker} {sp:<30} {count:>5}  {bar}")

    # --- إحصائيات الثقة ---
    print("\n── إحصائيات الثقة / Confidence Stats ───────────────")
    stats = confidence_stats(crow_records)
    print(f"  أدنى / Min  : {stats['min']}")
    print(f"  أعلى / Max  : {stats['max']}")
    print(f"  متوسط / Mean: {stats['mean']}")

    # --- النشاط بالساعة ---
    print("\n── النشاط بالساعة / Hourly Activity ────────────────")
    hourly = hourly_activity(crow_records)
    if hourly:
        peak_hour = max(hourly, key=hourly.get)
        for hour, count in hourly.items():
            bar = "█" * min(count, 30)
            marker = " ◄ ذروة / peak" if hour == peak_hour else ""
            print(f"  {hour:02d}:00  {count:>4}  {bar}{marker}")
    else:
        print("  لا توجد بيانات بعد / No data yet")

    # --- آخر 5 اكتشافات ---
    recent = sorted(crow_records, key=lambda r: r["timestamp"], reverse=True)[:5]
    if recent:
        print("\n── آخر اكتشافات الغربان / Recent Crow Detections ──")
        for r in recent:
            ts = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            print(f"  [{ts}] {r['species']} ({r['confidence']:.2f})")

    print("\n══════════════════════════════════════════════════════")


def main():
    """نقطة الدخول الرئيسية / Main entry point."""
    records = load_detections(DATA_FILE)
    if not records:
        print("لا توجد بيانات للتحليل. شغّل phase0_listener.py أولاً.")
        print("No data to analyze. Run phase0_listener.py first.")
        return
    print_report(records)


if __name__ == "__main__":
    main()
