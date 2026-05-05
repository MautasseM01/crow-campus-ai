"""
phase0_listener.py — المستمع الرئيسي للمرحلة الصفرية
Phase 0: Passive listening — record 3s segments, classify with BirdNET, log to CSV.
Architecture: 3 threads — recorder / analyzer / stats
"""

import csv
import queue
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer

# ═══════════════════════════════════════════════════════════════════
# الإعدادات / Configuration
# ═══════════════════════════════════════════════════════════════════

SAMPLE_RATE = 48_000          # Hz — BirdNET requirement
SEGMENT_DURATION = 3          # seconds per recording chunk
CHANNELS = 1
MIN_CONFIDENCE = 0.25         # minimum BirdNET confidence to log

TARGET_SPECIES = {
    "Carrion Crow",
    "Hooded Crow",
    "Rook",
    "Common Raven",
    "Eurasian Jackdaw",
}

OUTPUT_CSV = Path("data") / "detections.csv"
CSV_HEADERS = ["timestamp", "species", "confidence", "lat", "lon"]

# الإحداثيات التقريبية للحرم الجامعي (تُحدَّث حسب الموقع الفعلي)
# Approximate campus coordinates — update to actual location
CAMPUS_LAT = 48.8566
CAMPUS_LON = 2.3522

# ═══════════════════════════════════════════════════════════════════
# الحالة المشتركة / Shared state
# ═══════════════════════════════════════════════════════════════════

audio_queue: queue.Queue = queue.Queue(maxsize=10)
stats_queue: queue.Queue = queue.Queue()
stop_event = threading.Event()

# إحصائيات الجلسة / Session statistics
session_stats: dict = {
    "segments_recorded": 0,
    "segments_analyzed": 0,
    "detections": 0,
    "crow_detections": 0,
}
stats_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════
# الخيط الأول: التسجيل / Thread 1: Recorder
# ═══════════════════════════════════════════════════════════════════

def recorder_thread():
    """
    يسجل مقاطع صوتية مدة كل منها 3 ثوانٍ ويضعها في الطابور.
    Records 3-second audio segments and puts them in the queue.
    """
    print("🎙️  بدأ خيط التسجيل / Recorder thread started")
    while not stop_event.is_set():
        try:
            audio = sd.rec(
                int(SEGMENT_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
            )
            sd.wait()
            timestamp = datetime.now().isoformat()

            with stats_lock:
                session_stats["segments_recorded"] += 1

            audio_queue.put((timestamp, audio), timeout=2)
        except queue.Full:
            print("تحذير: الطابور ممتلئ — تم تجاهل المقطع / Warning: queue full, segment dropped")
        except Exception as exc:
            print(f"خطأ في التسجيل / Recorder error: {exc}")

    print("⏹️  انتهى خيط التسجيل / Recorder thread stopped")


# ═══════════════════════════════════════════════════════════════════
# الخيط الثاني: التحليل / Thread 2: Analyzer
# ═══════════════════════════════════════════════════════════════════

def analyzer_thread(analyzer: Analyzer):
    """
    يسحب المقاطع من الطابور، يحللها بـ BirdNET، ويسجل النتائج.
    Pulls segments from the queue, analyzes with BirdNET, logs results.

    Args:
        analyzer: كائن BirdNET المُحمَّل مسبقاً / Pre-loaded BirdNET Analyzer instance.
    """
    print("🔬  بدأ خيط التحليل / Analyzer thread started")
    _ensure_csv()

    while not stop_event.is_set() or not audio_queue.empty():
        try:
            timestamp, audio = audio_queue.get(timeout=1)
        except queue.Empty:
            continue

        detections = _classify(analyzer, audio, timestamp)

        with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            for det in detections:
                writer.writerow(det)

        with stats_lock:
            session_stats["segments_analyzed"] += 1
            session_stats["detections"] += len(detections)
            crow_count = sum(1 for d in detections if d["species"] in TARGET_SPECIES)
            session_stats["crow_detections"] += crow_count

        if detections:
            for det in detections:
                marker = "🐦" if det["species"] in TARGET_SPECIES else "  "
                print(
                    f"{marker} [{det['timestamp']}] {det['species']} "
                    f"({det['confidence']:.2f})"
                )

        stats_queue.put(dict(session_stats))
        audio_queue.task_done()

    print("⏹️  انتهى خيط التحليل / Analyzer thread stopped")


def _classify(analyzer: Analyzer, audio: np.ndarray, timestamp: str) -> list[dict]:
    """
    يصنف مقطعاً صوتياً باستخدام BirdNET ويعيد قائمة الاكتشافات.
    Classifies an audio segment using BirdNET and returns detections list.
    """
    detections = []

    # نكتب الصوت في ملف مؤقت لأن birdnetlib تتطلب ملفاً
    # birdnetlib requires a file path, so we use a temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        sf.write(tmp_path, audio, SAMPLE_RATE, subtype="PCM_16")
        recording = Recording(
            analyzer,
            tmp_path,
            lat=CAMPUS_LAT,
            lon=CAMPUS_LON,
            date=datetime.now(),
            min_conf=MIN_CONFIDENCE,
        )
        recording.analyze()

        for det in recording.detections:
            detections.append({
                "timestamp": timestamp,
                "species": det["common_name"],
                "confidence": round(det["confidence"], 4),
                "lat": CAMPUS_LAT,
                "lon": CAMPUS_LON,
            })
    except Exception as exc:
        print(f"خطأ في التصنيف / Classification error: {exc}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return detections


def _ensure_csv():
    """ينشئ ملف CSV مع الترويسات إذا لم يكن موجوداً / Create CSV with headers if missing."""
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not OUTPUT_CSV.exists():
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
        print(f"تم إنشاء ملف البيانات: {OUTPUT_CSV} / Data file created")


# ═══════════════════════════════════════════════════════════════════
# الخيط الثالث: الإحصائيات / Thread 3: Stats
# ═══════════════════════════════════════════════════════════════════

def stats_thread(interval: int = 60):
    """
    يطبع ملخصاً للإحصائيات كل فترة محددة.
    Prints a session summary at a regular interval.

    Args:
        interval: الفترة الزمنية بين كل تقرير (بالثواني) / Seconds between reports.
    """
    print("📊  بدأ خيط الإحصائيات / Stats thread started")
    last_print = time.time()

    while not stop_event.is_set():
        time.sleep(1)
        if time.time() - last_print >= interval:
            _print_stats()
            last_print = time.time()

    _print_stats()  # تقرير نهائي عند التوقف / final report on shutdown
    print("⏹️  انتهى خيط الإحصائيات / Stats thread stopped")


def _print_stats():
    """يطبع الإحصائيات الحالية للجلسة / Print current session statistics."""
    with stats_lock:
        s = dict(session_stats)
    print(
        f"\n── إحصائيات الجلسة / Session Stats ──────────────────\n"
        f"  مقاطع مسجلة / Recorded  : {s['segments_recorded']}\n"
        f"  مقاطع محللة / Analyzed  : {s['segments_analyzed']}\n"
        f"  اكتشافات / Detections   : {s['detections']}\n"
        f"  غربان / Crow detections : {s['crow_detections']}\n"
        f"──────────────────────────────────────────────────────\n"
    )


# ═══════════════════════════════════════════════════════════════════
# نقطة الدخول الرئيسية / Main entry point
# ═══════════════════════════════════════════════════════════════════

def main():
    """تشغيل نظام الاستماع السلبي / Launch the passive listening system."""
    print("══════════════════════════════════════════════════════")
    print("  🐦‍⬛ CrowCampus AI — المرحلة الصفرية / Phase 0")
    print("  الاستماع السلبي / Passive Listening")
    print("══════════════════════════════════════════════════════")
    print("اضغط Ctrl+C للإيقاف / Press Ctrl+C to stop\n")

    print("جارٍ تحميل BirdNET... / Loading BirdNET model...")
    analyzer = Analyzer()
    print("تم تحميل النموذج! / Model loaded!\n")

    threads = [
        threading.Thread(target=recorder_thread, name="Recorder", daemon=True),
        threading.Thread(target=analyzer_thread, args=(analyzer,), name="Analyzer", daemon=True),
        threading.Thread(target=stats_thread, name="Stats", daemon=True),
    ]

    for t in threads:
        t.start()

    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nجارٍ الإيقاف... / Shutting down...")
        stop_event.set()

    for t in threads:
        t.join(timeout=10)

    print("\nتم الإيقاف بنجاح! / Shutdown complete!")


if __name__ == "__main__":
    main()
