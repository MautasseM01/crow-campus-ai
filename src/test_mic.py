"""
test_mic.py — التحقق من أن الميكروفون يعمل قبل أي شيء آخر
Verify that the microphone is working before running the main listener.
"""

import sounddevice as sd
import numpy as np


def list_devices():
    """طباعة قائمة بجميع أجهزة الصوت المتاحة / Print all available audio devices."""
    print("=== أجهزة الصوت المتاحة / Available Audio Devices ===")
    print(sd.query_devices())


def record_test(duration: float = 3.0, sample_rate: int = 48000) -> np.ndarray:
    """
    تسجيل عينة صوتية قصيرة للتحقق من الميكروفون.
    Record a short audio sample to verify the microphone.

    Args:
        duration: مدة التسجيل بالثواني / Recording duration in seconds.
        sample_rate: معدل أخذ العينات / Sample rate in Hz.

    Returns:
        مصفوفة numpy تحتوي على بيانات الصوت / numpy array of audio data.
    """
    print(f"\nجارٍ التسجيل لمدة {duration} ثانية... / Recording for {duration}s...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    print("اكتمل التسجيل! / Recording complete!")
    return audio


def check_signal(audio: np.ndarray, threshold: float = 0.001) -> bool:
    """
    التحقق من أن الإشارة الصوتية ليست صامتة.
    Check that the recorded audio signal is not silent.

    Args:
        audio: بيانات الصوت المسجلة / Recorded audio data.
        threshold: الحد الأدنى لمستوى الصوت / Minimum acceptable RMS level.

    Returns:
        True إذا كان الصوت مسموعاً / True if audio is audible.
    """
    rms = float(np.sqrt(np.mean(audio ** 2)))
    print(f"مستوى الصوت (RMS): {rms:.6f}")
    return rms > threshold


def main():
    """نقطة الدخول الرئيسية / Main entry point."""
    list_devices()

    audio = record_test(duration=3.0)

    if check_signal(audio):
        print("\n✓ الميكروفون يعمل بشكل صحيح! / Microphone is working correctly!")
    else:
        print("\n✗ تحذير: مستوى الصوت منخفض جداً — تحقق من الميكروفون.")
        print("  Warning: signal too weak — check your microphone.")


if __name__ == "__main__":
    main()
