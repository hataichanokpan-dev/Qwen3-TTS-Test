"""Real TTS engine wrapper — bridges Qwen3-TTS VoiceCloner to the engine_fn interface.

engine_fn signature: (text: str, speed: float) -> (audio: np.ndarray, sr: int)

Speed control is applied via librosa.effects.time_stretch post-processing because
Qwen3-TTS does not expose a native speed parameter.
"""

import os
import sys
import gc
import numpy as np
import librosa

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from voice_clone.cloner import VoiceCloner

TARGET_SR = 24000


def create_real_engine(
    ref_audio: str = "ref_best.wav",
    ref_text: str = None,
    preprocess: bool = False,
    device: str = "cuda:0",
) -> callable:
    """Create a real TTS engine function backed by Qwen3-TTS VoiceCloner.

    Args:
        ref_audio: Path to reference audio for voice cloning.
        ref_text: Path to reference text file, or text string.
                  If None, auto-detected from ref_audio name.
        preprocess: Enable English→Thai text preprocessing.
        device: CUDA device string.

    Returns:
        engine_fn(text, speed=1.0) -> (np.ndarray, sample_rate)
    """
    # Auto-detect ref_text: look for .txt with same stem as ref_audio
    if ref_text is None:
        stem = os.path.splitext(ref_audio)[0]
        candidates = [
            stem + ".txt",
            stem + "_text.txt",
            stem + ".ref.txt",
            os.path.join(os.path.dirname(stem), "ref_text.txt"),
        ]
        for candidate in candidates:
            if os.path.isfile(candidate):
                ref_text = candidate
                break

    cloner = VoiceCloner(
        ref_audio=ref_audio,
        ref_text=ref_text,
        config={"device": device},
        preprocess=preprocess,
    )

    # Lazy-load: model loads on first clone() call
    _loaded = [False]

    def engine_fn(text: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
        """Generate real Thai speech via Qwen3-TTS.

        Args:
            text: Thai text to synthesize.
            speed: Speaking rate multiplier (>1.0 = faster, <1.0 = slower).
                   Applied via librosa.effects.time_stretch post-processing.

        Returns:
            (audio_array, sample_rate) — always 24kHz mono float64.
        """
        if not _loaded[0]:
            cloner.load_model()
            _loaded[0] = True

        # Generate at native speed
        audio, sr = cloner.clone(text)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio[:, 0]

        # Resample if needed
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        # Apply speed adjustment via time-stretch
        if abs(speed - 1.0) > 0.01:
            # librosa.effects.time_stretch rate >1.0 means faster (shorter)
            # Our speed >1.0 means faster, so pass directly
            audio = librosa.effects.time_stretch(audio, rate=speed)

        return audio.astype(np.float64), TARGET_SR

    # Attach unload helper for cleanup
    engine_fn.unload = lambda: (
        cloner.unload_model(),
        gc.collect(),
        _loaded.__setitem__(0, False),
    )
    engine_fn.cloner = cloner

    return engine_fn


def create_real_engine_cpu(
    ref_audio: str = "ref_best.wav",
    ref_text: str = None,
    preprocess: bool = False,
) -> callable:
    """Create a real TTS engine using CPU (slower but no GPU required)."""
    return create_real_engine(
        ref_audio=ref_audio,
        ref_text=ref_text,
        preprocess=preprocess,
        device="cpu",
    )


if __name__ == "__main__":
    import soundfile as sf
    import time

    print("=== Real Engine Test ===")

    ref = os.path.join(BASE_DIR, "ref_best.wav")
    if not os.path.isfile(ref):
        print(f"ERROR: {ref} not found")
        sys.exit(1)

    print(f"Reference: {ref}")
    print("Creating engine...")
    engine = create_real_engine(ref_audio=ref)

    test_text = "สวัสดีครับ วันนี้อากาศดีมาก"
    print(f"Generating: '{test_text}'")

    t0 = time.time()
    audio, sr = engine(test_text, speed=1.0)
    elapsed = time.time() - t0
    duration = len(audio) / sr
    rtf = elapsed / duration

    print(f"Output: {len(audio)} samples, {sr}Hz, {duration:.2f}s")
    print(f"RTF: {rtf:.3f} ({1/rtf:.1f}x realtime)")
    assert len(audio) > 0, "Audio is empty"
    assert sr == TARGET_SR, f"Wrong SR: {sr}"

    out_path = os.path.join(BASE_DIR, "test_real_engine.wav")
    sf.write(out_path, audio, sr)
    print(f"Saved: {out_path}")

    # Test speed variation
    print("\nTesting speed=1.2 (faster)...")
    audio_fast, sr2 = engine(test_text, speed=1.2)
    dur_fast = len(audio_fast) / sr2
    print(f"Speed 1.2: {dur_fast:.2f}s (vs {duration:.2f}s at 1.0)")
    assert dur_fast < duration, "Speed 1.2 should be shorter"

    print("\nTesting speed=0.8 (slower)...")
    audio_slow, sr3 = engine(test_text, speed=0.8)
    dur_slow = len(audio_slow) / sr3
    print(f"Speed 0.8: {dur_slow:.2f}s (vs {duration:.2f}s at 1.0)")
    assert dur_slow > duration, "Speed 0.8 should be longer"

    engine.unload()
    print("\nAll tests PASSED")
