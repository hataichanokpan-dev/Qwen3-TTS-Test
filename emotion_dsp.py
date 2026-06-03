"""
Emotion DSP - Synthesize emotional variants from a single reference audio.

Applies pitch shift, time stretch, and parametric EQ to create reference audio
variants for ICL mode. The TTS model picks up the emotional character from the
modified reference audio.

This module is model-agnostic — it works with any TTS engine that uses
reference audio for voice cloning (Qwen3-TTS ICL, etc.).

Emotion presets:
  dramatic_curious   +1 semitone, x0.95 speed, +1dB @ 3-5kHz
  alarming           +2 semitone, x1.05 speed, +2dB @ 4-6kHz
  confident           0 shift,    x0.98 speed, +1dB @ 200-400Hz
  reassuring         -1 semitone, x0.95 speed, +1dB @ 200Hz, -1dB @ 4kHz
  reflective_hopeful -1 semitone, x0.90 speed, +2dB @ 150-300Hz
  default            (unchanged)

Usage:
    from emotion_dsp import apply_emotion, generate_emotional_variants

    # Single variant
    audio, sr = apply_emotion("ref_best.wav", "confident")

    # Generate all variants
    paths = generate_emotional_variants("ref_best.wav", "emotion_variants/")
"""

import os
import numpy as np
import soundfile as sf

try:
    import librosa
except ImportError:
    librosa = None


# ---------------------------------------------------------------------------
# Emotion presets
# ---------------------------------------------------------------------------

EMOTIONS = {
    "dramatic_curious": {
        "pitch_semitones": 1,
        "time_stretch": 0.95,
        "eq": [{"freq_low": 3000, "freq_high": 5000, "gain_db": 1}],
    },
    "alarming": {
        "pitch_semitones": 2,
        "time_stretch": 1.05,
        "eq": [{"freq_low": 4000, "freq_high": 6000, "gain_db": 2}],
    },
    "confident": {
        "pitch_semitones": 0,
        "time_stretch": 0.98,
        "eq": [{"freq_low": 200, "freq_high": 400, "gain_db": 1}],
    },
    "reassuring": {
        "pitch_semitones": -1,
        "time_stretch": 0.95,
        "eq": [
            {"freq_low": 200, "freq_high": 200, "gain_db": 1},
            {"freq_low": 4000, "freq_high": 4000, "gain_db": -1},
        ],
    },
    "reflective_hopeful": {
        "pitch_semitones": -1,
        "time_stretch": 0.90,
        "eq": [{"freq_low": 150, "freq_high": 300, "gain_db": 2}],
    },
    "default": {
        "pitch_semitones": 0,
        "time_stretch": 1.0,
        "eq": [],
    },
}


# ---------------------------------------------------------------------------
# Internal DSP functions
# ---------------------------------------------------------------------------

def _apply_eq(y: np.ndarray, sr: int, freq_low: float, freq_high: float, gain_db: float) -> np.ndarray:
    """Apply parametric EQ using FFT-based Gaussian bandpass filter.

    Args:
        y: Audio signal (1D numpy array).
        sr: Sample rate.
        freq_low: Lower frequency bound (Hz).
        freq_high: Upper frequency bound (Hz).
        gain_db: Gain in dB (positive = boost, negative = cut).

    Returns:
        Equalized audio signal.
    """
    n = len(y)
    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, 1 / sr)

    center = (freq_low + freq_high) / 2
    width = max(freq_high - freq_low, 100)  # minimum bandwidth 100Hz

    # Convert dB to linear gain
    gain_linear = 10 ** (gain_db / 20)

    # Gaussian-shaped gain mask: 1 + (gain-1) * exp(-0.5 * ((f-center)/(width/2))^2)
    mask = 1 + (gain_linear - 1) * np.exp(-0.5 * ((freqs - center) / (width / 2)) ** 2)

    spectrum *= mask
    return np.fft.irfft(spectrum, n)


def _apply_pitch_shift(y: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Shift pitch by given number of semitones using librosa."""
    if librosa is None:
        raise ImportError("librosa is required for pitch shifting. Install with: pip install librosa")
    if semitones == 0:
        return y
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=semitones)


def _apply_time_stretch(y: np.ndarray, sr: int, rate: float) -> np.ndarray:
    """Time-stretch audio by given rate using librosa.

    Args:
        rate: >1 = faster/shorter, <1 = slower/longer.
    """
    if librosa is None:
        raise ImportError("librosa is required for time stretching. Install with: pip install librosa")
    if rate == 1.0:
        return y
    return librosa.effects.time_stretch(y, rate=rate)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_emotion(
    audio_input,
    emotion: str,
    output_path: str = None,
) -> tuple:
    """Apply emotion DSP to audio.

    Processes: time stretch -> pitch shift -> EQ (in order).

    Args:
        audio_input: Path to WAV file, or (numpy_array, sr) tuple.
        emotion: Emotion preset name from EMOTIONS dict.
        output_path: Optional path to save result WAV.

    Returns:
        (numpy_array, sample_rate) tuple.

    Raises:
        ValueError: Unknown emotion name.
        ImportError: librosa not installed and DSP is needed.
    """
    if emotion not in EMOTIONS:
        raise ValueError(
            f"Unknown emotion '{emotion}'. Choose from: {list(EMOTIONS.keys())}"
        )

    params = EMOTIONS[emotion]

    # Load audio
    if isinstance(audio_input, str):
        y, sr = librosa.load(audio_input, sr=None)
    else:
        y, sr = audio_input

    # Ensure mono
    if y.ndim > 1:
        y = y[:, 0]

    # Step 1: Time stretch
    if params["time_stretch"] != 1.0:
        y = _apply_time_stretch(y, sr, params["time_stretch"])

    # Step 2: Pitch shift
    if params["pitch_semitones"] != 0:
        y = _apply_pitch_shift(y, sr, params["pitch_semitones"])

    # Step 3: Parametric EQ
    for eq_band in params["eq"]:
        y = _apply_eq(y, sr, eq_band["freq_low"], eq_band["freq_high"], eq_band["gain_db"])

    # Save if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        sf.write(output_path, y, sr)
        duration = len(y) / sr
        print(f"Saved: {output_path} ({duration:.1f}s, emotion={emotion})")

    return y, sr


def generate_emotional_variants(
    ref_path: str,
    output_dir: str,
) -> dict:
    """Generate all emotion variants from a reference audio file.

    Creates one WAV per emotion preset in output_dir.

    Args:
        ref_path: Path to reference WAV file.
        output_dir: Directory to write variant WAVs.

    Returns:
        Dict mapping emotion name -> output file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(ref_path))[0]
    results = {}

    for emotion in EMOTIONS:
        out_path = os.path.join(output_dir, f"{base}_{emotion}.wav")
        audio, sr = apply_emotion(ref_path, emotion, output_path=out_path)
        results[emotion] = out_path

    print(f"\nGenerated {len(results)} emotion variants in {output_dir}")
    return results


def list_emotions() -> list[str]:
    """Return list of available emotion preset names."""
    return list(EMOTIONS.keys())


def get_emotion_params(emotion: str) -> dict:
    """Return parameters for a given emotion preset."""
    if emotion not in EMOTIONS:
        raise ValueError(f"Unknown emotion '{emotion}'. Choose from: {list(EMOTIONS.keys())}")
    return EMOTIONS[emotion].copy()
