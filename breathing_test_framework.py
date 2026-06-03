"""Breathing test framework for evaluating natural pauses in TTS audio.

Provides utilities to:
- Split Thai text into sentences
- Generate audio with pauses at sentence boundaries
- Splice recorded breath sounds at boundaries
- Evaluate breathing quality of generated audio
- Detect sentence boundaries via silence gaps

Standalone — no TTS engine dependency required.
"""

import os
import sys
import re
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def split_thai_sentences(text: str) -> list[str]:
    """Split Thai text into sentences using common Thai delimiters.

    Splits on: Thai iteration mark (ฯ), spaces between sentences,
    and standard punctuation (. ? !) followed by space or end-of-string.

    Args:
        text: Thai text to split.

    Returns:
        List of sentence strings, whitespace-stripped, non-empty.
    """
    # Split on Thai sentence-ending markers and standard punctuation
    # ฯ = Thai punctuation mark (paiyannoi), used as sentence terminator
    # Also split on multiple spaces (2+) as Thai sentences are often space-separated
    parts = re.split(r'(?:ฯ|(?<=[.?!])\s+|\s{2,})', text)
    # If no punctuation splits happened, fall back to single-space splitting
    sentences = [s.strip() for s in parts if s.strip()]
    if len(sentences) <= 1:
        # Thai text often has no punctuation — split on single spaces as fallback
        parts = text.split()
        # Regroup: treat each space-separated token as a sentence
        # But only if there are multiple tokens
        if len(parts) > 1:
            sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def generate_with_pauses(
    text: str,
    pause_duration_ms: int = 400,
    engine_fn=None,
) -> np.ndarray:
    """Generate audio with silence pauses inserted at Thai sentence boundaries.

    Args:
        text: Thai text to synthesize.
        pause_duration_ms: Duration of pause in milliseconds (300-600ms for natural breathing).
        engine_fn: Callable that takes text string and returns tuple (np.ndarray, sample_rate).

    Returns:
        Combined audio with pauses as np.ndarray at 24kHz.
    """
    if engine_fn is None:
        raise ValueError("engine_fn is required — pass a callable(text) -> (audio, sr)")

    target_sr = 24000
    sentences = split_thai_sentences(text)
    pause_samples = int(pause_duration_ms / 1000.0 * target_sr)
    silence = np.zeros(pause_samples, dtype=np.float64)

    parts: list[np.ndarray] = []
    for i, sentence in enumerate(sentences):
        audio, sr = engine_fn(sentence)
        # Resample if needed
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        if audio.ndim > 1:
            audio = audio[:, 0]
        parts.append(audio)
        # Add pause between sentences (not after the last one)
        if i < len(sentences) - 1:
            parts.append(silence)

    if not parts:
        return np.zeros(0, dtype=np.float64)

    return np.concatenate(parts)


def splice_breath_at_boundaries(
    audio: np.ndarray,
    sr: int,
    breath_sample: np.ndarray,
    breath_sr: int,
    boundaries: list[int],
    crossfade_ms: float = 30.0,
) -> np.ndarray:
    """Splice a recorded breath sound at specified sample boundaries with crossfade.

    Args:
        audio: Original audio array (1D).
        sr: Sample rate of the original audio.
        breath_sample: Recorded breath sound (1D).
        breath_sr: Sample rate of the breath sample.
        boundaries: List of sample indices where breaths should be inserted.
        crossfade_ms: Crossfade duration in milliseconds.

    Returns:
        Audio with breath sounds spliced in.
    """
    # Resample breath to match audio SR if needed
    if breath_sr != sr:
        import librosa
        breath_sample = librosa.resample(breath_sample, orig_sr=breath_sr, target_sr=sr)

    cf_samples = int(crossfade_ms / 1000.0 * sr)
    breath_len = len(breath_sample)

    # Sort boundaries descending so insertions don't shift earlier indices
    result = audio.copy()
    for boundary in sorted(boundaries, reverse=True):
        if boundary < 0 or boundary >= len(result):
            continue

        # Build the spliced region with crossfade
        # Before boundary (keep as-is up to cf_samples before boundary)
        pre = result[:boundary]
        # After boundary (keep from boundary onward, with fade-in)
        post = result[boundary:]

        if cf_samples > 0 and len(pre) >= cf_samples and breath_len >= cf_samples:
            fade_out = np.linspace(1.0, 0.0, cf_samples, dtype=np.float64)
            fade_in = np.linspace(0.0, 1.0, cf_samples, dtype=np.float64)

            # Crossfade pre-tail with breath head
            pre_tail = pre[-cf_samples:] * fade_out + breath_sample[:cf_samples] * fade_in
            pre = np.concatenate([pre[:-cf_samples], pre_tail])

            # If breath is longer than cf_samples, append the middle portion
            mid_breath = breath_sample[cf_samples:]
            if len(mid_breath) > 0:
                pre = np.concatenate([pre, mid_breath])

            # Crossfade breath tail with post head
            if len(post) >= cf_samples:
                breath_tail_fade = np.zeros(cf_samples, dtype=np.float64)
                post_head = post[:cf_samples] * fade_in
                post_cross = breath_tail_fade + post_head
                result = np.concatenate([pre, post_cross, post[cf_samples:]])
            else:
                result = np.concatenate([pre, post])
        else:
            # No crossfade — simple splice
            result = np.concatenate([pre, breath_sample, post])

    return result


def evaluate_breathing(audio_path: str) -> dict:
    """Evaluate breathing quality of generated audio.

    Args:
        audio_path: Path to WAV file to evaluate.

    Returns:
        Dict with:
            - silence_ratio: ratio of silence to total duration (target: 0.08-0.15)
            - pause_durations_ms: list of pause durations in ms
            - pauses_at_boundaries: bool, whether pauses align with sentence-like boundaries
            - mean_pause_ms: average pause duration (target: 300-600ms)
    """
    import soundfile as sf

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    # Detect silence regions (below -40 dBFS)
    threshold_db = -40
    threshold_amp = 10.0 ** (threshold_db / 20.0)

    # RMS in small windows
    window_ms = 30  # 30ms windows
    window_samples = int(sr * window_ms / 1000.0)
    n_windows = len(audio) // window_samples

    is_silent = np.zeros(n_windows, dtype=bool)
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        rms = np.sqrt(np.mean(audio[start:end] ** 2))
        peak = np.max(np.abs(audio[start:end]))
        is_silent[i] = peak < threshold_amp

    # Find contiguous silent regions
    pause_durations_ms = []
    in_silence = False
    silence_start = 0
    for i in range(n_windows):
        if is_silent[i] and not in_silence:
            in_silence = True
            silence_start = i
        elif not is_silent[i] and in_silence:
            in_silence = False
            dur_ms = (i - silence_start) * window_ms
            # Only count pauses > 100ms (ignore tiny gaps)
            if dur_ms >= 100:
                pause_durations_ms.append(dur_ms)
    # Handle trailing silence
    if in_silence:
        dur_ms = (n_windows - silence_start) * window_ms
        if dur_ms >= 100:
            pause_durations_ms.append(dur_ms)

    total_samples = len(audio)
    silent_samples = int(np.sum(is_silent)) * window_samples
    silence_ratio = silent_samples / total_samples if total_samples > 0 else 0.0

    mean_pause_ms = float(np.mean(pause_durations_ms)) if pause_durations_ms else 0.0

    # Check if pauses are at reasonable boundaries (evenly-ish spaced)
    # A simple heuristic: pauses should be at intervals of 2-10 seconds
    pauses_at_boundaries = False
    if len(pause_durations_ms) >= 2:
        total_dur_ms = total_samples / sr * 1000
        expected_interval_ms = total_dur_ms / (len(pause_durations_ms) + 1)
        pauses_at_boundaries = 2000 <= expected_interval_ms <= 10000

    return {
        "silence_ratio": round(float(silence_ratio), 4),
        "pause_durations_ms": pause_durations_ms,
        "pauses_at_boundaries": pauses_at_boundaries,
        "mean_pause_ms": round(mean_pause_ms, 1),
    }


def find_sentence_boundaries(
    audio: np.ndarray,
    sr: int,
    threshold_db: float = -40,
    min_silence_ms: int = 200,
) -> list[int]:
    """Find likely sentence boundaries in audio by detecting silence gaps.

    Args:
        audio: Audio array (1D).
        sr: Sample rate.
        threshold_db: Threshold in dB below which audio is considered silent.
        min_silence_ms: Minimum silence duration to count as a boundary.

    Returns:
        List of sample indices where sentences likely end (start of silence gap).
    """
    threshold_amp = 10.0 ** (threshold_db / 20.0)
    min_silence_samples = int(sr * min_silence_ms / 1000.0)

    # Compute envelope using absolute value smoothed over small windows
    window_ms = 10
    window_samples = int(sr * window_ms / 1000.0)
    n_windows = len(audio) // window_samples

    is_silent = np.zeros(n_windows, dtype=bool)
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples
        peak = np.max(np.abs(audio[start:end]))
        is_silent[i] = peak < threshold_amp

    # Find silence gaps >= min_silence_ms
    boundaries = []
    in_silence = False
    silence_start = 0
    for i in range(n_windows):
        if is_silent[i] and not in_silence:
            in_silence = True
            silence_start = i * window_samples
        elif not is_silent[i] and in_silence:
            silence_end = i * window_samples
            silence_dur = silence_end - silence_start
            if silence_dur >= min_silence_samples:
                # Boundary is the midpoint of the silence gap
                boundary = silence_start + silence_dur // 2
                boundaries.append(boundary)
            in_silence = False

    return boundaries


if __name__ == "__main__":
    # Quick self-test
    print("=== split_thai_sentences test ===")
    test_text = "สวัสดีครับ วันนี้อากาศดีมาก แต่เหนื่อยเหมือนกัน"
    sentences = split_thai_sentences(test_text)
    print(f"Input: {test_text}")
    print(f"Sentences ({len(sentences)}): {sentences}")
    assert len(sentences) == 3, f"Expected 3 sentences, got {len(sentences)}: {sentences}"
    print("PASS\n")

    print("=== evaluate_breathing test on ref_best.wav ===")
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_best.wav")
    if os.path.exists(ref_path):
        result = evaluate_breathing(ref_path)
        print(f"  silence_ratio: {result['silence_ratio']}")
        print(f"  mean_pause_ms: {result['mean_pause_ms']}")
        print(f"  pause_count: {len(result['pause_durations_ms'])}")
        print(f"  pauses_at_boundaries: {result['pauses_at_boundaries']}")
        print("PASS")
    else:
        print(f"  SKIP: {ref_path} not found")
    print()

    print("=== merge_wavs crossfade test ===")
    import soundfile as sf
    import tempfile

    # Create two short sine-wave test segments
    dur_s = 0.5
    test_sr = 24000
    crossfade_s = 0.05
    gap_s = 0.2
    t = np.linspace(0, dur_s, int(test_sr * dur_s), endpoint=False)
    seg1 = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float64)
    seg2 = (np.sin(2 * np.pi * 660 * t) * 0.3).astype(np.float64)

    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = os.path.join(tmpdir, "seg1.wav")
        p2 = os.path.join(tmpdir, "seg2.wav")
        out = os.path.join(tmpdir, "merged.wav")

        sf.write(p1, seg1, test_sr)
        sf.write(p2, seg2, test_sr)

        # Import the fixed merge_wavs
        from generate_full_script import merge_wavs
        merge_wavs([p1, p2], out, gap_s=gap_s, crossfade_s=crossfade_s)

        merged_audio, merged_sr = sf.read(out)
        print(f"  Merged shape: {merged_audio.shape}, sr: {merged_sr}")
        print(f"  Expected duration: ~{(dur_s * 2 - 0.05):.2f}s, actual: {len(merged_audio) / merged_sr:.2f}s")
        assert merged_sr == test_sr, f"SR mismatch: {merged_sr} != {test_sr}"
        assert len(merged_audio) > 0, "Merged audio is empty"
        # 2 segments: seg1 + crossfaded seg2, no gap (gap only between non-last segments)
        # seg1=0.5s, crossfade=0.05s overlap, seg2 contributes 0.45s after overlap
        expected_samples = int((dur_s + dur_s - crossfade_s) * test_sr)
        actual_samples = len(merged_audio)
        print(f"  Samples: {actual_samples} (expected ~{expected_samples})")
        assert abs(actual_samples - expected_samples) < test_sr * 0.05, \
            f"Duration off: {actual_samples} vs expected ~{expected_samples}"
        print("PASS")
