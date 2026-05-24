"""Audio analysis utilities for voice similarity measurement."""

import numpy as np
import soundfile as sf


def spectral_signature(audio, sr, n_seg=10):
    seg_len = len(audio) // n_seg
    if seg_len < 256:
        seg_len = min(256, len(audio))
        n_seg = len(audio) // seg_len
    features = []
    for i in range(n_seg):
        s = i * seg_len
        e = s + seg_len
        if e > len(audio):
            break
        fft = np.abs(np.fft.rfft(audio[s:e]))
        fft = fft / (np.max(fft) + 1e-10)
        features.append(fft)
    if not features:
        return None
    return np.mean(features, axis=0)


def cosine_similarity(path_a, path_b):
    """Spectral cosine similarity between two audio files."""
    a, sr_a = sf.read(path_a)
    b, sr_b = sf.read(path_b)
    if a.ndim > 1:
        a = a[:, 0]
    if b.ndim > 1:
        b = b[:, 0]
    if sr_a != sr_b:
        import librosa
        b = librosa.resample(b, orig_sr=sr_b, target_sr=sr_a)

    sig_a = spectral_signature(a, sr_a)
    sig_b = spectral_signature(b, sr_b)
    if sig_a is None or sig_b is None:
        return 0.0

    ml = min(len(sig_a), len(sig_b))
    sig_a, sig_b = sig_a[:ml], sig_b[:ml]
    return float(
        np.dot(sig_a, sig_b)
        / (np.linalg.norm(sig_a) * np.linalg.norm(sig_b) + 1e-10)
    )


def spectral_centroid(filepath):
    """Spectral centroid (brightness) in Hz."""
    audio, sr = sf.read(filepath)
    if audio.ndim > 1:
        audio = audio[:, 0]
    n_fft = min(4096, len(audio))
    fft_data = np.fft.rfft(audio[:n_fft])
    magnitude = np.abs(fft_data)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mag_norm = magnitude / (magnitude.sum() + 1e-10)
    return float(np.sum(freqs * mag_norm))


def find_best_segment(ref_full_path, segment_s=15, hop_s=10):
    """Cut long audio into overlapping segments, return paths sorted by quality.

    Returns list of (path, start_s, end_s) sorted by centroid stability.
    """
    import os, tempfile

    audio, sr = sf.read(ref_full_path)
    if audio.ndim > 1:
        audio = audio[:, 0]

    seg_samples = int(segment_s * sr)
    hop_samples = int(hop_s * sr)
    segments = []
    pos = 0
    idx = 0
    while pos + seg_samples <= len(audio):
        seg = audio[pos : pos + seg_samples]
        tmp = os.path.join(tempfile.gettempdir(), f"_seg_{idx}.wav")
        sf.write(tmp, seg, sr)
        segments.append(
            {"path": tmp, "start_s": pos / sr, "end_s": (pos + seg_samples) / sr}
        )
        pos += hop_samples
        idx += 1

    ref_cent = spectral_centroid(ref_full_path)
    for seg in segments:
        seg["centroid_diff"] = abs(spectral_centroid(seg["path"]) - ref_cent)

    segments.sort(key=lambda x: x["centroid_diff"])
    return segments
