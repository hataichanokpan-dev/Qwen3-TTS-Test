"""Benchmark chunk sizes for OmniVoice Thai TTS.

Tests chunk sizes 8, 10, 12, 15 Thai words and reports:
- Resemblyzer similarity scores (avg/min/max)
- Generation time
- Pass rate (sim >= 0.90)
- Punctuation pause test (comma/period as pause markers)

Usage:
    python benchmark_chunk_size.py
"""

import os
import sys
import time
import gc
import numpy as np
import soundfile as sf
import librosa
from pythainlp.tokenize import word_tokenize
from pythainlp.util import normalize as pythainlp_normalize

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# Constants (same as omnivoice_long.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_AUDIO = os.path.join(BASE_DIR, "ref_best.wav")
TARGET_SR = 24000
SIM_THRESHOLD = 0.90

# Test sentences (diverse lengths from omnivoice_long.py TEXT)
TEST_SENTENCES = [
    "วันนี้เราจะมาคุยกันถึงเรื่องที่น่าสนใจมาก "
    "เรื่องของปัญญาประดิษฐ์และผลกระทบที่มีต่อสังคมมนุษย์",
    "ถ้าย้อนกลับไปเมื่อสิบปีก่อน "
    "เราอาจไม่เคยคิดเลยว่าจะมีวันที่คอมพิวเตอร์สามารถพูดคุยกับเราได้เหมือนมนุษย์",
    "ลองมองดูสิครับว่าปัญญาประดิษฐ์ได้เข้ามาอยู่ในชีวิตประจำวันของเรามากขนาดไหน",
    "ในด้านการศึกษาเอไอก็เข้ามามีบทบาทสำคัญ",
    "ด้านการแพทย์ถือว่าเป็นอีกหนึ่งสาขาที่เอไอสร้างผลกระทบอย่างมหาศาล "
    "ระบบเอไอสามารถวิเคราะห์ภาพเอกซเรย์ได้แม่นยำไม่แพ้แพทย์ผู้เชี่ยวชาญ",
]

CHUNK_SIZES = [8, 10, 12, 15]
RETRIES = 3


def thai_word_count(text):
    """Count actual Thai words using pythainlp word_tokenize."""
    return len([t for t in word_tokenize(text, engine="newmm") if t.strip()])


def make_chunk(text, size):
    """Extract exactly `size` Thai words from text."""
    words = word_tokenize(text, engine="newmm")
    words = [w for w in words if w.strip()]
    return "".join(words[:size])


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def insert_test_pauses(chunk):
    """Insert comma/period for punctuation test."""
    words = word_tokenize(chunk, engine="newmm")
    words = [w for w in words if w.strip()]
    mid = len(words) // 2
    return "".join(words[:mid]) + ", " + "".join(words[mid:]) + "."


def normalize_rms(audio, target_rms):
    """Scale audio to match target RMS level."""
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms < 1e-8:
        return audio
    gain = target_rms / current_rms
    return audio * gain


def generate_chunk(model, text, ref_audio_path):
    """Generate audio for a chunk. Returns (audio, gen_time) or raises."""
    t0 = time.time()
    audio_list = model.generate(
        text=text,
        ref_audio=ref_audio_path,
        language_id="th",
        speed=0.9,
    )
    audio = audio_list[0]
    if audio.ndim > 1:
        audio = audio[:, 0]
    gen_time = time.time() - t0
    return audio, gen_time


def measure_similarity(encoder, ref_emb, audio_24k):
    """Measure Resemblyzer similarity against reference embedding."""
    from resemblyzer import preprocess_wav
    wav_16k = librosa.resample(audio_24k, orig_sr=TARGET_SR, target_sr=16000)
    chunk_emb = encoder.embed_utterance(preprocess_wav(wav_16k))
    return cosine_sim(ref_emb, chunk_emb)


def detect_pause(audio_24k, sr=TARGET_SR, min_pause_dur=0.15):
    """Detect if there's a noticeable pause (>min_pause_dur) in the audio.

    Uses RMS energy to find silence regions. Returns True if a pause
    longer than min_pause_dur is found in the middle 60% of the audio.
    """
    frame_length = int(sr * 0.02)  # 20ms frames
    hop_length = int(sr * 0.01)    # 10ms hop

    rms = librosa.feature.rms(y=audio_24k, frame_length=frame_length, hop_length=hop_length)[0]
    silence_threshold = np.mean(rms) * 0.1

    # Only check middle 60% of audio (skip start/end natural pauses)
    start_idx = int(len(rms) * 0.2)
    end_idx = int(len(rms) * 0.8)
    middle_rms = rms[start_idx:end_idx]

    # Find silent frames
    is_silent = middle_rms < silence_threshold

    # Find contiguous silent regions
    silent_regions = []
    in_silence = False
    silence_start = 0
    for i, s in enumerate(is_silent):
        if s and not in_silence:
            silence_start = i
            in_silence = True
        elif not s and in_silence:
            silent_regions.append((silence_start, i))
            in_silence = False
    if in_silence:
        silent_regions.append((silence_start, len(is_silent)))

    # Check if any silent region exceeds min_pause_dur
    hop_dur_sec = hop_length / sr
    for start, end in silent_regions:
        dur = (end - start) * hop_dur_sec
        if dur >= min_pause_dur:
            return True

    return False


def run_benchmark():
    from omnivoice import OmniVoice
    import torch
    from resemblyzer import VoiceEncoder, preprocess_wav

    print("=" * 65)
    print("OmniVoice Thai TTS Chunk Size Benchmark")
    print("=" * 65)
    print()

    # --- Load OmniVoice ---
    print("Loading OmniVoice...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda:0",
        dtype=torch.float16,
    )
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # --- Load Resemblyzer ---
    print("Loading Resemblyzer...")
    encoder = VoiceEncoder()
    ref_wav, ref_sr = sf.read(REF_AUDIO)
    if ref_wav.ndim > 1:
        ref_wav = ref_wav[:, 0]
    if ref_sr != 16000:
        ref_wav = librosa.resample(ref_wav, orig_sr=ref_sr, target_sr=16000)
    ref_emb = encoder.embed_utterance(preprocess_wav(ref_wav))

    # Compute ref RMS for normalization
    ref_audio_full, _ = sf.read(REF_AUDIO)
    if ref_audio_full.ndim > 1:
        ref_audio_full = ref_audio_full[:, 0]
    ref_rms = np.sqrt(np.mean(ref_audio_full ** 2))
    print(f"  Ref RMS: {ref_rms:.4f}")
    print()

    # --- Prepare test chunks ---
    print("Preparing test chunks...")
    test_chunks = {}  # chunk_size -> list of (text, source_idx)
    for size in CHUNK_SIZES:
        chunks_for_size = []
        for i, sent in enumerate(TEST_SENTENCES):
            wc = thai_word_count(sent)
            if wc >= size:
                chunk = make_chunk(sent, size)
                actual_wc = thai_word_count(chunk)
                chunks_for_size.append((chunk, i))
        test_chunks[size] = chunks_for_size
        print(f"  Size {size:2d} words: {len(chunks_for_size)} test sentences available")
    print()

    # --- Main benchmark: chunk sizes × sentences × retries ---
    print("Running benchmark...")
    print("-" * 65)

    results = {}  # size -> {sims: [], times: [], pass_count: int, total: int}

    for size in CHUNK_SIZES:
        chunks_for_size = test_chunks[size]
        if not chunks_for_size:
            print(f"\n  Chunk size {size}: no suitable test sentences, skipping")
            continue

        print(f"\n--- Chunk size: {size} words ---")
        size_sims = []
        size_times = []
        size_pass = 0
        size_total = 0

        for chunk_text, sent_idx in chunks_for_size:
            actual_wc = thai_word_count(chunk_text)
            print(f"  Sentence {sent_idx + 1} ({actual_wc} words): {chunk_text[:50]}...")

            for retry in range(RETRIES):
                try:
                    audio, gen_time = generate_chunk(model, chunk_text, REF_AUDIO)
                    sim = measure_similarity(encoder, ref_emb, audio)
                    dur = len(audio) / TARGET_SR
                    passed = sim >= SIM_THRESHOLD

                    size_sims.append(sim)
                    size_times.append(gen_time)
                    size_total += 1
                    if passed:
                        size_pass += 1

                    status = "PASS" if passed else "FAIL"
                    print(f"    retry {retry + 1}/{RETRIES}: "
                          f"sim={sim:.4f} time={gen_time:.1f}s dur={dur:.1f}s [{status}]")
                except Exception as e:
                    print(f"    retry {retry + 1}/{RETRIES}: ERROR - {e}")
                    size_total += 1

        results[size] = {
            "sims": size_sims,
            "times": size_times,
            "pass_count": size_pass,
            "total": size_total,
        }

    # --- Punctuation test ---
    print("\n" + "=" * 65)
    print("Punctuation Test")
    print("=" * 65)

    # Use a sentence that has enough words for chunk size 10
    punc_size = 10
    punc_sent_idx = None
    punc_plain_text = None
    for i, sent in enumerate(TEST_SENTENCES):
        if thai_word_count(sent) >= punc_size:
            punc_plain_text = make_chunk(sent, punc_size)
            punc_sent_idx = i
            break

    punc_results = {"plain": [], "with_punc": []}

    if punc_plain_text:
        punc_with_pauses = insert_test_pauses(punc_plain_text)
        print(f"  Sentence {punc_sent_idx + 1} ({punc_size} words)")
        print(f"  Plain:       {punc_plain_text[:60]}...")
        print(f"  With pauses: {punc_with_pauses[:60]}...")

        for label, text in [("plain", punc_plain_text), ("with_punc", punc_with_pauses)]:
            print(f"\n  Testing [{label}]:")
            for retry in range(RETRIES):
                try:
                    audio, gen_time = generate_chunk(model, text, REF_AUDIO)
                    sim = measure_similarity(encoder, ref_emb, audio)
                    dur = len(audio) / TARGET_SR
                    has_pause = detect_pause(audio)

                    punc_results[label].append({
                        "sim": sim,
                        "time": gen_time,
                        "dur": dur,
                        "pause": has_pause,
                    })
                    print(f"    retry {retry + 1}/{RETRIES}: "
                          f"sim={sim:.4f} time={gen_time:.1f}s dur={dur:.1f}s pause={has_pause}")
                except Exception as e:
                    print(f"    retry {retry + 1}/{RETRIES}: ERROR - {e}")
    else:
        print("  No suitable sentence for punctuation test")

    # --- Print results table ---
    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)
    print()
    print(f"{'Chunk Size':>10} | {'Avg Sim':>7} | {'Min Sim':>7} | {'Max Sim':>7} | "
          f"{'Avg Time':>8} | {'Pass Rate':>14}")
    print("-" * 10 + "-+-" + "-" * 7 + "-+-" + "-" * 7 + "-+-" + "-" * 7 + "-+-" +
          "-" * 8 + "-+-" + "-" * 14)

    for size in CHUNK_SIZES:
        if size not in results or not results[size]["sims"]:
            print(f"{size:>10} | {'N/A':>7} | {'N/A':>7} | {'N/A':>7} | "
                  f"{'N/A':>8} | {'N/A':>14}")
            continue

        r = results[size]
        sims = r["sims"]
        times = r["times"]
        avg_sim = np.mean(sims)
        min_sim = np.min(sims)
        max_sim = np.max(sims)
        avg_time = np.mean(times)
        pass_rate = f"{r['pass_count']}/{r['total']} ({r['pass_count']/max(r['total'],1)*100:.0f}%)"

        print(f"{size:>10} | {avg_sim:>7.3f} | {min_sim:>7.3f} | {max_sim:>7.3f} | "
              f"{avg_time:>7.1f}s | {pass_rate:>14}")

    # Punctuation test summary
    if punc_results["plain"] and punc_results["with_punc"]:
        print()
        print("Punctuation Test:")
        plain_sims = [r["sim"] for r in punc_results["plain"]]
        plain_times = [r["time"] for r in punc_results["plain"]]
        punc_sims = [r["sim"] for r in punc_results["with_punc"]]
        punc_times = [r["time"] for r in punc_results["with_punc"]]
        punc_pauses = [r["pause"] for r in punc_results["with_punc"]]
        plain_pauses = [r["pause"] for r in punc_results["plain"]]

        print(f"  Plain chunk:          sim={np.mean(plain_sims):.3f}  "
              f"time={np.mean(plain_times):.1f}s  "
              f"pause_detected={any(plain_pauses)}")
        print(f"  With comma/period:    sim={np.mean(punc_sims):.3f}  "
              f"time={np.mean(punc_times):.1f}s  "
              f"pause_detected={any(punc_pauses)}")
        print()
        if any(punc_pauses) and not any(plain_pauses):
            print("  -> OmniVoice interprets punctuation as pauses")
        elif any(punc_pauses) and any(plain_pauses):
            print("  -> Pauses detected in both (punctuation may not be the cause)")
        else:
            print("  -> No significant pauses detected with punctuation")

    # Cleanup
    print()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Benchmark complete. GPU memory freed.")


if __name__ == "__main__":
    run_benchmark()
