"""Quick engine comparison: OmniVoiceEngine vs real_engine (VoiceCloner wrapper).

Generates the same Thai text with both engines, compares:
  - Speed (RTF, generation time)
  - Audio quality (via Whisper transcription)
  - Audio metrics (duration, silence ratio)

Usage:
    python compare_engines.py
"""

import os
import sys
import time
import gc
import numpy as np
import soundfile as sf
import importlib.util

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

OUTPUT_DIR = os.path.join(BASE_DIR, "compare_engines_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 24000

TEST_TEXT = "สวัสดีครับทุกคน วันนี้เราจะมาคุยกันเรื่องที่น่าสนใจมาก เรื่องของปัญญาประดิษฐ์ หรือที่เรารู้จักกันในชื่อ AI ซึ่งในปัจจุบันนี้ AI ได้เข้ามามีบทบาทสำคัญในชีวิตประจำวันของเรามากขึ้นเรื่อยๆ ลองคิดดูสิครับ ตอนเช้าที่เราตื่นนอน สิ่งแรกที่หลายคนทำคือเปิดมือถือ"

REF_AUDIO = os.path.join(BASE_DIR, "ref_best.wav")


def load_omnivoice():
    """Load OmniVoiceEngine from .pyc."""
    spec = importlib.util.spec_from_file_location(
        "omnivoice_engine",
        os.path.join(BASE_DIR, "__pycache__", "omnivoice_engine.cpython-312.pyc"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.OmniVoiceEngine


def test_omnivoice():
    """Test OmniVoice (official API)."""
    print("=" * 60)
    print("ENGINE 1: OmniVoice (official API from k2-fsa)")
    print("=" * 60)

    from omnivoice import OmniVoice
    import torch

    print("Loading model from HuggingFace...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda:0",
        dtype=torch.float16,
    )
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")

    # Generate (voice cloning mode)
    print(f"\nGenerating: '{TEST_TEXT[:60]}...'")
    t0 = time.time()
    audio_list = model.generate(
        text=TEST_TEXT,
        ref_audio=REF_AUDIO,
        language_id="th",
    )
    gen_time = time.time() - t0
    audio = audio_list[0]
    sr = 24000
    duration = len(audio) / sr
    rtf = gen_time / duration

    print(f"Output: {len(audio)} samples, {sr}Hz, {duration:.2f}s")
    print(f"Generation time: {gen_time:.1f}s")
    print(f"RTF: {rtf:.3f} ({1/rtf:.1f}x realtime)")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "omnivoice.wav")
    if audio.ndim > 1:
        audio = audio[:, 0]
    sf.write(out_path, audio, sr)
    print(f"Saved: {out_path}")

    result = {
        "engine": "OmniVoice (k2-fsa)",
        "load_time": load_time,
        "gen_time": gen_time,
        "duration": duration,
        "rtf": rtf,
        "sr": sr,
        "samples": len(audio),
    }

    print("Unloading...")
    del model
    gc.collect()
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result, out_path


def test_real_engine():
    """Test real_engine (VoiceCloner wrapper)."""
    print("\n" + "=" * 60)
    print("ENGINE 2: real_engine (VoiceCloner wrapper)")
    print("=" * 60)

    from real_engine import create_real_engine

    engine = create_real_engine(ref_audio=REF_AUDIO)
    print("Creating engine (lazy load)...")

    # Generate (triggers model load on first call)
    print(f"\nGenerating: '{TEST_TEXT[:60]}...'")
    t0 = time.time()
    audio, sr = engine(TEST_TEXT)
    total_time = time.time() - t0
    duration = len(audio) / sr
    rtf = total_time / duration

    print(f"Output: {len(audio)} samples, {sr}Hz, {duration:.2f}s")
    print(f"Total time (incl. load): {total_time:.1f}s")
    print(f"RTF: {rtf:.3f} ({1/rtf:.1f}x realtime)")

    # Second generation (model already loaded)
    print(f"\nGenerating again (model cached)...")
    t0 = time.time()
    audio2, sr2 = engine(TEST_TEXT)
    gen_time = time.time() - t0
    duration2 = len(audio2) / sr2
    rtf2 = gen_time / duration2
    print(f"Generation time (cached): {gen_time:.1f}s")
    print(f"RTF (cached): {rtf2:.3f} ({1/rtf2:.1f}x realtime)")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "real_engine.wav")
    sf.write(out_path, audio, sr)
    print(f"Saved: {out_path}")

    result = {
        "engine": "real_engine (VoiceCloner)",
        "total_time_incl_load": total_time,
        "gen_time_cached": gen_time,
        "duration": duration,
        "rtf_first": rtf,
        "rtf_cached": rtf2,
        "sr": sr,
        "samples": len(audio),
    }

    print("Unloading...")
    engine.unload()
    gc.collect()

    return result, out_path


def transcribe(path, label):
    """Transcribe with faster-whisper (CPU to avoid GPU contention)."""
    from faster_whisper import WhisperModel

    print(f"\nTranscribing {label} (CPU mode)...")
    t0 = time.time()
    model = WhisperModel("medium", device="cpu", compute_type="int8")
    segments, info = model.transcribe(path, language="th", beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    elapsed = time.time() - t0
    print(f"  Transcribed in {elapsed:.1f}s")
    print(f"  Preview: {text[:80]}...")
    return text


def main():
    print("ENGINE COMPARISON TEST")
    print(f"Text: {len(TEST_TEXT.split())} space-separated tokens")
    print(f"Ref audio: {REF_AUDIO}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Phase 1: Generate with both engines (sequential to save VRAM)
    omni_result, omni_path = test_omnivoice()
    real_result, real_path = test_real_engine()

    # Phase 2: Whisper transcription (CPU, no GPU contention)
    print("\n" + "=" * 60)
    print("WHISPER QA (CPU mode)")
    print("=" * 60)

    omni_transcription = transcribe(omni_path, "OmniVoiceEngine")
    real_transcription = transcribe(real_path, "real_engine")

    # Compute CER
    def normalize_thai(text):
        import re
        text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"[,.!?;:ฯ]", "", text)
        return text

    def cer(ref, hyp):
        ref, hyp = normalize_thai(ref), normalize_thai(hyp)
        if not ref:
            return 0.0
        n, m = len(ref), len(hyp)
        prev = list(range(m + 1))
        for i in range(1, n + 1):
            curr = [i] + [0] * m
            for j in range(1, m + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            prev = curr
        return prev[m] / n

    omni_cer = cer(TEST_TEXT, omni_transcription)
    real_cer = cer(TEST_TEXT, real_transcription)

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':30s} {'OmniVoice':>15s} {'real_engine':>15s}")
    print("-" * 60)
    print(f"{'Duration':30s} {omni_result['duration']:>14.2f}s {real_result['duration']:>14.2f}s")
    print(f"{'RTF (first call)':30s} {omni_result['rtf']:>15.3f} {real_result['rtf_first']:>15.3f}")
    print(f"{'RTF (cached)':30s} {'N/A':>15s} {real_result['rtf_cached']:>15.3f}")
    print(f"{'CER (vs original)':30s} {omni_cer:>15.4f} {real_cer:>15.4f}")
    print(f"{'Sample rate':30s} {omni_result['sr']:>15d} {real_result['sr']:>15d}")

    # Write report
    report_path = os.path.join(OUTPUT_DIR, "comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Engine Comparison Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Text:** {TEST_TEXT[:80]}...\n\n")
        f.write("| Metric | OmniVoiceEngine | real_engine (VoiceCloner) |\n")
        f.write("|--------|----------------|--------------------------|\n")
        f.write(f"| Duration | {omni_result['duration']:.2f}s | {real_result['duration']:.2f}s |\n")
        f.write(f"| RTF (first) | {omni_result['rtf']:.3f} | {real_result['rtf_first']:.3f} |\n")
        f.write(f"| RTF (cached) | N/A | {real_result['rtf_cached']:.3f} |\n")
        f.write(f"| CER | {omni_cer:.4f} | {real_cer:.4f} |\n")
        f.write(f"| Sample rate | {omni_result['sr']} | {real_result['sr']} |\n\n")
        f.write("## Transcriptions\n\n")
        f.write(f"**OmniVoice:** {omni_transcription[:300]}\n\n")
        f.write(f"**real_engine:** {real_transcription[:300]}\n")

    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
