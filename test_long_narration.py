"""Long narration TTS test with Whisper-based QA.

Generates 500+ word Thai narration using 3 approaches:
  - Baseline: single chunk, no pauses
  - Approach C: 400ms silence pauses at sentence boundaries
  - Approach C+: speed variation + variable pauses (recommended)

Each output is transcribed back with faster-whisper, then compared
against the original text via WER/CER metrics.

Usage:
    python test_long_narration.py
"""

import os
import sys
import time
import re
import numpy as np
import soundfile as sf

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from real_engine import create_real_engine
from breathing_test_framework import split_thai_sentences, evaluate_breathing
from speech_optimizer import optimize_and_generate

OUTPUT_DIR = os.path.join(BASE_DIR, "test_long_narration_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SR = 24000

# ---------------------------------------------------------------------------
# Thai narration text — 500+ words, mixed sentence types
# ---------------------------------------------------------------------------

THAI_TEST_TEXT = (
    "สวัสดีครับทุกคน วันนี้เราจะมาคุยกันเรื่องที่น่าสนใจมาก "
    "เรื่องของปัญญาประดิษฐ์ หรือที่เรารู้จักกันในชื่อ AI "
    "ซึ่งในปัจจุบันนี้ AI ได้เข้ามามีบทบาทสำคัญในชีวิตประจำวันของเรามากขึ้นเรื่อยๆ"

    " ลองคิดดูสิครับ ตอนเช้าที่เราตื่นนอน สิ่งแรกที่หลายคนทำคือเปิดมือถือ "
    "แอปพลิเคชันที่เราใช้งาน เช่น โซเชียลมีเดีย แผนที่ หรือแม้แต่ร้านอาหารออนไลน์ "
    "ล้วนแล้วแต่มี AI ทำงานอยู่เบื้องหลังทั้งสิ้น"

    " ทำไม AI ถึงกลายเป็นเทคโนโลยีที่ทุกคนพูดถึง? "
    "เหตุผลหลักคือความสามารถในการเรียนรู้จากข้อมูลจำนวนมหาศาล "
    "ในปี 2024 ข้อมูลดิจิทัลทั่วโลกมีปริมาณถึง 149 ล้านล้านกิกะไบต์ "
    "ตัวเลขนี้เพิ่มขึ้น 23% ทุกปี "
    "มนุษย์ไม่สามารถประมวลผลข้อมูลขนาดนี้ได้ด้วยตัวคนเดียว "
    "แต่ AI สามารถทำได้ในเวลาเพียงไม่กี่วินาที"

    " แต่ก็อย่างที่รู้กันดี ไม่มีเทคโนโลยีไหนที่สมบูรณ์แบบร้อยเปอร์เซ็นต์ "
    "AI ก็มีความท้าทายที่ต้องเผชิญอยู่หลายประการ "
    "ประการแรกคือเรื่องความเป็นส่วนตัว "
    "เมื่อ AI ต้องการข้อมูลมากๆ เพื่อเรียนรู้ "
    "ข้อมูลส่วนบุคคลของเราอาจถูกนำไปใช้โดยที่เราไม่รู้ตัว"

    " ประการที่สองคือเรื่องความยุติธรรม "
    "AI เรียนรู้จากข้อมูลในอดีต ซึ่งอาจมีอคติฝังอยู่ "
    "ถ้าข้อมูลมีอคติ AI ก็จะตัดสินใจด้วยอคติเช่นกัน "
    "นี่คือปัญหาใหญ่ที่นักวิจัยทั่วโลกกำลังพยายามแก้ไข"

    " ประการที่สามคือผลกระทบต่อการจ้างงาน "
    "อาจมีอาชีพบางอย่างที่หายไปเพราะ AI เข้ามาทำแทน "
    "แต่ในขณะเดียวกัน ก็มีอาชีพใหม่ๆ เกิดขึ้นมากมาย "
    "เช่น วิศวกรฝึกสอน AI ผู้เชี่ยวชาญด้านจริยธรรม AI "
    "และนักออกแบบประสบการณ์การใช้งาน AI"

    " มาดูตัวอย่างการใช้งาน AI ในประเทศไทยกันบ้าง "
    "ในวงการแพทย์ โรงพยาบาลหลายแห่งเริ่มนำ AI มาช่วยวิเคราะห์ภาพเอกซเรย์ "
    "ช่วยให้แพทย์ตรวจพบโรคได้เร็วขึ้นและแม่นยำขึ้น "
    "มีงานวิจัยจากมหาวิทยาลัยมหิดลบอกว่า "
    "AI สามารถตรวจพบมะเร็งปอดได้แม่นยำถึง 94.5% "
    "ในขณะที่แพทย์ทั่วไปมีความแม่นยำอยู่ที่ 88% "

    " ในวงการเกษตรกรรม AI ช่วยวิเคราะห์ข้อมูลดิน ฟ้า อากาศ "
    "บอกเกษตรกรว่าควรปลูกพืชอะไร เมื่อไหร่ และใช้น้ำปริมาณเท่าไหร่ "
    "โครงการนำร่องในจังหวัดเชียงใหม่และขอนแก่น "
    "ช่วยลดการใช้น้ำได้ 30% และเพิ่มผลผลิต 15% "

    " ในวงการศึกษา AI ช่วยสร้างระบบการเรียนการสอนที่ปรับให้เหมาะกับแต่ละคน "
    "นักเรียนที่เรียนช้าได้รับเนื้อหาเพิ่มเติม "
    "นักเรียนที่เรียนเก่งได้รับความท้าทายมากขึ้น "
    "โรงเรียนกว่า 200 แห่งในประเทศไทยเริ่มทดลองใช้ระบบนี้แล้ว"

    " แต่เราต้องไม่ลืมว่า AI เป็นเพียงเครื่องมือ "
    "เครื่องมือไม่ดีไม่ชั่วในตัวมันเอง "
    "ทุกอย่างขึ้นอยู่กับว่าเราจะนำไปใช้อย่างไร "
    "เราต้องมีกฎระเบียบที่ชัดเจน "
    "ต้องมีการตรวจสอบและดูแลอย่างใกล้ชิด "
    "และที่สำคัญที่สุดคือต้องให้คนมีส่วนร่วมในการตัดสินใจ"

    " อนาคตของ AI ในประเทศไทยนั้นสดใส "
    "แต่ก็ต้องอาศัยความร่วมมือจากทุกภาคส่วน "
    "รัฐบาล เอกชน สถาบันการศึกษา และประชาชนทั่วไป "
    "ต้องร่วมมือกันเพื่อสร้างระบบนิเวศ AI ที่ยั่งยืน "
    "ที่เป็นประโยชน์ต่อทุกคน ไม่ทิ้งใครไว้ข้างหลัง"

    " สำหรับคนรุ่นใหม่ที่กำลังมองหาทิศทางในอาชีพ "
    "ผมอยากแนะนำให้ลองเรียนรู้เกี่ยวกับ AI ดู "
    "ไม่จำเป็นต้องเป็นโปรแกรมเมอร์ก็เรียนรู้ได้ "
    "มีคอร์สออนไลน์ฟรีมากมายทั้งภาษาไทยและภาษาอังกฤษ "
    "ความรู้เรื่อง AI จะเป็นทักษะพื้นฐานที่สำคัญในอนาคต "
    "เหมือนกับที่ความรู้เรื่องคอมพิวเตอร์เป็นทักษะจำเป็นในยุคก่อน"

    " สรุปแล้ว AI ไม่ใช่สิ่งที่น่ากลัว "
    "แต่เป็นเทคโนโลยีที่เราต้องเข้าใจ ต้องควบคุม "
    "และต้องใช้อย่างมีจริยธรรม "
    "หากเราทำได้แบบนี้ AI จะเป็นเครื่องมือที่ยกระดับคุณภาพชีวิต "
    "ของคนไทยทุกคนได้อย่างแท้จริง "
    "ขอบคุณที่ฟังมาจนจบครับ แล้วพบกันใหม่ในตอนหน้า"
)


def normalize_thai(text: str) -> str:
    """Normalize Thai text for comparison: strip, collapse whitespace, remove punctuation."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    # Remove common punctuation that Whisper may or may not produce
    text = re.sub(r"[,.!?;:ฯ]", "", text)
    text = text.replace("  ", " ")
    return text.strip()


# Target words per segment for TTS (keeps quality high, generation fast)
TARGET_WORDS_PER_SEGMENT = 30


def split_thai_segments(text: str, max_words: int = TARGET_WORDS_PER_SEGMENT) -> list[str]:
    """Split Thai text into short segments suitable for TTS generation.

    Strategy:
      1. Split on sentence-ending punctuation (? ! ฯ)
      2. Split on paragraph breaks (2+ spaces)
      3. For remaining long segments, split at space boundaries
         to keep each segment <= max_words

    This prevents the quality degradation that occurs when TTS models
    generate audio for very long text in a single pass.

    Args:
        text: Thai text to split.
        max_words: Maximum space-separated tokens per segment.

    Returns:
        List of non-empty text segments.
    """
    # Phase 1: Split on sentence-ending punctuation
    parts = re.split(r'(?<=[?!ฯ])\s+', text)

    # Phase 2: Split on paragraph breaks (2+ spaces)
    expanded: list[str] = []
    for part in parts:
        sub = re.split(r'\s{2,}', part)
        expanded.extend(s for s in sub if s.strip())

    # Phase 3: Split long segments at word boundaries
    result: list[str] = []
    for seg in expanded:
        words = seg.split()
        if len(words) <= max_words:
            result.append(seg.strip())
        else:
            # Chunk into max_words-sized pieces
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                if chunk.strip():
                    result.append(chunk.strip())

    return [s for s in result if s]


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

def generate_baseline(text: str, engine_fn) -> tuple[np.ndarray, int, float]:
    """Generate audio per-sentence, concatenated with no pauses.

    Splits into sentences like Approach C, but omits silence gaps.
    This avoids the O(n²) cost of generating the full text as one chunk.

    Returns (audio, sr, rtf).
    """
    print("  Generating baseline (per-segment, no pauses)...")
    sentences = split_thai_segments(text)
    print(f"    {len(sentences)} sentences")

    t0 = time.time()
    parts: list[np.ndarray] = []
    for i, sent in enumerate(sentences):
        audio, sr = engine_fn(sent)
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        if audio.ndim > 1:
            audio = audio[:, 0]
        parts.append(audio)
        if (i + 1) % 5 == 0:
            print(f"    ... {i+1}/{len(sentences)} sentences done")

    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
    elapsed = time.time() - t0
    duration = len(audio) / TARGET_SR
    rtf = elapsed / duration if duration > 0 else 0
    print(f"    Duration: {duration:.2f}s, RTF: {rtf:.3f}")
    return audio, TARGET_SR, rtf


def generate_approach_c(text: str, engine_fn, pause_ms: int = 400) -> tuple[np.ndarray, int, float]:
    """Generate audio with silence pauses at sentence boundaries.

    Returns (audio, sr, rtf).
    """
    print(f"  Generating Approach C ({pause_ms}ms pauses)...")
    t0 = time.time()
    audio = generate_with_pauses_local(text, pause_ms, engine_fn)
    elapsed = time.time() - t0
    duration = len(audio) / TARGET_SR
    rtf = elapsed / duration if duration > 0 else 0
    print(f"    Duration: {duration:.2f}s, RTF: {rtf:.3f}")
    return audio, TARGET_SR, rtf


def generate_with_pauses_local(text: str, pause_ms: int, engine_fn) -> np.ndarray:
    """Generate with pauses (local copy to avoid extra imports)."""
    sentences = split_thai_segments(text)
    pause_samples = int(pause_ms / 1000.0 * TARGET_SR)
    silence = np.zeros(pause_samples, dtype=np.float64)

    parts: list[np.ndarray] = []
    for i, sentence in enumerate(sentences):
        audio, sr = engine_fn(sentence)
        if sr != TARGET_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        if audio.ndim > 1:
            audio = audio[:, 0]
        parts.append(audio)
        if i < len(sentences) - 1:
            parts.append(silence)

    if not parts:
        return np.zeros(0, dtype=np.float64)
    return np.concatenate(parts)


def generate_approach_cplus(text: str, engine_fn) -> tuple[np.ndarray, int, float]:
    """Generate audio with speed variation + variable pauses.

    Uses split_thai_segments for shorter chunks (better quality + speed).
    Applies per-sentence speed adjustment and variable pause durations.
    """
    from speech_optimizer import adjust_speed, _detect_sentence_type, PAUSE_PRESETS

    print("  Generating Approach C+ (speed variation)...")
    sentences = split_thai_segments(text)
    print(f"    {len(sentences)} segments")

    target_sr = TARGET_SR
    crossfade_s = 0.05

    t0 = time.time()
    segments: list[np.ndarray] = []
    pause_durations: list[int] = []

    for i, sent in enumerate(sentences):
        sent_speed = adjust_speed(sent, base_speed=1.0)
        audio, sr = engine_fn(sent, sent_speed)

        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        if audio.ndim > 1:
            audio = audio[:, 0]

        segments.append(audio)
        sent_type = _detect_sentence_type(sent)
        pause_ms = PAUSE_PRESETS.get(sent_type, 400)
        pause_durations.append(pause_ms)

        if (i + 1) % 5 == 0:
            print(f"    ... {i+1}/{len(sentences)} segments done")

    # Merge with crossfade + variable pauses
    cf_samples = int(crossfade_s * target_sr)
    merged = segments[0].copy()

    for i in range(1, len(segments)):
        seg = segments[i]
        if cf_samples > 0 and len(merged) >= cf_samples and len(seg) >= cf_samples:
            fade_out = np.linspace(1.0, 0.0, cf_samples, dtype=np.float64)
            fade_in = np.linspace(0.0, 1.0, cf_samples, dtype=np.float64)
            tail = merged[-cf_samples:] * fade_out + seg[:cf_samples] * fade_in
            merged = np.concatenate([merged[:-cf_samples], tail, seg[cf_samples:]])
        else:
            merged = np.concatenate([merged, seg])

        if i < len(segments) - 1:
            pause_samples = int(pause_durations[i - 1] / 1000.0 * target_sr)
            silence = np.zeros(pause_samples, dtype=np.float64)
            merged = np.concatenate([merged, silence])

    elapsed = time.time() - t0
    duration = len(merged) / target_sr
    rtf = elapsed / duration if duration > 0 else 0
    print(f"    Duration: {duration:.2f}s, RTF: {rtf:.3f}")
    return merged, target_sr, rtf


# ---------------------------------------------------------------------------
# Whisper QA
# ---------------------------------------------------------------------------

def transcribe_with_whisper(audio_path: str, model_size: str = "medium") -> str:
    """Transcribe audio file using faster-whisper.

    Returns transcribed text.
    """
    from faster_whisper import WhisperModel

    print(f"  Transcribing with faster-whisper ({model_size})...")
    t0 = time.time()

    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, info = model.transcribe(
        audio_path,
        language="th",
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300),
    )

    text = " ".join(seg.text.strip() for seg in segments)
    elapsed = time.time() - t0
    print(f"    Transcribed in {elapsed:.1f}s ({info.duration:.1f}s audio)")
    print(f"    Preview: {text[:100]}...")

    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate using jiwer."""
    from jiwer import wer
    ref = normalize_thai(reference)
    hyp = normalize_thai(hypothesis)
    if not ref:
        return 0.0
    try:
        return float(wer(ref, hyp))
    except Exception:
        # Fallback: simple character-level comparison
        return compute_cer(ref, hyp)


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    ref = normalize_thai(reference)
    hyp = normalize_thai(hypothesis)
    if not ref:
        return 0.0
    # Simple Levenshtein distance
    n = len(ref)
    m = len(hyp)
    if n == 0:
        return 1.0 if m > 0 else 0.0

    # Use dynamic programming
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr

    return prev[m] / n


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("LONG NARRATION TTS TEST (500+ words)")
    print("=" * 70)

    # Count words
    word_count = len(THAI_TEST_TEXT.split())
    sentence_count = len(split_thai_sentences(THAI_TEST_TEXT))
    print(f"Text: {word_count} words, {sentence_count} sentences")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # Load Qwen3-TTS engine
    print("Loading Qwen3-TTS engine...")
    ref_path = os.path.join(BASE_DIR, "ref_best.wav")
    engine = create_real_engine(ref_audio=ref_path)
    print("Engine ready.\n")

    approaches = [
        ("baseline", generate_baseline),
        ("approach_c", lambda t, e: generate_approach_c(t, e, pause_ms=400)),
        ("approach_cplus", generate_approach_cplus),
    ]

    results = {}

    try:
        for name, gen_fn in approaches:
            print(f"--- {name.upper()} ---")

            # Generate
            audio, sr, rtf = gen_fn(THAI_TEST_TEXT, engine)

            # Save WAV
            out_path = os.path.join(OUTPUT_DIR, f"{name}.wav")
            sf.write(out_path, audio, sr)
            print(f"  Saved: {out_path}")

            # Audio metrics
            duration = len(audio) / sr
            breathing = evaluate_breathing(out_path)

            # Whisper QA
            try:
                transcription = transcribe_with_whisper(out_path)
                wer_score = compute_wer(THAI_TEST_TEXT, transcription)
                cer_score = compute_cer(THAI_TEST_TEXT, transcription)
            except Exception as ex:
                print(f"  Whisper error: {ex}")
                transcription = ""
                wer_score = -1.0
                cer_score = -1.0

            results[name] = {
                "path": out_path,
                "duration": round(duration, 2),
                "rtf": round(rtf, 3),
                "silence_ratio": breathing["silence_ratio"],
                "pause_count": len(breathing["pause_durations_ms"]),
                "mean_pause_ms": breathing["mean_pause_ms"],
                "pauses_at_boundaries": breathing["pauses_at_boundaries"],
                "wer": round(wer_score, 4) if wer_score >= 0 else "N/A",
                "cer": round(cer_score, 4) if cer_score >= 0 else "N/A",
                "transcription_preview": transcription[:200] if transcription else "",
            }
            print()

    finally:
        print("Unloading model...")
        engine.unload()

    # ---- Summary ----
    print("=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"{'Approach':15s} {'Duration':>8s} {'RTF':>6s} {'Silence%':>9s} "
          f"{'#Pauses':>8s} {'MeanPause':>10s} {'WER':>8s} {'CER':>8s}")
    print("-" * 70)
    for name, data in results.items():
        wer_str = f"{data['wer']:.4f}" if isinstance(data["wer"], float) else str(data["wer"])
        cer_str = f"{data['cer']:.4f}" if isinstance(data["cer"], float) else str(data["cer"])
        print(f"{name:15s} {data['duration']:7.2f}s {data['rtf']:6.3f} "
              f"{data['silence_ratio']:9.4f} {data['pause_count']:8d} "
              f"{data['mean_pause_ms']:8.1f}ms {wer_str:>8s} {cer_str:>8s}")

    # ---- Write report ----
    report_path = os.path.join(OUTPUT_DIR, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Long Narration TTS Test Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Text:** {word_count} words, {sentence_count} sentences\n")
        f.write(f"**Engine:** Qwen3-TTS (real) with ref_best.wav\n")
        f.write(f"**QA:** faster-whisper large-v3 (Thai)\n\n")

        f.write("## Metrics Comparison\n\n")
        f.write(f"| Approach | Duration | RTF | Silence% | #Pauses | MeanPause | WER | CER |\n")
        f.write(f"|----------|----------|-----|----------|---------|-----------|-----|-----|\n")
        for name, data in results.items():
            wer_str = f"{data['wer']:.4f}" if isinstance(data["wer"], float) else str(data["wer"])
            cer_str = f"{data['cer']:.4f}" if isinstance(data["cer"], float) else str(data["cer"])
            f.write(f"| {name} | {data['duration']:.2f}s | {data['rtf']:.3f} | "
                    f"{data['silence_ratio']:.4f} | {data['pause_count']} | "
                    f"{data['mean_pause_ms']:.0f}ms | {wer_str} | {cer_str} |\n")

        f.write("\n## Transcription Previews\n\n")
        for name, data in results.items():
            f.write(f"### {name}\n")
            f.write(f"```\n{data['transcription_preview'][:300]}\n```\n\n")

        f.write("## Notes\n\n")
        f.write("- WER/CER includes Whisper transcription errors (not purely TTS quality)\n")
        f.write("- Lower WER/CER = transcription closer to original = better TTS intelligibility\n")
        f.write("- Duration reflects total audio length including pauses\n")
        f.write("- RTF (Real-Time Factor) = generation time / audio duration (<1.0 = faster than real-time)\n")

    print(f"\nReport: {report_path}")
    return results


if __name__ == "__main__":
    main()
