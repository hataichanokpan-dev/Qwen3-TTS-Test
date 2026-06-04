"""Generate long-form Thai narration with OmniVoice + Resemblyzer QA gate.

Chunks text into configurable Thai word counts (default 15), generates each
with quality gate: generate → Resemblyzer similarity → retry if < 0.90.
Includes post-chunk breathing pause insertion and equal-power crossfade merge.

Usage:
    python omnivoice_long.py                    # default run
    python omnivoice_long.py --dry-run          # preview chunks without GPU
    python omnivoice_long.py --chunk-size 10    # custom chunk size
    python omnivoice_long.py --preprocess       # enable text preprocessing
    python omnivoice_long.py --text "ทดสอบ"     # custom text
"""

import os
import sys
import time
import gc
import argparse
import numpy as np
import soundfile as sf
import librosa
from pythainlp.tokenize import word_tokenize, sent_tokenize
from pythainlp.util import normalize as pythainlp_normalize

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "omnivoice_long_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

REF_AUDIO = os.path.join(BASE_DIR, "ref_best.wav")
TARGET_SR = 24000
SIM_THRESHOLD = 0.90
MAX_RETRIES = 5
CHUNK_SIZE = 15  # Thai words (~3-5s audio per chunk, benchmark-verified optimal at 92% pass rate)

# ~500-word Thai narration about AI and society
# Note: avoid greeting-style opening ("สวัสดีครับทุกคน") which causes voice drift
TEXT = (
    "วันนี้เราจะมาคุยกันถึงเรื่องที่น่าสนใจมาก "
    "เรื่องของปัญญาประดิษฐ์และผลกระทบที่มีต่อสังคมมนุษย์ในปัจจุบันและอนาคต "
    "หลายคนอาจสงสัยว่าปัญญาประดิษฐ์คืออะไรกันแน่ "
    "ในความเป็นจริงแล้วปัญญาประดิษฐ์หรือเอไอคือเทคโนโลยีที่พยายามจำลองความคิดและการตัดสินใจของมนุษย์ "
    "ผ่านระบบคอมพิวเตอร์ที่สามารถเรียนรู้และปรับปรุงตัวเองได้"

    " ถ้าย้อนกลับไปเมื่อสิบปีก่อน "
    "เราอาจไม่เคยคิดเลยว่าจะมีวันที่คอมพิวเตอร์สามารถพูดคุยกับเราได้เหมือนมนุษย์ "
    "สร้างภาพศิลปะได้สวยงาม "
    "หรือแม้กระทั่งแต่งเพลงได้ด้วยตัวเอง "
    "แต่วันนี้สิ่งเหล่านี้เกิดขึ้นจริงแล้ว "
    "และเกิดขึ้นอย่างรวดเร็วจนหลายคนตามไม่ทัน"

    " ลองมองดูสิครับว่าปัญญาประดิษฐ์ได้เข้ามาอยู่ในชีวิตประจำวันของเรามากขนาดไหน "
    "ตั้งแต่ตื่นนอนเช้ามืด "
    "โทรศัพท์มือถือของเราก็ใช้เอไอในการแนะนำข่าวที่น่าสนใจ "
    "แอปพลิเคชันนำทางใช้เอไอคำนวณเส้นทางที่ดีที่สุด "
    "ร้านค้าออนไลน์ใช้เอไอแนะนำสินค้าที่เหมาะกับเรา "
    "แม้แต่ธนาคารก็ใช้เอไอตรวจสอบการทุจริตในระบบ"

    " ในด้านการศึกษาเอไอก็เข้ามามีบทบาทสำคัญ "
    "นักเรียนสามารถเรียนรู้จากระบบที่ปรับเนื้อหาให้เหมาะกับความสามารถของแต่ละคน "
    "ครูอาจารย์สามารถใช้เอไอช่วยสร้างสื่อการสอน "
    "วิเคราะห์จุดอ่อนของนักเรียน "
    "และออกแบบกิจกรรมที่เหมาะสมได้ "
    "แต่ในขณะเดียวกันก็มีความกังวลว่า "
    "นักเรียนอาจพึ่งพาเอไอมากเกินไปจนไม่ฝึกทักษะการคิดวิเคราะห์ด้วยตัวเอง"

    " ด้านการแพทย์ถือว่าเป็นอีกหนึ่งสาขาที่เอไอสร้างผลกระทบอย่างมหาศาล "
    "ระบบเอไอสามารถวิเคราะห์ภาพเอกซเรย์ได้แม่นยำไม่แพ้แพทย์ผู้เชี่ยวชาญ "
    "ช่วยค้นพบโรคในระยะเริ่มต้นที่มนุษย์อาจมองข้าม "
    "การพัฒนายาใหม่ก็เร็วขึ้นมากเมื่อใช้เอไอช่วยคำนวณและจำลองสถานการณ์ "
    "ทำให้ผู้ป่วยได้รับการรักษาที่รวดเร็วและมีประสิทธิภาพมากขึ้น"

    " อย่างไรก็ตามสิ่งที่ต้องคิดอย่างจริงจังคือผลกระทบต่อตลาดแรงงาน "
    "อาชีพจำนวนมากอาจถูกแทนที่ด้วยระบบอัตโนมัติ "
    "พนักงานขับรถอาจต้องแข่งขันกับรถยนต์ไร้คนขับ "
    "พนักงานธนาคารอาจถูกแทนด้วยระบบการเงินดิจิทัล "
    "แม้แต่นักข่าวและนักเขียนบางส่วนก็ต้องปรับตัวเมื่อเอไอสามารถเขียนบทความได้ดีขึ้นเรื่อยๆ"

    " แต่เราไม่ควรมองเอไอในแง่ร้ายอย่างเดียว "
    "เพราะทุกครั้งที่เทคโนโลยีใหม่เกิดขึ้น "
    "อาชีพใหม่ๆ ก็เกิดตามมาเช่นกัน "
    "เราต้องปรับตัวโดยการพัฒนาทักษะที่เอไอทำแทนเราไม่ได้ "
    "เช่น ความคิดสร้างสรรค์ การทำงานเป็นทีม ความเข้าใจในอารมณ์ของมนุษย์ "
    "และการแก้ปัญหาที่ซับซ้อน"

    " ในอนาคตเอไอจะยิ่งก้าวหน้ามากขึ้นเรื่อยๆ "
    "เราอาจเห็นระบบที่สามารถคิดค้นนวัตกรรมใหม่ๆ ได้ด้วยตัวเอง "
    "หุ่นยนต์ที่ช่วยเหลือผู้สูงอายุในบ้าน "
    "หรือระบบขนส่งที่ฉลาดพอที่จะลดการจราจรติดขัดลงได้อย่างมีนัยสำคัญ "
    "แต่สิ่งสำคัญที่สุดคือเราต้องใช้เอไออย่างมีจริยธรรม "
    "ไม่ใช้ไปในทางที่ทำร้ายผู้อื่นหรือสังคม"

    " สรุปได้ว่าปัญญาประดิษฐ์เป็นดาบสองคม "
    "ถ้าเราใช้อย่างถูกต้องก็จะนำพามนุษยชาติไปสู่การพัฒนาที่ยิ่งใหญ่ "
    "แต่ถ้าใช้ผิดวิธีก็อาจสร้างความเสียหายได้เช่นกัน "
    "ดังนั้นการเรียนรู้และเข้าใจเอไอจึงเป็นสิ่งจำเป็นสำหรับทุกคนในยุคนี้ "
    "ไม่ว่าจะเป็นเด็กนักเรียน นักศึกษา วัยทำงาน หรือผู้สูงอายุ "
    "เพราะเอไอจะอยู่กับเราไปอีกนาน "
    "และอนาคตของเราขึ้นอยู่กับว่าเราจะใช้มันอย่างไร "
    "ขอบคุณที่ฟังกันนะครับ แล้วพบกันใหม่ครับ"
)


def thai_word_count(text):
    """Count actual Thai words using pythainlp word_tokenize (filters whitespace tokens)."""
    return len([t for t in word_tokenize(text, engine="newmm") if t.strip()])


# Thai clause boundaries and transition words that naturally precede a pause
_PAUSE_AFTER_WORDS = {
    "แล้ว", "ดังนั้น", "ดังนั้นจึง", "เพราะ", "เพราะว่า",
    "ถ้า", "หาก", "แต่", "อย่างไรก็ตาม", "ในทางกลับกัน",
    "เช่น", "ตัวอย่างเช่น", "กล่าวคือ", "กล่าวคือว่า",
    "โดยเฉพาะ", "ในขณะเดียวกัน", "นอกจากนี้",
    "เช่นเดียวกัน", "ในทำนองเดียวกัน", "กล่าวโดยสรุป",
    "สรุปแล้ว", "สรุปได้ว่า", "เป็นที่น่าสังเกตว่า",
}

# Conjunctions that join clauses — insert comma before them when between clauses
_PAUSE_BEFORE_WORDS = {
    "และ", "หรือ", "แต่ว่า", "เมื่อ", "ขณะที่", "ในขณะที่",
}


def insert_breathing_pauses(chunk_text):
    """Insert comma/period as breathing pause markers in a single chunk.

    Post-chunk operation: called AFTER chunk_text(), BEFORE model.generate().
    Uses universal punctuation that TTS engines interpret as pauses.

    Rules:
    - Insert comma after clause boundary words (แล้ว, ดังนั้น, เพราะ, etc.)
    - Insert comma before conjunctions when they join clauses
    - Do NOT insert duplicate markers where punctuation already exists
    - Keep markers minimal — at most 1-2 per chunk
    """
    words = word_tokenize(chunk_text, engine="newmm")
    words = [w for w in words if w.strip()]

    if len(words) <= 4:
        return chunk_text  # Too short for pause insertion

    result = []
    pauses_inserted = 0
    max_pauses = max(1, len(words) // 5)  # ~1 pause per 5 words

    for i, w in enumerate(words):
        # Skip if previous token is already punctuation
        if result and result[-1].rstrip() in {",", ".", "ๆ", "ฯ"}:
            result.append(w)
            continue

        # Insert comma after clause boundary words
        if pauses_inserted < max_pauses and w in _PAUSE_AFTER_WORDS:
            result.append(w)
            # Check next word isn't punctuation
            if i + 1 < len(words) and words[i + 1].strip() not in {",", ".", "ๆ", "ฯ", ""}:
                result.append(", ")
                pauses_inserted += 1
            continue

        # Insert comma before conjunctions (when not at start)
        if pauses_inserted < max_pauses and i > 0 and w in _PAUSE_BEFORE_WORDS:
            # Only if previous word is content (not another conjunction/punctuation)
            prev = result[-1].rstrip() if result else ""
            if prev not in {",", ".", "ๆ", "ฯ", ""} and prev not in _PAUSE_BEFORE_WORDS:
                result.append(", ")
                result.append(w)
                pauses_inserted += 1
                continue

        result.append(w)

    text = "".join(result)
    # Clean up any double punctuation
    text = text.replace(", , ", ", ").replace(", .", ".").replace("..", ".")
    return text


def chunk_text(text, max_words=CHUNK_SIZE):
    """Split Thai text into chunks by sentence boundaries.

    Uses pythainlp sent_tokenize for boundary detection, then groups
    sentences into chunks up to max_words real Thai words each.
    Long sentences that exceed max_words are split by word boundaries.
    """
    sentences = sent_tokenize(text)
    chunks = []
    current_sentences = []
    current_word_count = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        sent_words = thai_word_count(sent)

        # If single sentence exceeds max_words, split it by words
        if sent_words > max_words:
            # Flush current chunk first
            if current_sentences:
                chunks.append("".join(current_sentences))
                current_sentences = []
                current_word_count = 0
            # Split long sentence into sub-chunks by word count
            words = word_tokenize(sent, engine="newmm")
            sub = []
            sub_count = 0
            for w in words:
                sub.append(w)
                sub_count += 1
                if sub_count >= max_words:
                    chunks.append("".join(sub))
                    sub = []
                    sub_count = 0
            if sub:
                # Append remainder to current
                current_sentences = ["".join(sub)]
                current_word_count = len(sub)
            continue

        if current_word_count + sent_words > max_words and current_sentences:
            chunks.append("".join(current_sentences))
            current_sentences = [sent]
            current_word_count = sent_words
        else:
            current_sentences.append(sent)
            current_word_count += sent_words

    if current_sentences:
        chunks.append("".join(current_sentences))

    return chunks


def normalize_rms(audio, target_rms):
    """Scale audio to match target RMS level."""
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms < 1e-8:
        return audio
    gain = target_rms / current_rms
    return audio * gain


def crossfade_merge(segments, crossfade_s=0.3):
    """Merge audio segments with crossfade (no silence gaps)."""
    cf_samples = int(crossfade_s * TARGET_SR)
    merged = segments[0]

    for seg in segments[1:]:
        if cf_samples > 0 and len(merged) >= cf_samples and len(seg) >= cf_samples:
            # Equal-power crossfade: cos²(x) + sin²(x) = 1 → constant energy
            t = np.linspace(0, np.pi / 2, cf_samples)
            fade_out = np.cos(t)
            fade_in = np.sin(t)
            overlap = merged[-cf_samples:] * fade_out + seg[:cf_samples] * fade_in
            merged = np.concatenate([merged[:-cf_samples], overlap, seg[cf_samples:]])
        else:
            merged = np.concatenate([merged, seg])

    return merged


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def parse_args():
    """Parse CLI arguments for omnivoice_long.py."""
    parser = argparse.ArgumentParser(
        description="Generate long-form Thai narration with OmniVoice + QA gate",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Preview chunks and pipeline without loading model or generating audio",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CHUNK_SIZE,
        help=f"Max Thai words per chunk (default: {CHUNK_SIZE})",
    )
    parser.add_argument(
        "--crossfade", type=float, default=0.3,
        help="Crossfade duration in seconds (default: 0.3)",
    )
    parser.add_argument(
        "--preprocess", action="store_true",
        help="Enable full text preprocessing (English→Thai, numbers→words) before chunking",
    )
    parser.add_argument(
        "--text", type=str, default=None,
        help="Custom text to generate (overrides built-in TEXT)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Custom output directory (default: omnivoice_long_output/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Select text source
    raw_text = args.text if args.text else TEXT

    # Optional full preprocessing (English→Thai, numbers→words)
    if args.preprocess:
        from voice_clone.text_preprocess import preprocess_thai_text
        raw_text = preprocess_thai_text(raw_text)

    # Normalize text with pythainlp before processing
    normalized_text = pythainlp_normalize(raw_text)

    word_count = thai_word_count(normalized_text)
    space_tokens = len(normalized_text.split())
    print(f"Text: {word_count} Thai words ({space_tokens} space-separated tokens)")

    chunks = chunk_text(normalized_text, max_words=args.chunk_size)
    print(f"Chunks: {len(chunks)} (max {args.chunk_size} Thai words each)")
    for i, c in enumerate(chunks):
        wc = thai_word_count(c)
        paused = insert_breathing_pauses(c)
        print(f"  [{i+1}] {wc} words: {c[:60]}...")
        if paused != c:
            print(f"       +pauses: {paused[:60]}...")
    print()

    # Dry-run: print chunks and exit without loading model
    if args.dry_run:
        print("[DRY RUN] Preview complete. No audio generated.")
        return

    from omnivoice import OmniVoice
    import torch
    from resemblyzer import VoiceEncoder, preprocess_wav

    # Resolve output directory
    out_dir = args.output_dir if args.output_dir else OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Load OmniVoice
    print("Loading OmniVoice...")
    t0 = time.time()
    model = OmniVoice.from_pretrained(
        "k2-fsa/OmniVoice",
        device_map="cuda:0",
        dtype=torch.float16,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s")

    # Load Resemblyzer for quality gate
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
    print(f"Ref RMS: {ref_rms:.4f}")
    print()

    # Generate with quality gate + adaptive chunk splitting
    segments = []
    total_gen = 0
    results = []

    def generate_with_gate(chunk_text, chunk_id, total, retries=MAX_RETRIES):
        """Generate a chunk with quality gate retries. Returns (audio, sim, passed)."""
        best_audio = None
        best_sim = -1

        for attempt in range(1, retries + 1):
            wc = thai_word_count(chunk_text)
            print(f"  [{chunk_id}/{total}] ({wc} words) attempt {attempt}/{retries}...", end=" ")

            t0 = time.time()
            audio_list = model.generate(
                text=chunk_text,
                ref_audio=REF_AUDIO,
                language_id="th",
                speed=0.9,
            )
            audio = audio_list[0]
            if audio.ndim > 1:
                audio = audio[:, 0]
            gen_time = time.time() - t0

            wav_16k = librosa.resample(audio, orig_sr=TARGET_SR, target_sr=16000)
            chunk_emb = encoder.embed_utterance(preprocess_wav(wav_16k))
            sim = cosine_sim(ref_emb, chunk_emb)

            dur = len(audio) / TARGET_SR
            passed = sim >= SIM_THRESHOLD
            status = "PASS" if passed else "FAIL"
            print(f"{dur:.1f}s gen:{gen_time:.1f}s sim:{sim:.4f} [{status}]")

            nonlocal total_gen
            total_gen += gen_time

            if sim > best_sim:
                best_sim = sim
                best_audio = audio

            if passed:
                return best_audio, best_sim, True

        return best_audio, best_sim, False

    # Process each chunk — insert breathing pauses, then generate with quality gate
    for i, chunk in enumerate(chunks):
        paused_chunk = insert_breathing_pauses(chunk)
        audio, sim, passed = generate_with_gate(paused_chunk, i+1, len(chunks))

        if not passed and thai_word_count(chunk) > 4:
            # Adaptive split: chunk failed, try splitting in half by sentences
            sents = sent_tokenize(chunk)
            mid = len(sents) // 2
            if mid > 0:
                half1 = " ".join(sents[:mid])
                half2 = " ".join(sents[mid:])
            else:
                # Single sentence, split by words
                words = word_tokenize(chunk, engine="newmm")
                mid = len(words) // 2
                half1 = "".join(words[:mid])
                half2 = "".join(words[mid:])
            print(f"  >> Chunk {i+1} failed ({sim:.4f}), splitting into {thai_word_count(half1)}+{thai_word_count(half2)} words")

            audio1, sim1, p1 = generate_with_gate(half1, i+1, len(chunks))
            audio2, sim2, p2 = generate_with_gate(half2, i+1, len(chunks))

            # Use split results if better
            combined_sim = (sim1 + sim2) / 2
            if combined_sim > sim:
                audio = np.concatenate([audio1, audio2])
                sim = combined_sim
                passed = p1 and p2
                print(f"  >> Split result: sim1={sim1:.4f} sim2={sim2:.4f} avg={combined_sim:.4f}")

        # Normalize RMS to match reference
        audio = normalize_rms(audio, ref_rms)
        segments.append(audio)
        results.append({"chunk": i+1, "sim": sim, "passed": sim >= SIM_THRESHOLD})

        status = "PASS" if sim >= SIM_THRESHOLD else "FAIL (best effort)"
        print(f"  >> Chunk {i+1} final: sim={sim:.4f} [{status}]")

        # Save chunk
        chunk_path = os.path.join(out_dir, f"chunk_{i+1:02d}.wav")
        sf.write(chunk_path, audio, TARGET_SR)

    # Merge with equal-power crossfade
    print("\nMerging with equal-power crossfade...")
    merged = crossfade_merge(segments, crossfade_s=args.crossfade)
    total_dur = len(merged) / TARGET_SR
    out_path = os.path.join(out_dir, "full_500words.wav")
    sf.write(out_path, merged, TARGET_SR)

    # Summary
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    print(f"\n{'='*55}")
    print(f"Chunk size: {args.chunk_size} words | Crossfade: {args.crossfade}s")
    print(f"Chunks:    {len(chunks)} ({passed} passed, {failed} failed)")
    print(f"Duration:  {total_dur:.1f}s ({total_dur/60:.1f} min)")
    print(f"Gen time:  {total_gen:.1f}s ({total_gen/60:.1f} min)")
    print(f"Avg RTF:   {total_gen/total_dur:.3f}")
    print(f"Preprocess: {'ON' if args.preprocess else 'OFF'}")
    print(f"Output:    {out_path}")
    print()
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  Chunk {r['chunk']:2d}: sim={r['sim']:.4f} [{status}]")

    # Cleanup
    del model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
