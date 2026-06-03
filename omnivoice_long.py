"""Generate ~500-word Thai narration with OmniVoice + Resemblyzer QA gate.

Chunks text into 8-token pieces, generates each with quality gate:
  generate → measure speaker similarity → if < 0.90, retry (max 3 attempts)
Then normalizes RMS to match ref and merges with crossfade.

Usage:
    python omnivoice_long.py
"""

import os
import sys
import time
import gc
import numpy as np
import soundfile as sf
import librosa

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
CHUNK_SIZE = 8

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


def chunk_text(text, max_words=CHUNK_SIZE):
    """Split text into chunks by spaces."""
    words = text.split()
    chunks = []
    current = []

    for w in words:
        current.append(w)
        if len(current) >= max_words:
            chunks.append(" ".join(current))
            current = []

    if current:
        chunks.append(" ".join(current))

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
            fade_out = np.linspace(1.0, 0.0, cf_samples)
            fade_in = np.linspace(0.0, 1.0, cf_samples)
            overlap = merged[-cf_samples:] * fade_out + seg[:cf_samples] * fade_in
            merged = np.concatenate([merged[:-cf_samples], overlap, seg[cf_samples:]])
        else:
            merged = np.concatenate([merged, seg])

    return merged


def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    from omnivoice import OmniVoice
    import torch
    from resemblyzer import VoiceEncoder, preprocess_wav

    word_count = len(TEXT.split())
    print(f"Text: {word_count} space-separated tokens")

    chunks = chunk_text(TEXT)
    print(f"Chunks: {len(chunks)} (max {CHUNK_SIZE} tokens each)")
    print()

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
            print(f"  [{chunk_id}/{total}] ({len(chunk_text.split())} tok) attempt {attempt}/{retries}...", end=" ")

            t0 = time.time()
            audio_list = model.generate(
                text=chunk_text,
                ref_audio=REF_AUDIO,
                language_id="th",
                speed=0.8,
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

    # Process each chunk — if fails after retries, split in half and retry each half
    for i, chunk in enumerate(chunks):
        audio, sim, passed = generate_with_gate(chunk, i+1, len(chunks))

        if not passed and len(chunk.split()) > 4:
            # Adaptive split: chunk failed, try splitting in half
            words = chunk.split()
            mid = len(words) // 2
            half1 = " ".join(words[:mid])
            half2 = " ".join(words[mid:])
            print(f"  >> Chunk {i+1} failed ({sim:.4f}), splitting into {len(half1.split())}+{len(half2.split())} tokens")

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
        chunk_path = os.path.join(OUTPUT_DIR, f"chunk_{i+1:02d}.wav")
        sf.write(chunk_path, audio, TARGET_SR)

    # Merge with crossfade
    print("\nMerging with crossfade...")
    merged = crossfade_merge(segments, crossfade_s=0.3)
    total_dur = len(merged) / TARGET_SR
    out_path = os.path.join(OUTPUT_DIR, "full_500words.wav")
    sf.write(out_path, merged, TARGET_SR)

    # Summary
    passed = sum(1 for r in results if r["passed"])
    failed = len(results) - passed
    print(f"\n{'='*55}")
    print(f"Chunks:    {len(chunks)} ({passed} passed, {failed} failed)")
    print(f"Duration:  {total_dur:.1f}s ({total_dur/60:.1f} min)")
    print(f"Gen time:  {total_gen:.1f}s ({total_gen/60:.1f} min)")
    print(f"Avg RTF:   {total_gen/total_dur:.3f}")
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
