# F5-TTS Thai Voice Clone (Qwen3-TTS)

Thai voice cloning using Qwen3-TTS following official guide best practices.

## Flow ที่ถูกต้อง (Official Guide)

```
1. เตรียม Reference Audio (15s)
   ├── ตัด segment ที่ดีที่สุดจาก source audio ยาว
   └── ใช้ find_best_segment() เพื่อหาช่วงที่มีคุณภาพเสียงดีที่สุด

2. Transcribe Reference Audio → ref_text
   └── ใช้ faster-whisper transcribe เป็นข้อความไทย (จำเป็นสำหรับ ICL mode)

3. สร้าง Voice Clone Prompt (ครั้งเดียว)
   ├── model.create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False)
   └── prompt นี้ reuse สำหรับทุกประโยค → เสียงสม่ำเสมอ

4. Generate ทีละ section
   ├── model.generate_voice_clone(text, voice_clone_prompt=prompt, ...)
   ├── แบ่งข้อความยาวเป็น chunk ย่อย (ไม่เกิน ~200 ตัวอักษร) ป้องกัน hallucination
   └── text preprocessing: แปลงคำอังกฤษเป็นไทยก่อน TTS
```

## โหมด Clone

| โหมด | x_vector_only_mode | ref_text | คุณภาพ | ความเร็ว | ความเสถียร |
|------|-------------------|----------|---------|----------|------------|
| **ICL (แนะนำ)** | False | จำเป็น | สูง | ช้า | ต้องแบ่ง chunk |
| xvec | True | ไม่จำเป็น | ต่ำกว่า | เร็ว 3-5x | เสถียรมาก |

## สิ่งสำคัญที่ต้องรู้

- **Thai ไม่อยู่ใน 10 ภาษาที่รองรับ** (CN/EN/JA/KO/DE/FR/RU/PT/ES/IT) → ใช้ `language="Auto"`
- **คำอังกฤษต้องแปลงเป็นไทยก่อน** → "AI" → "เอไอ", "chatbot" → "แชทบอท"
- **Reference audio ไม่เกิน 15 วินาที** → เกินนี้จะช้าและอาจ hallucination
- **ข้อความยาวต้องแบ่ง chunk** → ไม่เกิน ~200 ตัวอักษรต่อครั้ง ป้องกันออกเสียงยาวเกิน

## การติดตั้ง

```bash
pip install qwen-tts faster-whisper soundfile torch
```

## การใช้งาน

### CLI

```bash
# พื้นฐาน - ICL mode (แนะนำ)
python clone_voice.py "สวัสดีครับ" -o output.wav

# ระบุ ref audio + ref text เอง
python clone_voice.py "ข้อความ" --ref ref.wav --ref-text transcript.txt

# เตรียม reference จาก audio ยาว
python clone_voice.py --prepare long_audio.wav

# xvec mode (เร็วกว่า เสียงเหมือนน้อยกว่า)
python clone_voice.py "ข้อความ" --xvec
```

### Python API

```python
from voice_clone import VoiceCloner

# สร้าง cloner (ICL mode - ต้องมี ref_text)
cloner = VoiceCloner(
    ref_audio="ref_best.wav",
    ref_text="ref_best_text.txt",  # หรือสตริงข้อความโดยตรง
)

# สร้าง reusable prompt (ครั้งเดียว)
cloner.build_prompt()

# Generate ทีละ section - เสียงสม่ำเสมอ
audio, sr = cloner.clone("ข้อความที่จะพูด")
cloner.save(audio, sr, "output.wav")

# Batch generate
results = cloner.clone_batch(["ประโยค 1", "ประโยค 2"])

# Cleanup GPU memory
cloner.unload_model()
```

### รันเต็มรูปแบบ

```bash
python test_full_script.py
```

## โครงสร้างโปรเจค

```
├── voice_clone/
│   ├── __init__.py          # Package exports
│   ├── cloner.py            # VoiceCloner class (ICL mode)
│   ├── config.py            # Default config
│   ├── audio_utils.py       # Segment search, similarity metrics
│   └── text_preprocess.py   # English→Thai transliteration
├── clone_voice.py           # CLI entry point
├── test_full_script.py      # Full script test
├── ref_best.wav             # Best 15s reference segment
├── ref_best_text.txt        # Whisper transcription of ref_best.wav
├── ref_full.wav             # Full source audio
└── script-v3.json           # Test script
```

## Config

```python
DEFAULTS = {
    "model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "device": "cuda:0",
    "dtype": "bfloat16",
    "language": "Auto",
    "x_vector_only_mode": False,  # ICL mode (recommended)
    "temperature": 0.8,
    "top_p": 0.9,
    "repetition_penalty": 1.05,
}
```

## Hardware

- GPU: NVIDIA RTX 3070 8GB
- ICL mode: ~8-11 min per section (~200 chars)
- xvec mode: ~2-3 min per section (~200 chars)
- Model: Qwen3-TTS-12Hz-1.7B-Base (~3.4GB VRAM)
