# F5-TTS-THAI-TEST — Thai Voice Cloning

> Thai TTS voice cloning project. Uses OmniVoice engine + Resemblyzer quality gate.

## Core Stack

| Component | What | Why |
|-----------|------|-----|
| **OmniVoice** (`k2-fsa/OmniVoice`) | TTS engine with voice cloning | Best Thai quality so far, supports `language_id="th"` |
| **Resemblyzer** | Speaker embedding (d-vector) | Accurate voice similarity measurement — matches human perception |
| **faster-whisper** (medium) | ASR for QA transcription | Verify pronunciation accuracy (CER) |
| **ref_best.wav** | Reference voice template | 15s, 24kHz — single speaker for all generation |

## Engine Status

| Engine | Status | Notes |
|--------|--------|-------|
| OmniVoice (`k2-fsa/OmniVoice`) | ✅ Working | Primary engine. ~3.1GB model in HF cache |
| Qwen3-TTS (`VoiceCloner` wrapper) | ❌ Broken | `pad_token_id` AttributeError in `Qwen3TTSTalkerConfig` |
| F5-TTS-TH-V2 (`VIZINTZOR/F5-TTS-TH-V2`) | ⏳ Not tested yet | Cached, may be better for Thai specifically |

## Architecture

```
clone_voice.py          CLI — Qwen3-TTS ICL mode (currently broken)
omnivoice_long.py       OmniVoice long-form generation (chunk + merge)
compare_engines.py      A/B test OmniVoice vs real_engine
real_engine.py          VoiceCloner wrapper → engine_fn interface
voice_qa.py             QA gate (needs Resemblyzer upgrade — see below)
voice_clone/
  cloner.py             VoiceCloner class (Qwen3-TTS)
  config.py             Model config
  text_preprocess.py    Thai text preprocessing
  audio_utils.py        Audio utilities
```

## OmniVoice: What Works

```python
from omnivoice import OmniVoice
import torch

model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
audio_list = model.generate(text=text, ref_audio="ref_best.wav", language_id="th")
audio = audio_list[0]  # numpy array, 24kHz
```

### Critical: Chunk Size

**Max 8 space-separated tokens per chunk** (~20s audio). Larger chunks cause:
- Hallucination (English/Chinese words mixed in)
- 30s+ silent gaps in output
- Pronunciation degradation

Chunk sizes tested:
| Tokens | Audio/chunk | Quality |
|--------|-------------|---------|
| 60 | ~100s | ❌ Heavy hallucination, 30s gaps |
| 35 | ~100s | ❌ Still broken |
| 15 | ~30-40s | ⚠️ Borderline, some fail |
| **8** | **~20s** | ✅ **Stable, passes quality gate** |

### Critical: Greeting Avoidance

**Never start text with greeting-style phrases** like "สวัสดีครับทุกคน". This causes consistent voice drift (sim 0.78-0.86). Start with content directly.

### Quality Gate Pipeline

Every chunk goes through: generate → Resemblyzer similarity → retry if < 0.90 → adaptive split if still failing.

```
chunk(8 tok) → generate → sim ≥ 0.90? → PASS → normalize → merge
                                  ↓ < 0.90
                              retry (max 5)
                                  ↓ still fail
                              split in half → retry each
```

- Threshold: 0.90 (Resemblyzer d-vector cosine similarity)
- Max retries: 5
- Adaptive split: halve failed chunks and retry each half
- RMS normalization: match ref_best.wav (0.090)
- Merge: crossfade 0.3s (no silence gaps)

### Generation Stats

- Load time: ~5s
- RTF: 0.67 (with quality gate overhead)
- Model size: 3,116 MB (2 files: 2,337 MB + 768 MB)
- 8/8 chunks pass at 8-token size

## Resemblyzer: Speaker Similarity

**This is the ground truth for voice quality — not MFCC.**

```python
from resemblyzer import VoiceEncoder, preprocess_wav

encoder = VoiceEncoder()
emb = encoder.embed_utterance(preprocess_wav(audio_16kHz))
sim = cosine_similarity(ref_emb, chunk_emb)
```

### Proven Results

Test with ref_best.wav + 5 chunks of 500-word narration:

| Chunk | Resemblyzer | MFCC | Human ear |
|-------|-------------|------|-----------|
| 5 | **0.945** | 0.990 | ✅ |
| 4 | **0.936** | 0.995 | ✅ Only one that sounds right |
| 1 | 0.905 | 0.957 | OK |
| 2 | 0.890 | 0.984 | OK |
| **3** | **0.765** | 0.984 | **❌ Worst** |

**Key insight:** MFCC spread = 0.04 (says everything's fine) vs Resemblyzer spread = 0.18 (correctly flags chunk 3). MFCC cannot be trusted for voice similarity.

### Quality Thresholds

| Score | Meaning |
|-------|---------|
| ≥ 0.90 | Good match with reference |
| 0.80-0.89 | Acceptable, slight drift |
| < 0.80 | Poor — should regenerate |

## DONE: voice_qa.py Upgrade ✅

`voice_qa.py` now uses **Resemblyzer d-vector** for `check_speaker_similarity()` (threshold 0.90). Previously used MFCC which gave false confidence (0.98 for chunks that sounded nothing like the reference).

## Environment

- Python 3.12 on Windows 11
- CUDA GPU required (model loads on `cuda:0`)
- HuggingFace cache: `~/.cache/huggingface/hub/`
- Output sample rate: 24kHz

## Known Issues

- **SoX not installed** — resampling falls back to librosa (slower)
- **flash-attn not installed** — OmniVoice uses manual PyTorch (slower but works)
- **Symlink warnings** — Windows Developer Mode not enabled, HF cache degraded
- **Qwen3-TTS broken** — `Qwen3TTSTalkerConfig` missing `pad_token_id` — version mismatch between `qwen_tts` package and `transformers`
