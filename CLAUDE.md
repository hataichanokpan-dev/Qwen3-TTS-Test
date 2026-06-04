# F5-TTS-THAI-TEST — Thai Voice Cloning

> Thai TTS voice cloning project. Uses OmniVoice engine + Resemblyzer quality gate.

## Core Stack

| Component | What | Why |
|-----------|------|-----|
| **OmniVoice** (`k2-fsa/OmniVoice`) | TTS engine with voice cloning | Best Thai quality so far, supports `language_id="th"` |
| **Resemblyzer** | Speaker embedding (d-vector) | Accurate voice similarity measurement — matches human perception |
| **faster-whisper** (medium) | ASR for QA transcription | Verify pronunciation accuracy (CER) |
| **pythainlp** | Thai NLP (tokenize, normalize, num-to-words) | Accurate word counting, sentence chunking, text normalization |
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
omnivoice_long.py       OmniVoice long-form generation (chunk + merge + QA gate)
  --dry-run             Preview chunks without loading model
  --chunk-size N        Override default 15 Thai words per chunk
  --crossfade N          Crossfade duration (default 0.3s, equal-power cos/sin)
  --preprocess          Enable English→Thai + number conversion
  --text "..."          Custom text input
  --output-dir          Custom output directory
benchmark_chunk_size.py Benchmark script for finding optimal chunk size
compare_engines.py      A/B test OmniVoice vs real_engine
real_engine.py          VoiceCloner wrapper → engine_fn interface
voice_qa.py             QA gate (uses Resemblyzer d-vector)
voice_clone/
  cloner.py             VoiceCloner class (Qwen3-TTS)
  config.py             Model config
  text_preprocess.py    Thai text preprocessing (English→Thai, numbers→words)
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

**Default 15 Thai words per chunk** (benchmark-verified optimal). Configurable via `--chunk-size`.
Larger chunks (>15) untested. Smaller chunks (8-12) have LOWER pass rates and slower generation.
Audio rate: ~3-5s per 15-word chunk (~196 words/min speaking rate).

Chunk sizes benchmarked (2026-06-04, 5 sentences × 3 retries, Resemblyzer sim ≥ 0.90):
| Size | Avg Sim | Min Sim | Pass Rate | Avg Gen Time | Audio dur |
|------|---------|---------|-----------|-------------|-----------|
| 8 words | 0.874 | 0.767 | 27% (4/15) | 21.0s | ~2.4s |
| 10 words | 0.894 | 0.838 | 58% (7/12) | 11.3s | ~2.7s |
| 12 words | 0.900 | 0.861 | 33% (4/12) | 4.5s | ~3.0s |
| **15 words** | **0.928** | **0.899** | **92% (11/12)** | **4.5s** | **~4.5s** |

Key finding: 15 words is optimal — highest sim, highest pass rate, fastest generation.
Sentence content matters more than chunk size (some sentences consistently fail across all sizes).

### Critical: Greeting Avoidance

**Never start text with greeting-style phrases** like "สวัสดีครับทุกคน". This causes consistent voice drift (sim 0.78-0.86). Start with content directly.

### Quality Gate Pipeline

Every chunk goes through: generate → Resemblyzer similarity → retry if < 0.90 → adaptive split if still failing.

```
chunk(15 tok) → insert_breathing_pauses → generate → sim ≥ 0.90? → PASS → normalize → merge
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

- Load time: ~5.8s
- RTF: ~0.67 (with quality gate overhead)
- Model size: 3,116 MB (2 files: 2,337 MB + 768 MB)
- Benchmark (2026-06-04): CHUNK_SIZE=15 → 92% pass rate, avg sim 0.928, avg gen 4.5s/chunk

### Breathing Pauses

`insert_breathing_pauses()` inserts commas at clause boundaries (post-chunk operation).
**Benchmark finding:** OmniVoice does NOT interpret comma/period as pauses in Thai text.
Pauses don't appear but also don't hurt quality — feature is kept as harmless.
Text content affects quality more than punctuation.

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
