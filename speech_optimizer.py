"""Speech optimizer integrating auto_optimizer logic with breathing framework.

Provides:
- Thai sentence splitting and classification
- Per-sentence speed adjustment (questions -5%, exclamations +3%, data-heavy -5%)
- Inline non-verbal tag insertion at semantic positions
- Pause duration variation by sentence type
- Combined pipeline: split → classify → generate per-segment → merge with pauses

Recovered from auto_optimizer.cpython-312.pyc (bytecode analysis) and
integrated with breathing_test_framework.py for the natural speech pipeline.
"""

import os
import sys
import re
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from breathing_test_framework import split_thai_sentences, evaluate_breathing

# ---------------------------------------------------------------------------
# Constants (from decompiled auto_optimizer)
# ---------------------------------------------------------------------------

INLINE_TAGS = frozenset({
    "question-ei", "surprise-yo", "question-en", "surprise-oh",
    "dissatisfaction-hnn", "sigh", "question-yi", "question-ah",
    "surprise-ah", "confirmation-en", "laughter", "question-oh",
    "surprise-wa",
})

EXISTING_TAG_RE = re.compile(r"\[(" + "|".join(re.escape(t) for t in INLINE_TAGS) + r")\]")

# Thai keywords for sentence classification (from bytecode)
HUMOR_KEYWORDS = ("หลอน", "กินหิน", "น้ำยาฟอกขาว", "แต่งเอง")
SHOCK_KEYWORDS = ("เลวร้าย", "น่าตกใจ", "พัง", "ทิ้งโปรเจกต์")
NEGATIVE_KEYWORDS = ("แต่ไม่มี", "ไม่ได้", "ผิดพลาด")

# Pause duration presets by sentence type (milliseconds)
PAUSE_PRESETS = {
    "question": 550,       # Longer pause after questions
    "exclamation": 350,    # Shorter after exclamations (energetic)
    "statement": 400,      # Default
    "data_heavy": 500,     # Slightly longer for processing time
}


# ---------------------------------------------------------------------------
# Sentence classification (recovered from _classify_sentence bytecode)
# ---------------------------------------------------------------------------

def classify_sentence(sent: str, idx: int = 0, total: int = 1) -> str | None:
    """Classify a sentence to determine which non-verbal tag fits best.

    Returns tag name or None.

    Rules (recovered from bytecode):
    - Position 0 (first sentence) → confirmation-en
    - Contains '?' → question-ah
    - Contains humor keywords → laughter
    - Contains shock keywords → surprise-ah
    - Contains negative keywords → sigh
    - Position >= 3 → None (skip to avoid over-tagging)
    """
    if idx >= 3:
        return None

    # Check for question mark
    if "?" in sent or "?" in sent:
        return "question-ah"

    # Check humor keywords
    if any(kw in sent for kw in HUMOR_KEYWORDS):
        return "laughter"

    # Check shock keywords
    if any(kw in sent for kw in SHOCK_KEYWORDS):
        return "surprise-ah"

    # Check negative keywords
    if any(kw in sent for kw in NEGATIVE_KEYWORDS):
        return "sigh"

    # First sentence gets a confirmation tag
    if idx == 1 and total > 1:
        return "confirmation-en"

    return None


# ---------------------------------------------------------------------------
# Speed adjustment (recovered from adjust_speed bytecode)
# ---------------------------------------------------------------------------

def adjust_speed(text: str, base_speed: float = 1.0) -> float:
    """Adjust speed based on text characteristics.

    Rules (recovered from bytecode):
    - Questions → slower by 5%
    - Exclamations → slightly faster by 3%
    - Long sentences (>100 chars) → slightly faster by 3%
    - Numbers/data heavy → slower by 5%
    - Default → base_speed unchanged

    Clamped to [0.7, 1.3] range.
    """
    speed = base_speed
    adjustments = 0.0

    if "?" in text or "?" in text:
        adjustments += -0.05

    if "!" in text:
        adjustments += 0.03

    if len(text) > 100:
        adjustments += -0.03

    if _is_data_heavy(text):
        adjustments += -0.05

    speed = base_speed + adjustments
    # Clamp to reasonable range
    return max(0.7, min(1.3, speed))


def _is_data_heavy(text: str) -> bool:
    """Check if text contains lots of numbers/data.

    Rules (from bytecode): 3+ digits, or contains '%' / 'ล้าน' / 'บาท'.
    """
    numbers = re.findall(r"\d+", text)
    if len(numbers) >= 3:
        return True
    if "%" in text or "ล้าน" in text or "บาท" in text:
        return True
    return False


# ---------------------------------------------------------------------------
# Tag insertion (recovered from insert_tags bytecode)
# ---------------------------------------------------------------------------

def insert_tags(text: str) -> str:
    """Insert non-verbal inline tags at natural points in Thai narration text.

    Strategy (recovered from bytecode):
    - After shocking/surprising statements → [surprise-ah]
    - Before emotional transitions → [sigh]
    - After rhetorical questions → [question-ah]
    - After ironic/humorous statements → [laughter]
    - After confirmation points → [confirmation-en]

    Does NOT insert duplicates near existing tags (2-sentence window).
    """
    sentences = _split_sentences_pyc(text)
    if not sentences:
        return text

    result_parts: list[str] = []
    last_tag_idx = -10  # Ensure first tag can be placed

    for i, sent in enumerate(sentences):
        tag = classify_sentence(sent, i, len(sentences))
        if tag and (i - last_tag_idx) >= 2:
            # Insert tag after the sentence
            result_parts.append(sent.rstrip() + f" [{tag}]")
            last_tag_idx = i
        else:
            result_parts.append(sent)

    return " ".join(result_parts)


def _split_sentences_pyc(text: str) -> list[str]:
    """Split Thai text into sentences on common delimiters.

    Uses the same patterns recovered from the original _split_sentences bytecode:
    - Split after punctuation [.!?ฯ。] followed by whitespace
    """
    parts = re.split(r"(?<=[.!?ฯ。])\s+", text)
    result: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Further split on question marks
        sub = re.split(r"(?<=[\?])\s+", part)
        for p in sub:
            p = p.strip()
            if p:
                result.append(p)
    return result


# ---------------------------------------------------------------------------
# Full segment optimization (recovered from optimize_segment bytecode)
# ---------------------------------------------------------------------------

def optimize_segment(text: str, base_speed: float = 1.0, auto_tags: bool = True) -> dict:
    """Full optimization for a segment.

    Returns dict with:
        text: optimized text (with tags if enabled)
        speed: adjusted speed
        tags_added: list of tags that were inserted
    """
    tags_added: list[str] = []

    if auto_tags:
        original = text
        new_text = insert_tags(text)
        # Find newly added tags
        new_tags = set(EXISTING_TAG_RE.findall(new_text))
        old_tags = set(EXISTING_TAG_RE.findall(original))
        tags_added = list(new_tags - old_tags)
        text = new_text

    speed = adjust_speed(text, base_speed)

    return {
        "text": text,
        "speed": speed,
        "tags_added": tags_added,
    }


# ---------------------------------------------------------------------------
# Sentence type detection for pause variation
# ---------------------------------------------------------------------------

def _detect_sentence_type(sent: str) -> str:
    """Detect the type of a sentence for pause duration selection."""
    if "?" in sent or "?" in sent:
        return "question"
    if "!" in sent:
        return "exclamation"
    if _is_data_heavy(sent):
        return "data_heavy"
    return "statement"


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def optimize_and_generate(
    text: str,
    engine_fn,
    base_speed: float = 1.0,
    auto_tags: bool = False,
    default_pause_ms: int = 400,
    crossfade_s: float = 0.05,
) -> tuple[np.ndarray, int]:
    """Optimize text and generate audio with natural speech variation.

    Pipeline: split → classify → generate per-segment with speed variation →
              merge with variable pauses via crossfade.

    Args:
        text: Thai text to synthesize.
        engine_fn: Callable(text: str, speed: float) -> tuple(np.ndarray, sample_rate).
                   The speed parameter allows the engine to adjust speaking rate.
        base_speed: Base speaking speed multiplier (default 1.0).
        auto_tags: Whether to insert non-verbal tags (for future engine support).
        default_pause_ms: Default pause duration in ms between sentences.
        crossfade_s: Crossfade duration in seconds between segments.

    Returns:
        Tuple of (combined_audio: np.ndarray, sample_rate: int).
    """
    target_sr = 24000

    # Step 1: Optimize the text (tags + speed hints)
    optimized = optimize_segment(text, base_speed=base_speed, auto_tags=auto_tags)

    # Step 2: Split into sentences using breathing framework
    sentences = split_thai_sentences(optimized["text"])

    if not sentences:
        return np.zeros(0, dtype=np.float64), target_sr

    # Step 3: Generate each segment with per-sentence speed variation
    segments: list[np.ndarray] = []
    pause_durations: list[int] = []

    for i, sent in enumerate(sentences):
        # Compute per-sentence speed
        sent_speed = adjust_speed(sent, base_speed)

        # Generate audio via engine
        audio, sr = engine_fn(sent, sent_speed)

        # Resample if needed
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        if audio.ndim > 1:
            audio = audio[:, 0]

        segments.append(audio)

        # Determine pause duration based on sentence type
        sent_type = _detect_sentence_type(sent)
        pause_ms = PAUSE_PRESETS.get(sent_type, default_pause_ms)
        pause_durations.append(pause_ms)

    # Step 4: Merge segments with variable pauses and crossfade
    cf_samples = int(crossfade_s * target_sr)
    merged = segments[0].copy()

    for i in range(1, len(segments)):
        seg = segments[i]

        # Crossfade between segments
        if cf_samples > 0 and len(merged) >= cf_samples and len(seg) >= cf_samples:
            fade_out = np.linspace(1.0, 0.0, cf_samples, dtype=np.float64)
            fade_in = np.linspace(0.0, 1.0, cf_samples, dtype=np.float64)
            tail = merged[-cf_samples:] * fade_out + seg[:cf_samples] * fade_in
            merged = np.concatenate([merged[:-cf_samples], tail, seg[cf_samples:]])
        else:
            merged = np.concatenate([merged, seg])

        # Add variable pause (not after last segment)
        if i < len(segments) - 1:
            pause_samples = int(pause_durations[i - 1] / 1000.0 * target_sr)
            silence = np.zeros(pause_samples, dtype=np.float64)
            merged = np.concatenate([merged, silence])

    return merged, target_sr


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Sentence classification test ===")
    test_sentences = [
        "สวัสดีครับ วันนี้เราจะมาคุยกันเรื่องน่าสนใจ",
        "ทำไมถึงเป็นแบบนี้?",
        "เลวร้ายมากเลย",
        "หลอนจังเลย",
        "แต่ไม่มีใครช่วยเรา",
    ]
    for i, s in enumerate(test_sentences):
        tag = classify_sentence(s, i, len(test_sentences))
        stype = _detect_sentence_type(s)
        speed = adjust_speed(s)
        print(f"  [{i}] '{s[:40]}...' tag={tag}, type={stype}, speed={speed:.2f}")
    print()

    print("=== Speed adjustment test ===")
    cases = [
        ("คำถามธรรมดา", "Normal statement"),
        ("ทำไมเป็นแบบนี้?", "Question"),
        ("ว้าว! สุดยอด!", "Exclamation"),
        ("ตัวเลขมีมาก 100 200 300 400", "Data-heavy"),
        ("x" * 120, "Long sentence (>100 chars)"),
    ]
    for text, desc in cases:
        sp = adjust_speed(text)
        print(f"  {desc:30s} speed={sp:.2f}")
    print()

    print("=== Tag insertion test ===")
    sample = "สวัสดีครับ ทำไมถึงเป็นแบบนี้? เลวร้ายมากเลย หลอนจังเลย แต่ไม่มีใครช่วย"
    tagged = insert_tags(sample)
    print(f"  Input:  {sample}")
    print(f"  Output: {tagged}")
    tags_found = EXISTING_TAG_RE.findall(tagged)
    print(f"  Tags:   {tags_found}")
    assert len(tags_found) >= 1, "Expected at least 1 tag to be inserted"
    print("PASS")
    print()

    print("=== optimize_and_generate test ===")
    if "--real" in sys.argv:
        # Real Qwen3-TTS engine
        print("  Mode: REAL (Qwen3-TTS)")
        from real_engine import create_real_engine
        ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref_best.wav")
        engine = create_real_engine(ref_audio=ref_path)
        test_text = "สวัสดีครับ วันนี้อากาศดีมาก แต่เหนื่อยเหมือนกัน"
        result_audio, result_sr = optimize_and_generate(
            test_text, engine, base_speed=1.0, auto_tags=False, default_pause_ms=400,
        )
        duration = len(result_audio) / result_sr
        print(f"  Text: {test_text}")
        print(f"  Output: {len(result_audio)} samples, {result_sr}Hz, {duration:.2f}s")
        assert len(result_audio) > 0, "Audio is empty"
        assert result_sr == 24000, f"Wrong SR: {result_sr}"
        print("PASS (real)")
        engine.unload()
    else:
        # Quick mock engine for unit testing (no GPU needed)
        print("  Mode: MOCK (unit test — use --real for Qwen3-TTS)")
        def mock_engine(text: str, speed: float = 1.0):
            sr = 24000
            dur = 0.3 / max(speed, 0.5)
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            audio = (np.sin(2 * np.pi * 440 * t) * 0.3).astype(np.float64)
            return audio, sr

        test_text = "สวัสดีครับ วันนี้อากาศดีมาก แต่เหนื่อยเหมือนกัน"
        result_audio, result_sr = optimize_and_generate(
            test_text, mock_engine, base_speed=1.0, auto_tags=False, default_pause_ms=300,
        )
        duration = len(result_audio) / result_sr
        print(f"  Text: {test_text}")
        print(f"  Output: {len(result_audio)} samples, {result_sr}Hz, {duration:.2f}s")
        print(f"  3 sentences → expect ~0.9s speech + ~0.7s pauses = ~1.6s total")
        assert len(result_audio) > 0, "Audio is empty"
        assert result_sr == 24000, f"Wrong SR: {result_sr}"
        assert duration > 1.0, f"Too short: {duration:.2f}s (expected >1.0s with pauses)"
        print("PASS")
    print()

    print("=== optimize_segment test ===")
    seg_result = optimize_segment("ทำไมเป็นแบบนี้? เลวร้ายมาก", base_speed=1.0, auto_tags=True)
    print(f"  text: {seg_result['text'][:60]}...")
    print(f"  speed: {seg_result['speed']:.2f}")
    print(f"  tags_added: {seg_result['tags_added']}")
    assert isinstance(seg_result, dict)
    assert "text" in seg_result and "speed" in seg_result and "tags_added" in seg_result
    print("PASS")
