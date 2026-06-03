"""Generate full script-v3.json with pauses and merge.

Uses ref_best.wav for all sections (single voice), adds natural
pause markers (...) for pacing. Merges into one continuous file.

Usage:
    python generate_full_script.py
"""

import os
import sys
import json
import time
import re
import numpy as np
import soundfile as sf

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from voice_clone import VoiceCloner

OUTPUT_DIR = os.path.join(BASE_DIR, "output_script")


def add_pauses(text, pace="medium"):
    """Add natural pause markers (...) to text for slower TTS delivery."""
    if pace == "slow":
        text = re.sub(r'\.\s*', '... ', text)
        text = re.sub(r'\?\s*', '?... ', text)
        text = re.sub(r',\s*', ',... ', text)
    elif pace == "fast":
        text = re.sub(r'\.\s*(?=[A-Za-zༀ-՟])', '. ', text)
        text = re.sub(r'\?\s*(?=[A-Za-zༀ-՟])', '?... ', text)
    else:  # medium
        text = re.sub(r'\.\s*', '... ', text)
        text = re.sub(r'\?\s*', '?... ', text)
    return text.strip()


def merge_wavs(wav_files, output_path, gap_s=1.0, crossfade_s=0.15):
    """Merge multiple WAV files into one with silence gaps and crossfade."""
    segments = []
    target_sr = None

    for path in wav_files:
        audio, sr = sf.read(path)
        if target_sr is None:
            target_sr = sr
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        if audio.ndim > 1:
            audio = audio[:, 0]
        segments.append(audio)

    if not segments:
        return

    gap_samples = int(gap_s * target_sr)
    silence = np.zeros(gap_samples)
    cf_samples = int(crossfade_s * target_sr)

    # Build merged audio with crossfade between segments
    merged = segments[0].copy()
    for i in range(1, len(segments)):
        seg = segments[i]

        # Apply crossfade: overlap end of previous with start of current
        if cf_samples > 0 and len(merged) >= cf_samples and len(seg) >= cf_samples:
            # Create linear fade curves
            fade_out = np.linspace(1.0, 0.0, cf_samples, dtype=np.float64)
            fade_in = np.linspace(0.0, 1.0, cf_samples, dtype=np.float64)

            # Blend the overlap region
            tail = merged[-cf_samples:] * fade_out + seg[:cf_samples] * fade_in
            merged = np.concatenate([merged[:-cf_samples], tail, seg[cf_samples:]])
        else:
            merged = np.concatenate([merged, seg])

        # Add gap silence (not after the last segment)
        if i < len(segments) - 1:
            merged = np.concatenate([merged, silence])

    sf.write(output_path, merged, target_sr)
    total_dur = len(merged) / target_sr
    print(f"\nMerged: {output_path} ({total_dur:.1f}s, {len(segments)} sections, {gap_s}s gaps)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    script_path = os.path.join(BASE_DIR, "script-v3.json")
    with open(script_path, "r", encoding="utf-8") as f:
        script = json.load(f)

    # Build section list
    sections = [("intro", script["intro"])]
    for ch in script["chapters"]:
        sections.append((f"ch{ch['id']}", ch))
    sections.append(("outro", script["outro"]))

    print(f"Script: {script['title_thai']}")
    print(f"Sections: {len(sections)}")
    print(f"Reference: ref_best.wav (single voice, no emotion switching)")
    print()

    # Single reference for all sections
    ref_wav = os.path.join(BASE_DIR, "ref_best.wav")
    ref_txt = os.path.join(BASE_DIR, "ref_best_text.txt")

    # Build prompt ONCE — reuse for all sections (official recommended approach)
    cloner = VoiceCloner(ref_audio=ref_wav, ref_text=ref_txt)
    cloner.load_model()
    cloner.build_prompt()
    print()

    results = []
    wav_files = []

    for i, (section_id, section) in enumerate(sections):
        text = section["narration"]
        pace = section.get("pace", "medium")
        emotion = section.get("emotion", "neutral")

        text_with_pauses = add_pauses(text, pace)
        out_path = os.path.join(OUTPUT_DIR, f"{section_id}.wav")

        print(f"[{i+1}/{len(sections)}] {section_id} (pace={pace})")
        print(f"  text: {text_with_pauses[:80]}...")

        try:
            t0 = time.time()
            audio, sr = cloner.clone(text_with_pauses)
            VoiceCloner.save(audio, sr, out_path)

            dur = len(audio) / sr
            elapsed = time.time() - t0
            print(f"  -> {dur:.1f}s ({elapsed:.0f}s)")

            wav_files.append(out_path)
            results.append({
                "section": section_id,
                "pace": pace,
                "duration": round(dur, 1),
                "output": out_path,
                "status": "ok",
            })
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "section": section_id,
                "status": "error",
                "error": str(e),
            })
        print()

    # Merge all sections
    if wav_files:
        merged_path = os.path.join(OUTPUT_DIR, "full_script.wav")
        merge_wavs(wav_files, merged_path, gap_s=1.5)

    # Save manifest
    manifest = {
        "title": script["title_thai"],
        "sections": results,
    }
    manifest_path = os.path.join(OUTPUT_DIR, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print("Summary:")
    total_dur = sum(r.get("duration", 0) for r in results)
    for r in results:
        if r["status"] == "ok":
            print(f"  {r['section']:10s} {r.get('pace',''):10s} {r['duration']:5.1f}s")
        else:
            print(f"  {r['section']:10s} ERROR")
    print(f"  {'TOTAL':10s} {'':10s} {total_dur:5.1f}s")
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
