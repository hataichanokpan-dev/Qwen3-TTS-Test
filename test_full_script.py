"""Test full script-v3.json with corrected ICL voice cloning flow."""

import os
import sys
import json
import time

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from voice_clone import VoiceCloner

# Load script
with open(os.path.join(BASE_DIR, "script-v3.json"), "r", encoding="utf-8") as f:
    script = json.load(f)

print(f"Script: {script['title_thai']}")
print(f"Version: {script['version']}")

# Collect all narrations in order
sections = []
sections.append(("intro", script["intro"]["narration"], script["intro"]["pace"]))
for ch in script["chapters"]:
    sections.append((f"ch{ch['id']}_{ch['title']}", ch["narration"], ch["pace"]))
sections.append(("outro", script["outro"]["narration"], script["outro"]["pace"]))

print(f"\nTotal sections: {len(sections)}")
total_chars = sum(len(s[1]) for s in sections)
print(f"Total characters: {total_chars}")
print()

# Output dir
out_dir = os.path.join(BASE_DIR, "output_script")
os.makedirs(out_dir, exist_ok=True)

# Initialize cloner with ICL mode (official flow)
ref_text_path = os.path.join(BASE_DIR, "ref_best_text.txt")
cloner = VoiceCloner(
    ref_audio=os.path.join(BASE_DIR, "ref_best.wav"),
    ref_text=ref_text_path,
)

# Build reusable prompt ONCE (official recommended)
print("Building voice clone prompt (ICL mode)...")
cloner.build_prompt()
print("Prompt ready - generating all sections with same voice\n")

print(f"{'Section':<25} {'Chars':<8} {'Duration':<10} {'Time':<10} {'File'}")
print("-" * 80)

total_start = time.time()
results = []

for i, (name, text, pace) in enumerate(sections):
    safe_name = name.replace(" ", "_").replace("?", "")
    out_path = os.path.join(out_dir, f"{i+1:02d}_{safe_name}.wav")

    print(f"  [{i+1}/{len(sections)}] {name[:25]}...", end="", flush=True)

    try:
        t0 = time.time()
        audio, sr = cloner.clone(text)
        gen_time = time.time() - t0

        VoiceCloner.save(audio, sr, out_path)
        duration = len(audio) / sr

        results.append({
            "section": name,
            "chars": len(text),
            "duration": round(duration, 1),
            "gen_time": round(gen_time, 1),
            "file": os.path.basename(out_path),
        })
        print(f" {duration:.1f}s ({gen_time:.1f}s) -> {os.path.basename(out_path)}")

    except Exception as e:
        print(f" FAILED: {e}")
        results.append({"section": name, "error": str(e)})

# Cleanup
cloner.unload_model()

total_time = time.time() - total_start

# Summary
print(f"\n{'='*60}")
print(f"DONE - {total_time:.0f}s total")
print(f"{'='*60}")

success = [r for r in results if "duration" in r]
if success:
    total_dur = sum(r["duration"] for r in success)
    print(f"Sections: {len(success)}/{len(sections)}")
    print(f"Total audio: {total_dur:.1f}s ({total_dur/60:.1f}min)")
    print(f"Output: {out_dir}")

# Save manifest
manifest = {
    "script": script["title_thai"],
    "version": script["version"],
    "mode": "ICL (official guide)",
    "total_duration": sum(r.get("duration", 0) for r in results),
    "total_gen_time": round(total_time, 1),
    "sections": results,
}
with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
