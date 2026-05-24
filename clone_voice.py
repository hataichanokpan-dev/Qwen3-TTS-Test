"""
Qwen3-TTS Voice Cloning CLI (ICL mode per official guide)

Usage:
    python clone_voice.py "ข้อความที่ต้องการพูด"
    python clone_voice.py "ข้อความ" -o output.wav
    python clone_voice.py "ข้อความ" --ref other_ref.wav --ref-text transcript.txt
    python clone_voice.py input.txt -o output.wav
    python clone_voice.py --prepare long_audio.wav
"""

import os
import sys
import time
import argparse

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from voice_clone import VoiceCloner


def main():
    parser = argparse.ArgumentParser(description="Qwen3-TTS Voice Cloning (ICL mode)")
    parser.add_argument("text", nargs="?", help="Text to synthesize (or .txt file path)")
    parser.add_argument("--output", "-o", help="Output WAV path")
    parser.add_argument("--ref", "-r", help="Reference audio (default: ref_best.wav)")
    parser.add_argument("--ref-text", "-rt", help="Reference text for ICL mode (.txt file or string)")
    parser.add_argument("--prepare", metavar="AUDIO", help="Auto-find best 15s segment from long audio")
    parser.add_argument("--temp", type=float, default=0.8, help="Temperature (default: 0.8)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (default: 0.9)")
    parser.add_argument("--rp", type=float, default=1.05, help="Repetition penalty (default: 1.05)")
    parser.add_argument("--xvec", action="store_true", help="Use xvec mode instead of ICL (lower quality)")
    parser.add_argument("--no-preprocess", action="store_true", help="Disable English→Thai text preprocessing")
    args = parser.parse_args()

    cloner = VoiceCloner()

    # Prepare reference mode
    if args.prepare:
        ref_path = cloner.prepare_reference(args.prepare)
        out = os.path.join(BASE_DIR, "ref_prepared.wav")
        audio, sr = __import__("soundfile").read(ref_path)
        __import__("soundfile").write(out, audio, sr)
        print(f"Prepared reference: {out}")
        return

    # Determine reference audio
    ref_audio = args.ref
    if ref_audio is None:
        ref_best = os.path.join(BASE_DIR, "ref_best.wav")
        ref_15s = os.path.join(BASE_DIR, "ref_15s.wav")
        ref_audio = ref_best if os.path.exists(ref_best) else ref_15s

    if not os.path.exists(ref_audio):
        print(f"Error: Reference not found: {ref_audio}")
        sys.exit(1)

    # Determine ref_text (required for ICL mode)
    ref_text = args.ref_text
    if ref_text is None and not args.xvec:
        # Auto-detect ref_best_text.txt
        ref_text_auto = os.path.join(BASE_DIR, "ref_best_text.txt")
        if os.path.exists(ref_text_auto):
            ref_text = ref_text_auto

    # Get text
    if not args.text:
        print("Error: Provide text or use --prepare")
        sys.exit(1)

    if os.path.isfile(args.text):
        with open(args.text, "r", encoding="utf-8") as f:
            text = f.read().strip()
    else:
        text = args.text

    if not text:
        print("Error: Empty text")
        sys.exit(1)

    # Output path
    output = args.output
    if output is None:
        os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        output = os.path.join(BASE_DIR, "output", f"clone_{ts}.wav")

    # Config overrides
    config_overrides = {}
    if args.xvec:
        config_overrides["x_vector_only_mode"] = True

    cloner = VoiceCloner(
        ref_audio=ref_audio,
        ref_text=ref_text,
        config=config_overrides if config_overrides else None,
        preprocess=not args.no_preprocess,
    )

    mode = "xvec" if args.xvec else "ICL"
    pp = "OFF" if args.no_preprocess else "ON"
    print(f"Ref: {ref_audio}")
    print(f"Ref text: {ref_text or '(none - xvec mode)'}")
    print(f"Text: {text[:80]}{'...' if len(text)>80 else ''}")
    print(f"Mode: {mode}, preprocess: {pp}, temp={args.temp}, top_p={args.top_p}")

    # Build prompt once (official recommended)
    if not args.xvec:
        cloner.build_prompt()

    audio, sr = cloner.clone(
        text,
        temperature=args.temp,
        top_p=args.top_p,
        repetition_penalty=args.rp,
    )
    VoiceCloner.save(audio, sr, output)


if __name__ == "__main__":
    main()
