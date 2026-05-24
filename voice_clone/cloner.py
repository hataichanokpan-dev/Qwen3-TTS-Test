"""
VoiceCloner - Voice cloning following official Qwen3-TTS guide.

Official recommended flow:
  1. Load model
  2. create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode=False)
  3. Reuse prompt for all generate_voice_clone() calls → consistent voice

Usage:
    cloner = VoiceCloner(ref_audio="ref_best.wav", ref_text="transcript.txt")
    audio, sr = cloner.clone("ข้อความที่ต้องการพูด")
    cloner.save(audio, sr, "output.wav")
"""

import os
import gc
import time
import torch
import numpy as np
import soundfile as sf

from .config import DEFAULTS
from .audio_utils import find_best_segment
from .text_preprocess import preprocess_thai_text


class VoiceCloner:
    def __init__(self, ref_audio=None, ref_text=None, config=None, preprocess=True):
        self.config = {**DEFAULTS, **(config or {})}
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.model = None
        self.preprocess = preprocess
        self._prompt = None  # cached reusable prompt

    def load_model(self):
        if self.model is not None:
            return

        from qwen_tts import Qwen3TTSModel

        print(f"Loading {self.config['model_name']}...")
        t0 = time.time()
        self.model = Qwen3TTSModel.from_pretrained(
            self.config["model_name"],
            device_map=self.config["device"],
            dtype=getattr(torch, self.config["dtype"]),
            attn_implementation=self.config["attn_implementation"],
        )
        print(f"Model loaded in {time.time()-t0:.1f}s")

    def unload_model(self):
        if self.model is not None:
            del self.model
            self.model = None
        self._prompt = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _read_ref_text(self, ref_text):
        """Read ref_text - can be a string or path to .txt file."""
        if ref_text is None:
            return None
        if os.path.isfile(ref_text):
            with open(ref_text, "r", encoding="utf-8") as f:
                return f.read().strip()
        return ref_text

    def build_prompt(self, ref_audio=None, ref_text=None):
        """Build reusable voice clone prompt (official recommended approach).

        Call once, reuse for all clone() calls → consistent voice.

        Args:
            ref_audio: Override reference audio path.
            ref_text: Override reference text (string or .txt path).
                      REQUIRED for ICL mode (x_vector_only_mode=False).

        Returns:
            List[VoiceClonePromptItem] - cached internally for reuse.
        """
        self.load_model()

        ref = ref_audio or self.ref_audio
        rt = self._read_ref_text(ref_text) if ref_text else self._read_ref_text(self.ref_text)

        if not self.config["x_vector_only_mode"] and rt is None:
            raise ValueError(
                "ref_text is REQUIRED for ICL mode (x_vector_only_mode=False). "
                "Use Whisper to transcribe your reference audio first."
            )

        print(f"Building voice clone prompt (mode={'ICL' if not self.config['x_vector_only_mode'] else 'xvec'})...")
        t0 = time.time()
        self._prompt = self.model.create_voice_clone_prompt(
            ref_audio=ref,
            ref_text=rt,
            x_vector_only_mode=self.config["x_vector_only_mode"],
        )
        print(f"Prompt built in {time.time()-t0:.1f}s")
        return self._prompt

    def clone(self, text, ref_audio=None, ref_text=None, **overrides):
        """Clone voice and return (audio_array, sample_rate).

        Follows official guide: uses cached prompt from build_prompt().
        Auto-builds prompt on first call if not built yet.

        Args:
            text: Text to synthesize (Thai text, auto-preprocessed).
            ref_audio: Override reference audio (used only for initial prompt build).
            ref_text: Override reference text (used only for initial prompt build).
            **overrides: Override any config param (temperature, top_p, etc.)

        Returns:
            (numpy_array, sample_rate)
        """
        self.load_model()

        if self.preprocess:
            text = preprocess_thai_text(text)

        params = {**self.config, **overrides}

        # Build prompt on first call if not cached
        if self._prompt is None:
            self.build_prompt(ref_audio=ref_audio, ref_text=ref_text)

        wavs, sr = self.model.generate_voice_clone(
            text=text,
            language=params["language"],
            voice_clone_prompt=self._prompt,
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
        )
        return wavs[0], sr

    def clone_batch(self, texts, ref_audio=None, ref_text=None, **overrides):
        """Clone multiple texts with one shared prompt, return list of (audio, sr).

        Official recommended approach: build prompt once, generate all texts.
        """
        self.load_model()

        if self.preprocess:
            texts = [preprocess_thai_text(t) for t in texts]

        params = {**self.config, **overrides}

        # Build prompt on first call if not cached
        if self._prompt is None:
            self.build_prompt(ref_audio=ref_audio, ref_text=ref_text)

        # Batch generate - pass all texts at once
        languages = [params["language"]] * len(texts)
        wavs, sr = self.model.generate_voice_clone(
            text=texts,
            language=languages,
            voice_clone_prompt=self._prompt,
            temperature=params["temperature"],
            top_k=params["top_k"],
            top_p=params["top_p"],
            repetition_penalty=params["repetition_penalty"],
        )
        return [(w, sr) for w in wavs]

    @staticmethod
    def save(audio, sr, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        sf.write(path, audio, sr)
        print(f"Saved: {path} ({len(audio)/sr:.1f}s)")

    def prepare_reference(self, audio_path, method="best_segment"):
        """Auto-prepare reference audio from a long source file.

        Args:
            audio_path: Path to source audio (any length).
            method: "best_segment" = cut best 15s segment

        Returns:
            Path to prepared reference audio.
        """
        audio, sr = sf.read(audio_path)
        if audio.ndim > 1:
            audio = audio[:, 0]
        duration = len(audio) / sr

        if duration <= 16:
            print(f"Audio is {duration:.1f}s - using as-is")
            return audio_path

        print(f"Audio is {duration:.1f}s - finding best 15s segment...")
        segments = find_best_segment(
            audio_path,
            segment_s=self.config["segment_length_s"],
            hop_s=self.config["segment_hop_s"],
        )

        best = segments[0]
        print(f"Best segment: {best['start_s']:.0f}s-{best['end_s']:.0f}s "
              f"(centroid_diff={best['centroid_diff']:.0f}Hz)")
        return best["path"]
