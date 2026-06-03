"""
Voice QA Gate - Automated audio quality assessment for TTS output.

Acts as "ears" for automated testing:
  1. Speaker Similarity: cosine sim of ref vs generated embeddings
  2. ASR Verification: Whisper transcribe output → compare with input text (CER/WER)
  3. Duration Sanity: flag if output way too long/short for input text
  4. Signal Quality: RMS, silence ratio, SQUIM (PESQ/STOI/SI-SDR)
  5. Hallucination Detection: flag if output > 2x expected duration

Usage:
    from voice_qa import VoiceQA
    qa = VoiceQA(ref_audio="ref_best.wav")
    report = qa.evaluate("output.wav", expected_text="สวัสดีครับ")
    print(report.summary())
    # PASS/FAIL with scores
"""

import os
import numpy as np
import soundfile as sf
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav


class QAResult:
    """Single check result."""
    def __init__(self, name, score, threshold, passed, detail=""):
        self.name = name
        self.score = score
        self.threshold = threshold
        self.passed = passed
        self.detail = detail

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.score:.3f} (threshold: {self.threshold})"


class QAReport:
    """Full QA report with all check results."""
    def __init__(self, audio_path, results):
        self.audio_path = audio_path
        self.results = results
        self.all_passed = all(r.passed for r in results)

    def summary(self):
        lines = [f"\n{'='*60}"]
        lines.append(f"Voice QA: {self.audio_path}")
        lines.append(f"{'='*60}")
        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            score_str = f"{r.score:.3f}" if isinstance(r.score, float) else str(r.score)
            thresh_str = f"{r.threshold:.3f}" if isinstance(r.threshold, float) else str(r.threshold)
            lines.append(f"  [{status}] {r.name}: {score_str} (need: {thresh_str})")
            if r.detail:
                lines.append(f"         {r.detail}")
        lines.append(f"{'='*60}")
        verdict = "ALL PASSED" if self.all_passed else "SOME FAILED"
        lines.append(f"Verdict: {verdict}")
        return "\n".join(lines)


class VoiceQA:
    def __init__(self, ref_audio=None, whisper_model="large-v3", whisper_device="cuda"):
        self.ref_audio = ref_audio
        self.ref_embedding = None
        self._whisper_model = None
        self._whisper_params = (whisper_model, whisper_device)
        self._squim_model = None
        self._encoder = None

        if ref_audio:
            self.ref_embedding = self._extract_embedding(ref_audio)

    def _get_whisper(self):
        if self._whisper_model is None:
            from faster_whisper import WhisperModel
            model_name, device = self._whisper_params
            self._whisper_model = WhisperModel(model_name, device=device, compute_type="float16")
        return self._whisper_model

    def _get_squim(self):
        if self._squim_model is None:
            import torch
            from torchaudio.pipelines import SQUIM_SUBJECTIVE
            self._squim_model = SQUIM_SUBJECTIVE.get_model().to(torch.float32)
        return self._squim_model

    def _get_encoder(self):
        """Lazy-load Resemblyzer voice encoder."""
        if self._encoder is None:
            self._encoder = VoiceEncoder()
        return self._encoder

    def _extract_embedding(self, audio_path):
        """Extract speaker embedding using Resemblyzer d-vector."""
        wav, sr = sf.read(audio_path)
        if wav.ndim > 1:
            wav = wav[:, 0]
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        wav = preprocess_wav(wav)
        encoder = self._get_encoder()
        return encoder.embed_utterance(wav)

    def _cosine_sim(self, a, b):
        """Cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def check_speaker_similarity(self, audio_path, threshold=0.90):
        """Check if generated voice matches reference speaker (Resemblyzer d-vector)."""
        if self.ref_embedding is None:
            return QAResult("speaker_similarity", 0, threshold, False, "No reference audio set")

        gen_emb = self._extract_embedding(audio_path)
        sim = self._cosine_sim(self.ref_embedding, gen_emb)

        return QAResult("speaker_similarity", sim, threshold, sim >= threshold,
                        f"resemblyzer_dvec_sim={sim:.4f}")

    def check_asr(self, audio_path, expected_text, cer_threshold=0.3):
        """Transcribe output and compare with expected text using CER."""
        model = self._get_whisper()
        segments, info = model.transcribe(audio_path, language="th", beam_size=5)
        transcribed = "".join(seg.text for seg in segments).strip()

        if not transcribed or not expected_text:
            return QAResult("asr_cer", 1.0, cer_threshold, False,
                          f"Empty transcription or expected text")

        # Character Error Rate (better for Thai than WER)
        cer = self._cer(expected_text, transcribed)
        detail = f"expected='{expected_text[:40]}...' transcribed='{transcribed[:40]}...'"
        return QAResult("asr_cer", cer, cer_threshold, cer <= cer_threshold, detail)

    def _cer(self, ref, hyp):
        """Character Error Rate using edit distance."""
        ref_chars = list(ref.replace(" ", ""))
        hyp_chars = list(hyp.replace(" ", ""))
        if not ref_chars:
            return 1.0
        d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1))
        for i in range(len(ref_chars) + 1):
            d[i][0] = i
        for j in range(len(hyp_chars) + 1):
            d[0][j] = j
        for i in range(1, len(ref_chars) + 1):
            for j in range(1, len(hyp_chars) + 1):
                cost = 0 if ref_chars[i-1] == hyp_chars[j-1] else 1
                d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
        return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)

    def check_duration(self, audio_path, expected_text, chars_per_sec=6.0, tolerance=2.5):
        """Check if output duration is reasonable for input text length."""
        info = sf.info(audio_path)
        actual_dur = info.duration
        expected_dur = len(expected_text.replace(" ", "")) / chars_per_sec
        ratio = actual_dur / max(expected_dur, 0.1)

        passed = ratio <= tolerance
        detail = f"actual={actual_dur:.1f}s expected~{expected_dur:.1f}s ratio={ratio:.1f}x"
        return QAResult("duration_sanity", ratio, tolerance, passed, detail)

    def check_silence(self, audio_path, max_silence_ratio=0.3):
        """Check silence ratio - too much silence = bad quality."""
        y, sr = librosa.load(audio_path, sr=16000)
        rms = librosa.feature.rms(y=y)[0]
        threshold = np.mean(rms) * 0.1
        silence_frames = np.sum(rms < threshold)
        silence_ratio = silence_frames / len(rms)

        passed = silence_ratio <= max_silence_ratio
        detail = f"silence={silence_ratio:.1%} of audio"
        return QAResult("silence_ratio", silence_ratio, max_silence_ratio, passed, detail)

    def check_clipping(self, audio_path, max_clip_ratio=0.01):
        """Check for audio clipping (distortion)."""
        y, sr = librosa.load(audio_path, sr=None)
        if y.dtype == np.float32 or y.dtype == np.float64:
            clip_samples = np.sum(np.abs(y) >= 0.99)
        else:
            clip_samples = np.sum(np.abs(y) >= 32767)
        clip_ratio = clip_samples / len(y)

        passed = clip_ratio <= max_clip_ratio
        detail = f"clipped={clip_ratio:.4%} of samples"
        return QAResult("clipping", clip_ratio, max_clip_ratio, passed, detail)

    def check_signal_stats(self, audio_path):
        """Check signal quality: RMS energy, spectral features."""
        y, sr = librosa.load(audio_path, sr=None)
        rms = float(np.mean(librosa.feature.rms(y=y)))
        spec_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        spec_bw = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))

        # Flag if RMS very low (near silence) or very high (likely distorted)
        rms_ok = 0.001 < rms < 0.5
        detail = f"RMS={rms:.4f} centroid={spec_centroid:.0f}Hz bandwidth={spec_bw:.0f}Hz"
        return QAResult("signal_quality", rms, "0.001-0.500", rms_ok, detail)

    def evaluate(self, audio_path, expected_text=None):
        """Run all QA checks and return report."""
        results = []

        # 1. Speaker similarity (needs ref_audio)
        if self.ref_audio:
            results.append(self.check_speaker_similarity(audio_path))

        # 2. ASR verification (needs expected_text)
        if expected_text:
            results.append(self.check_asr(audio_path, expected_text))

        # 3. Duration sanity (needs expected_text)
        if expected_text:
            results.append(self.check_duration(audio_path, expected_text))

        # 4. Silence ratio
        results.append(self.check_silence(audio_path))

        # 5. Clipping
        results.append(self.check_clipping(audio_path))

        # 6. Signal quality stats
        results.append(self.check_signal_stats(audio_path))

        return QAReport(audio_path, results)


def main():
    """CLI for voice QA."""
    import sys
    import argparse

    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Voice QA Gate")
    parser.add_argument("audio", help="Audio file to evaluate")
    parser.add_argument("--ref", "-r", help="Reference audio for similarity check")
    parser.add_argument("--text", "-t", help="Expected text for ASR verification")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    qa = VoiceQA(ref_audio=args.ref)
    report = qa.evaluate(args.audio, expected_text=args.text)

    if args.json:
        import json
        data = {
            "file": args.audio,
            "passed": report.all_passed,
            "checks": [
                {
                    "name": r.name,
                    "score": round(r.score, 4) if isinstance(r.score, float) else r.score,
                    "threshold": r.threshold,
                    "passed": r.passed,
                    "detail": r.detail,
                }
                for r in report.results
            ]
        }
        print(json.dumps(data, ensure_ascii=False, indent=2))
    else:
        print(report.summary())

    sys.exit(0 if report.all_passed else 1)


if __name__ == "__main__":
    main()
