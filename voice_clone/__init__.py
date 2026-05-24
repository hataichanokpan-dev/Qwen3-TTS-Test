"""
Qwen3-TTS Voice Cloning Package
Best config from systematic optimization (cos_sim=0.9634)

Usage:
    from voice_clone import VoiceCloner

    cloner = VoiceCloner(ref_audio="ref_best.wav")
    audio, sr = cloner.clone("สวัสดีครับ")
    cloner.save(audio, sr, "output.wav")
"""

from .cloner import VoiceCloner

__all__ = ["VoiceCloner"]
__version__ = "1.0.0"
