"""Default configuration - ICL mode per official Qwen3-TTS guide."""

DEFAULTS = {
    # Model
    "model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "device": "cuda:0",
    "dtype": "bfloat16",
    "attn_implementation": "sdpa",

    # Generation - ICL mode (official recommended default)
    "language": "Auto",
    "x_vector_only_mode": False,  # ICL mode - requires ref_text
    "temperature": 0.8,
    "top_k": 50,
    "top_p": 0.9,
    "repetition_penalty": 1.05,

    # Audio
    "target_sr": 24000,

    # Reference segment search
    "segment_length_s": 15,
    "segment_hop_s": 10,
}
