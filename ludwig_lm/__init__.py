"""Ludwig 2: a gated-attention MoE language model trained from scratch on free GPUs.

The trunk is serialized in the Qwen3-Next format (all layers full attention),
so checkpoints load in Hugging Face transformers and convert to GGUF for llama.cpp.
"""

__version__ = "0.1.0"
