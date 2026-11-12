"""Run llama.cpp's convert_hf_to_gguf.py, with one extension.

llama.cpp identifies a BPE tokenizer's pre-tokenizer by hashing how it encodes a
fixed test string. A brand-new tokenizer (like Ludwig's 64K one) has a hash that
llama.cpp has never seen, so conversion stops. If LUDWIG_TOKENIZER_PRE is set
(for example "qwen2" or "llama-bpe"), we use that pre-tokenizer instead.
Only set it after checking that the tokenizer's split regex is identical to that
pre-tokenizer's regex (scripts/smoke_format.py does this), and always confirm
with the tokenization parity test.

Usage: LLAMA_CPP_DIR=/path/to/llama.cpp python convert_gguf.py <hf_dir> --outfile x.gguf --outtype f32
"""
import os
import runpy
import sys

llama_dir = os.environ.get("LLAMA_CPP_DIR", "llama.cpp")
sys.path.insert(0, os.path.join(llama_dir, "gguf-py"))
sys.path.insert(0, llama_dir)

import conversion.base as cb  # noqa: E402

_original = cb.TextModel.get_vocab_base_pre


def _patched(self, tokenizer):
    try:
        return _original(self, tokenizer)
    except NotImplementedError:
        pre = os.environ.get("LUDWIG_TOKENIZER_PRE")
        if not pre:
            raise
        print(f"[ludwig] pre-tokenizer hash unknown to llama.cpp; using '{pre}' (regex checked beforehand)")
        return pre


cb.TextModel.get_vocab_base_pre = _patched
script = os.path.join(llama_dir, "convert_hf_to_gguf.py")
sys.argv = [script] + sys.argv[1:]
runpy.run_path(script, run_name="__main__")
