# llama.cpp patch: Qwen3-Next with no recurrent layers

`llama-cpp-hybrid-without-recurrent-layers.patch` lets llama.cpp run a `qwen3next` model whose layers are **all** `full_attention`, which is how Ludwig 2 is stored (format A, see [`docs/DECISIONS.md`](../docs/DECISIONS.md#d02-file-format)).

```bash
git -C llama.cpp apply /path/to/ludwig-lm/patches/llama-cpp-hybrid-without-recurrent-layers.patch
cmake --build llama.cpp/build -j --target llama-server llama-cli
```

`scripts/smoke_format.py --patch` applies it and rebuilds for you.

## Symptom

The checkpoint converts to GGUF without errors, MTP layer included. Then `llama-server` aborts while loading the model:

```
ggml/src/ggml-backend.cpp:345: GGML_ASSERT(buffer) failed
```

The backtrace goes through `llm_graph_input_mem_hybrid::set_input`, at the line `GGML_ASSERT(ggml_backend_buffer_is_host(inp_rs->s_copy->buffer))`.

## Cause

- llama.cpp picks the memory type by architecture. `qwen3next` always gets the hybrid memory (KV cache + recurrent state), whatever `layer_types` says.
- The hybrid graph input creates the `s_copy` tensor for the recurrent state. With zero recurrent layers, no node of the graph reads it, so the graph allocator never gives it a buffer.
- `set_input` only checks that the tensor exists (`if (inp_rs->s_copy)`), then asserts that its buffer is in host memory. The buffer is null, so the assert fires.

## Fix

Three identical one-line changes in `src/llama-graph.cpp`, one in each hybrid input class (`llm_graph_input_mem_hybrid`, `_k` and `_iswa`):

```cpp
// before
if (inp_rs->s_copy) {
// after: skip it when the tensor was never allocated
if (inp_rs->s_copy && inp_rs->s_copy->buffer) {
```

When a model has recurrent layers, `s_copy` is allocated and the code path is the same as before.

## Evidence

`scripts/smoke_format.py` builds a tiny model with exactly Ludwig 2's structure (gated full attention in every layer, partial RoPE, MoE + shared expert, tied embeddings, one MTP module), converts it and compares llama.cpp with transformers. On llama.cpp `dd6b945` (CPU), Nov 11, 2026:

| Check | Stock llama.cpp | Patched |
|---|---|---|
| Convert to GGUF with MTP | pass | pass |
| Next-token probabilities vs transformers | crash on load | max abs diff 1.9e-6 |
| Greedy continuation, token by token | crash | 4/4 identical |
| Speculative decoding with the MTP head | crash | 4/4 identical, 58 of 91 drafts accepted |

Raw reports: [`results/t0_1_stock.json`](../results/t0_1_stock.json) and [`results/t0_1_patched.json`](../results/t0_1_patched.json).

## What the patch does not change

llama.cpp still treats the model as hybrid. So `llama-server` cannot shift the context when it fills up, and it reuses prompts through checkpoints instead of the full KV cache. A cleaner upstream fix may be to give `qwen3next` plain KV memory when no layer is recurrent; that is a maintainers' call. Other engines (vLLM, MLX, Ollama) have not been tested yet.

## Sending it upstream

llama.cpp accepts AI-assisted code under strict rules ([CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md), [AGENTS.md](https://github.com/ggml-org/llama.cpp/blob/master/AGENTS.md)). This patch was written with AI assistance, so whoever submits it must:

1. **Understand every line** and be able to explain it to a reviewer without AI help. The sections above are meant for that.
2. **Search existing issues and PRs first.** If one covers this, comment there instead of opening a duplicate.
3. **Re-run the reproduction on the latest `master`.** The bug may already be fixed.
4. **Write the issue, the PR description and the commit message yourself.** AI-written ones are prohibited and get the PR closed.
5. **Disclose the AI usage** in the PR template.

That is why this folder has a technical note and no ready-made PR text.
