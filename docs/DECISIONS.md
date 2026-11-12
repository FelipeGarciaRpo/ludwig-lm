# Ludwig 2: decision log

Every design decision, why it was taken, and the evidence or test behind it. The full reasoning, with numbers and sources, is in the design document (Spanish): [`docs/es/DISENO.md`](es/DISENO.md), v1.1.

**Status:** ✅ decided · 📏 fixed by a measurement · ⏳ open, with the test that closes it · 🟡 proposed, waiting for approval

---

## Proposed v1.2 (🟡 waiting for Pipe's approval)

Two pieces of evidence came in on November 11, 2026 (details in [`docs/es/BITACORA.md`](es/BITACORA.md)):

- **Bench v1 on Kaggle 2×T4** (`results/banco_t4_v1/banco_t4_v1.py`, `results/banco_t4_v1/`): a separate PyTorch implementation measured **3,856 tokens/s** for 26 × 1024 with 40 experts, with peak memory 13.2 GiB (13.8 reserved) and 3.2 GB/s between the GPUs. That is **1,801 h for 25B tokens**. The same model with a sigmoid router, a bias and an output gate trained stably in FP16 for 60 steps.
- **T0.1-C** (`scripts/smoke_format_afmoe.py`, `results/t0_1c_afmoe.json`): the **AFMoE format** (`AfmoeForCausalLM`, Arcee Trinity) passes every check on **stock** llama.cpp.

| ID | Proposal | Replaces |
|---|---|---|
| P1 | Store Ludwig 2 as `AfmoeForCausalLM` ([`configs/ludwig2_afmoe.json`](../configs/ludwig2_afmoe.json)): gated attention + QK-norm; 3 sliding-window layers (2048, RoPE θ = 10K) per global layer (no RoPE); depth-scaled sandwich norms; 1 dense layer; sigmoid router with expert bias; shared expert; tied embeddings scaled by √d. **1.906B total, 0.491B active.** Runs in llama.cpp unpatched. KV cache 136 MiB at 16K, against 416. Precedent: Trinity Nano (1B active, width 1024, Muon, same block). | D02, D03 (the A/B), D04 (partial RoPE everywhere), D05's softmax router |
| P2 | Balance with an aux-loss-free expert bias, updated with SMEBU (Trinity: tanh-clamped, momentum 0.5), plus a 1e-4 sequence-wise auxiliary loss. λ calibrated in T0.3. | D05's global balancing loss |
| P3 | Drop the MTP head (AFMoE has none in llama.cpp). | D07 |
| P4 | Keep Block AttnRes and KDA out of v1. In llama.cpp they exist only inside the Kimi-K3 architecture (MLA + KDA + latent MoE), and transformers has no native class for them. | — |
| P5 | Train with the transformers class plus our own speed pieces: padded-bmm experts registered through transformers' experts interface, FP32 residual stream, FP16 gradient reduce-scatter, explicit FSDP2 prefetch, and optional Muon momentum in CPU RAM. Measured in **T0.2 v2**. | D10 details |

If approved, the entries below are rewritten. Until then they stand.

---

## Model

### D01. Shape
✅ 26 layers × width 1024; GQA with 16 query heads, 4 KV heads, head size 64; MoE in every layer. **1.963B total parameters, 0.491B active per token**, plus 75M for the MTP head.

- **Why:** this is MobileMoE-M's shape (Meta, 2026): deep and thin, the right trade-off at this size. The states fit two 15 GiB T4s with FSDP2 (D10), and 0.49B active parameters run fast on a CPU.
- **Rejected:** the earlier 6 × 2048 design. At this size, shallow and wide loses to deep and thin.
- 📏 **Bench v1 (2×T4):** 26 × 1024 runs at 3,856 tokens/s, 1.11× slower than the fastest shape (11 × 1536 at 4,287), and as fast as 6 × 2048 (3,866). The T4 is power-limited, so large matrices are no faster than small ones (21.8 vs 27.0 TFLOPS). Depth stays.
- **Plan B:** 32 experts (1.64B total) if T0.2 leaves less than 1.5 GiB free per GPU.

### D02. File format
✅ Stored as Hugging Face `Qwen3NextForCausalLM` with `layer_types` = 26 × `full_attention` (**format A**).

- **Why:** that class already has every piece we want (gated attention, QK-norm, shared expert, MTP weights) in a format transformers loads and llama.cpp converts. We maintain no modeling code of our own that could drift.
- **Evidence (T0.1, CPU):** exported and reloaded logits are identical; the GGUF converts with the MTP layer. **Stock llama.cpp crashes on load** (`GGML_ASSERT(buffer)`); with a 3-line patch, every check passes: next-token probabilities within 1.9e-6, greedy 4/4 identical, MTP speculative decoding 4/4 identical. See [`patches/README.md`](../patches/README.md).
- **Known limit:** llama.cpp still treats the model as hybrid: no context shift, and prompt reuse goes through checkpoints.

### D03. Format A/B
⏳ Two identical short runs of the real model (~0.3B tokens, same data order, same LR), in T2:

- **A** = Qwen3-Next format: output gate + softmax router with a global balancing loss.
- **B** = GLM-4.5 format (`Glm4MoeForCausalLM`): no gate; sigmoid router with a bias and no auxiliary loss (DeepSeek-style, MobileMoE's recipe); QK-norm; partial RoPE; MTP. Runs in llama.cpp without patches.
- **Rule:** A goes if it beats B by ≥1.5% in validation loss, or if B is unstable in FP16. Otherwise B goes, because it runs everywhere unpatched. Expert balance, router entropy, under-used experts, loss spikes and tokens/s are also compared.

### D04. Attention
✅ QK-norm (RMSNorm per head on q and k); an elementwise sigmoid gate on the attention output (`q_proj` outputs the query and the gate); partial RoPE on 50% of the dimensions with θ = 500,000.

- **Why:** QK-norm bounds attention logits, which matters in FP16. The gate removes attention sinks and massive activations and tolerates a higher LR (Qwen, *Gated Attention*, NeurIPS 2025). Partial RoPE makes context extension smoother.

### D05. Mixture of experts
✅ 40 routed experts (SwiGLU, 512 wide), top-4 with normalized weights, plus one shared expert (SwiGLU, 2048) with a sigmoid gate. FFN expansion (40 × 512 + 2048) / (4 × 512 + 2048) = 5.5, inside MobileMoE's useful range of 4–8.

- **Router:** softmax computed in FP32.
- **Balancing:** global-batch load-balancing loss (coefficient 1e-3; expert frequencies are summed across GPUs with one all-reduce) + router z-loss 1e-4. No token dropping.
- **Kernel:** on the T4, transformers' `grouped_mm` falls back to one matmul per expert, `eager` does the same, and `batched_mm` copies the expert weights for every token, which does not fit at training batch sizes. Bench v1 measured padded bmm 16% faster than one matmul per expert, with balanced routing. See P5.
- 📏 **Expert count (bench v1):** 48 experts fit only with half the micro-batch (2,926 tokens/s, −24%); 60 only with Muon's momentum in CPU RAM (2,670, −31%). Every micro-batch moves the whole model between the GPUs at 3.2 GB/s, so more experts cost speed. **40 stays.**

### D06. Tokenizer
✅ Our own 64K BPE (the same as the larger Ludwig plan), with tied input and output embeddings.

- **Why 64K:** the output head is 12% of the compute, against ~23% with a 128K vocabulary.
- **Special tokens from day one:** chat template, `<think>` and `</think>`, FIM, tool calls, end of text.
- **Constraint:** the pre-tokenizer regex must be one llama.cpp implements. T0.1 checks it (T1–T2) and must be re-run with the final tokenizer. ⏳

### D07. MTP head
✅ One multi-token-prediction module (projection 2048 → 1024, two norms, one decoder layer, the shared head), used as the draft model for speculative decoding (`llama-server --spec-type draft-mtp`).

- **Training:** only on the last ~1B tokens of the v1 branch, with stop-gradient into the trunk, the embeddings and the head. It learns from the model's representations but cannot make the model worse.
- **Evidence:** T0.1 A5, outputs identical with and without speculation; 64% of drafts accepted on the tiny test model.

---

## Training

### D08. Optimizer
✅ **Muon** for matrices: attention, each expert separately, the shared expert and the MTP projection. Nesterov momentum 0.95, 5 Newton–Schulz steps, Moonlight's scale 0.2·√max(m, n) so AdamW's LR and weight decay carry over. The query and gate halves of `q_proj`, and the gate and up halves of each expert, are orthogonalized separately. **AdamW** (β 0.9/0.95) for embeddings, norms, router and shared-expert gate. Weight decay 0.1 on matrices.

- **Muon momentum in bf16:** fp32 range at half the memory.

### D09. Precision
✅ FP16 with dynamic loss scaling (the T4 has no BF16), FP32 master weights; router, softmax, norms and loss in FP32; output z-loss 1e-4 as an FP16 guard; non-finite steps are skipped.

### D10. Parallelism
✅ **FSDP2, everything sharded,** one unit per layer. The root module is not resharded after the forward pass, because the chunked loss uses the tied embedding as the output head.

- 📏 **Bench v1:** 3.2 GB/s between the two T4s. At 8 sequences of 2K per GPU, the reserved memory was 13.8 GiB of ~14.5 usable: too tight for 3 months. P5 lists the fixes: FP16 gradients, and momentum in RAM (−1.7 GiB per GPU).
- **Rejected for now:** experts split across the GPUs (20 each) + DDP on the dense part. The ~0.33B dense parameters would be replicated: **+1.71 GiB per GPU** (9.33 → 11.05 GiB of states). It is built only if FSDP2 misses the G0 throughput (D16).

### D11. Loss
✅ Chunked cross-entropy with recomputation: the full logits are never materialized (16K tokens × 64K vocabulary would be 4.3 GB in FP32).

### D12. Schedule, batch and LR
- ✅ **WSD:** 2,000 warm-up steps; constant LR in the trunk; each branch decays to 0 with a 1−√ curve.
- ✅ **Batch:** 0.25M tokens per step for the first ~2B tokens, then 0.5M (~54,000 steps in v1).
- ⏳ **Peak LR:** Ludwig 1's (same width and attention), confirmed with three short runs at ×0.5, ×1 and ×2 in T2.
- ⏳ **Init:** Normal(0, 0.02), `o_proj` and `down_proj` scaled by 1/√(2·26), norm weights at 0 (scale 1 + w). T0.2 uses transformers' default init, which does not change speed or memory.

### D13. Context length
✅ 2K up to 50B tokens, 4K after that; the last ~0.5B tokens of the v1 branch at 16K; 32K from 250B.

### D14. One trunk, cooldown branches
✅ A single trunk with constant LR and a stationary data mix (D17). Each public version is a cooldown branch of ~10% of its milestone, taken from the latest trunk checkpoint; the trunk continues from before the branch. Averaging trunk checkpoints gives free previews between milestones.

### D15. Relay on Kaggle
✅ Each friend runs turns **on their own Kaggle account, without sharing credentials**. A lock with expiry keeps one session training at a time. Full-state checkpoint every 1–2 h (~12 GB with bf16 momentum): two rotating private checkpoints plus public milestones. Each resume is verified by hashing the first batch.

### D16. Gate G0
✅ No trunk hour is spent until the real model and the real data loader pass all of this:

1. A sustained run of ≥6 h, with no NaN and a stable loss scaler.
2. ≥2,600 tokens/s sustained over the last hour.
3. Peak memory ≤13.5 GiB, evaluation and saving included.
4. ≥2 checkpoints uploaded to Hugging Face and one resume on another account, with the expected first-batch hash.
5. 50 steps at 16K context without running out of memory.

---

## Data

### D17. Mixes
✅ **Trunk** (stationary from 0 to 900B tokens):

| English web | Code | Math | Knowledge | Synthetic textbooks | Spanish | Instructions and short reasoning |
|---|---|---|---|---|---|---|
| 50% | 20% | 10% | 5% | 3% | 10% | 2% |

✅ **Branches** (~10% of each milestone):

| English web | Code | Math | Reasoning and instructions | Knowledge | Spanish |
|---|---|---|---|---|---|
| 26% | 25% | 22% | 13% | 4% | 10% |

- **Code** is mostly Python (45%) and JS/TS (25%). ⏳ Source: Stack-Edu or The Stack v3 filtered, decided before the trunk.
- **Rule:** at most ~4 passes over any source up to 1T tokens; scarce, high-quality data goes only into the branches.

### D18. Spanish: 10%
✅ 10% in the trunk and the branches, 25–30% in SFT. That leaves the trunk's natural language at about 86% English and 14% Spanish.

- **Not 50/50:** in controlled studies, a dominant language helps the less frequent one, and at scale a 90/10 mix beat 50/50 in both languages. English also has the best data and nearly all the benchmarks.
- **Not 4%:** ~1B Spanish tokens in v1 is too little to hold a conversation.
- **Precedent:** SmolLM3 (3B) used 12% multilingual web across 5 languages and beat Qwen2.5-3B and Llama 3.2 3B on Spanish HellaSwag.
- **Measured with:** Spanish bits per byte, tokenizer compression, MGSM, Belebele, Global-MMLU, Spanish HellaSwag and Flores (EN↔ES). If Spanish lags, the branches raise it without touching the trunk.

### D19. Synthetic data
✅ At most 10% of the trunk and 35% of each branch; at least 3 teacher families; near-duplicate removal; a style watch (n-gram diversity, teacher tics such as "Certainly!").

### D20. Licenses
✅ 🟢 open human text and outputs of permissively licensed open models (DeepSeek-R1/V3, Qwen3, QwQ, gpt-oss, Mixtral). 🟡 models whose license sets conditions (for example, Llama's naming clause): out by default. 🔴 outputs of closed APIs, whose terms forbid training competing models (for example TinyGSM, made with GPT-3.5): never.

- **Code:** Apache-2.0 now.
- **Weights:** Apache-2.0 ⏳ only after auditing the data card source by source. This gates every release.

### D21. Reasoning
✅ Yes, with a realistic scope: short chain-of-thought for grade-school math, Python as a calculator (tool-integrated reasoning), `<think>` from day one. General SFT first, then reasoning SFT, which beat mixing both in MobileLLM-R1. Mostly short traces with some long ones ("mix distillation"). RL is optional and late: for small models, SFT beat RL in MobileLLM-R1.

- **Expectation for v1 (25B tokens):** 2–4-step problems, irregularly; GSM8K ~10–25% after SFT. Not competition math.

### D22. Distillation
✅ From **data**: yes. OpenMathReasoning (CC-BY-4.0) and a small data factory with 🟢 teachers. From **logits**: no. Our tokenizer differs from every teacher's, and on a T4 a teacher's forward pass costs as much as training the student.

---

## Engineering

### D23. Names
✅ Repository `ludwig-lm`, Python package `ludwig_lm`. `ludwig` on PyPI is the declarative deep-learning framework started at Uber, a different project.

### D24. Source of truth
✅ Scripts are the source of truth; the Kaggle notebooks are generated from `scripts/make_notebooks.py`. Every test writes a raw JSON report to `results/`, and every number in the README comes from one of them.
