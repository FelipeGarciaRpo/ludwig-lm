# Ludwig 2: decision log

Every design decision, why it was taken, and the evidence or test behind it. The full reasoning, with numbers and sources, is in the design document (Spanish): [`docs/es/DISENO.md`](es/DISENO.md), v1.1.

**Status:** ✅ decided · 📏 fixed by a measurement · ⏳ open, with the test that closes it · ⛔ superseded

---

## v1.2 (✅ approved by Pipe on November 12, 2026)

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

P1–P4 are decided. P5 is decided as the plan and 📏 measured piece by piece in T0.2 v2 (`notebooks/t0_2v2_afmoe_2xT4.ipynb`, `scripts/bench_afmoe.py`, `ludwig_lm/afmoe.py`). The entries below marked ⛔ are replaced by them; the rest still stand.

### After T0.2 v2 (November 13, 2026)

T0.2 v2 ran on Kaggle 2×T4 on November 12 (`results/t0_2v2_intento2/`): 13 measurements, all finished, no skipped FP16 step. Each piece was switched off once against the base (padded-bmm experts, FP32 residual, FP16 reduce-scatter, prefetch, Muon momentum on the GPU, 8 sequences of 2K per GPU: 2,930 tokens/s, 14.33 GiB reserved).

| ID | Decision | Evidence |
|---|---|---|
| P5a | 📏 **Training configuration:** one matmul per expert (transformers' `grouped_mm`, a loop on the T4), FP32 residual stream, FP16 reduce-scatter, explicit prefetch, **Muon momentum in pinned CPU RAM**, 8 sequences of 2K per GPU. These are the defaults of `scripts/train.py`. The two pieces were measured separately: together they should give ~3,300 tokens/s (~2,100 h for 25B) at ~12.7 GiB reserved. T0.3 + T2 measure the combination over hours. | Loop +17.3% (3,436 tokens/s); momentum in RAM −1.57 GiB reserved for −3.4%; FP16 residual −0.66 GiB but −0.4%; FP32 reduce −5.7%; no prefetch −6.7%; 4 sequences −23.5%. Only momentum in RAM and 4 sequences stay ≤ 13.5 GiB reserved. 4K and 16K fit (2,240 and 1,228 tokens/s). Checkpoint 11.29 GiB, save 54 s, load 48 s, exact |
| P2a | ✅ **SMEBU λ = 0.01.** Reviewed with real text in T0.3 + T2: it stays while the busiest expert stays under 3× the mean and no expert dies. | 60 steps of synthetic text: λ = 0.01 ended at 2.41× the mean load; λ = 0.001 at 6.86×; the fixed-sign update at 7.57×. Same loss in the three |
| D16a | ✅ **G0's memory limit (13.5 GiB) is reserved memory**, what PyTorch's allocator holds on the GPU; bench v1 measured the same. The T4 gives PyTorch ~14.56 GiB, so 13.5 leaves ~1 GiB for what a 3-month run can add. | With allocated memory every T0.2 v2 measurement would pass (12.17 GiB at the base), which says nothing about running out |
| P6 | ✅ **T0.3 and T2 run together**, before the tokenizer is ready: three runs of the real model and training loop at LR ×0.5, ×1 and ×2 around Ludwig 1's recipe (D12), 400 steps of 0.25M tokens each (~105M tokens; 100 warm-up steps, 1−√ decay over the last 80, the last 20% as in Ludwig 1), same data order, one Kaggle session each, in parallel on three accounts. **Proxy data:** FineWeb-Edu `sample-10BT` and SmolLM2's tokenizer (49,152 tokens, Apache-2.0, EOS = 0 like ours), compared in bits per byte, which do not depend on the tokenizer. The ×2 run is the FP16 stress test. **Decision rule, fixed before the results:** a run counts only if stable (no NaN, no stop for spikes, a steady FP16 scale); the lowest validation bits per byte after the decay wins; within 0.5% the lower LR wins, because short runs favour high LRs and the trunk is ~200× longer; if an end of the sweep wins, one more run beyond it before deciding. The real tokenizer and reader are then checked by G0. | Notebook `notebooks/t0_3_t2_lr_2xT4.ipynb`, results in `results/t0_3_t2/` |
| P7 | ✅ **The training loop** (`scripts/train.py`) is the design's recipe on the T0.2 v2 stack: WSD schedule, batch ramp, evaluation in bits per byte on fixed held-out windows, the alarms of the design's table (DISENO 5.4: a loss spike above mean + 4σ of the last 200 steps is skipped; 3 within 200 steps stop the run), checkpoints with **a verified resume**: the loaded state is compared bit for bit with a fingerprint saved with it, and the first batch with the hash of the batch that followed the save. A time limit saves and stops before Kaggle's 12 h. The provisional reader (`ludwig_lm/data.py`) packs parquet documents with EOS into windows and resumes exactly from three integers; T1 replaces it with Ludwig 1's reader, extended to the mix. | `python scripts/test_train_cpu.py`: a run resumed at step 3, or after a pause, reproduces the uninterrupted run bit for bit on CPU |
| P8 | ✅ **Two fixes found while writing the loop.** (1) transformers turns `pad_token_id` (0, our EOS) into the embedding's `padding_idx`: that row started at zero and never got a gradient as an input, although packed training has an EOS at every document boundary. `init_ludwig` now initialises it like the other rows and drops `padding_idx`; the checkpoint format does not change. (2) `torch.distributed.checkpoint`'s `set_state_dict` reloads optimizer state through `Optimizer.load_state_dict`, which casts it to the parameter dtype and device: Muon's bf16 momentum came back in FP32 (on the GPU: +1.7 GiB per T4). `dcp.load` already restores everything in place, so `ludwig_lm/checkpoint.py` does not call it. T0.2 v2's checkpoint test (J) checked the weights, the bias and the balancer, not the momentum's dtype. | Reproduced on CPU with the tiny model |

---

## Model

### D01. Shape
✅ 26 layers × width 1024; GQA with 16 query heads, 4 KV heads, head size 64; MoE in every layer. **1.963B total parameters, 0.491B active per token**, plus 75M for the MTP head.

- **Why:** this is MobileMoE-M's shape (Meta, 2026): deep and thin, the right trade-off at this size. The states fit two 15 GiB T4s with FSDP2 (D10), and 0.49B active parameters run fast on a CPU.
- **Rejected:** the earlier 6 × 2048 design. At this size, shallow and wide loses to deep and thin.
- 📏 **Bench v1 (2×T4):** 26 × 1024 runs at 3,856 tokens/s, 1.11× slower than the fastest shape (11 × 1536 at 4,287), and as fast as 6 × 2048 (3,866). The T4 is power-limited, so large matrices are no faster than small ones (21.8 vs 27.0 TFLOPS). Depth stays.
- **Plan B:** 32 experts (1.64B total) if T0.2 leaves less than 1.5 GiB free per GPU.

### D02. File format
⛔ *Replaced by P1 (AFMoE).* Stored as Hugging Face `Qwen3NextForCausalLM` with `layer_types` = 26 × `full_attention` (**format A**).

- **Why:** that class already has every piece we want (gated attention, QK-norm, shared expert, MTP weights) in a format transformers loads and llama.cpp converts. We maintain no modeling code of our own that could drift.
- **Evidence (T0.1, CPU):** exported and reloaded logits are identical; the GGUF converts with the MTP layer. **Stock llama.cpp crashes on load** (`GGML_ASSERT(buffer)`); with a 3-line patch, every check passes: next-token probabilities within 1.9e-6, greedy 4/4 identical, MTP speculative decoding 4/4 identical. See [`patches/README.md`](../patches/README.md).
- **Known limit:** llama.cpp still treats the model as hybrid: no context shift, and prompt reuse goes through checkpoints.

### D03. Format A/B
⛔ *Not needed: AFMoE has the gate of A and runs unpatched like B (P1).* Two identical short runs of the real model (~0.3B tokens, same data order, same LR), in T2:

- **A** = Qwen3-Next format: output gate + softmax router with a global balancing loss.
- **B** = GLM-4.5 format (`Glm4MoeForCausalLM`): no gate; sigmoid router with a bias and no auxiliary loss (DeepSeek-style, MobileMoE's recipe); QK-norm; partial RoPE; MTP. Runs in llama.cpp without patches.
- **Rule:** A goes if it beats B by ≥1.5% in validation loss, or if B is unstable in FP16. Otherwise B goes, because it runs everywhere unpatched. Expert balance, router entropy, under-used experts, loss spikes and tokens/s are also compared.

### D04. Attention
✅ QK-norm and the gate stay. ⛔ *Partial RoPE everywhere is replaced by RoPE on local layers only (P1).* QK-norm (RMSNorm per head on q and k); an elementwise sigmoid gate on the attention output (`q_proj` outputs the query and the gate); partial RoPE on 50% of the dimensions with θ = 500,000.

- **Why:** QK-norm bounds attention logits, which matters in FP16. The gate removes attention sinks and massive activations and tolerates a higher LR (Qwen, *Gated Attention*, NeurIPS 2025). Partial RoPE makes context extension smoother.

### D05. Mixture of experts
✅ Experts, shared expert and kernel stay. ⛔ *The softmax router and its global balancing loss are replaced by P1–P2.* 40 routed experts (SwiGLU, 512 wide), top-4 with normalized weights, plus one shared expert (SwiGLU, 2048) with a sigmoid gate. FFN expansion (40 × 512 + 2048) / (4 × 512 + 2048) = 5.5, inside MobileMoE's useful range of 4–8.

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
⛔ *Dropped (P3): AFMoE has no MTP in llama.cpp.* One multi-token-prediction module (projection 2048 → 1024, two norms, one decoder layer, the shared head), used as the draft model for speculative decoding (`llama-server --spec-type draft-mtp`).

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
- ⏳ **Peak LR:** Ludwig 1's (same width and attention), confirmed with three short runs at ×0.5, ×1 and ×2 in T2 (P6). Ludwig 1's recipe: AdamW 5e-4 for embeddings and norms; Muon 3× that, 1.5e-3 (in Ludwig 1's A/B of October 31 the ×3 beat AdamW by 0.37 nats and ×1 lost); weight decay 0.1 applied at the AdamW LR to Muon's matrices too; 1,000 warm-up steps; WSD with a 1−√ decay over the last 20%; 131K tokens per step. Both Muons use Moonlight's scale (0.2·√max(rows, columns)), so the LRs compare directly. `scripts/train.py` uses these peaks and this weight decay by default (`--muon-wd-lr adamw`); decaying at Muon's own LR would be 3× faster.
- ✅ **Init (v1.2):** transformers' Normal(0, 0.02), plus `init_ludwig`: the norm after each sublayer starts at 1/√26 (depth-scaled sandwich norm, Trinity), the router starts random instead of at zero, and the EOS row trains like the others (P8). ⛔ *The format-A recipe (`o_proj` and `down_proj` scaled by 1/√(2·26), norm weights at 0 with scale 1 + w) does not apply: AFMoE's norms are `weight · x`, and the post-sublayer norm already sets each sublayer's scale.*

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
3. Peak memory ≤13.5 GiB **reserved** (D16a), evaluation and saving included.
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
