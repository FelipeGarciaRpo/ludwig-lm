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
| P2a | ⛔ *P30: λ 0.01 for the first 500 steps, then 0.001.* ✅ **SMEBU λ = 0.01.** Reviewed with real text in T0.3 + T2: it stays while the busiest expert stays under 3× the mean and no expert dies. | 60 steps of synthetic text: λ = 0.01 ended at 2.41× the mean load; λ = 0.001 at 6.86×; the fixed-sign update at 7.57×. Same loss in the three |
| D16a | ✅ **G0's memory limit (13.5 GiB) is reserved memory**, what PyTorch's allocator holds on the GPU; bench v1 measured the same. The T4 gives PyTorch ~14.56 GiB, so 13.5 leaves ~1 GiB for what a 3-month run can add. | With allocated memory every T0.2 v2 measurement would pass (12.17 GiB at the base), which says nothing about running out |
| P6 | ✅ **T0.3 and T2 run together**, before the tokenizer is ready: three runs of the real model and training loop at LR ×0.5, ×1 and ×2 around Ludwig 1's recipe (D12), 400 steps of 0.25M tokens each (~105M tokens; 100 warm-up steps, 1−√ decay over the last 80, the last 20% as in Ludwig 1), same data order, one Kaggle session each, in parallel on three accounts. **Proxy data:** FineWeb-Edu `sample-10BT` and SmolLM2's tokenizer (49,152 tokens, Apache-2.0, EOS = 0 like ours), compared in bits per byte, which do not depend on the tokenizer. The ×2 run is the FP16 stress test. **Decision rule, fixed before the results:** a run counts only if stable (no NaN, no stop for spikes, a steady FP16 scale); the lowest validation bits per byte after the decay wins; within 0.5% the lower LR wins, because short runs favour high LRs and the trunk is ~200× longer; if an end of the sweep wins, one more run beyond it before deciding. The real tokenizer and reader are then checked by G0. | Notebook `notebooks/t0_3_t2_lr_2xT4.ipynb`, results in `results/t0_3_t2/` |
| P7 | ✅ **The training loop** (`scripts/train.py`) is the design's recipe on the T0.2 v2 stack: WSD schedule, batch ramp, evaluation in bits per byte on fixed held-out windows, the alarms of the design's table (DISENO 5.4: a loss spike above mean + 4σ of the last 200 steps is skipped; 3 within 200 steps stop the run), checkpoints with **a verified resume**: the loaded state is compared bit for bit with a fingerprint saved with it, and the first batch with the hash of the batch that followed the save. A time limit saves and stops before Kaggle's 12 h. The provisional reader (`ludwig_lm/data.py`) packs parquet documents with EOS into windows and resumes exactly from three integers; T1 replaces it with Ludwig 1's reader, extended to the mix. | `python scripts/test_train_cpu.py`: a run resumed at step 3, or after a pause, reproduces the uninterrupted run bit for bit on CPU |
| P8 | ✅ **Two fixes found while writing the loop.** (1) transformers turns `pad_token_id` (0, our EOS) into the embedding's `padding_idx`: that row started at zero and never got a gradient as an input, although packed training has an EOS at every document boundary. `init_ludwig` now initialises it like the other rows and drops `padding_idx`; the checkpoint format does not change. (2) `torch.distributed.checkpoint`'s `set_state_dict` reloads optimizer state through `Optimizer.load_state_dict`, which casts it to the parameter dtype and device: Muon's bf16 momentum came back in FP32 (on the GPU: +1.7 GiB per T4). `dcp.load` already restores everything in place, so `ludwig_lm/checkpoint.py` does not call it. T0.2 v2's checkpoint test (J) checked the weights, the bias and the balancer, not the momentum's dtype. | Reproduced on CPU with the tiny model |

### After T0.3 + T2, first round (November 14, 2026)

Three 400-step runs at ×0.5, ×1 and ×2 Ludwig 1's learning rates ([`results/t0_3_t2/`](../results/t0_3_t2/), code `52f5b0e`). All stable: no NaN, no skipped step, FP16 scale fixed at 32,768, 12.92 GiB reserved, bit-exact resume.

| ID | Decision | Evidence |
|---|---|---|
| P6a | 📏 **×2 wins the short horizon**: final validation 1.3470 / 1.2223 / 1.1516 bits per byte, ×1 6.1% above ×2. As P6 asks for an end of the sweep, **×4 runs before deciding**. The speed gap between runs (3,601 / 3,428 / 3,293 tokens/s) comes from the machines, not from the LR: within each run the speed does not move with the expert load (correlation −0.07 / +0.17 / +0.01 from step 100), and the ×2 run is as fast at a load of 7.2× as at 2.1×. The qk gain grows about linearly while the LR is constant (×2: +0.002 per step) and stops with the decay: the 1,200-step runs show whether it levels off. | Steps and evaluations of the three `.jsonl` files |
| P2b | ✅ **Expert balance is judged per layer over the 50-step evaluation window.** A single step's maximum also reflects what that batch is about. **λ scales with the LR: λ = 0.01 × the multiplier.** ⛔ *P29: the ×2 and ×4 episodes were the router z-loss, not the LR; P30 sets λ.* | ×1: every window at or below 1.6× after step 150. ×2: 65 steps above 3× (27 in a row), and the windows ending at steps 300 and 320 at 3.6× and 4.2×: SMEBU took ~30 steps to catch up. Every layer within 1.11× by step 400. No dead expert |
| P9 | ✅ **The trunk's LR rule, fixed before the second round.** The best LR falls as training gets longer, as a power law of the tokens, with exponent ~0.32 measured for AdamW and for Muon. The trunk is ~240× longer than these runs, so the short winner overstates its LR: at that exponent, ×2 to ×4 at 105M tokens is ~×1 at 25B. The second round measures the exponent here: **×4** at 400 steps (λ = 0.04) closes the short sweep, and **×1 and ×2 continue from step 320 to 1,200 steps** (3× the horizon, same λ as before, decay over the last 240). `scripts/lr_horizon.py` fits the best LR at each horizon (a parabola in log2 LR), extrapolates it to 25B tokens plus 0.5 in log2 for the trunk's batch doubling, and takes the nearest of ×0.5, ×0.7, ×1, ×1.4 and ×2. The trunk never goes above ×2, the largest LR tested over a long run. With the first round and the literature's exponent it gives ×1, Ludwig 1's recipe; if the ×1–×2 gap does not shrink at 1,200 steps, it gives ×2. | Bjorck et al., *Scaling Optimal LR Across Token Horizons* (ICLR 2025); the same exponent for Muon in arXiv 2603.28743 |

### After the 64K tokenizer (November 14, 2026)

The tokenizer trained on Kaggle's CPU in 56 min ([`results/tokenizador_64k/con_transweb/`](../results/tokenizador_64k/con_transweb/)) and passed all its checks. It now lives in [`tokenizer/`](../tokenizer/), and the script that made it in [`scripts/train_tokenizer.py`](../scripts/train_tokenizer.py), unchanged.

| ID | Decision | Evidence |
|---|---|---|
| P10 | ✅ **The 64K tokenizer is fixed as trained**: `tokenizer/tokenizer.json`, sha1 `0b7da8fdb062`. 4.159 bytes per token on the trunk mix (Qwen3 4.034, OLMoE 3.719, SmolLM2 3.551) and 1.33 tokens per Spanish word (SmolLM2 2.08). Its sample included TransWeb-Edu (9%), against what had been agreed while its translator was unconfirmed. **It is not retrained without it.** TransWeb-Edu is FineWeb-Edu translated with Mistral-7B-Instruct-v0.1 (Apache-2.0), released under ODC-By: the output of a permissively licensed open model, 🟢 under D20. It stays in the Spanish mix. A different, later corpus called TransWebEdu was translated with NLLB-200 under a non-commercial license: it stays out. | TransWeb-Edu's paper, arXiv 2410.23956. The mixes compared in the run keep the trunk figure within 0.2%, so the sample's composition barely moves the tokenizer |
| P11 | ✅ **Data mode turns every added token into text.** The 64K tokenizer keeps `<think>`, `</think>` and the tool tags (ids 4–9) as added tokens that are not special, as Qwen3 does, so decoding shows them in chat. `encode_special_tokens` covers only the special ones, so with the provisional reader a `<think>` written in a web page would have become id 4. `load_tokenizer` (`ludwig_lm/data.py`) now does what the tokenizer's own loader does: no control id comes from text. The pipeline places them: EOS after each document now, and in T1 FIM on code and the chat format of instruction and reasoning data (DISENO 4.3.7: `<think>` as id 4, never as plain text). | `scripts/test_train_cpu.py` encodes every control string with the real tokenizer: no control id, exact round trip. The LR rounds keep their own code zips |
| P12 | ✅ **Padding is id 1 (`<\|pad\|>`), and BOS is id 0, the EOS, never added**, in `configs/ludwig2_afmoe.json`. Packed pre-training never pads, but in SFT the usual collators set the label to −100 wherever the input is the padding id. With padding = EOS, the real EOS of every answer would be masked too, and the model would never learn to stop. P8's fix (the `padding_idx` row starts like the others) now applies to row 1. BOS = EOS is Qwen's convention, and it matches packed training, where every document starts after an EOS. Without a BOS id, llama.cpp falls back to id 11, which here is `<\|fim_middle\|>`, for any template or tool that asks for the BOS. | transformers' `DataCollatorForLanguageModeling` masks labels equal to `pad_token_id`. T0.1b: with no `bos_token_id`, llama.cpp reported BOS = 11; with 0, BOS = 0 and `add_bos` false |

### The trunk's data (November 15, 2026)

Pipe chose mix v3 on November 13. An independent review closed its sources and internal splits on November 14 and 15 without moving any block's share ([`results/datos_tronco_v3/`](../results/datos_tronco_v3/)). The full tables are in DISENO 4.2–4.10.

| ID | Decision | Evidence |
|---|---|---|
| P13 | ✅ **Trunk mix v3:**<br>• English 36%: FineWeb-Edu dedup 14, DCLM dedup 11, FinePhrase 6, English Wikipedia and StackExchange 3, Cosmopedia v2 2.<br>• Spanish 24%: FineWeb2-HQ 13, TransWeb-Edu 8.5, Spanish Wikipedia 2, parallel text 0.5.<br>• Code 28%, math 10% (FineMath 4+ and InfiWebMath 4+) and instructions 2%.<br>English:Spanish is 60:40 in natural language. The sources are sized for a 100B horizon (trunk to 90B) with at most ~4 passes over any of them; going past 100B means revisiting them. Replaces D17's trunk table and D18's 10%. | Pipe's goals (DISENO 1): "totalmente bilingüe", Python and JS/TS at a beginner's level, reasoning and basic math |
| P14 | ✅ **Code: Stack-Edu's permissively licensed files, with their content from `common-pile/stackv2_edu_filtered`, matched by `blob_id`.**<br>• It holds ~90% of them in Python (89.2% of the files), JavaScript (88.3%) and TypeScript (90.4%): ≈5.30B, 2.39B and 1.39B tokens, which are 1.9, 2.7 and 2.5 passes at 90B.<br>• The filter is Stack-Edu's list of permissive `blob_id`s, never Common Pile's score, which is not comparable with Stack-Edu's.<br>• No download from Software Heritage and no pass over The Stack v3. Software Heritage stays optional, for the ~10% Common Pile lacks.<br>• These are the files the tokenizer saw. HTML and CSS come from The Stack v1 dedup (`data/html`, `data/css`), as in the tokenizer. | [`common_pile.txt`](../results/datos_tronco_v3/common_pile.txt); [`revision_fuentes.txt`](../results/datos_tronco_v3/revision_fuentes.txt), points 2 and 3 |
| P15 | ✅ **The remaining sources and internal splits.**<br>• DCLM (Zyphra's dedup) stays: it is CC-BY-4.0, and its card's "intended for research use only" sits under out-of-scope use, explained by performance. The sentence goes into the data card.<br>• StackExchange: `HuggingFaceH4/stack-exchange-preferences`, without Stack Overflow.<br>• Notebooks: `bigcode/starcoderdata`, `jupyter-*`.<br>• Parallel text: UN 0.35 + Europarl 0.15, without their test sets.<br>• Instructions: OpenMathReasoning 1.9 (only solutions that fit in 2K) + OpenAssistant 0.1. OpenThoughts3 stays out until audited.<br>• Stack Overflow: `mikex86/stackoverflow-posts` (June 14, 2023, CC BY-SA per row), with `Id` and `OwnerUserId` in the manifest for attribution.<br>• Splits: FinePhrase 1.5 per format; Wikipedia 1.5 + StackExchange 1.5; notebooks 1.1 + Markdown 1.0 + Stack Overflow 1.0; JS 7.3 + TS 3.9; HTML 0.7 + CSS 0.7; SQL 0.55 + Shell 0.55. | [`revision_fuentes.txt`](../results/datos_tronco_v3/revision_fuentes.txt). OLMo 2, SmolLM2 and SmolLM3 trained on DCLM and released Apache-2.0 weights |
| P16 | ✅ **Caps on model-written text, by type** (replaces D19's 10%):<br>• Free generation ≤5% of the trunk.<br>• Rewriting ≤10%, and ≤30% of the English web.<br>• Machine translation ≤10%, and ≤40% of the Spanish.<br>• Total ≤20%, with no single generator above 10%. Branches ≤35%.<br>v3 sits at ~4 / 6 / 8.5, 18.5% in total, and its largest generator is Mistral-7B at 8.5%. | Meta (arXiv 2510.01631): the good dose of rewritten text is ~30%. FinePhrase (arXiv 2604.13977): from 10% to 90% rewritten, every mix beats web alone. TransWeb-Edu (arXiv 2410.23956). The translation cap is judgment, not measurement |
| P17 | ✅ **No proxy mix test before the trunk.** Its rule could barely fail, a failure would move ~4 points that the branches can make up, and it costs 64–80 h of T4 and ~3 days. Instead, validation bits per byte per domain from step 0, reviewed at 1B and 2.25B trunk tokens. Two 0.5B runs on spare accounts during the first week remain optional. | DataDecide: comparisons with 150M models pick the winner at 1B ~80% of the time. [`revision_fuentes.txt`](../results/datos_tronco_v3/revision_fuentes.txt), point 8 |

### T1: the pre-tokenized shards (November 15, 2026)

Pipe approved pre-tokenizing on November 15. The builder is [`scripts/build_shards.py`](../scripts/build_shards.py) (`ludwig_lm/sources.py`, `ludwig_lm/build.py`), the reader [`ludwig_lm/shards.py`](../ludwig_lm/shards.py), the recipe [`configs/data_v3.json`](../configs/data_v3.json). [`scripts/test_shards_cpu.py`](../scripts/test_shards_cpu.py) tests them on a small local Hub with every kind of source.

| ID | Decision | Evidence |
|---|---|---|
| P18 | ✅ **The shard format.**<br>• Token ids as uint16, 2^27 per shard (256 MiB, about one 12-hour session).<br>• Documents joined with EOS; windows cut one after another without dropping tails.<br>• A manifest with every shard's sha256, and an index per shard: source, key, offset, length and attribution (URL, repository, path, license, author).<br>• The reader's state is one integer, the global token offset, so `seq`, `micro` and the number of GPUs may change between sessions.<br>• `build_id` = recipe + tokenizer + builder version. A build continued with another commit of the code is refused.<br>• Built on Kaggle's CPU in 3 sessions of 56 shards: the v1 trunk is 168 shards. | Built in one session or in two (6 + 4), the test's 10 shards are identical byte for byte. Training on them and resuming is bit-exact. 460–760K tokens per second on 2 CPU cores |
| P19 | ✅ **Validation: the first documents of each source's last file**, 2M tokens for the main sources. They are dropped from training by key, and **the rest of that file does train** (refines DISENO 4.3.8). Holding out whole files would cost the code 8% of the Python, 6% of the JavaScript and 20% of the TypeScript: one Common Pile shard each. Bits per byte for 5 domains: English (FineWeb-Edu + DCLM), Spanish (FineWeb2-HQ), Python, JavaScript (JS + TS) and math (FineMath + InfiWebMath). | The language ranges in [`common_pile.txt`](../results/datos_tronco_v3/common_pile.txt): 13, 16 and 5 shards. The test checks that no validation key reaches a train shard |
| P20 | ✅ **Decontamination by exact 64-bit hashes of 13-grams of tokens**, instead of a Bloom filter.<br>• A document that shares one 13-gram with a benchmark item is dropped.<br>• A 13-gram found in more than 10 items is a template ("Which of the following…") and does not count.<br>• 16 benchmarks: those of DISENO 4.3.5 plus MATH-500. | 10M 13-grams take 80 MB, with no false positives. The check costs 65 µs per 1,000-token document, ~25 min for the whole trunk. Test: the planted benchmark question is removed, the template is not |
| P21 | ✅ **MeCo tags as spans of reserved ids:** `<\|reserved_0\|>` + "URL: domain" + `<\|reserved_1\|>` before each trunk document (ids 16 and 17).<br>• Out of the loss, as in MeCo.<br>• The page's domain when there is a URL; a fixed label otherwise (`finephrase`, `transweb`, `cosmopedia`, `github.com`, `stackoverflow.com`, `un.org`…).<br>• None on chat or FIM documents, and none in validation.<br>• `--meta strip` removes the spans, so the same shards can train without tags: the 🧪 stays reversible without rebuilding. | MeCo (arXiv 2501.01956): "URL: en.wikipedia.org", domain only, the loss only on the document's tokens, metadata on the first 90% of training. The test covers the mask, spans cut by a window, and the strip mode with an exact resume |
| P22 | ✅ **The mix: the source furthest behind its share of tokens goes next.**<br>• Exact shares at every scale, with no sampling noise.<br>• Each source has a seeded shuffle buffer of 2,048 documents and a seeded file order per epoch.<br>• A source smaller than its share repeats with its files reshuffled (design 4.6). | Test: every source within 0.12 points of its weight at 82K tokens |
| P23 | ✅ **One owner per page.** Within the `cc-en` group (FineWeb-Edu, DCLM, FinePhrase, TransWeb-Edu, FineMath, InfiWebMath), a page whose normalized URL or Common Crawl record id another source already used is dropped. A source may repeat its own pages in later epochs. This is DISENO 4.3.2 and 4.3.3: FineWeb-Edu against DCLM, and the original against its rewrite and its translation. | Test: no URL in two sources, no page next to its rewrite |
| P24 | ✅ **How each kind of document is written:**<br>• **FIM** in prefix-suffix-middle order (Qwen2.5-Coder's), on 10% of the code files (Python, JS, TS, SQL, Shell, notebook scripts, HTML, CSS), split at two random characters.<br>• **Traces:** ChatML with `<think>` as ids 4 and 5. OpenMathReasoning ≤2,048 tokens; OpenAssistant takes the best-ranked path of each tree, in English and Spanish.<br>• **Stack Overflow:** a question with score ≥1 and its accepted answer, joined inside each file.<br>• **Stack Exchange:** a question and up to 3 answers with a positive score.<br>• **Parallel text:** 8–32 consecutive pairs, with the direction drawn per document.<br>• **Code:** license headers stripped, minified and generated files out, keys and passwords replaced by `<KEY>` and `<PASSWORD>`.<br>• **HTML and CSS:** emails and public IPs replaced with Common Pile's placeholders. | Test: the FIM pieces give back the original file; the planted key, header, minified file, emails and IPs are handled; `logo@2x.png` is not an email |

### The trunk's LR, the relay and the gate G0 (November 15, 2026)

| ID | Decision | Evidence |
|---|---|---|
| P25 | ✅ **The trunk trains at ×1: Ludwig 1's recipe** (Muon 1.5e-3, AdamW 5e-4), at constant LR after 2,000 warm-up steps, for 46,800 steps ([`configs/tronco_v1.json`](../configs/tronco_v1.json)). T2 is closed.<br>• Rule P9, with the literature's exponent and the completed runs only, gives ×1.<br>• At equal steps and constant LR, ×2's lead over ×1 falls from 6.65% (step 300) to 1.01% (step 800), and keeps shrinking.<br>• ×4 is the best at 400 steps, but its balance diverges instead of converging. At step 800, ×2's qk gain is 2.48, against 1.69 for ×1.<br>• The locally measured exponent (~1.0 between steps 320 and 800) is too steep to extrapolate: it would give ×0.02 at 25B. | [`results/t0_3_t2/`](../results/t0_3_t2/): `lr-x4`, `lr-x2-320-800` and `lr-x1-1200`. ×4's windows in layers 4–15 rise from 1.08 to 1.90 (max 2.30), and its load passes 3 in 289 of 300 steps. ⚠️ P29: both 1,200-step runs had a broken balance by step 800 (from steps 544 and 470) |
| P26 | ✅ **Relay storage: Hugging Face first, Kaggle as the backup** ([`ludwig_lm/relay.py`](../ludwig_lm/relay.py), [`notebooks/tronco_relevo_2xT4.ipynb`](../notebooks/tronco_relevo_2xT4.ipynb)).<br>• **The lock and its fence:** a private repo of the project's organization holds the lock, a 2-hour lease renewed every 30 min and with each upload. Every write is a commit whose parent is the session's last head, so a session whose lease was taken over fails its next commit and stops.<br>• **Uploads:** a checkpoint every 3 h, each in one commit with its index, the rotation (the newest 2 stay) and the lock's renewal.<br>• **Storage:** `super_squash_history` after each turn. That keeps it at ≤~68 GB of the 100 GB private.<br>• **The backup:** the turn's last checkpoint goes to whichever of two private Kaggle datasets holds the older one, as one tar plus its meta.<br>• **Disk:** `/tmp` never holds more than two checkpoints at once (~23 GiB of Kaggle's ~73 GB). The starting checkpoint is deleted after the turn's first save, train.py keeps only the newest periodic one (`keep` 1), and before the backup only the last stays.<br>• **Resume:** from the newest complete checkpoint among the three places; a tie goes to the copy already on disk.<br>• **Without the lock:** only when Hugging Face is down and it was agreed on WhatsApp (`SIN_CANDADO`). Then the turn reads and writes Kaggle only.<br>• **Credentials:** each account uses its own (HF token with write on the repo; Kaggle API token). | [`scripts/test_relay_cpu.py`](../scripts/test_relay_cpu.py): the lock, the fence after an expired lease, a stale commit refused, the squash, the backups taking turns. Three turns on three accounts give the uninterrupted run bit for bit. Hugging Face: `parent_commit` makes a commit fail if the branch moved; 100 GB of private storage on the free plan |
| P27 | ✅ **The gate G0 in four parts.** It is Pipe's proposal, with a dry run added first: a relay bug found in the first trunk turn would cost that turn.<br>• **G0-0:** `prueba-relevo`, ~15 min. The whole relay with the tiny model against a test repo: upload, lock, backup, squash and an exact resume.<br>• **G0-1:** `g0-16k`, ~30–60 min, on a spare account. 50 steps of the real model at 16K, without saving: condition 5.<br>• **G0-2:** the trunk's first turn. Conditions 1–4 except the resume: ≥6 h without NaN and with a stable scale, ≥2,600 tokens/s over the last hour, ≤13.5 GiB reserved, ≥2 uploads. If it passes, its checkpoints count.<br>• **G0-3:** the second turn, on another account. The resume, with the state and the first batch exact.<br>Before G0-2, read the build's `muestras.txt`. | DISENO 5.7; D16 |
| P28 | ✅ **GPU runs are not bit-for-bit repeatable, and the relay does not need them to be.**<br>• **Where it starts:** step 0 of the two ×1 runs is identical (ce 11.32779, gradient norm 3.4302, same batch), and step 1 differs at 1e-6. So the first backward starts it.<br>• **The likely sources:** atomic additions in the backward pass. The gradient of each token gathered to its 4 experts is summed in no fixed order, as are the embedding's rows. Also the backward of the memory-efficient attention SDPA uses on a T4, which PyTorch documents as non-deterministic by default.<br>• **G0 is unaffected:** a resume compares the restored state bit for bit and the first batch's hash, and both are exact.<br>• **The noise floor:** identical runs differ by ≤0.11% at 300 steps. An A/B gap below ~0.3% is noise; T2's gaps were 1–6.65%.<br>• **Deterministic mode** would cost speed for a property the relay does not use. `torch.use_deterministic_algorithms(True, warn_only=True)` on a 2-step run names each non-deterministic op, if it is ever needed. | `lr-x1.jsonl` and `lr-x1-1200.jsonl`, steps 0–3; evaluations at steps 50–300 within 0.11% |

### The router broke the first trunk; the trunk restarts (November 25, 2026)

| ID | Decision | Evidence |
|---|---|---|
| P29 | 📏 **Why the expert balance broke at step ~1250: the router z-loss lowered the router's level until SMEBU's bias decided.** *Mechanism corrected after P31's probe.*<br>• **What happened:** in the first 5 MoE layers, the busiest expert of a step went from ~1.5× the mean to ~7.8× (78% of the tokens; the ceiling is 10×), with an expert at zero tokens in every step from step 1400. Over a window the load looked even, because the hoarding experts took turns. The run stopped on loss spikes at step 3348.<br>• **The mechanism:** the router z-loss (1e-4 × logsumexp(logits)²) was made for softmax routers. With sigmoid scores it pushes every logit down, and nothing resists. The gate weights are normalized over the 4 chosen experts, so the cross-entropy cannot see the logits' level, and AdamW turns that small, steady gradient into LR-sized steps. The scores sink into the sigmoid's tail. In the first 5 MoE layers the routers' preferences were common to all tokens (P31): the tokens shared one boundary, so a bias step re-routed most of them at once, even with λ only ~1× each token's 4th–5th score gap. The busiest experts were then punished, the next group won, and the loop cycled.<br>• **The bias SMEBU had to hold shrank with the router's level, then grew chasing the imbalance:** its largest size averaged 0.32 over steps 0–550, 0.23 over 550–1100, 0.15 at the break (1150–1300) and 0.39 after step 2000. With λ fixed at 0.01, each step went from ~3% of the bias to ~7%. There is no clean threshold: just before the break, λ was 6.9% of the bias in the trunk and 5.2% in `lr-x1-1200`, but ~3% in the ×2 runs; the LR counts too. So the warning is z_router itself: in the trunk it fell below 1.5 ~250 steps before the break.<br>• **The break depends on the router's level, not on the step or the LR:** it began at the same z_router (~0.85, with λ 0.01) in two runs with very different schedules: the trunk at step 1249 and T2's `lr-x1-1200` (100 warm-up steps) at step 544. `lr-x2` broke at z ≈ 1.0–1.3, and `lr-x4` (λ 0.04) at z ≈ 2.2: a 4× larger step breaks at a higher level. The runs that never went below z ≈ 1.5 never broke.<br>• **A toy of the loop reproduces it:** sigmoid scores, the top 4 of score plus bias, and SMEBU. With λ 0.01, the busiest expert stays at 1.1× while the median gap is ≥0.0026, and collapses (9.8× of 10, in turns) at 0.0004. With λ 0.001, it stays balanced down to 0.0004. The toy gives each token its own boundary, so in it the loop breaks once λ is ~5–10× the gap; the trunk's routers did not work that way (P31).<br>• **T2's second round compared broken runs:** `lr-x1-1200` broke at step 544 and `lr-x2-320-800` at 470, before the 800 steps that P25 compares. ×1 stays (P30): it is the safe choice, and redoing the comparison would cost days. | [`results/tronco/resumen_t01-t07.txt`](../results/tronco/resumen_t01-t07.txt) and the turns' `.jsonl` (`bias_abs_max` and `z_router` per step); [`results/t0_3_t2/`](../results/t0_3_t2/). The probe: P31. The toy: [`scripts/toy_router_loop.py`](../scripts/toy_router_loop.py), output in [`results/tronco/toy_router_loop.txt`](../results/tronco/toy_router_loop.txt). ST-MoE (Zoph et al., 2022) made the router z-loss for softmax routers. DeepSeek-V3's sigmoid router has none, and its bias moves by γ = 0.001 |
| P30 | ⛔ *P32: after its first turn, `tronco3` (3 dense layers and 44 experts, the same recipe) replaces `tronco2`.* ✅ **The trunk restarts from step 0 as `tronco2`** ([`configs/tronco_v2.json`](../configs/tronco_v2.json)).<br>• **No router z-loss** (`router_z` 0, now the default). The output z-loss stays: the head is a softmax.<br>• **SMEBU: λ 0.01 for the first 500 steps, then 0.001.** 0.01 balances the start fast (P2a: 0.001 left 6.86× at step 60), and 0.001 is DeepSeek-V3's γ, with 10× the margin.<br>• **`seq_aux` 1e-3** instead of 1e-4: a balancing force that acts per token, through the router's weights. It is still lighter than the usual 1e-2.<br>• **Watched: the router's level.** Every step logs and prints z_router, the mean over the MoE layers, and an alarm fires when it stays below 1.5 for 50 steps: the first trunk crossed it ~250 steps before its break, and without the z-loss nothing should push it down. Each evaluation, every 250 steps, logs per MoE layer the level as z and as signed lse (a mean over 25 layers can hide 5 that sink, and z, a square, would rise again below 0), the score gap, how often the bias decides, the bias range and the busiest expert. It flags a layer where the bias decides for more than 80% of the tokens, the bias range passes 20× the gap, or the busiest expert passes 2× the mean: P31's probe put the healthy layers at up to 66%, 17× and 1.85×, and the broken ones from 92%, 22× and 4.49×. λ over the gap is still logged, with its alarm, but it is not a health signal (P31).<br>• **A balance stop:** the run stops and saves when the busiest expert is above 3× the mean in 80% of 50 steps, from step 200 on. It would have stopped the first trunk at step ~1385 instead of 3348. A spike stop saves too.<br>• **Unchanged:** the LR (×1, 2,000 warm-up steps), the data, and everything else.<br>• **Why restart, not resume from 3270:**<br>  – No checkpoint from before the break survives: Hugging Face has steps 3131 and 3270, and the backups 1608 and 2151.<br>  – At 3270, the bias had chosen the experts of every token in the first 5 MoE layers for ~2,000 steps. Each of those experts saw everything, in turns, and their routers learned little about tokens.<br>  – Restarting costs ~3,350 steps: ~3 days of relay, 0.84B tokens, 3.7% of the trunk. It gives a run with one configuration from step 0, and it tests the fix where the first run failed. Pipe preferred retraining for clean experts.<br>• **In the relay, each run rotates only its own checkpoints,** under `checkpoints/<tag>/`. The first trunk's 3131 and 3270 stay on Hugging Face. | P29. [`scripts/test_train_cpu.py`](../scripts/test_train_cpu.py): the λ switch, the balance stop with its save, and `--eval-only`. [`scripts/test_relay_cpu.py`](../scripts/test_relay_cpu.py): a run under a new tag starts from scratch and leaves the other run's checkpoints |
| P31 | ❌ **The probe of checkpoint 3270 failed its test** (`MODO = "sonda"`; `train.py --eval-only`). The test was: P29 holds if λ over the gap is high (≳5) in the first 5 MoE layers and low in the rest.<br>• **λ/gap was low everywhere:** 0.6–1.6 in MoE layers 0–4 and 0.5–2.5 in the rest, and the median gap was no smaller in the broken layers (0.0096 against 0.011).<br>• **The broken layers stood out by the bias:** its range was 0.23–0.42 (22–60× the gap) against 0.03–0.14 (4–17×), it changed the experts of 92–100% of the tokens against 37–66%, and their busiest expert took 4.5–7.6× the mean against 1.4–1.85×.<br>• **The instrument was wrong, not only the moment:** λ/gap assumes each token has its own boundary, so that a bias step moves only the tokens near it. These layers' routers had preferences common to all tokens, which share one boundary, and one step moves them all even at λ/gap ~1. P29's core stands, corrected; `tronco2` keeps its recipe and watches the router's level instead (P30).<br>• **Not needed:** probing 1608 and 2151. Both come after the break (1608 is ~270 steps after step 1335), so they would show the same final state. | [`results/tronco/sonda-tronco-3270_salida_kaggle.txt`](../results/tronco/sonda-tronco-3270_salida_kaggle.txt) |

### The first MoE layers become dense; the trunk restarts as tronco3 (November 25, 2026)

| ID | Decision | Evidence |
|---|---|---|
| P32 | ✅ **3 dense layers instead of 1 and 44 experts instead of 40; the trunk restarts from step 0 as `tronco3`** ([`configs/tronco_v3.json`](../configs/tronco_v3.json); the model, [`configs/ludwig2_afmoe_d3.json`](../configs/ludwig2_afmoe_d3.json)).<br>• **The model:** 1.938B total parameters (1.906B before). The active ones stay at 0.491B, because a dense layer's FFN (4,096) is as wide as a MoE layer's 4 shared and 4 routed experts of 512. MoE starts at layer 3 instead of 1, after the two MoE layers that broke worst in the first trunk. The AFMoE format takes it unchanged (Trinity Nano and Mini have 2 dense layers, Large 6), and a tiny model with 3 passes T0.1-C on stock llama.cpp.<br>• **44 experts per MoE layer instead of 40: Pipe's condition,** no fewer total parameters than `tronco2`. With 40, the 3 dense layers would leave 1.793B; 43 experts give 1.902B, so 44 is the smallest count above 1.906B. Each token still goes through 4, so the compute per token is the same: bench v1's expert matmuls ran at 27.0 TFLOPS with 40 experts and 26.7 with 60, and its −24% with 48 came from halving the micro-batch for memory. Memory, estimated at 5.26 bytes per parameter per GPU (9.33 GiB of states for 1.906B): +0.15 GiB, so ~13.1 GiB reserved at 2K (the first trunk peaked at 12.91) and ~13.5 at 16K (G0-1: 13.40 with 1.906B), G0's limit; 46 experts (2.010B) would put 16K at ~13.9. The router's neutral level moves from ln 40 = 3.69 to ln 44 = 3.78.<br>• **Why the first MoE layers:** they are where the bias ends up choosing the experts.<br>  – In the first trunk, MoE layers 0–4 broke, and 0 and 1 worst.<br>  – Sigma-MoE-Tiny saw the same in its lower layers with DeepSeek's bias: the biases keep growing until the router's scores no longer matter, and the expert with the highest bias takes most tokens. They fixed it by activating more experts there.<br>  – Trinity Large's routing drifted and its experts collapsed. Its fix included going from 3 to 6 dense layers (of 60), among six changes made at once; Trinity Nano and Mini use 2. DeepSeek-V3 uses 3 of 61.<br>• **The rest is `tronco2`'s recipe (P30):** no router z-loss, λ 0.01 then 0.001 from step 500, `seq_aux` 1e-3, the balance stop, and the watch of the router's level (P31).<br>• **Why now:** a restart costs only `tronco2`'s first turn (~490 steps: at ~80 s per step, its 11 hours end before step 500), and every later turn makes it dearer. `tronco2` was healthy at step 220, but so was the first trunk until step ~1150. Even without the z-loss, its router's level falls at ~45% of the first trunk's pace (z_router 13.31 at step 220, against 11.01).<br>• **Without an ablation,** like Trinity's: it is the field's usual choice for the layers that broke, not a measured one. | Sigma-MoE-Tiny technical report (Microsoft, 2025), §2.2.2 and §3.5.2. Arcee Trinity Large technical report (2026), §6 and table 2. DeepSeek-V3 technical report. `tronco2`'s turn 1, steps 0–220. [`scripts/test_train_cpu.py`](../scripts/test_train_cpu.py) builds both models with their counts. [`results/t0_1c_afmoe_d3.json`](../results/t0_1c_afmoe_d3.json): with 3 dense layers, stock llama.cpp still matches transformers (C1–C5). [`results/banco_t4_v1/reporte.txt`](../results/banco_t4_v1/reporte.txt): the expert matmuls and the 48-expert run. [`results/g0/g0_1_salida_kaggle.txt`](../results/g0/g0_1_salida_kaggle.txt): 13.40 GiB at 16K. The first trunk's peak: `peak_reserved_GiB` in `results/tronco/turno0*_cuenta-1.jsonl` |
| P33 | ⏳ **A floor for the router's level, built and off** (`--router-floor`, `--router-floor-lse`).<br>• **What it is:** a penalty (τ − mean lse)² for each MoE layer whose mean level is below τ, and zero above it. It acts on the layer's level, not on each token.<br>• **τ = 1.2,** the level of the z_router alarm (1.2² ≈ 1.5). The first trunk crossed it ~250 steps before its break.<br>• **Off in `tronco_v3.json`** (weight 0). Switching it on is a config change and a new commit, decided if the z_router alarm fires or a layer's lse sinks toward 1.2.<br>• No paper uses it: it is plan B, not the recipe. | [`scripts/test_train_cpu.py`](../scripts/test_train_cpu.py): with weight 0 it adds nothing, and with τ above the level it raises the level within 8 steps |

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
✅ Our own 64K BPE (the same as the larger Ludwig plan), with tied input and output embeddings. 📏 Fixed on November 14, 2026 (P10): [`tokenizer/`](../tokenizer/).

- **Why 64K:** the output head is 12% of the compute, against ~23% with a 128K vocabulary.
- **Special tokens from day one:** chat template, `<think>` and `</think>`, FIM, tool calls, end of text.
- **Constraint:** the pre-tokenizer regex must be one llama.cpp implements. 📏 **T0.1b** checked the final tokenizer on November 14, 2026 ([`scripts/smoke_tokenizer.py`](../scripts/smoke_tokenizer.py), [`results/t0_1b_tokenizer.json`](../results/t0_1b_tokenizer.json), llama.cpp `61dbf49`):
  - The regex is Qwen2's (`qwen2`).
  - In the AFMoE format, 17 of 17 strings tokenize as in Hugging Face, chat, reasoning, tool-call and FIM markup included.
  - llama.cpp detects every special id: EOS, end of turn, padding and FIM.
  - The vocabulary's hash is new to llama.cpp, so conversion goes through `scripts/convert_gguf.py` until the hash is registered upstream. That PR is one Pipe writes himself, as llama.cpp's rules require.

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
- ✅ **Peak LR: ×1, Ludwig 1's** (same width and attention): confirmed by T2's two rounds and rule P9 (P25). Ludwig 1's recipe: AdamW 5e-4 for embeddings and norms; Muon 3× that, 1.5e-3 (in Ludwig 1's A/B of October 31 the ×3 beat AdamW by 0.37 nats and ×1 lost); weight decay 0.1 applied at the AdamW LR to Muon's matrices too; 1,000 warm-up steps; WSD with a 1−√ decay over the last 20%; 131K tokens per step. Both Muons use Moonlight's scale (0.2·√max(rows, columns)), so the LRs compare directly. `scripts/train.py` uses these peaks and this weight decay by default (`--muon-wd-lr adamw`); decaying at Muon's own LR would be 3× faster.
- ✅ **Init (v1.2):** transformers' Normal(0, 0.02), plus `init_ludwig`: the norm after each sublayer starts at 1/√26 (depth-scaled sandwich norm, Trinity), the router starts random instead of at zero, and the EOS row trains like the others (P8). ⛔ *The format-A recipe (`o_proj` and `down_proj` scaled by 1/√(2·26), norm weights at 0 with scale 1 + w) does not apply: AFMoE's norms are `weight · x`, and the post-sublayer norm already sets each sublayer's scale.*

### D13. Context length
✅ 2K up to 50B tokens, 4K after that; the last ~0.5B tokens of the v1 branch at 16K; 32K from 250B.

### D14. One trunk, cooldown branches
✅ A single trunk with constant LR and a stationary data mix (D17). Each public version is a cooldown branch of ~10% of its milestone, taken from the latest trunk checkpoint; the trunk continues from before the branch. Averaging trunk checkpoints gives free previews between milestones.

### D15. Relay on Kaggle
✅ Each friend runs turns **on their own Kaggle account, without sharing credentials**. A lock with expiry keeps one session training at a time. Full-state checkpoint every 1–2 h (~12 GB with bf16 momentum): two rotating private checkpoints plus public milestones. Each resume is verified by hashing the first batch. ✅ How: P26.

### D16. Gate G0
✅ No trunk hour is spent until the real model and the real data loader pass all of this:

1. A sustained run of ≥6 h, with no NaN and a stable loss scaler.
2. ≥2,600 tokens/s sustained over the last hour.
3. Peak memory ≤13.5 GiB **reserved** (D16a), evaluation and saving included.
4. ≥2 checkpoints uploaded to Hugging Face and one resume on another account, with the expected first-batch hash.
5. 50 steps at 16K context without running out of memory.

✅ **How (P27):** the sustained run is the trunk's first turn, and its checkpoints count if it passes. A dry run of the relay (G0-0) and the 16K test (G0-1) come before it.

✅ **Passed** with the first trunk: G0-0 and G0-1 on November 17, conditions 1–4 except the resume in turn 1, and the resume, exact and on another account, in turn 5. The restarted trunk (P30) keeps the same relay and loop, so it does not repeat G0.

---

## Data

### D17. Mixes
✅ **Trunk**, mix v3 (P13), stationary and sized for a 100B horizon:

| English web | Code | Math | Knowledge | Synthetic textbooks | Spanish | Instructions and short reasoning |
|---|---|---|---|---|---|---|
| 31% | 28% | 10% | 3% | 2% | 24% | 2% |

✅ **Branches** (~10% of each milestone):

| English web | Code | Math | Reasoning and instructions | Knowledge | Spanish |
|---|---|---|---|---|---|
| 26% | 25% | 22% | 13% | 4% | 10% |

- **Code:** Python and JS/TS get the same tokens (11.2% each). ✅ Source: Stack-Edu's permissively licensed files, with their content from Common Pile (P14).
- **Rule:** at most ~4 passes over any source up to a 90B trunk; scarce, high-quality data goes only into the branches.

### D18. Spanish: 24%
✅ 24% of the trunk (P13), 10% in the branches, 25–30% in SFT. That puts the trunk's natural language at 60% English and 40% Spanish.

- **The goal:** a "fully bilingual" model (DISENO 1).
- **Not 90/10:** the result that a dominant language helps the other, with 90/10 beating 50/50, comes from cloned languages. The precedent for a small, truly bilingual model is CroissantLLM (1.3B), trained 1:1 on English and French.
- **Precedent:** SmolLM3 (3B) used 12% multilingual web across 5 languages and beat Qwen2.5-3B and Llama 3.2 3B on Spanish HellaSwag.
- **Cost:** human-written English web drops to 25% of the trunk, a risk for general knowledge and English benchmarks.
- **Measured with:** bits per byte per domain from step 0, reviewed at 1B and 2.25B trunk tokens (P17); tokenizer compression; MGSM, Belebele, Global-MMLU, Spanish HellaSwag and Flores (EN↔ES). If Spanish lags, the branches raise it without touching the trunk.

### D19. Synthetic data
✅ Caps by type (P16): free generation ≤5% of the trunk, rewriting ≤10%, machine translation ≤10%, ≤20% in total, no single generator above 10%, and ≤35% of each branch. At least 3 teacher families; near-duplicate removal; a style watch (n-gram diversity, teacher tics such as "Certainly!").

### D20. Licenses
✅ 🟢 open human text and outputs of permissively licensed open models (DeepSeek-R1/V3, Qwen3, QwQ, gpt-oss, Mixtral, Mistral-7B, SmolLM2). 🟡 models whose license sets conditions (for example, Llama's naming clause): out by default. 🔴 outputs of closed APIs, whose terms forbid training competing models (for example TinyGSM, made with GPT-3.5): never.

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
