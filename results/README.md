# Results

Raw reports written by the test scripts. Every number in the READMEs and the design document comes from one of these files.

| File | Test | Where it ran | What it shows |
|---|---|---|---|
| `t0_1_stock.json` | T0.1, `scripts/smoke_format.py` | CPU, llama.cpp `dd6b945` without changes | Export, reload and GGUF conversion pass; `llama-server` crashes on load (`GGML_ASSERT(buffer)`) |
| `t0_1_patched.json` | T0.1, `scripts/smoke_format.py --patch` | CPU, same llama.cpp with `patches/` applied | Every check passes: probabilities within 1.9e-6, greedy and MTP speculative decoding identical |
| `t0_2_cpu.jsonl` | T0.2 logic, `scripts/bench_train.py --config configs/tiny.json` | CPU, 2 processes (gloo) | FSDP2 + Muon + loss scaling work; checkpoints restore exactly with and without MTP; the loss falls on a fixed batch. **Speed and memory numbers in this file mean nothing**: tiny model on CPU |
| `t0_1c_afmoe.json` | T0.1-C, `scripts/smoke_format_afmoe.py` | CPU, llama.cpp `dd6b945` without changes | The AFMoE format matches transformers in llama.cpp, with prompts longer than the sliding window and a large expert bias: probabilities within 5.9e-5, greedy 4/4 |
| `banco_t4_v1/` | Bench v1, `results/banco_t4_v1/banco_t4_v1.py` | Kaggle 2×T4, torch 2.11 + CUDA 13.0 | `salida_kaggle.txt`: the full run log with its report. `resultados.json`, `reporte.txt` and `logs/` come from the notebook's output on Kaggle. Separate PyTorch model: 3,856 tokens/s for 26 × 1024 with 40 experts |
| `t0_2.jsonl` | T0.2 (format A), `notebooks/t0_2_velocidad_memoria_2xT4.ipynb` | — | Superseded by T0.2 v2 before it ran |
| `t0_2v2_intento1/` | T0.2 v2, first attempt, `notebooks/t0_2v2_afmoe_2xT4.ipynb` on code `10a7d88` | Kaggle 2×T4, torch 2.11 + CUDA 13.0 | `salida_kaggle.txt`: the run log, copied from Kaggle's log viewer. All 13 measurements ran out of memory: the router started at zero and SDPA with `enable_gqa` kept the attention weights in FP32 (fixed in `3ac380f`). No speed or memory numbers to keep |
| `t0_2v2_intento2/` | T0.2 v2, second attempt, `notebooks/t0_2v2_afmoe_2xT4.ipynb` on code `60a341f` | Kaggle 2×T4, torch 2.11 + CUDA 13.0 | `salida_kaggle.txt`: the run log, copied from Kaggle's log viewer; `t0_2v2.jsonl`: the notebook's output, one row per measurement (it adds allocated memory, MFU and the training curves). All 13 measurements ran: base 2,930 tokens/s at 14.33 GiB reserved; the per-expert loop is 17.3% faster; momentum in CPU RAM frees 1.57 GiB for 3.4%; 4K and 16K fit (2,240 and 1,228 tokens/s); checkpoint 11.29 GiB with exact restore; SMEBU λ = 0.01 ends at 2.41× the mean load |
