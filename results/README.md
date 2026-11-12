# Results

Raw reports written by the test scripts. Every number in the READMEs and the design document comes from one of these files.

| File | Test | Where it ran | What it shows |
|---|---|---|---|
| `t0_1_stock.json` | T0.1, `scripts/smoke_format.py` | CPU, llama.cpp `dd6b945` without changes | Export, reload and GGUF conversion pass; `llama-server` crashes on load (`GGML_ASSERT(buffer)`) |
| `t0_1_patched.json` | T0.1, `scripts/smoke_format.py --patch` | CPU, same llama.cpp with `patches/` applied | Every check passes: probabilities within 1.9e-6, greedy and MTP speculative decoding identical |
| `t0_2_cpu.jsonl` | T0.2 logic, `scripts/bench_train.py --config configs/tiny.json` | CPU, 2 processes (gloo) | FSDP2 + Muon + loss scaling work; checkpoints restore exactly with and without MTP; the loss falls on a fixed batch. **Speed and memory numbers in this file mean nothing**: tiny model on CPU |
| `t0_2.jsonl` | T0.2, `notebooks/t0_2_velocidad_memoria_2xT4.ipynb` | Kaggle 2×T4 | ⏳ Pending: tokens/s and peak memory of the real model |
