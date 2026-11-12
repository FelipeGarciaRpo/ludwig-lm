"""T0.1-C: can Ludwig 2 be published in the AFMoE format (Arcee Trinity) with no patches?

Same idea as smoke_format.py (format A), on a tiny `AfmoeForCausalLM` with the layout proposed for Ludwig 2:
gated attention + QK-norm; 3 local layers (sliding window, RoPE) per global layer (no RoPE); sandwich norms;
1 dense layer; sigmoid router with an expert bias; a shared expert; tied embeddings.

Checks, all against **stock** llama.cpp (Ludwig's qwen3next patch is reverted if it is applied):
  C1  export with save_pretrained and reload: identical logits
  C2  llama.cpp's converter writes the GGUF
  C3  next-token probabilities match, with prompts longer than the sliding window
  C4  greedy continuations match token by token
  C5  the expert bias changes which experts are picked, and llama.cpp follows it

    python scripts/smoke_format_afmoe.py --llama-cpp /path/to/llama.cpp
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import smoke_format as sf  # noqa: E402  (tokenizer, conversion and llama-server helpers)

ROOT = sf.ROOT
log = sf.log


def stock_llama_cpp(llama_dir: Path, jobs: int) -> bool:
    """Revert Ludwig's qwen3next patch if it is applied, so the result holds for stock llama.cpp.
    Returns True when the llama.cpp checkout has no local changes at all."""
    patch = ROOT / "patches" / "llama-cpp-hybrid-without-recurrent-layers.patch"
    applied = subprocess.run(["git", "-C", str(llama_dir), "apply", "--reverse", "--check", str(patch)],
                             capture_output=True).returncode == 0
    if applied:
        log("[build] reverting Ludwig's qwen3next patch: this test is for stock llama.cpp")
        subprocess.run(["git", "-C", str(llama_dir), "apply", "--reverse", str(patch)], check=True)
        subprocess.run(["cmake", "--build", str(llama_dir / "build"), "-j", str(jobs), "--target",
                        "llama-server", "llama-tokenize"], check=True)
    diff = subprocess.run(["git", "-C", str(llama_dir), "status", "--porcelain", "--untracked-files=no"],
                          capture_output=True, text=True).stdout.strip()
    return diff == ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-cpp", default=os.environ.get("LLAMA_CPP_DIR", str(ROOT / "llama.cpp")))
    ap.add_argument("--config", default=str(ROOT / "configs" / "tiny_afmoe.json"))
    ap.add_argument("--work", default=str(ROOT / "work" / "t0_1c"))
    ap.add_argument("--out", default=str(ROOT / "results" / "t0_1c_afmoe.json"))
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    args = ap.parse_args()

    torch.manual_seed(0)
    t0 = time.time()
    work = Path(args.work)
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    llama_dir = Path(args.llama_cpp)
    bin_dir = sf.ensure_llama_cpp(llama_dir, args.jobs)
    stock = stock_llama_cpp(llama_dir, args.jobs)

    import transformers
    from transformers import AfmoeConfig, AfmoeForCausalLM, AutoModelForCausalLM
    report = {"format": "AfmoeForCausalLM", "llama_cpp": sf.llama_commit(llama_dir), "llama_cpp_stock": stock,
              "torch": torch.__version__, "transformers": transformers.__version__}

    # ---- tiny tokenizer + tiny model with Ludwig 2's AFMoE layout
    tok = sf.train_tiny_tokenizer(work / "tiny_tokenizer", 1024, sf.KNOWN_PRE["qwen2"])
    cfg_dict = json.loads(Path(args.config).read_text())
    cfg_dict.pop("architectures", None)
    cfg_dict.pop("model_type", None)
    cfg_dict["vocab_size"] = len(tok)
    cfg = AfmoeConfig(**cfg_dict)
    window = cfg.sliding_window
    report["layer_types"] = cfg.layer_types
    model = AfmoeForCausalLM(cfg)

    ids = [i for s in sf.CORPUS for i in tok.encode(s + "<|endoftext|>", add_special_tokens=False)]
    seq = 4 * window                                   # every sequence is longer than the window
    data = torch.tensor((ids * (1 + 64 * seq // len(ids)))[: 64 * seq]).view(64, seq)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    for _ in range(args.steps):
        batch = data[torch.randint(0, data.size(0), (8,))]
        loss = model(input_ids=batch, labels=batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    log(f"[train] {args.steps} steps: loss={loss.item():.3f}")
    model.eval()

    # C5 setup: a large expert bias reroutes many tokens; selection uses score + bias, weights use the score only
    probe = data[:2, : 2 * window]
    with torch.no_grad():
        moe = [layer.mlp for layer in model.model.layers if hasattr(layer.mlp, "expert_bias")]
        before = model(input_ids=probe).logits
        g = torch.Generator().manual_seed(1)
        for m in moe:
            m.expert_bias.copy_(torch.randn(m.expert_bias.shape, generator=g) * 0.2)
        after = model(input_ids=probe).logits
    bias_effect = (before - after).abs().max().item()
    log(f"[C5] the expert bias moves the logits by up to {bias_effect:.3f}")

    # ---- C1: export and reload
    hf_dir = work / "tiny_hf"
    model.save_pretrained(hf_dir)
    tok.save_pretrained(hf_dir)
    reloaded = AutoModelForCausalLM.from_pretrained(hf_dir, dtype=torch.float32).eval()
    with torch.no_grad():
        diff = (model(input_ids=probe).logits - reloaded(input_ids=probe).logits).abs().max().item()
    report["C1_export_reload"] = {"pass": diff < 1e-5, "max_abs_diff": diff}
    log(f"[C1] reload max |diff| = {diff:.2e}")

    # ---- C2: GGUF conversion
    gguf = work / "tiny-afmoe-f32.gguf"
    r = sf.convert(llama_dir, hf_dir, gguf, pre="qwen2")
    ok = r.returncode == 0 and gguf.exists()
    report["C2_convert"] = {"pass": ok, "log_tail": (r.stdout + r.stderr)[-1500:]}
    log(f"[C2] convert: {'OK' if ok else 'FAILED'}")
    if not ok:
        log((r.stdout + r.stderr)[-3000:])
    else:
        # prompts of 2.5 windows, then 24 new tokens: local layers must forget what is outside the window
        plen, n_new = int(2.5 * window), 24
        prompts = [ids[o: o + plen] for o in (0, 37, 101, 211)]
        banned = sorted({tok.convert_tokens_to_ids(t) for t in sf.SPECIALS})
        bias = [[i, False] for i in banned]

        def hf_greedy(p):
            seqt = torch.tensor([p])
            with torch.no_grad():
                for _ in range(n_new):
                    logits = model(input_ids=seqt).logits[0, -1]
                    logits[banned] = float("-inf")
                    seqt = torch.cat([seqt, logits.argmax().view(1, 1)], dim=1)
            return seqt[0, len(p):].tolist()

        greedy_ref = [hf_greedy(p) for p in prompts]
        try:
            srv = sf.LlamaServer(bin_dir, gguf, extra=["-t", str(args.jobs)])
        except Exception as e:
            srv = None
            msg = str(e)
            for k in ("C3_next_token_probs", "C4_greedy", "C5_expert_bias"):
                report[k] = {"pass": False, "error": msg.splitlines()[0][:400], "log_tail": msg[-1500:]}
            log("[C3-C5] llama-server crashed loading the model: " + msg.splitlines()[0][:300])
        if srv is not None:
            try:
                worst, id_mismatch = 0.0, 0
                for p in prompts:
                    with torch.no_grad():
                        probs = model(input_ids=torch.tensor([p])).logits[0, -1].float().softmax(-1)
                    ref = torch.topk(probs, 5)
                    res = srv.post("/completion", {"prompt": p, "n_predict": 1, "temperature": 0, "n_probs": 5,
                                                   "cache_prompt": False})
                    got = sf.top_probs(res["completion_probabilities"][0], 5)
                    if [i for i, _ in got] != ref.indices.tolist():
                        id_mismatch += 1
                    for (i, pr) in got:
                        worst = max(worst, abs(pr - probs[i].item()))
                report["C3_next_token_probs"] = {"pass": id_mismatch == 0 and worst < 2e-3,
                                                 "max_abs_prob_diff": worst, "top5_order_mismatches": id_mismatch,
                                                 "prompt_tokens": plen, "sliding_window": window}
                log(f"[C3] top-5 order mismatches={id_mismatch}, max |p diff|={worst:.2e}")
                greedy_llama = []
                for p in prompts:
                    res = srv.post("/completion", {"prompt": p, "n_predict": n_new, "temperature": 0,
                                                   "logit_bias": bias, "return_tokens": True, "cache_prompt": False})
                    greedy_llama.append(res["tokens"])
                same = sum(a == b for a, b in zip(greedy_ref, greedy_llama))
                report["C4_greedy"] = {"pass": same == len(prompts), "identical": f"{same}/{len(prompts)}"}
                log(f"[C4] greedy identical: {same}/{len(prompts)}")
                # C5: C3 and C4 ran with the large bias, so llama.cpp matched a routing the bias had changed
                report["C5_expert_bias"] = {"pass": bias_effect > 1e-3 and report["C3_next_token_probs"]["pass"],
                                            "logit_change_from_bias": bias_effect}
            finally:
                srv.stop()

    checks = {k: v for k, v in report.items() if isinstance(v, dict) and "pass" in v}
    report["all_pass"] = all(v["pass"] for v in checks.values())
    report["minutes"] = round((time.time() - t0) / 60, 1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    log("\n=== T0.1-C summary (AFMoE, stock llama.cpp) ===")
    for k, v in checks.items():
        log(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}")
    log(f"  -> {'ALL PASS' if report['all_pass'] else 'SOME CHECKS FAILED'} ({report['minutes']} min). Report: {args.out}")
    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
