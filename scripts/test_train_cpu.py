#!/usr/bin/env python3
"""CPU logic test of the training loop (scripts/train.py) and its data reader. No GPU, no downloads.

    python scripts/test_train_cpu.py            # ~2–4 min; prints PASS or what failed

1. Writes tiny parquet files (with empty rows, accents and literal "<|endoftext|>" strings) and trains a
   1,000-token byte-level BPE tokenizer on them, in work/test_train_cpu/.
2. Reader (and Muon's weight decay): same windows on every run; a reader restarted from a saved state continues exactly; the two
   ranks get different data; epochs wrap; raw text never yields the EOS id; validation windows are fixed.
3. The quantile rule of the balancer on a fixed batch (P35): it balances scores around 0.5 and scores 250 times
   smaller with the same setting.
4. Training, 2 processes on CPU with configs/tiny_afmoe.json:
   A  8 steps with WSD decay, saves before step 3 and before the decay (step 5);
   B  resumes A at step 3: its steps 3–7 and evaluations must equal A's bit for bit, the restored state must
      match its fingerprint and the first batch its saved hash;
   P0 hits the time limit before its first step: stops without saving;
   P1 pauses before step 2 and saves it; P2 resumes it with --resume latest and must equal A;
   S  treats every step as a loss spike: the steps are skipped and the third one stops the run;
   L  switches SMEBU's step from 0.01 to 0.001 at step 3 (P30);
   G  stops on a broken expert balance (a threshold no step can meet) and saves where it stopped;
   E  evaluates A's step-3 checkpoint with --eval-only: B's evaluation there, bit for bit, and no step;
   F  turns on the floor for the router's level above the level, which must rise against A's (P33).
   F0 turns it on below the level, as the trunk runs it: nothing may change against A (P34).
   Q  balances with the quantile rule (P35): better than A's SMEBU from the second step, the same bias on both
      processes, and QR resumes it at step 3 bit for bit.
   O  equalises the router's experts' offsets in full at every step (P35, --router-offsets-rate 1): the offsets on
      the validation text fall to less than half of A's, the bias decides for fewer tokens, and OR resumes it bit
      for bit. O4 takes a quarter per step: its offsets end between A's and O's.
   X  runs both and zeroes the bias at step 4 (--bias-reset-step); XR resumes it before that step and must equal
      it through the reset. XS skips that step as a loss spike: the offsets and the bias are still reset.
   Before the training: the trunk's models build with their dense layers and parameter counts (P32).
   Every step logs the router's health, and every evaluation logs it per MoE layer (P29), with the router's
   level as z and signed lse; L also sets the z_router alarm (P30) so that it must fire.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
WORK = ROOT / "work" / "test_train_cpu"
FAILS: list[str] = []


def check(cond, what):
    print(("  ok    " if cond else "  FAIL  ") + what, flush=True)
    if not cond:
        FAILS.append(what)


def make_data(d: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    rng = random.Random(0)
    words = [f"w{i}" for i in range(300)] + ["año", "niño", "café", "über", "naïve", "研究", "😀"]

    def doc():
        r = rng.random()
        if r < 0.03:
            return None
        if r < 0.06:
            return ""
        text = " ".join(rng.choice(words) for _ in range(rng.randint(3, 160)))
        if rng.random() < 0.1:
            text += " <|endoftext|> " + " ".join(rng.choice(words) for _ in range(5))
        return text

    d.mkdir(parents=True, exist_ok=True)
    texts = []
    for name, n in [("train-000.parquet", 300), ("train-001.parquet", 260), ("val-000.parquet", 200)]:
        col = [doc() for _ in range(n)]
        texts += [t for t in col if t]
        pq.write_table(pa.table({"text": col, "id": list(range(n))}), d / name, row_group_size=100)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=1000, special_tokens=["<|endoftext|>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                                       show_progress=False))
    tok.save(str(d / "tokenizer.json"))


def test_reader(d: Path):
    import torch

    from ludwig_lm.data import PackedReader, expand, load_tokenizer, peek_batch, val_windows

    print("reader", flush=True)
    tok = load_tokenizer(d / "tokenizer.json")
    check(tok.token_to_id("<|endoftext|>") == 0, "EOS is id 0")
    check(0 not in tok.encode("w1 <|endoftext|> w2", add_special_tokens=False).ids,
          "data mode: a literal <|endoftext|> in the text is not the EOS id")
    real = ROOT / "tokenizer" / "tokenizer.json"
    if real.exists():
        big = load_tokenizer(real)
        control = {a["id"] for a in json.loads(real.read_text(encoding="utf-8"))["added_tokens"]}
        ids = big.encode("a <think> b </think> <tool_call> <|im_start|>user <|endoftext|> <|fim_prefix|> c",
                         add_special_tokens=False).ids
        check(not control & set(ids) and big.decode(ids) == "a <think> b </think> <tool_call> <|im_start|>user "
              "<|endoftext|> <|fim_prefix|> c", "the 64K tokenizer in data mode: no control id from text, exact round trip")
    files = expand([str(d / "train-*.parquet")])
    kw = dict(files=files, tok=tok, seq=64, micro=2, world=2, seed=0, eos_id=0, rows_per_slice=16, column="text")

    def read(n, **extra):
        r = PackedReader(**kw, **extra)
        out = [r.next() for _ in range(n)]
        st = r.state_dict()
        r.close()
        return out, st

    a, _ = read(300, rank=0)
    again, _ = read(300, rank=0)
    check(all(torch.equal(x, y) for x, y in zip(a, again)), "same windows on every run")
    r = PackedReader(**kw, rank=0)
    first = [r.next() for _ in range(7)]
    st = r.state_dict()
    r.close()
    check(all(torch.equal(x, y) for x, y in zip(first, a[:7])), "state_dict does not disturb the stream")
    rest, _ = read(293, rank=0, state=st)
    check(all(torch.equal(x, y) for x, y in zip(rest, a[7:])), "a reader restarted from its state continues exactly")
    check(torch.equal(peek_batch(**kw, rank=0, state=st), a[7]), "peek_batch gives the batch that comes next")
    long, end = read(1500, rank=0)
    check(end["epoch"] >= 2, f"epochs wrap (1,500 batches reach epoch {end['epoch']})")
    mid = PackedReader(**kw, rank=0)
    for _ in range(1100):
        mid.next()
    st_mid = mid.state_dict()
    mid.close()
    rest_mid, _ = read(400, rank=0, state=st_mid)
    check(st_mid["epoch"] >= 1 and all(torch.equal(x, y) for x, y in zip(rest_mid, long[1100:])),
          f"exact resume in a later epoch too (epoch {st_mid['epoch']})")
    b, _ = read(20, rank=1)
    check(not any(torch.equal(x, y) for x, y in zip(a[:20], b)), "the two ranks get different windows")
    check(all(int(x.max()) < tok.get_vocab_size() and x.shape == (2, 64) for x in a), "shape and id range")
    try:
        PackedReader(**{**kw, "seed": 1}, rank=0, state=st).close()
        check(False, "a state from another data seed is refused")
    except ValueError:
        check(True, "a state from another data seed is refused")
    v0, b0 = val_windows([str(d / "val-000.parquet")], tok, 64, 4, 0, 2, 0, 16)
    v0b, b0b = val_windows([str(d / "val-000.parquet")], tok, 64, 4, 0, 2, 0, 16)
    v1, b1 = val_windows([str(d / "val-000.parquet")], tok, 64, 4, 1, 2, 0, 16)
    check(torch.equal(v0, v0b) and b0 == b0b and b0 > 0, f"validation windows are fixed ({b0} bytes on rank 0)")
    check(not torch.equal(v0, v1), "the ranks share the validation windows out")


def test_optimizer():
    import torch

    from ludwig_lm.afmoe import LudwigAfmoe, load_config
    from ludwig_lm.muon import build_optimizers

    print("optimizer", flush=True)
    cfg = load_config(ROOT / "configs" / "tiny_afmoe.json")
    torch.manual_seed(0)
    model = LudwigAfmoe(cfg)
    lr, adamw_lr, wd = 1.5e-3, 5e-4, 0.1
    muon, _ = build_optimizers(model, lr=lr, weight_decay=wd, n_heads=cfg.num_attention_heads, head_dim=cfg.head_dim,
                               fused_qgate=False, adamw_lr=adamw_lr, muon_weight_decay=wd * adamw_lr / lr)
    p = muon.param_groups[0]["params"][0]
    before = p.detach().clone()
    for q in model.parameters():
        q.grad = torch.zeros_like(q)
    muon.step()                                   # zero gradient: only the weight decay moves the matrix
    check(torch.allclose(p.detach(), before * (1 - adamw_lr * wd), rtol=0, atol=1e-9),
          "Muon decays its matrices at the AdamW LR x wd (Ludwig 1), not at its own LR")


def test_trunk_model():
    import torch

    from ludwig_lm.afmoe import LudwigAfmoe, load_config, moe_blocks

    print("the trunk's model", flush=True)
    for name, dense, total in (("ludwig2_afmoe.json", 1, 1.906), ("ludwig2_afmoe_d3.json", 3, 1.938)):
        cfg = load_config(ROOT / "configs" / name)
        with torch.device("meta"):
            m = LudwigAfmoe(cfg)
        n = sum(p.numel() for p in m.parameters()) / 1e9
        check(cfg.num_dense_layers == dense and len(moe_blocks(m.lm)) == cfg.num_hidden_layers - dense
              and round(n, 3) == total, f"{name}: {dense} dense layers, {cfg.num_hidden_layers - dense} MoE of {cfg.num_experts} experts, {n:.3f}B")


def test_balancer():
    import torch

    from ludwig_lm.afmoe import bias_targets

    print("the quantile rule (P35)", flush=True)
    g = torch.Generator().manual_seed(0)
    n, e, k = 4096, 8, 2
    # each expert has its own offset, common to all tokens, next to what each token prefers
    logits = torch.randn(e, generator=g) + 0.6 * torch.randn(n, e, generator=g)
    for what, level in (("around 0.5", 0.0), ("around 0.002", -6.0)):
        scores = torch.sigmoid(logits + level)

        def load(bias):
            c = torch.bincount(torch.topk(scores + bias, k, dim=1).indices.reshape(-1), minlength=e).float()
            return c / c.mean()

        bias = torch.zeros(e)
        before = load(bias)
        for _ in range(20):
            bias = bias + 0.5 * (bias_targets(scores, bias, k) - bias)
            bias = bias - bias.mean()
        after = load(bias)
        check(float(before.max()) > 2 and float(after.max()) < 1.05 and float(after.min()) > 0.95,
              f"scores {what}: 20 half-way moves take the busiest expert from {float(before.max()):.2f}× the mean to "
              f"{float(after.max()):.2f}× and the least busy to {float(after.min()):.2f}× (bias range "
              f"{float(bias.max() - bias.min()):.4f})")

    from ludwig_lm.afmoe import ExpertBalancer, LudwigAfmoe, load_config

    cfg = load_config(ROOT / "configs" / "tiny_afmoe.json")
    torch.manual_seed(0)
    balancer = ExpertBalancer(LudwigAfmoe(cfg), rule="quantile")
    lg = [torch.randn(64, cfg.num_experts, generator=g) for _ in balancer.blocks]
    chosen = [torch.topk(x, cfg.num_experts_per_tok, dim=1).indices for x in lg]
    balancer.accumulate(chosen, lg)
    balancer.step()
    held = [b.expert_bias.clone() for b in balancer.blocks]
    lg[0][0, 0] = float("nan")
    balancer.accumulate(chosen, lg)
    balancer.step()
    check(all(float(h.abs().max()) > 0 for h in held)
          and all(torch.equal(b.expert_bias, h) for b, h in zip(balancer.blocks, held)),
          "a step with a non-finite score leaves the bias as it was")
    torch.manual_seed(0)
    balancer = ExpertBalancer(LudwigAfmoe(cfg), rule="quantile", offsets="equal", offsets_rate=1.0)
    rows = [b.router.gate.weight for b in balancer.blocks]
    xs = [torch.randn(cfg.hidden_size, generator=g) for _ in rows]
    spread = lambda: max(float((w.detach() @ x).std()) for w, x in zip(rows, xs))
    start = spread()
    balancer.accumulate(chosen, lg, xs)
    balancer.equalise_offsets()
    equal = [w.detach().clone() for w in rows]
    done = spread()
    balancer.accumulate(chosen, lg, [x * float("inf") for x in xs])
    balancer.equalise_offsets()
    check(start > 0.01 and done < 1e-6 and all(torch.equal(w.detach(), e) for w, e in zip(rows, equal)),
          f"equal offsets at rate 1 take the spread of w · x_mean from {start:.3f} to {done:.1e} in one step, and a "
          "non-finite input leaves the rows as they were")


def test_public_paths():
    sys.path.insert(0, str(ROOT / "scripts"))
    from train import public

    print("paths in logs", flush=True)
    check(public('{"resume": "/kaggle/input/notebooks/juan_k/ludwig-t2/checkpoint"}')
          == '{"resume": "/kaggle/input/notebooks/<usuario>/ludwig-t2/checkpoint"}'
          and public("/kaggle/input/datasets/ana99/ludwig-code/x") == "/kaggle/input/datasets/<usuario>/ludwig-code/x"
          and public("/tmp/ludwig_ckpt/step_000320") == "/tmp/ludwig_ckpt/step_000320",
          "the Kaggle username of an attached input never reaches the logs")


def torchrun(tag, out, *args):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2",
           str(ROOT / "scripts" / "train.py"), "--config", str(ROOT / "configs" / "tiny_afmoe.json"),
           "--tokenizer", str(WORK / "tokenizer.json"), "--train", str(WORK / "train-*.parquet"),
           "--val", str(WORK / "val-000.parquet"), "--seq", "64", "--micro", "2", "--tokens-per-step", "512",
           "--rows-per-slice", "16", "--steps", "8", "--warmup", "2", "--decay-steps", "3", "--eval-every", "4",
           "--eval-windows", "4", "--log-every", "1", "--tag", tag, "--out", str(out), *args]
    env = dict(os.environ, OMP_NUM_THREADS="1")
    if os.name == "nt":
        env["USE_LIBUV"] = "0"           # torchrun's store on Windows
    p = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:], p.stderr[-3000:], sep="\n")
    rows = [json.loads(line) for line in open(out)] if Path(out).exists() else []
    return p.returncode, rows


def by(rows, kind):
    return [r for r in rows if r["type"] == kind]


def same(r1, r2, keys):
    return all(r1.get(k) == r2.get(k) for k in keys)


def test_training():
    print("training (2 processes on CPU)", flush=True)
    keys = ["step", "tokens", "ce", "grad_norm", "scale", "lr_mult", "batch_sha", "load_max_over_mean", "bias_abs_max",
            "margin_min", "bias_decides_max", "lam"]
    code, a = torchrun("A", WORK / "a.jsonl", "--save-dir", str(WORK / "ckpt_a"), "--save-at", "3")
    check(code == 0 and by(a, "end") and by(a, "end")[-1]["status"] == "done", "A runs to the end")
    sa = {r["step"]: r for r in by(a, "step")}
    ea = {r["step"]: r for r in by(a, "eval")}
    check(sorted(sa) == list(range(8)), "A logs steps 0–7")
    check(sorted(ea) == [0, 4, 5, 8], f"A evaluates at 0, 4, 5 (before the decay) and 8: {sorted(ea)}")
    check([r["reason"] for r in by(a, "save")] == ["requested", "pre_decay"], "A saves at step 3 and before the decay")
    check(sa[7]["lr_mult"] < sa[5]["lr_mult"] == 1.0, "the LR decays over the last 3 steps")
    check(all(r["ce"] == r["ce"] for r in sa.values()), "no NaN")
    check(all(r["margin_min"] > 0 and 0 <= r["bias_decides_max"] <= 1 and r["lam_over_margin"] > 0 and r["lam"] == 0.01
              for r in sa.values()), "every step logs the router's health: score gap, λ over it, how often the bias decides")
    per_layer = ("router_margin", "bias_decides", "bias_range", "val_load_max_over_mean", "router_z", "router_lse",
                 "router_common", "router_offset_std", "router_specific_std", "score_mean_min", "score_mean_max",
                 "score_sd_min", "score_sd_max", "bias_score_corr")
    check(all(r[k] and len(r[k]) == len(ea[0]["router_margin"]) for r in ea.values() for k in per_layer),
          f"every evaluation logs the router per MoE layer ({len(ea[0]['router_margin'])} layers)")
    check(all(z >= lse * lse - 1e-4 for r in ea.values() for z, lse in zip(r["router_z"], r["router_lse"])),
          "each layer's level: z, the mean of lse², is at least the square of the signed mean lse")
    check(all(0 < c < 1 and o > 0 and s > 0 and 0 < lo <= hi < 1 and abs(cr) <= 1
              for r in ea.values() for c, o, s, lo, hi, cr in zip(r["router_common"], r["router_offset_std"],
                                                                 r["router_specific_std"], r["score_mean_min"],
                                                                 r["score_mean_max"], r["bias_score_corr"]))
          and all(cr == 0 for cr in ea[0]["bias_score_corr"]) and all(cr < 0 for cr in ea[8]["bias_score_corr"]),
          f"every evaluation logs what the bias has to cancel (P35): at step 8, {min(ea[8]['router_common']):.0%} to "
          f"{max(ea[8]['router_common']):.0%} of the router's input is common to all tokens, and the bias runs against "
          f"the experts' mean scores (correlation {max(ea[8]['bias_score_corr']):+.2f} at most)")

    code, b = torchrun("B", WORK / "b.jsonl", "--resume", str(WORK / "ckpt_a" / "step_000003"),
                       "--save-dir", str(WORK / "ckpt_b"))
    check(code == 0, "B resumes and runs to the end")
    res, chk = by(b, "resume"), by(b, "resume_check")
    check(bool(res) and res[0]["state_exact"], "B: the loaded state matches its fingerprint bit for bit")
    check(bool(chk) and chk[0]["batch_exact"], "B: the first batch matches the saved hash")
    sb = {r["step"]: r for r in by(b, "step")}
    eb = {r["step"]: r for r in by(b, "eval")}
    check(sorted(sb) == [3, 4, 5, 6, 7], "B runs steps 3–7")
    check(all(same(sa[s], sb[s], keys) for s in sb), "B's steps 3–7 equal A's bit for bit")
    check(all(eb[s]["val_bpb"] == ea[s]["val_bpb"] for s in (4, 5, 8)), "B's evaluations equal A's")
    check(3 in eb and eb[3]["after_resume"], "B evaluates right after resuming")

    code, p0 = torchrun("P0", WORK / "p0.jsonl", "--save-dir", str(WORK / "ckpt_p0"), "--max-hours", "1e-6")
    check(code == 0 and by(p0, "end")[-1]["status"] == "paused" and not list((WORK / "ckpt_p0").glob("step_*")),
          "P0: a time limit before the first step stops without saving (nothing to save yet)")
    code, p1 = torchrun("P1", WORK / "p1.jsonl", "--save-dir", str(WORK / "ckpt_p"), "--pause-at-step", "2")
    check(code == 0 and by(p1, "end")[-1]["status"] == "paused" and (WORK / "ckpt_p" / "step_000002" / "meta.json").exists(),
          "P1 pauses before step 2 and saves it, as on the time limit")
    code, p2 = torchrun("P2", WORK / "p2.jsonl", "--save-dir", str(WORK / "ckpt_p"), "--resume", "latest")
    sp = {r["step"]: r for r in by(p2, "step")}
    check(code == 0 and sorted(sp) == list(range(2, 8)), "P2 resumes the latest checkpoint and runs steps 2–7")
    check(all(same(sa[s], sp[s], keys) for s in sp), "P2 equals A bit for bit")
    check(bool(by(p2, "resume_check")) and by(p2, "resume_check")[0]["batch_exact"], "P2: first batch hash")
    code, sp_rows = torchrun("S", WORK / "s.jsonl", "--spike-min-history", "2", "--spike-sigma", "-100")
    end_s = (by(sp_rows, "end") or [{}])[-1]
    skipped = [r for r in by(sp_rows, "step") if r["skipped_spike"]]
    check(code == 0 and end_s.get("status") == "stopped_spikes" and end_s.get("spikes_skipped") == 3
          and [r["step"] for r in skipped] == [2, 3, 4] and all(r["grad_norm"] is None for r in skipped),
          "S: forced loss spikes are skipped and the third one stops the run")
    end = by(p2, "end")[-1]
    check(end["steps_done"] == 8 and end["val_bpb_final"] == ea[8]["val_bpb"] and end["val_bpb_pre_decay"] == ea[5]["val_bpb"],
          "P2's summary has A's final and pre-decay validation")

    code, lr = torchrun("L", WORK / "l.jsonl", "--steps", "5", "--lam", "0.01", "--lam-final", "0.001", "--lam-steps", "3",
                        "--z-router-alarm", "1e9", "--z-router-window", "2")
    check(code == 0 and [r["lam"] for r in by(lr, "step")] == [0.01, 0.01, 0.01, 0.001, 0.001],
          "L: SMEBU's step is 0.01 for steps 0–2 and 0.001 from step 3")
    zal = [r for r in by(lr, "alarm") if r.get("alarm") == "z_router_below_alarm"]
    check(code == 0 and len(zal) == 1 and zal[0]["step"] == 1 and zal[0]["window"] == 2,
          "L: the z_router alarm fires once its window is full of steps below the threshold (P30)")
    code, g = torchrun("G", WORK / "g.jsonl", "--save-dir", str(WORK / "ckpt_g"), "--balance-stop", "0.5",
                       "--balance-window", "3", "--balance-frac", "0.6", "--balance-min-step", "0")
    end_g = (by(g, "end") or [{}])[-1]
    check(code == 0 and end_g.get("status") == "stopped_balance" and end_g.get("steps_done") == 3
          and [r["reason"] for r in by(g, "save")] == ["stopped_balance"] and (WORK / "ckpt_g" / "step_000003").exists()
          and any(r.get("alarm") == "expert_balance_broke" for r in by(g, "alarm")),
          "G: a broken balance stops the run after its window, and it saves where it stopped")
    check(all(r["router_floor"] >= 0 for r in sa.values()), "every step logs the level floor's term (P33; off: weight 0)")
    code, f = torchrun("F", WORK / "f.jsonl", "--router-floor", "1", "--router-floor-lse", "50")
    sf, ef = {r["step"]: r for r in by(f, "step")}, {r["step"]: r for r in by(f, "eval")}
    mean = lambda xs: sum(xs) / len(xs)
    check(code == 0 and all(r["router_floor"] > 0 and r["ce"] == r["ce"] for r in sf.values())
          and mean(ef[8]["router_lse"]) > mean(ea[8]["router_lse"]),
          f"F: a floor above the level raises it (mean lse at step 8: {mean(ef[8]['router_lse']):.3f} with it, "
          f"{mean(ea[8]['router_lse']):.3f} without)")
    code, f0 = torchrun("F0", WORK / "f0.jsonl", "--router-floor", "0.01", "--router-floor-lse", "-50")
    s0, e0 = {r["step"]: r for r in by(f0, "step")}, {r["step"]: r for r in by(f0, "eval")}
    check(code == 0 and s0.keys() == sa.keys()
          and all(s0[k]["ce"] == sa[k]["ce"] and s0[k]["grad_norm"] == sa[k]["grad_norm"] and s0[k]["router_floor"] == 0
                  for k in sa)
          and e0[8]["val_bpb"] == ea[8]["val_bpb"] and e0[8]["router_lse"] == ea[8]["router_lse"],
          "F0: a floor below the level changes nothing: the same loss, gradient norm and router levels as A (P34)")
    check(all(r["bias_ranks_equal"] and 0 < r["bias_step_max"] < 2 * r["lam"] for r in sa.values()),
          "A: both processes hold the same bias after every step, and SMEBU never moves it by 2λ or more (λ from "
          "its clamp, and as much again from the re-centring)")
    code, q = torchrun("Q", WORK / "q.jsonl", "--balance", "quantile", "--save-dir", str(WORK / "ckpt_q"),
                       "--save-at", "3")
    sq = {r["step"]: r for r in by(q, "step")}
    check(code == 0 and sorted(sq) == list(range(8)) and all(r["ce"] == r["ce"] and r["lam"] == 0 for r in sq.values()),
          "Q: the quantile rule runs to the end, with no step size to log (λ 0)")
    check(all(r["bias_ranks_equal"] and r["bias_step_max"] > 0 for r in sq.values())
          and not any(r.get("alarm") == "expert_bias_differs_between_gpus" for r in by(q, "alarm")),
          "Q: both processes reach the same bias at every step, from the same pooled sample")
    late = range(2, 8)
    q_min, a_min = mean([sq[i]["load_min_over_mean"] for i in late]), mean([sa[i]["load_min_over_mean"] for i in late])
    q_max, a_max = mean([sq[i]["load_max_over_mean"] for i in late]), mean([sa[i]["load_max_over_mean"] for i in late])
    check(q_min > a_min + 0.1 and q_max < a_max,
          f"Q balances faster than A's SMEBU: over steps 2–7 its least busy expert has {q_min:.2f}× the mean "
          f"({a_min:.2f}× in A) and its busiest {q_max:.2f}× ({a_max:.2f}×)")
    code, qr = torchrun("QR", WORK / "qr.jsonl", "--balance", "quantile", "--resume",
                        str(WORK / "ckpt_q" / "step_000003"), "--save-dir", str(WORK / "ckpt_qr"))
    sqr = {r["step"]: r for r in by(qr, "step")}
    check(code == 0 and sorted(sqr) == [3, 4, 5, 6, 7] and by(qr, "resume")[0]["state_exact"]
          and by(qr, "resume_check")[0]["batch_exact"]
          and all(same(sq[i], sqr[i], keys + ["load_min_over_mean", "bias_step_max"]) for i in sqr),
          "QR: a run resumed with the quantile rule equals the uninterrupted one bit for bit (its sample needs no "
          "random numbers)")
    more = keys + ["load_min_over_mean", "bias_step_max"]
    equal = ("--router-offsets", "equal", "--router-offsets-rate", "1")
    code, o = torchrun("O", WORK / "o.jsonl", *equal, "--save-dir", str(WORK / "ckpt_o"), "--save-at", "3")
    so, eo = {r["step"]: r for r in by(o, "step")}, {r["step"]: r for r in by(o, "eval")}
    check(code == 0 and sorted(so) == list(range(8)) and all(r["bias_ranks_equal"] for r in so.values()),
          "O: equal offsets run to the end under FSDP2, each process editing its own rows of the router")
    check(all(x < 0.5 * y for i in (4, 5, 8) for x, y in zip(eo[i]["router_offset_std"], ea[i]["router_offset_std"]))
          and eo[0]["router_offset_std"] == ea[0]["router_offset_std"],
          f"O: on the validation text every MoE layer's offsets are under half of A's (at step 8: "
          f"{max(eo[8]['router_offset_std']):.3f} at most, against {min(ea[8]['router_offset_std']):.3f} at least)")
    o_min = mean([so[i]["load_min_over_mean"] for i in late])
    check(mean(eo[8]["bias_decides"]) < mean(ea[8]["bias_decides"]) and o_min > a_min + 0.1,
          f"O: with nothing common to cancel, the bias decides for {mean(eo[8]['bias_decides']):.0%} of the tokens at "
          f"step 8 ({mean(ea[8]['bias_decides']):.0%} in A), and the same SMEBU balances better (least busy expert "
          f"{o_min:.2f}× the mean over steps 2–7, {a_min:.2f}× in A)")
    code, orr = torchrun("OR", WORK / "or.jsonl", *equal, "--resume", str(WORK / "ckpt_o" / "step_000003"),
                         "--save-dir", str(WORK / "ckpt_or"))
    sor, eor = {r["step"]: r for r in by(orr, "step")}, {r["step"]: r for r in by(orr, "eval")}
    check(code == 0 and sorted(sor) == [3, 4, 5, 6, 7] and by(orr, "resume")[0]["state_exact"]
          and all(same(so[i], sor[i], more) for i in sor)
          and all(eor[i]["val_bpb"] == eo[i]["val_bpb"] and eor[i]["router_offset_std"] == eo[i]["router_offset_std"]
                  for i in (4, 5, 8)),
          "OR: a run resumed with equal offsets equals the uninterrupted one bit for bit (they keep no state)")
    code, o4 = torchrun("O4", WORK / "o4.jsonl", "--router-offsets", "equal", "--router-offsets-rate", "0.25")
    eo4 = {r["step"]: r for r in by(o4, "eval")}
    check(code == 0 and all(lo < x < hi for lo, x, hi in zip(eo[8]["router_offset_std"], eo4[8]["router_offset_std"],
                                                           ea[8]["router_offset_std"])),
          f"O4: taking a quarter of the differences per step leaves the offsets between O's and A's at step 8 "
          f"({max(eo4[8]['router_offset_std']):.3f} at most)")
    fix = ("--balance", "quantile", "--router-offsets", "equal", "--bias-reset-step", "4")
    code, x = torchrun("X", WORK / "x.jsonl", *fix, "--save-dir", str(WORK / "ckpt_x"), "--save-at", "3")
    sx, ex = {r["step"]: r for r in by(x, "step")}, {r["step"]: r for r in by(x, "eval")}
    check(code == 0 and sorted(sx) == list(range(8)) and sx[3]["bias_abs_max"] > 0 and sx[4]["bias_abs_max"] == 0
          and sx[5]["bias_abs_max"] > 0 and all(d == 0 for d in ex[5]["bias_decides"])
          and all(r["bias_ranks_equal"] for r in sx.values()),
          "X: --bias-reset-step 4 leaves the bias at zero after step 4, on both processes, and the rule takes it "
          "from there")
    before, after = ex[4]["router_offset_std"], ex[5]["router_offset_std"]
    check(all(y < x for x, y in zip(before, after)) and mean(after) < 0.5 * mean(before),
          f"X: that step also equalises the offsets in full: on the validation text they go from {mean(before):.3f} "
          f"(after four steps of a tenth each) to {mean(after):.3f}, mean of the MoE layers")
    code, xr = torchrun("XR", WORK / "xr.jsonl", *fix, "--resume", str(WORK / "ckpt_x" / "step_000003"),
                        "--save-dir", str(WORK / "ckpt_xr"))
    sxr, exr = {r["step"]: r for r in by(xr, "step")}, {r["step"]: r for r in by(xr, "eval")}
    check(code == 0 and sorted(sxr) == [3, 4, 5, 6, 7] and by(xr, "resume")[0]["state_exact"]
          and all(same(sx[i], sxr[i], more) for i in sxr)
          and all(exr[i]["val_bpb"] == ex[i]["val_bpb"] for i in (4, 5, 8)),
          "XR: resumed before the reset, the run equals the uninterrupted one through it, bit for bit: the reset "
          "belongs to a step, not to the resume")
    code, xs = torchrun("XS", WORK / "xs.jsonl", "--router-offsets", "equal", "--bias-reset-step", "2",
                        "--spike-min-history", "2", "--spike-sigma", "-100", "--save-dir", str(WORK / "ckpt_xs"))
    code2, xe = torchrun("XE", WORK / "xe.jsonl", "--router-offsets", "equal", "--resume", "latest", "--save-dir",
                         str(WORK / "ckpt_xs"), "--eval-only")
    exe = (by(xe, "eval") or [{}])[-1]
    check(code == 0 and code2 == 0 and (by(xs, "end") or [{}])[-1].get("status") == "stopped_spikes"
          and exe.get("step") == 5 and all(b == 0 for b in exe["bias_range"])
          and all(v < 0.5 * y for v, y in zip(exe["router_offset_std"], ea[0]["router_offset_std"])),
          "XS: when the reset step is skipped as a loss spike, the offsets are still equalised and the bias zeroed "
          "(the stop's checkpoint has no bias and small offsets)")
    code, e = torchrun("E", WORK / "e.jsonl", "--resume", str(WORK / "ckpt_a" / "step_000003"), "--eval-only")
    ee = by(e, "eval")
    check(code == 0 and (by(e, "end") or [{}])[-1].get("status") == "evaluated" and not by(e, "step")
          and len(ee) == 1 and ee[0]["step"] == 3 and ee[0]["val_bpb"] == eb[3]["val_bpb"]
          and ee[0]["router_margin"] == eb[3]["router_margin"] and ee[0]["router_common"] == eb[3]["router_common"]
          and ee[0]["router_offset_std"] == eb[3]["router_offset_std"],
          "E: --eval-only evaluates A's step 3 as B did after resuming, router included, and takes no step")


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    make_data(WORK)
    test_reader(WORK)
    test_optimizer()
    test_trunk_model()
    test_public_paths()
    test_balancer()
    test_training()
    print("PASS" if not FAILS else f"FAIL: {len(FAILS)} checks", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
