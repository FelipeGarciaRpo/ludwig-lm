#!/usr/bin/env python3
"""CPU logic test of the training loop (scripts/train.py) and its data reader. No GPU, no downloads.

    python scripts/test_train_cpu.py            # ~2–4 min; prints PASS or what failed

1. Writes tiny parquet files (with empty rows, accents and literal "<|endoftext|>" strings) and trains a
   1,000-token byte-level BPE tokenizer on them, in work/test_train_cpu/.
2. Reader: same windows on every run; a reader restarted from a saved state continues exactly; the two
   ranks get different data; epochs wrap; raw text never yields the EOS id; validation windows are fixed.
3. Training, 2 processes on CPU with configs/tiny_afmoe.json:
   A  8 steps with WSD decay, saves before step 3 and before the decay (step 5);
   B  resumes A at step 3: its steps 3–7 and evaluations must equal A's bit for bit, the restored state must
      match its fingerprint and the first batch its saved hash;
   P0 hits the time limit before its first step: stops without saving;
   P1 pauses before step 2 and saves it; P2 resumes it with --resume latest and must equal A;
   S  treats every step as a loss spike: the steps are skipped and the third one stops the run.
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
    keys = ["step", "tokens", "ce", "grad_norm", "scale", "lr_mult", "batch_sha", "load_max_over_mean", "bias_abs_max"]
    code, a = torchrun("A", WORK / "a.jsonl", "--save-dir", str(WORK / "ckpt_a"), "--save-at", "3")
    check(code == 0 and by(a, "end") and by(a, "end")[-1]["status"] == "done", "A runs to the end")
    sa = {r["step"]: r for r in by(a, "step")}
    ea = {r["step"]: r for r in by(a, "eval")}
    check(sorted(sa) == list(range(8)), "A logs steps 0–7")
    check(sorted(ea) == [0, 4, 5, 8], f"A evaluates at 0, 4, 5 (before the decay) and 8: {sorted(ea)}")
    check([r["reason"] for r in by(a, "save")] == ["requested", "pre_decay"], "A saves at step 3 and before the decay")
    check(sa[7]["lr_mult"] < sa[5]["lr_mult"] == 1.0, "the LR decays over the last 3 steps")
    check(all(r["ce"] == r["ce"] for r in sa.values()), "no NaN")

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


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    make_data(WORK)
    test_reader(WORK)
    test_training()
    print("PASS" if not FAILS else f"FAIL: {len(FAILS)} checks", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
