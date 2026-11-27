#!/usr/bin/env python3
"""Four more views of the lab's runs (moe_lab.py), for the diagnosis of the expert balance (P35).

    python results/diagnostico_moe/lab_extra.py start    runs/A_base.jsonl
    python results/diagnostico_moe/lab_extra.py offsets  runs/Q_quantile.jsonl
    python results/diagnostico_moe/lab_extra.py switch   runs/A_base.jsonl runs/M_midrun.jsonl 250
    python results/diagnostico_moe/lab_extra.py compare  250 runs/A_base.jsonl runs/R_routescale4.jsonl ...

start    the router at step 0, per MoE layer, to compare with the trunk's first evaluation
offsets  what moved the experts' offsets between probes: the router's own rows, or the mean of its input
switch   two runs step by step around a step (the second one changes something there)
compare  the training and validation loss of several runs around a step and at the end
"""
import json
import sys

import numpy as np


def load(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def steps(path):
    return {r["step"]: r for r in load(path) if r["type"] == "step"}


def name(path):
    return path.replace("\\", "/").split("/")[-1].replace(".jsonl", "")


mode, args = sys.argv[1], sys.argv[2:]
if mode == "start":
    p = [r for r in load(args[0]) if r["type"] == "probe"][0]
    print(f"{name(args[0])}, step {p['step']}, per MoE layer, before any bias:")
    for key, what in (("lse", "level (lse)"), ("gap", "median gap between the 4th and 5th score"),
                      ("load_max", "busiest expert over the mean"), ("rho", "share of the router's input common to all tokens"),
                      ("off_std", "spread of the experts' offsets, logits"), ("spec_std", "spread between tokens, logits")):
        print(f"  {what}: " + ", ".join(f"{v:.3f}" for v in p[key]))
elif mode == "offsets":
    for path in args:
        probes = [r for r in load(path) if r["type"] == "probe" and "off_speed_by_w" in r]
        print(f"{name(path)}: what moved the experts' offsets, in logits per optimizer step since the previous probe "
              "(fixed tokens, means over the MoE layers)")
        print(" step | offsets: total   by the router's rows   by its mean input | level: by rows   by input | "
              "common share of the input | cos(mean input, 25 steps before)")
        for p in probes:
            if p["step"] % 50:
                continue
            m = lambda k: float(np.mean(p[k]))
            print(f"{p['step']:5d} |          {m('off_speed_rms'):.5f}        {m('off_speed_by_w'):.5f}             "
                  f"{m('off_speed_by_x'):.5f}  |      {m('level_speed_by_w'):+.5f}   {m('level_speed_by_x'):+.5f} |"
                  f"          {m('rho'):.3f}            |       {m('xm_cos_prev'):.4f}")
elif mode == "switch":
    a, b, at = steps(args[0]), steps(args[1]), int(args[2])
    print(f"{name(args[0])} against {name(args[1])} around step {at} (ce: training loss; load: busiest / least busy "
          "expert over the mean)")
    print(f" step | {name(args[0])[:22]:>22}: ce   load      bias max  decides | {name(args[1])[:22]:>22}: ce   load      "
          "bias max  decides")
    for s in list(range(at - 4, at + 16)) + list(range(at + 20, at + 60, 10)):
        if s in a and s in b:
            x, y = a[s], b[s]
            print(f"{s:5d} | {x['ce']:28.3f}   {x['load_max']:.2f}/{x['load_min']:.2f}  {x['bias_abs_max']:.3f}     "
                  f"{np.mean(x['decides']):.2f}    | {y['ce']:28.3f}   {y['load_max']:.2f}/{y['load_min']:.2f}  "
                  f"{y['bias_abs_max']:.3f}     {np.mean(y['decides']):.2f}")
elif mode == "compare":
    at, paths = int(args[0]), args[1:]
    runs = {name(p): steps(p) for p in paths}
    last = min(max(s) for s in runs.values())
    head = " | ".join(f"{k[:20]:>20}" for k in runs)
    print("training loss around step", at)
    print(f"{'step':>5} | {head}")
    for s in list(range(at - 2, at + 11)) + [at + 15, at + 20, at + 30, at + 50, at + 100]:
        if all(s in r for r in runs.values()):
            print(f"{s:5d} | " + " | ".join(f"{r[s]['ce']:20.3f}" for r in runs.values()))
    print("50-step means of the training loss")
    for a0 in range(at - 50, last - 48, 50):
        print(f"{a0:5d} | " + " | ".join(f"{np.mean([r[s]['ce'] for s in range(a0, a0 + 50)]):20.3f}" for r in runs.values()))
    print("validation loss")
    for s in sorted({s for r in runs.values() for s in r if "val_ce" in r[s]}):
        if s >= at - 50 and all(s in r and "val_ce" in r[s] for r in runs.values()):
            print(f"{s:5d} | " + " | ".join(f"{r[s]['val_ce']:20.3f}" for r in runs.values()))
    print(f"busiest / least busy expert over the mean, steps {at + 50}-{last}")
    print("      | " + " | ".join(
        f"{np.mean([r[s]['load_max'] for s in range(at + 50, last + 1)]):13.2f} / "
        f"{np.mean([r[s]['load_min'] for s in range(at + 50, last + 1)]):.2f}" for r in runs.values()))
