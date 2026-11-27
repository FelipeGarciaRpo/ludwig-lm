"""Summarise moe_lab.py runs: python lab_report.py runs/A_base.jsonl [more.jsonl ...]"""
import json
import sys

import numpy as np


def load(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return ([r for r in rows if r["type"] == "step"], [r for r in rows if r["type"] == "probe"],
            [r for r in rows if r["type"] == "config"][0])


def ac1(x):
    x = np.asarray(x, float) - np.mean(x)
    return float((x[:-1] * x[1:]).mean() / max(x.var(), 1e-12))


def mean(v):
    return float(np.mean(v))


for path in sys.argv[1:]:
    steps, probes, cfg = load(path)
    a = cfg["args"]
    print(f"\n=== {path}  balance={a['balance']} lam={a['lam']}->{a['lam_final']}@{a['lam_steps']} offsets={a['offsets']} "
          f"router_lr x{a['router_lr_mult']} seq_aux={a['seq_aux']} route_scale={a['route_scale']} shared={a['shared']} "
          f"({len(steps)} steps)")
    print("probes on fixed tokens (means over the MoE layers; speeds per optimizer step, since the previous probe)")
    print(" step  rho  lvl   off_sd spec_sd | score mean min/med/max | sd min/max  | b_rng  b*_rng err_max dec  gate_eff |"
          " b*spd rms/max    off spd  (W only)  lvl spd  | CE lvl push")
    for p in probes:
        if p["step"] % 50 and p is not probes[-1]:
            continue
        g = lambda k_, d=float("nan"): mean(p[k_]) if k_ in p else d
        print(f"{p['step']:5d} {g('rho'):.3f} {g('level'):+.2f} {g('off_std'):.3f} {g('spec_std'):.3f}  | "
              f"{g('score_mean_min'):.3f} {g('score_mean_med'):.3f} {g('score_mean_max'):.3f}  | {g('score_sd_min'):.3f} "
              f"{g('score_sd_max'):.3f} | {g('bias_range'):.3f}  {g('bstar_range'):.3f}  {g('bias_err_max'):.3f} "
              f"{g('decides'):.2f}   {g('gate_eff'):.2f}   | {g('bstar_speed_rms'):.5f} {g('bstar_speed_max'):.5f} "
              f" {g('off_speed_rms'):.5f} {g('step_off_rms'):.5f}  {g('level_speed'):+.5f} | {g('ce_level_push'):+.2e}")
    print("training steps, 50-step windows (load: max over layers and experts of that step's load / mean)")
    print(" from   ce    val   lr   lam    load_max(mean,p95) load_min  ac1(load_max)  bias_max  lse(min..max)  decides")
    for s0 in range(0, len(steps), 50):
        w = steps[s0:s0 + 50]
        if len(w) < 10:
            continue
        lm = [r["load_max"] for r in w]
        vals = [r["val_ce"] for r in w if "val_ce" in r]
        print(f"{w[0]['step']:5d} {mean([r['ce'] for r in w]):.3f} {(vals[0] if vals else float('nan')):.3f} "
              f"{w[-1]['lr_mult']:.2f} {w[-1]['lam']:.4f}   {mean(lm):.2f}  {np.percentile(lm, 95):.2f}        "
              f"{mean([r['load_min'] for r in w]):.2f}      {ac1(lm):+.2f}        {mean([r['bias_abs_max'] for r in w]):.3f}    "
              f"{min(min(r['lse']) for r in w):.2f}..{max(max(r['lse']) for r in w):.2f}   "
              f"{mean([np.mean(r['decides']) for r in w]):.2f}")
