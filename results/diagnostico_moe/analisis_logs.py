#!/usr/bin/env python3
"""Every number the diagnosis of the expert balance takes from the trunk's own logs (P35), recomputed.

    python results/diagnostico_moe/analisis_logs.py > results/diagnostico_moe/analisis_logs.txt

Reads results/tronco/ (the first trunk and tronco3) and results/t0_3_t2/lr-x1-1200.jsonl. numpy only.

Two definitions used below:
* lag-1 autocorrelation of a per-step series: ~0 when each step is independent of the one before (noise), near 1
  when the series moves in slow waves.
* sign flips: the share of consecutive step-to-step changes with opposite signs. Independent steps give 2/3; a
  series that goes up and down on alternate steps gives ~1 (the mark of a loop whose step is too large).
A step skipped as a loss spike is logged with load_max_over_mean 0.0 and load_min_over_mean 1.0: it is not a
measurement, and both statistics are given with and without those rows where one falls in the window.
"""
import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TRONCO = ROOT / "results" / "tronco"


def rows(files):
    out = []
    for f in files:
        out += [json.loads(line) for line in open(f, encoding="utf-8")]
    return out


def steps(rs):
    return {r["step"]: r for r in rs if r["type"] == "step"}


def series(s, a, b, key, skipped=False):
    """Steps a..b inclusive."""
    return [s[i][key] for i in range(a, b + 1) if i in s and (skipped or not s[i]["skipped_spike"])]


def ac1(x):
    x = np.asarray(x, float) - np.mean(x)
    return float((x[:-1] * x[1:]).mean() / x.var())


def flips(x):
    d = np.diff(np.asarray(x, float))
    n = int(np.sum(np.sign(d[:-1]) != np.sign(d[1:])))
    return n / (len(d) - 1), n, len(d) - 1


t3_rows = rows([TRONCO / "tronco3" / "turno01_cuenta-1.jsonl", TRONCO / "tronco3" / "turno02_cuenta-1.jsonl"])
t3, ev3 = steps(t3_rows), {r["step"]: r for r in t3_rows if r["type"] == "eval"}
t1 = steps(rows(sorted(glob.glob(str(TRONCO / "turno0*_cuenta-1.jsonl")))))
t2 = steps(rows([ROOT / "results" / "t0_3_t2" / "lr-x1-1200.jsonl"]))

print("== tronco3 around step 500, where SMEBU's step went from 0.01 to 0.001 (results/tronco/tronco3/)")
print("what changed at step 500: lam", t3[499]["lam"], "->", t3[500]["lam"], "| the floor's term (router_floor) is 0 "
      f"until step {min(i for i in t3 if t3[i]['router_floor'] > 0) - 1}")
for a, b in ((485, 494), (495, 504), (505, 514), (515, 524)):
    print(f"  steps {a}-{b}: least busy expert {np.mean(series(t3, a, b, 'load_min_over_mean')):.3f}x the mean, "
          f"busiest {np.mean(series(t3, a, b, 'load_max_over_mean')):.3f}x")
print("  50-step means from step 500: busiest "
      + ", ".join(f"{np.mean(series(t3, a, a + 49, 'load_max_over_mean')):.2f}" for a in range(500, 945, 50))
      + " | least busy " + ", ".join(f"{np.mean(series(t3, a, a + 49, 'load_min_over_mean')):.2f}" for a in range(500, 945, 50)))
print("lag-1 autocorrelation of load_max_over_mean:")
for a, b in ((300, 497), (500, 699), (700, 944)):
    skipped = [i for i in range(a, b + 1) if t3[i]["skipped_spike"]]
    line = f"  steps {a}-{b}: {ac1(series(t3, a, b, 'load_max_over_mean')):+.3f}"
    if skipped:
        line += (f" without the skipped spike of step {skipped[0]}; {ac1(series(t3, a, b, 'load_max_over_mean', True)):+.3f} "
                 "with that row (logged as 0.0)")
    print(line)

print("\n== tronco3: the bias between evaluations (rows eval, per MoE layer)")
for a, b in ((250, 498), (498, 750)):
    d = np.array(ev3[b]["bias_range"]) - np.array(ev3[a]["bias_range"])
    print(f"  bias range, step {a} -> {b}: mean {d.mean():+.3f}, from {d.min():+.3f} to {d.max():+.3f}; it shrank in "
          f"{int((d < 0).sum())} of {len(d)} layers")
for step, lam in ((250, 0.01), (750, 0.001)):
    e = ev3[step]
    mx, mn = np.array(e["layer_load_max_over_mean"]), np.array(e["layer_load_min_over_mean"])
    span = int(e["window_assignments_per_layer"] / (4 * 262144))
    travel = lambda v: span * lam * math.tanh(2 * abs(v))
    print(f"  the {span} steps before step {step} (lam {lam}): per layer, the window's busiest expert had {mx.min():.2f}-"
          f"{mx.max():.2f}x the mean and the least busy {mn.min():.2f}-{mn.max():.2f}x. SMEBU's rule, lam*tanh(2v) per "
          f"step, moves their bias {travel(mx.min() - 1):.3f}-{travel(mx.max() - 1):.3f} and "
          f"{travel(1 - mn.max()):.3f}-{travel(1 - mn.min()):.3f} in that window; the most it allows is {span * lam:.2f}")
print("  largest bias (bias_abs_max): " + ", ".join(f"step {i}: {t3[i]['bias_abs_max']:.2f}" for i in (60, 100, 240, 400, 497, 750, 944)))

print("\n== tronco3: the router's level (lse), without a router z-loss")
print("  mean lse over the MoE layers: " + ", ".join(f"step {i}: {np.mean(ev3[i]['router_lse']):.2f}" for i in sorted(ev3)))
print(f"  at step 750 by layer: {min(ev3[750]['router_lse']):.2f} to {max(ev3[750]['router_lse']):.2f}")
for name, s in (("tronco3", t3), ("first trunk (router z-loss 1e-4)", t1)):
    z0, z1 = np.mean(series(s, 0, 9, "z_router")), np.mean(series(s, 935, 944, "z_router"))
    print(f"  {name}: mean z_router of steps 0-9 {z0:.2f} (lse ~{math.sqrt(z0):.2f}), of steps 935-944 {z1:.2f} "
          f"(lse ~{math.sqrt(z1):.2f}): a fall of ~{math.sqrt(z0) - math.sqrt(z1):.2f}")

print("\n== tronco3 at step 0, before any bias (for calib_init.py)")
v = sorted(ev3[0]["val_load_max_over_mean"])
print(f"  busiest expert on the validation text, per MoE layer: {v[0]:.2f}x to {v[-1]:.2f}x the mean, median {v[len(v) // 2]:.2f}x; "
      f"router_margin {np.mean(ev3[0]['router_margin']):.4f}, lse {np.mean(ev3[0]['router_lse']):.3f}")
for step in (250, 498, 750):
    e = ev3[step]
    print(f"  step {step}: router_margin {min(e['router_margin']):.3f}-{max(e['router_margin']):.3f} (median "
          f"{np.median(e['router_margin']):.3f}), bias_range {min(e['bias_range']):.2f}-{max(e['bias_range']):.2f} "
          f"(median {np.median(e['bias_range']):.2f}), bias_decides {min(e['bias_decides']):.3f}-{max(e['bias_decides']):.3f} "
          f"(median {np.median(e['bias_decides']):.3f}), lse {min(e['router_lse']):.2f}-{max(e['router_lse']):.2f}; "
          f"busiest on the validation text {min(e['val_load_max_over_mean']):.2f}-{max(e['val_load_max_over_mean']):.2f}x")

print("\n== the first trunk at its break (SMEBU's step 0.01 throughout; results/tronco/turno0*_cuenta-1.jsonl)")
print("sign flips of the step-to-step changes of load_min_over_mean (independent steps: 0.667):")
for a, b in ((600, 1149), (1150, 1199), (1200, 1229), (1200, 1230), (1230, 1259), (1260, 1299)):
    f, n, m = flips(series(t1, a, b, "load_min_over_mean"))
    g = flips(series(t1, a, b, "load_max_over_mean"))[0]
    print(f"  steps {a}-{b}: {f:.3f} ({n} of {m}); for load_max_over_mean {g:.3f}; mean z_router "
          f"{np.mean(series(t1, a, b, 'z_router')):.2f}, mean bias_abs_max {np.mean(series(t1, a, b, 'bias_abs_max')):.3f}, "
          f"least busy {np.mean(series(t1, a, b, 'load_min_over_mean')):.2f}x")
print("lag-1 autocorrelation of load_max_over_mean: " + ", ".join(
    f"steps {a}-{b}: {ac1(series(t1, a, b, 'load_max_over_mean')):+.2f}" for a, b in ((300, 548), (700, 999), (1150, 1249), (1350, 1499))))

print("\n== T2's lr-x1-1200 at its break (SMEBU's step 0.01, router z-loss 1e-4; results/t0_3_t2/lr-x1-1200.jsonl)")
for a, b in ((400, 479), (480, 519), (520, 559), (560, 599), (600, 639)):
    f, n, m = flips(series(t2, a, b, "load_min_over_mean"))
    print(f"  steps {a}-{b}: sign flips of load_min_over_mean {f:.3f} ({n} of {m}); mean z_router "
          f"{np.mean(series(t2, a, b, 'z_router')):.2f}, least busy {np.mean(series(t2, a, b, 'load_min_over_mean')):.2f}x")
