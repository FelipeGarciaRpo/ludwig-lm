#!/usr/bin/env python3
"""The trunk's learning rate from T2's runs (rule P9 of docs/DECISIONS.md).

Short runs favour high learning rates: the best LR falls as training gets longer, roughly as a power law
of the number of tokens. This script measures that fall with T2's runs and extrapolates it to the trunk.

1. At the shortest horizon (400 steps) it fits a parabola in log2(LR multiplier) to the final validation
   bits per byte of the finished runs: the vertex is the best LR there, and the curvature says how fast the
   loss grows away from it.
2. At a longer horizon (the ×1 and ×2 runs continued to 1,200 steps) it takes the same curvature: the gap
   between ×1 and ×2 then gives the best LR there. With three or more LRs at that horizon it fits them.
3. The exponent between the two horizons, carried to the trunk (25B tokens) plus 0.5 in log2 for the
   trunk's batch doubling (0.25M -> 0.5M tokens), gives the trunk's best LR. The trunk takes the nearest of
   ×0.5, ×0.7, ×1, ×1.4 and ×2: it never goes above ×2, the largest LR that ran long in FP16.

With only the short horizon, it uses the exponent 0.32 measured for AdamW and for Muon in the literature.

    python scripts/lr_horizon.py                    # results/t0_3_t2/*.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = {"×0.5": -1.0, "×0.7": -0.5, "×1": 0.0, "×1.4": 0.5, "×2": 1.0}   # log2 of the multiplier
PRIOR_EXPONENT = 0.32


def finished_runs(paths):
    """{(horizon in tokens, lr_mult): final val bpb} for runs that ended their schedule without a problem."""
    out, notes = {}, []
    for p in paths:
        rows = [json.loads(line) for line in open(p)]
        for tag in sorted({r["tag"] for r in rows}):
            mine = [r for r in rows if r["tag"] == tag]
            ends = [r for r in mine if r["type"] == "end" and r.get("status") == "done"]
            cfg = [r for r in mine if r["type"] == "config"]
            if not ends or not cfg or ends[-1].get("val_bpb_final") is None:
                continue
            end, args = ends[-1], cfg[-1]["args"]
            if not math.isfinite(end["val_bpb_final"]) or end.get("spikes_skipped", 0) > 0:
                notes.append(f"{tag}: left out (unstable)")
                continue
            out[(int(end["tokens"]), float(args["lr_mult"]))] = (float(end["val_bpb_final"]), tag)
    return out, notes


def vertex(points):
    """Least-squares parabola y = a + b x + c x² over (x, y): returns (x*, c), or None if it opens down."""
    x, y = np.array([p[0] for p in points]), np.array([p[1] for p in points])
    c, b, _ = np.polyfit(x, y, 2)
    return (-b / (2 * c), c) if c > 0 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="*", default=sorted(glob.glob(str(ROOT / "results" / "t0_3_t2" / "lr-*.jsonl"))))
    ap.add_argument("--trunk-tokens", type=float, default=25e9)
    ap.add_argument("--batch-shift", type=float, default=0.5, help="log2 shift for the trunk's batch doubling")
    a = ap.parse_args()
    runs, notes = finished_runs([f for f in a.files if not f.endswith("-prueba.jsonl")])
    for n in notes:
        print(n)
    horizons = sorted({h for h, _ in runs})
    if not horizons:
        raise SystemExit("no finished run")
    print("finished runs (horizon, LR, final val bpb):")
    for (h, m), (bpb, tag) in sorted(runs.items()):
        print(f"  {h / 1e6:7.1f}M tokens  ×{m:<4g} {bpb:.4f}  {tag}")

    short = horizons[0]
    pts = [(math.log2(m), bpb) for (h, m), (bpb, _) in runs.items() if h == short]
    ref = min(y for _, y in pts)
    fit = vertex([(x, y / ref) for x, y in pts]) if len(pts) >= 3 else None
    if fit is None:
        raise SystemExit(f"need three LRs whose losses curve upwards at {short / 1e6:.0f}M tokens")
    x_short, curv = fit
    inside = min(x for x, _ in pts) <= x_short <= max(x for x, _ in pts)
    print(f"\n{short / 1e6:.0f}M tokens: best LR ×{2 ** x_short:.2f}{'' if inside else ' (outside the LRs run: an extrapolation)'}, "
          f"curvature {curv:.4f} per log2 step²")

    exponent, base_x, base_h = PRIOR_EXPONENT, x_short, short
    longer = [h for h in horizons if h > short * 1.5]
    if longer:
        h = longer[-1]
        at = {m: bpb for (hh, m), (bpb, _) in runs.items() if hh == h}
        if len(at) >= 3 and (f := vertex([(math.log2(m), y / min(at.values())) for m, y in at.items()])):
            x_long = f[0]
        elif 1.0 in at and 2.0 in at:
            gap = at[1.0] / at[2.0] - 1
            x_long = (gap / curv + 1) / 2           # same curvature: gap = c (2 x* - 1)
            print(f"{h / 1e6:.0f}M tokens: ×1 is {100 * gap:.2f}% above ×2")
        else:
            x_long = None
        if x_long is not None:
            exponent = (x_short - x_long) / math.log2(h / short)
            base_x, base_h = x_long, h
            print(f"{h / 1e6:.0f}M tokens: best LR ×{2 ** x_long:.2f}; exponent measured {exponent:.3f} "
                  f"(literature: {PRIOR_EXPONENT})")
    else:
        print(f"only one horizon: exponent {PRIOR_EXPONENT} from the literature")

    x_trunk = base_x - exponent * math.log2(a.trunk_tokens / base_h) + a.batch_shift
    choice = min(GRID, key=lambda k: abs(GRID[k] - min(x_trunk, 1.0)))
    print(f"\ntrunk ({a.trunk_tokens / 1e9:.0f}B tokens, batch doubling +{a.batch_shift} in log2): best LR "
          f"×{2 ** x_trunk:.2f} -> rule P9: {choice} of Ludwig 1's recipe (Muon 1.5e-3, AdamW 5e-4), "
          f"SMEBU λ = {0.01 * 2 ** GRID[choice]:.3g}")


if __name__ == "__main__":
    main()
