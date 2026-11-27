"""One line per lab run: python lab_table.py runs/A_base.jsonl ... [--from 300 --to 500]"""
import json
import sys

import numpy as np

args = [a for a in sys.argv[1:] if a.endswith(".jsonl")]
lo = int(sys.argv[sys.argv.index("--from") + 1]) if "--from" in sys.argv else 300
hi = int(sys.argv[sys.argv.index("--to") + 1]) if "--to" in sys.argv else 500


def zig(x):
    d = np.diff(np.asarray(x, float))
    if len(d) < 4 or d.std() == 0:
        return float("nan"), float("nan")
    return float(np.corrcoef(d[:-1], d[1:])[0, 1]), float(np.mean(np.sign(d[:-1]) != np.sign(d[1:])))


def ac1(x):
    x = np.asarray(x, float) - np.mean(x)
    return float((x[:-1] * x[1:]).mean() / max(x.var(), 1e-12))


print(f"steps {lo}-{hi}: load = each step's busiest (least busy) expert over the mean, max (min) over the MoE layers")
print(f"{'run':24s} busiest mean  p95   max | least mean  min | ac1   zigzag(min) flips | bias range  decides | "
      f"b* speed rms/max  | lse min..max | val ce (step)")
for path in args:
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    st = [r for r in rows if r["type"] == "step" and lo <= r["step"] < hi]
    pr = [r for r in rows if r["type"] == "probe" and lo <= r["step"] <= hi]
    if not st:
        continue
    lm, ln = [r["load_max"] for r in st], [r["load_min"] for r in st]
    z, fl = zig(ln)
    m = lambda k: float(np.mean([np.mean(p[k]) for p in pr if k in p])) if pr else float("nan")
    val = [(r["val_ce"], r["step"]) for r in rows if r["type"] == "step" and "val_ce" in r and r["step"] <= hi]
    name = path.replace("\\", "/").split("/")[-1].replace(".jsonl", "")
    print(f"{name:24s}    {np.mean(lm):5.2f}   {np.percentile(lm, 95):5.2f} {np.max(lm):5.2f} |   {np.mean(ln):5.2f}   {np.min(ln):5.2f} | "
          f"{ac1(lm):+.2f}   {z:+.2f}      {fl:.2f}  |   {m('bias_range'):.3f}     {m('decides'):.2f}   | "
          f"{m('bstar_speed_rms'):.5f} {m('bstar_speed_max'):.5f} | {min(min(r['lse']) for r in st):.2f}..{max(max(r['lse']) for r in st):.2f}  | "
          f"{val[-1][0]:.3f} ({val[-1][1]})")
