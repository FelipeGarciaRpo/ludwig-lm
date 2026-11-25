#!/usr/bin/env python3
"""Summary of the trunk's turns from results/tronco/turnoNN_*.jsonl and .relevo.json: each turn, its alarms,
the evaluations (with the per-layer expert load of the last window), and 50-step means of the series that show
the expert balance. Reads only; prints to stdout.

    python scripts/resumen_tronco.py > results/tronco/resumen_t01-tNN.txt
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "results" / "tronco"
SERIES = ("ce", "load_max_over_mean", "load_min_over_mean", "router_entropy", "grad_norm", "z_router", "z_out",
          "seq_aux", "bias_abs_max", "tokens_per_s")


def main():
    steps, evals, alarms = {}, [], []
    print("== turns")
    for path in sorted(ROOT.glob("turno[0-9][0-9]_*.jsonl")):
        turn = int(path.name[5:7])
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        rel_path = path.with_suffix(".relevo.json")
        rel = json.loads(rel_path.read_text(encoding="utf-8")) if rel_path.exists() else {"events": [], "by": "?"}
        st_rows = [r for r in rows if r["type"] == "step"]
        for r in st_rows:
            steps[r["step"]] = dict(r, turn=turn)
        evals += [dict(r, turn=turn) for r in rows if r["type"] == "eval"]
        alarms += [dict(r, turn=turn) for r in rows if r["type"] == "alarm"]
        end = [r for r in rows if r["type"] == "end"][-1]
        res = [r for r in rows if r["type"] == "resume"]
        print(f"turn {turn} ({rel['by']}): steps {st_rows[0]['step']}-{st_rows[-1]['step']}, end {end['status']} at "
              f"{end['steps_done']}, {end['hours_this_session']} h, {end['tokens_per_s_last_hour']} tokens/s, "
              f"scale {end['loss_scale_last']:.0f}, FP16 skips {end['loss_scale_skips']}, spikes {end['spikes_skipped']}")
        if res:
            print(f"    resume at {res[0]['step']}: state exact {res[0]['state_exact']}, {res[0]['seconds']} s")
        for e in rel["events"]:
            if e["kind"] in ("resume_from", "download", "in_place", "backup", "backup_failed", "release_failed",
                             "lock_lost", "upload_failed"):
                info = {k: v for k, v in e.items() if k in ("where", "step", "dataset", "seconds", "error")}
                if "error" in info:
                    info["error"] = info["error"].split("\n")[0][:110]
                print(f"    {e['kind']} {json.dumps(info, ensure_ascii=False)}")

    print("\n== alarms")
    for a in alarms:
        print(f"step {a['step']} (turn {a['turn']}): {a['alarm']} " + ", ".join(
            f"{k} {v:.3f}" if isinstance(v, float) else f"{k} {v}" for k, v in a.items()
            if k not in ("type", "tag", "time", "step", "alarm", "turn")))

    print("\n== evaluations (layer load: the window's busiest expert over the mean, per MoE layer)")
    for e in evals:
        lay = e.get("layer_load_max_over_mean")
        print(f"step {e['step']:5d} (turn {e['turn']}): val bpb {e['val_bpb']:.4f}, router entropy "
              f"{e['router_entropy']:.3f}, qk gain {e['qk_gain_max']:.2f}, dead experts {e.get('dead_experts_window')}"
              + (f"\n    layer load: {' '.join(f'{x:.2f}' for x in lay)}" if lay else ""))

    print("\n== 50-step means (max load: the bucket's highest; >3: steps with load max/mean above 3)")
    print("from turn lr   " + " ".join(f"{k[:10]:>10s}" for k in SERIES) + "     scale max load  >3")
    for s0 in range(0, max(steps) + 1, 50):
        b = [steps[s] for s in range(s0, s0 + 50) if s in steps]
        if not b:
            continue
        means = []
        for k in SERIES:
            xs = [r[k] for r in b if r.get(k) is not None]
            means.append(st.mean(xs) if xs else float("nan"))
        loads = [r.get("load_max_over_mean") or 0.0 for r in b]
        print(f"{s0:4d} {b[0]['turn']:4d} {b[0]['lr_mult']:.2f} " + " ".join(f"{v:10.4f}" for v in means)
              + f" {b[-1]['scale']:9.0f} {max(loads):8.2f} {sum(x > 3 for x in loads):3d}")

    order = sorted(steps)

    def first_sustained(threshold, window=50, share=0.8):
        for i in range(len(order) - window):
            w = [steps[s].get("load_max_over_mean") or 0.0 for s in order[i:i + window]]
            if sum(x > threshold for x in w) >= share * window:
                return order[i]
        return None

    print("\nfirst step from which 80% of the next 50 steps have load max/mean above 2, 3 and 5:",
          first_sustained(2), first_sustained(3), first_sustained(5))


if __name__ == "__main__":
    main()
