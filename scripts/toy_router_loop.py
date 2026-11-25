#!/usr/bin/env python3
"""A toy of the router and SMEBU loop (P29): why the trunk's expert balance broke at step ~1250.

    python scripts/toy_router_loop.py        # ~5 min on a CPU, numpy only

Each step routes N tokens over E experts, as AFMoE does: sigmoid scores, the top k of score + bias, the bias
updated by SMEBU from the step's counts (tanh-clamped, re-centred, momentum 0.5). The logits are
    mu + a + spread · z
where z is fresh for every token (what the token wants), a is fixed per expert (what the router prefers
whatever the token, which the bias must cancel) and mu is the router's level, which the trunk's router z-loss
pushed down. Lowering mu sinks the scores into the sigmoid's tail and shrinks the gap between each token's
k-th and (k+1)-th score. Once SMEBU's step λ is ~5–10× that gap, one bias step re-routes most tokens at
once, and the busiest experts take turns: the trunk's collapse.

Per mu and λ it prints, over the last 300 of 1,500 steps: the busiest expert over the mean (mean and max;
the ceiling is E/k = 10), the median gap, how often the bias changes a token's experts, and the bias range.
"""
import numpy as np


def run(mu, lam, spread=1.0, E=40, k=4, N=8192, steps=1500, seed=0):
    r = np.random.default_rng(seed)
    a = r.normal(0, 0.5, E)
    b, m = np.zeros(E), np.zeros(E)
    loads, decides, gaps = [], [], []
    for t in range(steps):
        s = 1 / (1 + np.exp(-(mu + a + spread * r.normal(0, 1, (N, E)))))
        sel = np.argpartition(-(s + b), k, axis=1)[:, :k]
        c = np.bincount(sel.ravel(), minlength=E).astype(float)
        mean = c.mean()
        d = lam * np.tanh(2 * (mean - c) / mean)
        d -= d.mean()
        m = 0.5 * m + 0.5 * d
        b += m
        if t >= steps - 300:
            loads.append(c.max() / mean)
            srt = -np.sort(-s, axis=1)
            gaps.append(np.median(srt[:, k - 1] - srt[:, k]))
            own = np.sort(np.argpartition(-s, k, axis=1)[:, :k], axis=1)
            decides.append((np.sort(sel, axis=1) != own).any(1).mean())
    return np.mean(loads), np.max(loads), np.mean(gaps), np.mean(decides), b.max() - b.min()


if __name__ == "__main__":
    print(f"{'mu':>4} {'score':>7} | " + " | ".join(
        f"λ {lam:<5}: busiest mean/max   gap      λ/gap  bias decides  bias range" for lam in (0.01, 0.001)))
    for mu in (0, -2, -3, -4, -5, -6, -7, -8):
        line = f"{mu:4d} {1 / (1 + np.exp(-mu)):7.4f} | "
        for lam in (0.01, 0.001):
            lm, lx, gap, dec, br = run(mu, lam)
            line += f"{lm:5.2f} / {lx:5.2f}       {gap:8.5f}  {lam / gap:6.1f}  {dec:6.1%}        {br:6.3f}   | "
        print(line.rstrip(" |"), flush=True)
