"""Which router gives the numbers tronco3's evaluations logged per MoE layer?
Model of one layer: logit[t, i] = m + o_i + sigma_t * z[t, i], o_i ~ N(0, sigma_o^2) fixed per expert (common to
all tokens), z fresh per token. Scores = sigmoid. The bias is the one that balances the load exactly.
Outputs to compare with the logs: lse, median gap between the 4th and 5th score, bias range, share of tokens
whose experts the bias changes, and the busiest expert without the bias. The targets are the medians over the MoE
layers of tronco3's evaluations at steps 250 and 498 (analisis_logs.txt), rounded. Rows within 0.6 of the targets
(the sum of the log ratios of gap and range, plus 4 times the difference in the share) are printed; ~3 min on a CPU.

    python results/diagnostico_moe/calib_real.py > results/diagnostico_moe/calib_real.txt
"""
import numpy as np

E, k, N = 44, 4, 8000


def balanced(s, iters=30, damp=0.5):
    b = np.zeros(s.shape[1])
    for _ in range(iters):
        tot = s + b
        srt = -np.sort(-tot, axis=1)
        inside = tot >= srt[:, k - 1:k]
        beat = np.where(inside, srt[:, k:k + 1], srt[:, k - 1:k])
        q = np.quantile(beat - s, k / E, axis=0)
        b += damp * (q - b)
        b -= b.mean()
    return b


def stats(lse_target, so, st, seeds=6):
    out = []
    for seed in range(seeds):
        r = np.random.default_rng(seed)
        o = so * r.normal(size=E)
        z = o + st * r.normal(size=(N, E))
        m = lse_target - np.log(np.exp(z).sum(1)).mean()          # the level that gives this lse
        s = 1 / (1 + np.exp(-(z + m)))
        b = balanced(s)
        srt = -np.sort(-s, axis=1)
        gap = np.median(srt[:, k - 1] - srt[:, k])
        own = np.sort(np.argpartition(-s, k, axis=1)[:, :k], axis=1)
        sel = np.sort(np.argpartition(-(s + b), k, axis=1)[:, :k], axis=1)
        dec = (own != sel).any(1).mean()
        c0 = np.bincount(own.ravel(), minlength=E)
        c1 = np.bincount(sel.ravel(), minlength=E)
        g = np.take_along_axis(s, sel, 1)
        g = g / g.sum(1, keepdims=True)
        out.append((gap, b.max() - b.min(), dec, c0.max() / c0.mean(), c1.max() / c1.mean(), (1 / (g ** 2).sum(1)).mean(),
                    np.take_along_axis(s, sel, 1).std(0).mean(), s.std(0).min(), s.std(0).max(), m))
    return np.mean(out, 0)


targets = {"step 250": (3.70, 0.017, 0.44, 0.975), "step 498": (3.20, 0.024, 0.56, 0.990)}
for name, (lse, gap, rng, dec) in targets.items():
    print(f"\n{name}: logs say lse {lse}, gap {gap}, bias range {rng}, bias decides {dec}")
    print(" sigma_o sigma_t |   gap   bias range  decides | busiest w/o bias | effective experts/token | score sd by expert (min-max)")
    best = None
    for so in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        for st in (0.3, 0.4, 0.5, 0.6):
            g, r_, d, c0, c1, eff, _, sdmin, sdmax, m = stats(lse, so, st, seeds=2)
            err = abs(np.log(g / gap)) + abs(np.log(r_ / rng)) + 4 * abs(d - dec)
            if best is None or err < best[0]:
                best = (err, so, st)
            if err < 0.6:
                print(f"   {so:4.1f}    {st:4.1f}  | {g:.4f}   {r_:.3f}     {d:.3f}  |      {c0:5.2f}       |          {eff:.2f}           | {sdmin:.3f}-{sdmax:.3f}")
    print("  best fit: sigma_o %.1f, sigma_t %.1f -> common share of the logit variance %.0f%%" % (best[1], best[2], 100 * best[1] ** 2 / (best[1] ** 2 + best[2] ** 2)))
