"""At init the logits are w_i.x_t with w ~ N(0, 0.02^2) and |x|^2 = 1024, so their std over (token, expert) is 0.64.
If a fraction rho of |x|^2 is a direction common to all tokens, each expert gets a fixed offset (std 0.64*sqrt(rho))
and each token a specific part (std 0.64*sqrt(1-rho)). What rho gives the busiest expert tronco3's evaluation of
step 0 shows, with no bias yet (analisis_logs.txt: 1.83x to 3.88x the mean per MoE layer, median 2.48x; median gap
0.0132; lse 3.978)?

    python results/diagnostico_moe/calib_init.py > results/diagnostico_moe/calib_init.txt
"""
import numpy as np
E, k, N = 44, 4, 60000
r = np.random.default_rng(0)
print(" rho | busiest/mean (median, min-max over 23 layers) | median gap 4th-5th | lse")
for rho in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    busy, gaps, lses = [], [], []
    for layer in range(23):
        o = 0.64 * np.sqrt(rho) * r.normal(size=E)
        z = o + 0.64 * np.sqrt(1 - rho) * r.normal(size=(N, E))
        s = 1 / (1 + np.exp(-z))
        sel = np.argpartition(-s, k, axis=1)[:, :k]
        c = np.bincount(sel.ravel(), minlength=E)
        busy.append(c.max() / c.mean())
        srt = -np.sort(-s, axis=1)
        gaps.append(np.median(srt[:, k - 1] - srt[:, k]))
        lses.append(np.log(np.exp(z).sum(1)).mean())
    print(f"{rho:4.2f} | {np.median(busy):5.2f}  {min(busy):5.2f}-{max(busy):5.2f}                     | {np.mean(gaps):.4f}             | {np.mean(lses):.3f}")
