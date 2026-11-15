"""Losses shared by every Ludwig 2 model format."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _ce_chunk(h, w, y):
    logits = F.linear(h, w).float()
    lse = torch.logsumexp(logits, dim=-1)
    ce = lse - logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return ce.sum(), lse.pow(2).sum()


def _ce_chunk_masked(h, w, y, m):
    logits = F.linear(h, w).float()
    lse = torch.logsumexp(logits, dim=-1)
    ce = lse - logits.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (ce * m).sum(), (lse.pow(2) * m).sum()


def chunked_cross_entropy(h, w, y, chunk_tokens: int = 4096, mask=None):
    """Mean CE and mean logsumexp^2 (for z-loss) without storing all logits:
    each chunk is recomputed in backward (activation checkpointing).

    ``mask`` (bool, the shape of ``y``): only the True targets count, and the means are over them (MeCo
    leaves the metadata tokens out of the loss). Without it the computation is the one it always was."""
    h = h.reshape(-1, h.size(-1))
    y = y.reshape(-1)
    n = h.size(0)
    ce_sum = h.new_zeros((), dtype=torch.float32)
    z_sum = h.new_zeros((), dtype=torch.float32)
    if mask is None:
        for i in range(0, n, chunk_tokens):
            c, z = checkpoint(_ce_chunk, h[i:i + chunk_tokens], w, y[i:i + chunk_tokens], use_reentrant=False)
            ce_sum = ce_sum + c
            z_sum = z_sum + z
        return ce_sum / n, z_sum / n
    m = mask.reshape(-1).to(torch.float32)
    for i in range(0, n, chunk_tokens):
        c, z = checkpoint(_ce_chunk_masked, h[i:i + chunk_tokens], w, y[i:i + chunk_tokens], m[i:i + chunk_tokens],
                          use_reentrant=False)
        ce_sum = ce_sum + c
        z_sum = z_sum + z
    count = m.sum().clamp_min(1.0)
    return ce_sum / count, z_sum / count
