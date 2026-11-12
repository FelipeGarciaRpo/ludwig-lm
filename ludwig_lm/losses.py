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


def chunked_cross_entropy(h, w, y, chunk_tokens: int = 4096):
    """Mean CE and mean logsumexp^2 (for z-loss) without storing all logits:
    each chunk is recomputed in backward (activation checkpointing)."""
    h = h.reshape(-1, h.size(-1))
    y = y.reshape(-1)
    n = h.size(0)
    ce_sum = h.new_zeros((), dtype=torch.float32)
    z_sum = h.new_zeros((), dtype=torch.float32)
    for i in range(0, n, chunk_tokens):
        c, z = checkpoint(_ce_chunk, h[i:i + chunk_tokens], w, y[i:i + chunk_tokens], use_reentrant=False)
        ce_sum = ce_sum + c
        z_sum = z_sum + z
    return ce_sum / n, z_sum / n
