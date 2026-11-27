#!/usr/bin/env python3
"""How much of a MoE block's output comes from the routed experts, at initialisation (P35's open question).

    python results/diagnostico_moe/potencia_elegidos.py > results/diagnostico_moe/potencia_elegidos.txt

One AfmoeSparseMoeBlock with the trunk's sizes (width 1024, experts of 512), random weights as transformers
initialises them (std 0.02), inputs with unit RMS like the output of the norm before the block. It prints the
routed experts' output power over the shared experts'. The norm after the block rescales their sum, so this
ratio is what decides how much the routed experts weigh.

The k chosen experts are averaged with weights that add up to route_scale, so their power is route_scale^2 / k
of one expert's; n shared experts are one MLP n times as wide, with n times one expert's power. CPU, ~1 minute.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from transformers.models.afmoe import modeling_afmoe as am  # noqa: E402

from ludwig_lm import afmoe  # noqa: E402


def ratio(route_scale, shared, experts=44, k=4):
    cfg = afmoe.load_config(str(ROOT / "configs" / "ludwig2_afmoe_d3.json"), route_scale=route_scale,
                            num_shared_experts=shared, num_experts=experts, num_experts_per_tok=k)
    cfg._experts_implementation = "eager"
    block = am.AfmoeSparseMoeBlock(cfg)
    with torch.no_grad():
        for _, p in block.named_parameters():
            if p.ndim >= 2:
                p.normal_(0.0, 0.02)
        block.expert_bias.zero_()
        x = torch.randn(1, 2048, 1024)
        flat = x.view(-1, 1024)
        _, top, chosen = block.router(x, block.expert_bias)
        shared_out = block.shared_experts(flat)
        routed_out = block.experts(flat, chosen, top)
    return float(routed_out.pow(2).sum(1).mean() / shared_out.pow(2).sum(1).mean())


torch.manual_seed(0)
torch.set_num_threads(2)
print("power of the routed experts' output over the shared experts' output, one MoE block at initialisation")
for what, rs, shared, experts, k in (
        ("tronco3: 4 shared, top-4 of 44, route_scale 1.0", 1.0, 4, 44, 4),
        ("the same with route_scale 2.5", 2.5, 4, 44, 4),
        ("the same with route_scale 4.0", 4.0, 4, 44, 4),
        ("1 shared, top-4 of 44, route_scale 1.0", 1.0, 1, 44, 4),
        ("1 shared, top-4 of 44, route_scale 2.448 (the value in the request for this review, Trinity Large)", 2.448, 1, 44, 4),
        ("1 shared, top-8 of 44, route_scale 2.5 (DeepSeek-V3's value, with its 8 active experts)", 2.5, 1, 44, 8),
        ("1 shared, top-8 of 44, route_scale 2.826", 2.826, 1, 44, 8)):
    print(f"  {what}: {ratio(rs, shared, experts, k):.3f}  (route_scale^2 / (k * shared) = {rs * rs / (k * shared):.3f})")
