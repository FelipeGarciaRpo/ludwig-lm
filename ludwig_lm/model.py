"""Ludwig 2 model: Hugging Face Qwen3-Next with every layer set to full (gated)
attention, plus an MTP head and the training losses.

Why the Hugging Face class: the checkpoint we train is byte-for-byte the format
that transformers loads and that llama.cpp converts, so there is no conversion
code of our own that could drift.

What we add on top:
* ``MTPHead``: a multi-token-prediction block (DeepSeek-V3 / Qwen3-Next style).
  Its parameter names are the ones llama.cpp's converter maps to NextN layers
  (``mtp.fc``, ``mtp.pre_fc_norm_embedding``, ``mtp.pre_fc_norm_hidden``,
  ``mtp.layers.0.*``, ``mtp.norm``). It is used for speculative decoding.
* Losses: chunked cross-entropy (never materialises the full logits), output
  z-loss, a load-balancing loss whose expert frequencies are summed across
  data-parallel ranks, router z-loss, and the MTP loss.
* The router runs in FP32.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import Qwen3NextConfig, Qwen3NextForCausalLM
from transformers.models.qwen3_next import modeling_qwen3_next as qn


# --------------------------------------------------------------------------- config
def load_config(path: str, **overrides) -> Qwen3NextConfig:
    """Read a Ludwig config JSON. Overrides win; ``layer_types`` always follows
    ``num_hidden_layers`` (all ``full_attention``)."""
    with open(path) as f:
        d = json.load(f)
    d.update(overrides)
    d["layer_types"] = ["full_attention"] * d["num_hidden_layers"]
    cfg = Qwen3NextConfig(**d)
    cfg._attn_implementation = "sdpa"
    return cfg


# --------------------------------------------------------------------------- router in FP32
def _router_forward_fp32(self, hidden_states):
    hidden_states = hidden_states.reshape(-1, self.hidden_dim)
    router_logits = F.linear(hidden_states.float(), self.weight.float())
    router_probs = torch.softmax(router_logits, dim=-1)
    top_value, top_index = torch.topk(router_probs, self.top_k, dim=-1)
    if self.norm_topk_prob:
        top_value = top_value / top_value.sum(dim=-1, keepdim=True)
    return router_logits, top_value.to(hidden_states.dtype), top_index


def patch_router_fp32() -> None:
    """Compute router logits and softmax in FP32 (MobileMoE / Switch practice)."""
    qn.Qwen3NextTopKRouter.forward = _router_forward_fp32


# --------------------------------------------------------------------------- MTP head
class MTPHead(nn.Module):
    """One MTP module: predicts token t+2 from the post-norm hidden state at t and
    the embedding of token t+1.

        x = fc([norm_e(emb(x_{t+1})) ; norm_h(h_t)])  ->  decoder layer  ->  norm  ->  shared LM head

    The concatenation order (embedding first) and the use of the post-norm hidden
    state match llama.cpp's ``qwen3next`` MTP graph.
    """

    def __init__(self, config: Qwen3NextConfig):
        super().__init__()
        d, eps = config.hidden_size, config.rms_norm_eps
        self.fc = nn.Linear(2 * d, d, bias=False)
        self.pre_fc_norm_embedding = qn.Qwen3NextRMSNorm(d, eps=eps)
        self.pre_fc_norm_hidden = qn.Qwen3NextRMSNorm(d, eps=eps)
        # layer_idx=0: every layer is full attention + MoE, so index 0 has the right type.
        self.layers = nn.ModuleList([qn.Qwen3NextDecoderLayer(config, layer_idx=0)])
        self.norm = qn.Qwen3NextRMSNorm(d, eps=eps)

    def forward(self, hidden, next_embeds, position_embeddings):
        x = torch.cat([self.pre_fc_norm_embedding(next_embeds), self.pre_fc_norm_hidden(hidden)], dim=-1)
        x = self.fc(x)
        out = self.layers[0](x, position_embeddings=position_embeddings, attention_mask=None)
        x = out[0] if isinstance(out, tuple) else out
        return self.norm(x)


# --------------------------------------------------------------------------- losses
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


def router_losses(router_logits, top_k: int, num_experts: int):
    """Load-balancing loss (Switch form, E * sum_i f_i * P_i) with the expert
    frequencies f summed over all data-parallel ranks in one all-reduce, plus the
    router z-loss. Also returns per-layer load (max/mean) for monitoring."""
    logits = torch.stack([lg.float() for lg in router_logits])          # (L, T, E)
    probs = logits.softmax(dim=-1)
    top_idx = probs.topk(top_k, dim=-1).indices                          # (L, T, k)
    counts = torch.zeros(logits.size(0), num_experts, device=logits.device)
    counts.scatter_add_(1, top_idx.flatten(1), torch.ones_like(top_idx.flatten(1), dtype=counts.dtype))
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(counts)
    freq = counts / counts.sum(dim=1, keepdim=True)                      # (L, E), no gradient
    mean_prob = probs.mean(dim=1)                                        # (L, E), local, differentiable
    lbl = (num_experts * (freq * mean_prob).sum(dim=1)).mean()
    rz = torch.logsumexp(logits, dim=-1).pow(2).mean()
    load_max_over_mean = (counts.max(dim=1).values / counts.mean(dim=1)).detach()
    return lbl, rz, load_max_over_mean


@dataclass
class LossWeights:
    balance: float = 1e-3        # load-balancing loss (Qwen3-Next default)
    router_z: float = 1e-4       # router z-loss (ST-MoE / MobileMoE)
    output_z: float = 1e-4       # output z-loss (PaLM), a guard for FP16
    mtp: float = 0.0             # 0 = MTP off; ~0.3-1.0 when training the MTP head
    mtp_detach: bool = True      # stop-gradient into the trunk, embeddings and head


# --------------------------------------------------------------------------- full model
class LudwigLM(nn.Module):
    def __init__(self, config: Qwen3NextConfig, with_mtp: bool = True):
        super().__init__()
        patch_router_fp32()
        self.config = config
        self.lm = Qwen3NextForCausalLM(config)
        self.mtp = MTPHead(config) if with_mtp else None
        if self.mtp is not None:
            self.mtp.apply(self.lm._init_weights)

    # -- training forward
    def forward(self, input_ids: torch.Tensor, weights: LossWeights = LossWeights(), ce_chunk: int = 4096):
        cfg = self.config
        out = self.lm.model(input_ids=input_ids, output_router_logits=True, use_cache=False)
        h = out.last_hidden_state                                         # post final norm
        w_head = self.lm.lm_head.weight                                   # tied to the embeddings

        ce, z = chunked_cross_entropy(h[:, :-1], w_head, input_ids[:, 1:], ce_chunk)
        lbl, rz, load = router_losses(out.router_logits, cfg.num_experts_per_tok, cfg.num_experts)
        loss = ce + weights.output_z * z + weights.balance * lbl + weights.router_z * rz
        metrics = {"ce": ce.detach(), "z_out": z.detach(), "balance": lbl.detach(),
                   "z_router": rz.detach(), "load_max_over_mean": load.max()}

        if self.mtp is not None and weights.mtp > 0:
            hm = h[:, :-2]
            emb = self.lm.model.embed_tokens(input_ids[:, 1:-1])
            if weights.mtp_detach:
                hm, emb, w_mtp = hm.detach(), emb.detach(), w_head.detach()
            else:
                w_mtp = w_head
            pos = torch.arange(hm.size(1), device=hm.device).unsqueeze(0)
            pos_emb = self.lm.model.rotary_emb(hm, pos)
            hm = self.mtp(hm, emb, pos_emb)
            mtp_ce, _ = chunked_cross_entropy(hm, w_mtp, input_ids[:, 2:], ce_chunk)
            loss = loss + weights.mtp * mtp_ce
            metrics["mtp_ce"] = mtp_ce.detach()
        return loss, metrics

    # -- export in the format transformers loads and llama.cpp converts
    @torch.no_grad()
    def save_hf(self, out_dir: str, tokenizer=None, dtype: torch.dtype = torch.bfloat16) -> None:
        from safetensors.torch import save_file

        os.makedirs(out_dir, exist_ok=True)
        sd = {}
        for k, v in self.lm.state_dict().items():
            if k == "lm_head.weight" and self.config.tie_word_embeddings:
                continue                                                  # tied: stored once
            sd[k] = v.detach().to(dtype).contiguous().cpu()
        if self.mtp is not None:
            for k, v in self.mtp.state_dict().items():
                sd[f"mtp.{k}"] = v.detach().to(dtype).contiguous().cpu()
        save_file(sd, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
        cfg = self.config.to_dict()
        cfg["architectures"] = ["Qwen3NextForCausalLM"]
        cfg["torch_dtype"] = str(dtype).replace("torch.", "")
        cfg.pop("_attn_implementation_autoset", None)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        if tokenizer is not None:
            tokenizer.save_pretrained(out_dir)


def count_params(model: LudwigLM) -> dict:
    cfg = model.config
    total = sum(p.numel() for p in model.lm.parameters())
    mtp = sum(p.numel() for p in model.mtp.parameters()) if model.mtp is not None else 0
    e, k = cfg.num_experts, cfg.num_experts_per_tok
    routed = sum(p.numel() for n, p in model.lm.named_parameters() if ".experts." in n)
    active = total - routed + routed * k // e
    return {"total": total, "active": active, "mtp": mtp}
