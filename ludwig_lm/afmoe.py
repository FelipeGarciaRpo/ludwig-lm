"""Ludwig 2 in the AFMoE format: transformers' ``AfmoeForCausalLM``, the block of Arcee's Trinity models.

The checkpoint we train is exactly the format transformers loads and llama.cpp converts (T0.1-C).
Everything here works around the model, never changes its parameters or their names:

* ``load_config``: the JSON in ``configs/`` -> ``AfmoeConfig``, with SDPA attention and our experts kernel.
* ``padded_bmm``: an experts kernel registered in transformers' experts interface. It sorts the
  token-expert pairs by expert, pads every expert to the largest load and runs two batched matmuls.
  On the T4 it beat one matmul per expert by 16% with balanced routing (bench v1).
* The router runs in FP32 outside autocast. Each forward records (router logits, chosen experts)
  for the losses and the load balancing; the recomputation inside activation checkpointing is
  not recorded twice.
* ``expert_bias`` becomes a replicated buffer: FSDP2 then leaves it whole on every rank, and the
  balancer updates it the same way everywhere. The state-dict key does not change.
* ``ExpertBalancer``: aux-loss-free balancing. SMEBU (Arcee Trinity: tanh-clamped, re-centred,
  momentum) or the plain sign update, for comparison.
* ``LudwigAfmoe.forward``: chunked cross-entropy with the tied head, output z-loss,
  sequence-wise auxiliary loss (DeepSeek-V3), router z-loss (0 since P30), and the router's health per
  MoE layer (P29). With ``fp32_residual`` the residual stream stays in FP32 when the caller runs it under
  FP16 autocast.
* ``init_ludwig``: depth-scaled sandwich norm (Trinity): output-norm gains start at 1/sqrt(L); the router
  starts random (transformers starts it at zero, which sends every token to the same 4 experts); and the
  padding row trains like any other (transformers turns pad_token_id into a frozen ``padding_idx`` row; until
  the 64K tokenizer that id was 0, the EOS).
* ``shard_model``: the FSDP2 layout T0.2 v2 measured, shared by the bench and the training script.
* T4 fixes for transformers' SDPA path (pre-Ampere GPUs): repeat the KV heads instead of asking SDPA for
  native GQA, which there can fall back to the math kernel and store B*H*T*T attention weights in FP32;
  and use plain causal attention when a sliding window covers the whole sequence (it is the same mask).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from transformers import AfmoeConfig, AfmoeForCausalLM
from transformers import masking_utils as mu
from transformers.integrations import sdpa_attention as sa
from transformers.integrations.moe import ExpertsInterface
from transformers.models.afmoe import modeling_afmoe as am

from .losses import chunked_cross_entropy


# --------------------------------------------------------------------------- config
def load_config(path: str, **overrides) -> AfmoeConfig:
    d = json.loads(Path(path).read_text())
    d.pop("architectures", None)
    d.pop("model_type", None)
    d.update(overrides)
    cfg = AfmoeConfig(**d)
    cfg._attn_implementation = "sdpa"
    cfg._experts_implementation = "padded_bmm"
    return cfg


# --------------------------------------------------------------------------- attention on the T4
_use_gqa_in_sdpa = sa.use_gqa_in_sdpa


def _use_gqa_unless_pre_ampere(attention_mask, key, value):
    if key.is_cuda and torch.cuda.get_device_capability(key.device)[0] < 8:
        return False                       # repeat_kv: 4x the K/V of one layer, ~70 MB at 8 x 2K
    return _use_gqa_in_sdpa(attention_mask, key, value)


sa.use_gqa_in_sdpa = _use_gqa_unless_pre_ampere

_ignore_causal_mask_sdpa = mu._ignore_causal_mask_sdpa


def _window_covers_sequence(padding_mask, q_length, kv_length, q_offset, kv_offset, local_attention_size=None):
    # transformers' window keeps keys with kv_idx > q_idx - window: with kv_length <= window and no offsets
    # that is every causal key, so SDPA can use is_causal instead of an explicit B x T x T mask
    if local_attention_size is not None and kv_length <= local_attention_size and q_offset == 0 and kv_offset == 0:
        local_attention_size = None
    return _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, q_offset, kv_offset, local_attention_size)


mu._ignore_causal_mask_sdpa = _window_covers_sequence


# --------------------------------------------------------------------------- experts kernel
MAX_PADDED_IMBALANCE = 2.0   # above this max/mean load, a layer skips the padding and loops over its experts


def _sorted_loop(self, x, exp_sorted, tok, w_sorted, counts, dt):
    """Dropless fallback without padding: one pair of matmuls per expert over its sorted tokens."""
    ys, start = [], 0
    for e, c in enumerate(counts.tolist()):
        if c == 0:
            continue
        xe = x[tok[start:start + c]]
        gate, up = F.linear(xe, self.gate_up_proj[e].to(dt)).chunk(2, dim=-1)
        ys.append(F.linear(self.act_fn(gate) * up, self.down_proj[e].to(dt)) * w_sorted[start:start + c, None])
        start += c
    return torch.cat(ys)


def padded_bmm_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """Same result as transformers' eager experts; weights stay in their (E, out, in) layout."""
    n_tok, d = hidden_states.shape
    n_exp = self.gate_up_proj.shape[0]
    k = top_k_index.shape[1]
    dev = hidden_states.device.type
    dt = torch.get_autocast_dtype(dev) if torch.is_autocast_enabled(dev) else self.gate_up_proj.dtype
    x = hidden_states.to(dt)
    flat = top_k_index.reshape(-1)
    order = torch.argsort(flat, stable=True)
    exp_sorted = flat[order]
    counts = torch.bincount(flat, minlength=n_exp)
    cap = int(counts.max().item())                   # one host sync per layer
    tok = order // k
    w_sorted = top_k_weights.reshape(-1)[order].to(dt)
    if cap > MAX_PADDED_IMBALANCE * flat.numel() / n_exp:
        # padding every expert to the busiest one would multiply this layer's memory: loop instead
        y = _sorted_loop(self, x, exp_sorted, tok, w_sorted, counts, dt)
        return x.new_zeros(n_tok, d).index_add_(0, tok, y).to(hidden_states.dtype)
    cap = max(8, (cap + 7) // 8 * 8)                 # multiple of 8 for the tensor cores
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(flat.numel(), device=flat.device) - start[exp_sorted]
    xpad = x.new_zeros(n_exp, cap, d)
    xpad[exp_sorted, pos] = x[tok]
    gate, up = torch.bmm(xpad, self.gate_up_proj.to(dt).transpose(1, 2)).chunk(2, dim=-1)
    ypad = torch.bmm(self.act_fn(gate) * up, self.down_proj.to(dt).transpose(1, 2))
    y = ypad[exp_sorted, pos] * w_sorted.unsqueeze(-1)
    out = x.new_zeros(n_tok, d).index_add_(0, tok, y)
    return out.to(hidden_states.dtype)


ExpertsInterface.register("padded_bmm", padded_bmm_experts_forward)


# --------------------------------------------------------------------------- router
_RECORD: list | None = None          # set by LudwigAfmoe.forward: [(router_logits, chosen experts), ...]


def _router_forward_fp32(self, hidden_states, expert_bias):
    x = hidden_states.reshape(-1, hidden_states.shape[-1])
    with torch.autocast(device_type=x.device.type, enabled=False):
        logits = F.linear(x.float(), self.gate.weight.float())
        scores = torch.sigmoid(logits)
        _, chosen = torch.topk(scores + expert_bias.float(), k=self.top_k, dim=1)   # the bias only picks
        top = scores.gather(1, chosen)
        top = top / (top.sum(dim=-1, keepdim=True) + 1e-20) * self.route_scale
    if _RECORD is not None and torch._C._current_graph_task_id() == -1:              # not a recomputation
        _RECORD.append((logits, chosen))
    return logits, top, chosen


am.AfmoeTokenChoiceRouter.forward = _router_forward_fp32


def bias_to_buffers(lm: AfmoeForCausalLM):
    for layer in lm.model.layers:
        mlp = layer.mlp
        if isinstance(getattr(mlp, "expert_bias", None), nn.Parameter):
            b = mlp.expert_bias.detach().float().clone()
            del mlp.expert_bias
            mlp.register_buffer("expert_bias", b, persistent=True)


def moe_blocks(lm: AfmoeForCausalLM):
    return [layer.mlp for layer in lm.model.layers if hasattr(layer.mlp, "expert_bias")]


# --------------------------------------------------------------------------- init
@torch.no_grad()
def init_ludwig(lm: AfmoeForCausalLM):
    """Depth-scaled sandwich norm: the norm after each sublayer starts at 1/sqrt(L)."""
    g = 1.0 / math.sqrt(lm.config.num_hidden_layers)
    for layer in lm.model.layers:
        layer.post_attention_layernorm.weight.fill_(g)
        layer.post_mlp_layernorm.weight.fill_(g)
    for m in moe_blocks(lm):
        m.expert_bias.zero_()
        # transformers starts the router at zero: every score is 0.5 and every token picks the same top-k
        m.router.gate.weight.normal_(0.0, lm.config.initializer_range)
    emb = lm.model.embed_tokens
    if emb.padding_idx is not None:
        # transformers passes pad_token_id as padding_idx: that row starts at zero and its lookup gets no
        # gradient. Packed training never pads; with configs where padding is EOS (the tests' tiny configs, the
        # runs before the 64K tokenizer), EOS is an input at every document boundary.
        # The checkpoint format does not change: padding_idx is not a weight.
        emb.weight[emb.padding_idx].normal_(0.0, lm.config.initializer_range)
        emb.padding_idx = None


def shard_model(model, mesh, mp_policy, prefetch: bool = True):
    """FSDP2 with one unit per decoder layer. The root keeps embeddings and head unsharded through backward,
    because the checkpointed chunked loss uses the tied head. With ``prefetch`` each layer all-gathers the
    next one (forward) or the previous one (backward) while it computes."""
    try:
        from torch.distributed.fsdp import fully_shard
    except ImportError:  # torch < 2.6
        from torch.distributed._composable.fsdp import fully_shard
    layers = list(model.lm.model.layers)
    for layer in layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False)
    if prefetch:
        for i in range(len(layers) - 1):
            layers[i].set_modules_to_forward_prefetch([layers[i + 1]])
        for i in range(1, len(layers)):
            layers[i].set_modules_to_backward_prefetch([layers[i - 1]])
    return model


# --------------------------------------------------------------------------- losses
@dataclass
class LossWeights:
    output_z: float = 1e-4
    seq_aux: float = 1e-4        # sequence-wise balance loss (DeepSeek-V3); the bias does the rest
    router_z: float = 0.0        # P30: made for softmax routers; with sigmoid scores it broke the trunk's balance


def moe_losses(records, batch: int, seq: int, n_exp: int, k: int):
    """Sequence-wise auxiliary loss and router z-loss, averaged over the MoE layers; the mean entropy of each
    token's normalized router scores (a log; at most ln E); and the router's health in each MoE layer (P29):

    * ``margin``: the median, over the tokens, of the gap between the k-th and the (k+1)-th score. SMEBU moves
      the bias by up to λ per step: once λ is several times this gap, one step of the bias re-routes most
      tokens at once, and the busiest experts take turns (the trunk from step ~1250);
    * ``decides``: the fraction of tokens whose k experts are not the k best scores, because of the bias."""
    aux = torch.zeros((), device=records[0][0].device)
    rz = torch.zeros((), device=records[0][0].device)
    ent = torch.zeros((), device=records[0][0].device)
    margins, decides = [], []
    for logits, chosen in records:
        s = torch.sigmoid(logits)
        pt = s / s.sum(-1, keepdim=True)                                                 # (B*T, E)
        p = pt.view(batch, seq, n_exp).mean(1)                                           # (B, E)
        cnt = torch.zeros(batch, n_exp, device=logits.device)
        cnt.scatter_add_(1, chosen.view(batch, -1), torch.ones(batch, seq * k, device=logits.device))
        aux = aux + ((cnt * (n_exp / (k * seq))) * p).sum(-1).mean()
        rz = rz + torch.logsumexp(logits, -1).pow(2).mean()
        with torch.no_grad():
            q = pt.detach()
            ent = ent + (-(q * torch.log(q.clamp_min(1e-20))).sum(-1).mean())
            top = torch.topk(s.detach(), k + 1, dim=-1)
            margins.append((top.values[:, k - 1] - top.values[:, k]).median())
            own = top.indices[:, :k].sort(-1).values
            decides.append((own != chosen.reshape(-1, k).sort(-1).values).any(-1).float().mean())
    n = len(records)
    return aux / n, rz / n, ent / n, torch.stack(margins), torch.stack(decides)


# --------------------------------------------------------------------------- model
class LudwigAfmoe(nn.Module):
    def __init__(self, cfg: AfmoeConfig, fp32_residual: bool = True):
        super().__init__()
        self.lm = AfmoeForCausalLM(cfg)
        bias_to_buffers(self.lm)
        init_ludwig(self.lm)
        self.fp32_residual = fp32_residual

    def forward(self, input_ids, weights: LossWeights = LossWeights(), ce_chunk: int = 4096, loss_mask=None):
        """``loss_mask`` [B, T-1] (bool): the targets that count (MeCo: not the metadata spans)."""
        global _RECORD
        cfg = self.lm.config
        batch, seq = input_ids.shape
        emb = self.lm.model.embed_tokens(input_ids)
        if self.fp32_residual:
            emb = emb.float()
        records, previous = [], _RECORD
        _RECORD = records
        try:
            h = self.lm.model(inputs_embeds=emb, use_cache=False).last_hidden_state      # after the final norm
        finally:
            _RECORD = previous
        ce, z = chunked_cross_entropy(h[:, :-1], self.lm.lm_head.weight, input_ids[:, 1:], ce_chunk, mask=loss_mask)
        aux, rz, ent, margin, decides = moe_losses(records, batch, seq, cfg.num_experts, cfg.num_experts_per_tok)
        loss = ce + weights.output_z * z + weights.seq_aux * aux + weights.router_z * rz
        metrics = {"ce": ce.detach(), "z_out": z.detach(), "seq_aux": aux.detach(), "z_router": rz.detach(),
                   "router_entropy": ent, "router_margin": margin, "bias_decides": decides,
                   "chosen": [c for _, c in records]}
        return loss, metrics


def count_params(model) -> dict:
    lm = model.lm if hasattr(model, "lm") else model
    cfg = lm.config
    total = sum(p.numel() for p in lm.parameters())
    experts = sum(p.numel() for n, p in lm.named_parameters() if ".experts." in n)
    active = total - experts + experts * cfg.num_experts_per_tok / cfg.num_experts
    return {"total": total, "active": int(active), "routed_experts": experts}


def flops_per_token(cfg: AfmoeConfig, active: int, seq: int) -> float:
    """6 x active parameters + attention, with local layers limited to their window (MFU accounting)."""
    attn = 0
    for t in cfg.layer_types:
        span = min(seq, cfg.sliding_window) if t == "sliding_attention" else seq
        attn += 3 * span * cfg.num_attention_heads * 2 * cfg.head_dim
    return 6 * active + attn


# --------------------------------------------------------------------------- balancing
class ExpertBalancer:
    """Updates every MoE layer's expert_bias once per optimizer step from the tokens it got.

    smebu (Arcee Trinity): v = (mean - n_i) / mean;  d = lam * tanh(kappa * v), re-centred;
                           m = beta * m + (1 - beta) * d;  b += m
    sign  (Wang et al. 2024, used by DeepSeek-V3): b += gamma * sign(mean - n_i)
    """

    def __init__(self, model, rule: str = "smebu", lam: float = 1e-3, beta: float = 0.5, kappa: float = 2.0,
                 gamma: float = 1e-3):
        lm = model.lm if hasattr(model, "lm") else model
        self.blocks = moe_blocks(lm)
        self.n_exp = lm.config.num_experts
        self.rule, self.lam, self.beta, self.kappa, self.gamma = rule, lam, beta, kappa, gamma
        dev = self.blocks[0].expert_bias.device if self.blocks else torch.device("cpu")
        self.momentum = torch.zeros(len(self.blocks), self.n_exp, device=dev)
        self.counts = torch.zeros(len(self.blocks), self.n_exp, device=dev)
        self.window = torch.zeros(len(self.blocks), self.n_exp, device=dev)   # since the last window_stats()

    @torch.no_grad()
    def accumulate(self, chosen_per_layer):
        for i, c in enumerate(chosen_per_layer):
            self.counts[i] += torch.bincount(c.reshape(-1), minlength=self.n_exp).float()

    @torch.no_grad()
    def step(self) -> dict:
        c = self.counts
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(c)
        mean = c.mean(1, keepdim=True).clamp_min(1.0)
        v = (mean - c) / mean                              # > 0: the expert got fewer tokens than the mean
        if self.rule == "smebu":
            d = self.lam * torch.tanh(self.kappa * v)
            d = d - d.mean(1, keepdim=True)
            self.momentum.mul_(self.beta).add_(d, alpha=1.0 - self.beta)
            delta = self.momentum
        else:
            delta = self.gamma * torch.sign(v)
        for i, block in enumerate(self.blocks):
            block.expert_bias.add_(delta[i].to(block.expert_bias.dtype))
        load = c / mean
        stats = {"load_max_over_mean": float(load.max()), "load_min_over_mean": float(load.min()),
                 "dead_experts": int((c == 0).sum()),
                 "bias_abs_max": float(max(b.expert_bias.abs().max() for b in self.blocks))}
        self.window += c
        self.counts.zero_()
        return stats

    @torch.no_grad()
    def window_stats(self) -> dict:
        """Per-layer load over every step since the previous call (already summed over ranks), then reset.
        An expert with no tokens over many steps is really dead; one step can miss it by chance."""
        w = self.window
        if not len(w) or float(w.sum()) == 0:          # no step since the last call
            return {"layer_load_max_over_mean": None, "layer_load_min_over_mean": None, "dead_experts_window": None,
                    "window_assignments_per_layer": 0.0}
        mean = w.mean(1, keepdim=True).clamp_min(1.0)
        stats = {"layer_load_max_over_mean": [round(float(x), 3) for x in (w / mean).max(1).values],
                 "layer_load_min_over_mean": [round(float(x), 3) for x in (w / mean).min(1).values],
                 "dead_experts_window": int((w == 0).sum()),
                 "window_assignments_per_layer": float(w[0].sum())}   # tokens × top-k
        self.window.zero_()
        return stats

    def skip(self):
        """Forget the counts of a step that is not applied (a skipped loss spike)."""
        self.counts.zero_()

    def state_dict(self):
        return {"momentum": self.momentum.clone(), "rule": self.rule, "lam": self.lam, "gamma": self.gamma}

    def load_state_dict(self, sd):
        self.momentum.copy_(sd["momentum"])
