"""Muon for Ludwig 2, aware of FSDP2 sharding.

* Update: Nesterov momentum, then Newton-Schulz orthogonalisation (5 steps,
  Jordan et al. coefficients), then Moonlight's scale 0.2 * sqrt(max(m, n)) so
  that AdamW's learning rate and weight decay transfer.
* Fused matrices are orthogonalised piece by piece: ``q_proj`` holds
  [query | gate] per head; ``experts.gate_up_proj`` holds [gate ; up].
* Expert tensors are 3-D (experts, m, n): one matrix per expert, batched.
  Under FSDP2 they are sharded on the expert dimension, so each rank already
  holds whole experts and needs no communication.
* 2-D DTensors are sharded by rows: the update is all-gathered, orthogonalised
  and the local rows are kept.
* Momentum is stored in bf16 by default (saves ~1.8 GiB per T4 for Ludwig 2).
  ``momentum_device="cpu"`` keeps it in pinned CPU RAM instead (another ~1.7 GiB per T4):
  each step copies it to the GPU and back. It stays a DTensor on a CPU mesh, so distributed
  checkpoints still save every shard.
* ``fused_qgate``: True for Qwen3-Next (``q_proj`` = [query | gate]); False for AFMoE, whose gate
  is its own matrix.
"""
from __future__ import annotations

import math

import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor
except Exception:  # pragma: no cover - torch < 2.5
    try:
        from torch.distributed._tensor import DTensor
    except Exception:
        DTensor = ()  # type: ignore

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(g: torch.Tensor, steps: int = 5, dtype: torch.dtype = torch.float16, eps: float = 1e-7):
    """Approximate the orthogonal factor of g (batched over leading dims)."""
    a, b, c = NS_COEFFS
    x = g.float()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    x = x.to(dtype)
    transposed = x.size(-2) > x.size(-1)
    if transposed:
        x = x.mT
    for _ in range(steps):
        s = x @ x.mT
        x = a * x + (b * s + c * (s @ s)) @ x
    if transposed:
        x = x.mT
    return x


def _orthogonalize(u: torch.Tensor, kind: str, n_heads: int, head_dim: int, ns_dtype) -> torch.Tensor:
    def ns(t):
        m, n = t.size(-2), t.size(-1)
        return newton_schulz(t, dtype=ns_dtype).float() * (0.2 * math.sqrt(max(m, n)))

    if kind == "qgate":                              # (H*2*hd, d): per head [q(hd) ; gate(hd)]
        d_in = u.size(-1)
        v = u.view(n_heads, 2, head_dim, d_in)
        q = ns(v[:, 0].reshape(n_heads * head_dim, d_in))
        g = ns(v[:, 1].reshape(n_heads * head_dim, d_in))
        return torch.stack([q.view(n_heads, head_dim, d_in), g.view(n_heads, head_dim, d_in)], dim=1).reshape(u.shape)
    if kind == "halves":                             # (..., 2*inter, d): [gate ; up]
        top, bottom = u.chunk(2, dim=-2)
        return torch.cat([ns(top), ns(bottom)], dim=-2)
    return ns(u)


def _kind(name: str, fused_qgate: bool = True) -> str:
    if fused_qgate and name.endswith("self_attn.q_proj.weight"):
        return "qgate"
    if name.endswith("experts.gate_up_proj"):
        return "halves"
    return "plain"


def _local(t):
    return t.to_local() if isinstance(t, DTensor) else t


class Muon(torch.optim.Optimizer):
    def __init__(self, named_params, lr=1e-3, momentum=0.95, nesterov=True, weight_decay=0.1,
                 n_heads=16, head_dim=64, momentum_dtype=torch.bfloat16, ns_dtype=torch.float16,
                 fused_qgate=True, momentum_device=None, cpu_mesh=None):
        named_params = list(named_params)
        params = [p for _, p in named_params]
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))
        self._kinds = {p: _kind(n, fused_qgate) for n, p in named_params}
        self._names = {p: n for n, p in named_params}
        self.n_heads, self.head_dim = n_heads, head_dim
        self.momentum_dtype, self.ns_dtype = momentum_dtype, ns_dtype
        self.momentum_cpu = momentum_device == "cpu"
        self.cpu_mesh = cpu_mesh

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = _local(p.grad).float()
                state = self.state[p]
                if "momentum_buffer" not in state:
                    on_cpu = self.momentum_cpu and g.device.type != "cpu"
                    zeros = torch.zeros(g.shape, dtype=self.momentum_dtype, device="cpu" if on_cpu else g.device,
                                        pin_memory=on_cpu)
                    if isinstance(p, DTensor):  # keep it a DTensor so distributed checkpoints save every shard
                        mesh = self.cpu_mesh if on_cpu else p.device_mesh
                        zeros = DTensor.from_local(zeros, mesh, p.placements, shape=p.shape, stride=p.stride())
                    state["momentum_buffer"] = zeros
                stored = _local(state["momentum_buffer"])
                buf = stored.to(g.device, non_blocking=True) if stored.device != g.device else stored
                buf.mul_(mu).add_(g.to(buf.dtype))
                if buf is not stored:
                    stored.copy_(buf, non_blocking=True)
                u = g.add(buf.float(), alpha=mu) if group["nesterov"] else buf.float()

                kind = self._kinds[p]
                sharded_rows = isinstance(p, DTensor) and p.ndim == 2 and u.shape != p.shape
                if sharded_rows:
                    # 2-D matrix split by rows across ranks: gather, orthogonalise, keep our rows.
                    full = DTensor.from_local(u, p.device_mesh, p.placements, shape=p.shape, stride=p.stride()).full_tensor()
                    o_full = _orthogonalize(full, kind, self.n_heads, self.head_dim, self.ns_dtype)
                    ws, rank = p.device_mesh.size(), p.device_mesh.get_local_rank()
                    o = o_full.chunk(ws, dim=0)[rank]
                else:
                    # plain tensor, or 3-D experts sharded on the expert dimension: local is enough.
                    o = _orthogonalize(u, kind, self.n_heads, self.head_dim, self.ns_dtype)

                x = _local(p)
                x.mul_(1.0 - lr * wd)
                x.add_(o.to(x.dtype), alpha=-lr)


def split_params(model):
    """Muon for matrices (attention, experts, shared expert, MTP projection);
    AdamW for embeddings, norms, router and the shared-expert gate."""
    muon, adam_decay, adam_no_decay = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_matrix = p.ndim >= 2
        is_special = ("embed_tokens" in n or "lm_head" in n or n.endswith("mlp.gate.weight")
                      or n.endswith("router.gate.weight") or "shared_expert_gate" in n)
        if is_matrix and not is_special:
            muon.append((n, p))
        elif is_matrix:
            adam_decay.append(p)
        else:
            adam_no_decay.append(p)
    return muon, adam_decay, adam_no_decay


def build_optimizers(model, lr: float, weight_decay: float = 0.1, n_heads: int = 16, head_dim: int = 64,
                     momentum_dtype=torch.bfloat16, fused_qgate: bool = True, adamw_lr: float | None = None,
                     momentum_device=None, cpu_mesh=None):
    muon_p, decay_p, no_decay_p = split_params(model)
    muon = Muon(muon_p, lr=lr, weight_decay=weight_decay, n_heads=n_heads, head_dim=head_dim,
                momentum_dtype=momentum_dtype, fused_qgate=fused_qgate, momentum_device=momentum_device,
                cpu_mesh=cpu_mesh)
    adamw = torch.optim.AdamW(
        [{"params": decay_p, "weight_decay": weight_decay}, {"params": no_decay_p, "weight_decay": 0.0}],
        lr=lr if adamw_lr is None else adamw_lr, betas=(0.9, 0.95), eps=1e-8, foreach=False,
    )
    return muon, adamw


class DynamicLossScaler:
    """Loss scaling for FP16 that works with plain tensors and DTensors.
    Skips the step when any rank sees a non-finite gradient."""

    def __init__(self, init_scale=2.0 ** 15, growth_interval=1000, min_scale=1.0):
        self.scale = init_scale
        self.growth_interval = growth_interval
        self.min_scale = min_scale
        self._good_steps = 0
        self.skipped = 0

    @torch.no_grad()
    def unscale_and_check(self, params) -> bool:
        inv = 1.0 / self.scale
        bad = None
        for p in params:
            if p.grad is None:
                continue
            g = _local(p.grad)
            g.mul_(inv)
            flag = (~torch.isfinite(g)).any().float()
            bad = flag if bad is None else torch.maximum(bad, flag)
        if bad is None:
            return False
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(bad, op=dist.ReduceOp.MAX)
        return bool(bad.item() > 0)

    def update(self, found_inf: bool):
        if found_inf:
            self.scale = max(self.scale / 2.0, self.min_scale)
            self._good_steps = 0
            self.skipped += 1
        else:
            self._good_steps += 1
            if self._good_steps % self.growth_interval == 0:
                self.scale *= 2.0

    def state_dict(self):
        return {"scale": self.scale, "good_steps": self._good_steps, "skipped": self.skipped}

    def load_state_dict(self, sd):
        self.scale, self._good_steps, self.skipped = sd["scale"], sd["good_steps"], sd["skipped"]
