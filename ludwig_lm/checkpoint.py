"""Full-state checkpoints for scripts/train.py, with a check of every resume.

One checkpoint is a directory written by both ranks:

  .metadata, __*.distcp   model, Muon, AdamW, loss scaler and expert balancer (torch.distributed.checkpoint)
  rank{r}.pt              this rank's data-reader state and RNG states, a fingerprint of its shards, and the
                          hash of the batch that comes next
  meta.json               step, tokens, loss history, run settings. Written last: a directory without it is
                          an interrupted save and is never resumed from

Loading relies on dcp.load restoring the state in place, and does not call set_state_dict for the
optimizers: torch.optim.Optimizer.load_state_dict casts every state tensor to its parameter's dtype and
device, which turns Muon's bf16 momentum into FP32 (on the GPU, +1.7 GiB per T4 with this model).

After loading, ``fingerprint`` is recomputed and compared bit for bit with the one saved, and the first
batch of the resumed run is compared with the saved hash: the two checks of the G0 gate (5.7, point 4).
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover - torch < 2.5
    from torch.distributed._tensor import DTensor


def _local(t):
    return t.to_local() if isinstance(t, DTensor) else t


@torch.no_grad()
def fingerprint(model, optimizers, balancer) -> torch.Tensor:
    """Float64 sums of this rank's shards: parameters, expert bias, optimizer states, balancer momentum."""
    vals = [_local(p).detach().double().sum() for p in model.parameters()]
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                st = opt.state.get(p, {})
                vals += [_local(st[k]).detach().double().sum() for k in sorted(st) if torch.is_tensor(st[k])]
    vals += [b.expert_bias.detach().double().sum() for b in balancer.blocks]
    vals.append(balancer.momentum.double().sum())
    return torch.stack([v.cpu() for v in vals])


def rng_state() -> dict:
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state()
    return st


def set_rng_state(st: dict):
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["cuda"])


def _state(model, optimizers, scaler, balancer):
    from torch.distributed.checkpoint.state_dict import get_state_dict

    msd, osd = get_state_dict(model, optimizers)   # on a fresh optimizer this creates its state (a zero step)
    return {"model": msd, "optim": osd, "scaler": scaler.state_dict(), "balancer": balancer.state_dict()}


def all_ok(ok: bool) -> bool:
    """True only if every rank says True. A CPU tensor: gloo carries it with either backend set."""
    t = torch.tensor([1.0 if ok else 0.0])
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item() == 1.0)


def local_step(fn) -> bool:
    try:
        fn()
        return True
    except Exception as e:  # noqa: BLE001 - reported through all_ok, so no rank waits at a barrier alone
        print(f"[checkpoint] rank {dist.get_rank()}: {type(e).__name__}: {e}", flush=True)
        return False


def save(path, model, optimizers, scaler, balancer, rank: int, rank_extra: dict, meta: dict):
    """Raises on every rank if any rank fails (torch.distributed.checkpoint does the same for its part)."""
    import torch.distributed.checkpoint as dcp

    path = Path(path)
    if not all_ok(local_step(lambda: shutil.rmtree(path) if rank == 0 and path.exists() else None)):
        raise RuntimeError(f"could not clear {path}")
    dcp.save(_state(model, optimizers, scaler, balancer), checkpoint_id=str(path))
    extra = {**rank_extra, "fingerprint": fingerprint(model, optimizers, balancer), "scaler": scaler.state_dict(),
             "rng": rng_state()}
    if not all_ok(local_step(lambda: torch.save(extra, path / f"rank{rank}.pt"))):
        raise RuntimeError(f"could not write the per-rank files of {path}")
    meta_ok = local_step(lambda: (path / "meta.json").write_text(json.dumps(meta, indent=1))) if rank == 0 else True
    if not all_ok(meta_ok):
        raise RuntimeError(f"could not write {path / 'meta.json'}")


def load(path, model, optimizers, scaler, balancer, rank: int):
    """Restores everything in place. Returns (meta, this rank's extras, state bit-exact?)."""
    import torch.distributed.checkpoint as dcp

    path = Path(path)
    meta = json.loads((path / "meta.json").read_text())
    state = _state(model, optimizers, scaler, balancer)
    dcp.load(state, checkpoint_id=str(path))
    scaler.load_state_dict(state["scaler"])
    balancer.load_state_dict(state["balancer"])
    extra = torch.load(path / f"rank{rank}.pt", weights_only=False)
    set_rng_state(extra["rng"])
    exact = torch.equal(extra["fingerprint"], fingerprint(model, optimizers, balancer)) \
        and extra["scaler"] == scaler.state_dict()
    return meta, extra, exact


def complete(save_dir) -> list[Path]:
    """Finished checkpoints in save_dir, oldest first."""
    d = Path(save_dir)
    return sorted(p.parent for p in d.glob("step_*/meta.json")) if d.is_dir() else []


def resolve(resume: str, save_dir: str) -> Path:
    if resume == "latest":
        done = complete(save_dir)
        if not done:
            raise FileNotFoundError(f"no finished checkpoint in {save_dir}")
        return done[-1]
    path = Path(resume)
    if not (path / "meta.json").is_file():
        raise FileNotFoundError(f"{path} is not a finished checkpoint (no meta.json)")
    return path


def rotate(save_dir, keep: int, protected: set[str]):
    """Rank 0: delete all but the newest ``keep`` periodic checkpoints, and interrupted saves."""
    done = complete(save_dir)
    periodic = [p for p in done if p.name not in protected]
    for p in periodic[:-keep] if keep > 0 else periodic:
        shutil.rmtree(p, ignore_errors=True)
    newest = done[-1].name if done else ""
    for p in Path(save_dir).glob("step_*"):
        if not (p / "meta.json").exists() and p.name < newest:
            shutil.rmtree(p, ignore_errors=True)
