#!/usr/bin/env python3
"""T0.2 — throughput and memory of the real Ludwig 2 on 2×T4, with synthetic tokens.

Everything that matters for the real run is real: the 1.96B model, FSDP2 over the
two GPUs, FP16 with loss scaling, Muon (bf16 momentum) + AdamW, the chunked loss,
the balancing and z-losses, optional MTP, optional activation checkpointing, and a
distributed checkpoint save + load. Only the data is random.

Launch (Kaggle 2×T4):
  torchrun --nproc_per_node 2 scripts/bench_train.py --seq 2048 --micro 4 --accum 2 --moe-impl grouped_mm
CPU logic test (tiny model, gloo):
  torchrun --nproc_per_node 2 scripts/bench_train.py --config configs/tiny.json --seq 64 --micro 2 --steps 3

Appends one JSON line per run to --out (default results/t0_2.jsonl).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ludwig_lm.model import LossWeights, LudwigLM, count_params, load_config  # noqa: E402
from ludwig_lm.muon import DynamicLossScaler, build_optimizers  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "ludwig2.json"))
    ap.add_argument("--experts", type=int, default=None, help="override num_experts (plan B: 32)")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--micro", type=int, default=4, help="sequences per GPU per micro-step")
    ap.add_argument("--accum", type=int, default=2, help="micro-steps per optimizer step")
    ap.add_argument("--steps", type=int, default=6, help="timed optimizer steps (after 2 warm-up)")
    ap.add_argument("--moe-impl", default="grouped_mm", choices=["eager", "batched_mm", "grouped_mm"])
    ap.add_argument("--no-ckpt", action="store_true", help="disable activation checkpointing")
    ap.add_argument("--momentum-dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--mtp", type=float, default=0.0, help="MTP loss weight (0 = MTP head not trained)")
    ap.add_argument("--ce-chunk", type=int, default=4096)
    ap.add_argument("--ckpt-test", action="store_true", help="time a distributed checkpoint save + load")
    ap.add_argument("--ckpt-dir", default="/tmp/ludwig_ckpt_test")
    ap.add_argument("--fixed-batch", action="store_true", help="reuse one batch: loss must fall (update sanity check)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(ROOT / "results" / "t0_2.jsonl"))
    return ap.parse_args()


def main():
    args = parse()
    use_cuda = torch.cuda.is_available()
    # short timeout: if only one rank runs out of memory, the other fails fast instead of hanging
    dist.init_process_group("nccl" if use_cuda else "gloo", timeout=timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    is0 = rank == 0

    def say(*a):
        if is0:
            print(*a, flush=True)

    # ---------------------------------------------------------------- model
    from torch.distributed.device_mesh import init_device_mesh
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    except ImportError:  # torch < 2.6
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

    overrides = {} if args.experts is None else {"num_experts": args.experts}
    cfg = load_config(args.config, **overrides)
    mesh = init_device_mesh(device.type, (world,))
    mp = MixedPrecisionPolicy(param_dtype=torch.float16 if use_cuda else torch.float32, reduce_dtype=torch.float32)

    def build_and_shard():
        torch.manual_seed(0)                       # same initial weights on every rank
        m = LudwigLM(cfg, with_mtp=True)
        m.lm.config._experts_implementation = args.moe_impl
        if args.mtp == 0:
            m.mtp.requires_grad_(False)            # as in the trunk: the MTP head is trained only at the end
        if not args.no_ckpt:
            m.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        for layer in m.lm.model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp)
        fully_shard(m.mtp.layers[0], mesh=mesh, mp_policy=mp)
        # root keeps embeddings/head unsharded through backward (needed by the checkpointed chunked loss)
        fully_shard(m, mesh=mesh, mp_policy=mp, reshard_after_forward=False)
        return m

    # Ranks build one after another: the full FP32 model (~8 GB) lives in CPU RAM only until it is
    # sharded onto the GPU, so two ranks never hold it at the same time (Kaggle has ~30 GB of RAM).
    t_build = time.time()
    model = None
    for r in range(world):
        if r == rank:
            model = build_and_shard()
        dist.barrier()
    counts = count_params(model)
    say(f"[model] total {counts['total']/1e9:.3f}B, active {counts['active']/1e9:.3f}B, mtp {counts['mtp']/1e6:.0f}M "
        f"| built+sharded in {time.time()-t_build:.0f}s | moe={args.moe_impl} | world={world} | device={device.type}")

    mom_dtype = torch.bfloat16 if args.momentum_dtype == "bf16" else torch.float32
    muon, adamw = build_optimizers(model, lr=args.lr, n_heads=cfg.num_attention_heads, head_dim=cfg.head_dim,
                                   momentum_dtype=mom_dtype)
    params = [p for p in model.parameters() if p.requires_grad]
    scaler = DynamicLossScaler(init_scale=2.0 ** 15 if use_cuda else 1.0)
    weights = LossWeights(mtp=args.mtp)
    gen = torch.Generator(device="cpu").manual_seed(1234 + rank)

    fixed = torch.randint(0, cfg.vocab_size, (args.micro, args.seq), generator=gen).to(device)

    def batch():
        if args.fixed_batch:
            return fixed
        return torch.randint(0, cfg.vocab_size, (args.micro, args.seq), generator=gen).to(device)

    def opt_step():
        mets = None
        for _ in range(args.accum):
            loss, mets = model(batch(), weights, ce_chunk=args.ce_chunk)
            (loss * (scaler.scale / args.accum)).backward()
        found_inf = scaler.unscale_and_check(params)
        gnorm = None
        if not found_inf:
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            gnorm = float(gnorm.full_tensor() if hasattr(gnorm, "full_tensor") else gnorm)
            muon.step()
            adamw.step()
        scaler.update(found_inf)
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        return mets, gnorm, found_inf

    def sync():
        if use_cuda:
            torch.cuda.synchronize()
        dist.barrier()

    # ---------------------------------------------------------------- warm-up + timed steps
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    status, err = "ok", None
    times, ces, loads, gnorms = [], [], [], []
    try:
        for i in range(2 + args.steps):
            sync()
            t = time.time()
            mets, gnorm, found_inf = opt_step()
            sync()
            if i >= 2:
                times.append(time.time() - t)
                ces.append(float(mets["ce"]))
                loads.append(float(mets["load_max_over_mean"]))
                gnorms.append(gnorm)
            say(f"  step {i:2d}  {time.time()-t:7.2f}s  ce={float(mets['ce']):.3f}  load_max/mean={float(mets['load_max_over_mean']):.2f}"
                f"  skipped={scaler.skipped}  scale={scaler.scale:.0f}")
    except torch.cuda.OutOfMemoryError as e:
        status, err = "oom", str(e)[:300]
    except RuntimeError as e:
        status, err = ("oom" if "out of memory" in str(e).lower() else "error"), str(e)[:500]

    peak_alloc = torch.tensor(torch.cuda.max_memory_allocated() / 2**30 if use_cuda else 0.0, device=device)
    peak_res = torch.tensor(torch.cuda.max_memory_reserved() / 2**30 if use_cuda else 0.0, device=device)
    dist.all_reduce(peak_alloc, op=dist.ReduceOp.MAX)
    dist.all_reduce(peak_res, op=dist.ReduceOp.MAX)

    tokens_per_step = world * args.micro * args.seq * args.accum
    step_s = sum(times) / len(times) if times else None
    result = {
        "tag": args.tag, "status": status, "error": err, "world": world, "gpu": torch.cuda.get_device_name(0) if use_cuda else "cpu",
        "experts": cfg.num_experts, "seq": args.seq, "micro": args.micro, "accum": args.accum, "moe_impl": args.moe_impl,
        "act_ckpt": not args.no_ckpt, "momentum_dtype": args.momentum_dtype, "mtp": args.mtp,
        "params_total_B": round(counts["total"] / 1e9, 3), "params_active_B": round(counts["active"] / 1e9, 3),
        "tokens_per_step": tokens_per_step, "step_s": step_s,
        "tokens_per_s": (tokens_per_step / step_s) if step_s else None,
        "peak_alloc_GiB": round(float(peak_alloc), 2), "peak_reserved_GiB": round(float(peak_res), 2),
        "loss_scale_skips": scaler.skipped, "ce_last": ces[-1] if ces else None,
        "load_max_over_mean": max(loads) if loads else None, "grad_norm_last": gnorms[-1] if gnorms else None,
        "torch": torch.__version__,
    }
    if result["tokens_per_s"]:
        h25 = 25e9 / result["tokens_per_s"] / 3600
        result["hours_for_25B"] = round(h25)

    # ---------------------------------------------------------------- checkpoint save + load
    if args.ckpt_test and status == "ok":
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        if is0:
            shutil.rmtree(args.ckpt_dir, ignore_errors=True)
        dist.barrier()
        probe = next(p for p in model.parameters() if p.ndim == 3)          # an expert tensor
        before = probe.to_local().detach().float().clone() if hasattr(probe, "to_local") else probe.detach().float().clone()
        msd, osd = get_state_dict(model, [muon, adamw])
        sync()
        t = time.time()
        dcp.save({"model": msd, "optim": osd, "scaler": scaler.state_dict()}, checkpoint_id=args.ckpt_dir)
        sync()
        save_s = time.time() - t
        with torch.no_grad():
            (probe.to_local() if hasattr(probe, "to_local") else probe).add_(1.0)  # corrupt, then restore
        msd, osd = get_state_dict(model, [muon, adamw])
        state = {"model": msd, "optim": osd, "scaler": scaler.state_dict()}
        sync()
        t = time.time()
        dcp.load(state, checkpoint_id=args.ckpt_dir)
        set_state_dict(model, [muon, adamw], model_state_dict=state["model"], optim_state_dict=state["optim"])
        sync()
        load_s = time.time() - t
        after = probe.to_local().detach().float() if hasattr(probe, "to_local") else probe.detach().float()
        ok = torch.tensor(float(torch.equal(before, after)), device=device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        size = sum(f.stat().st_size for f in Path(args.ckpt_dir).rglob("*") if f.is_file()) / 2**30 if is0 else 0
        result.update({"ckpt_save_s": round(save_s, 1), "ckpt_load_s": round(load_s, 1),
                       "ckpt_GiB": round(size, 2), "ckpt_restore_exact": bool(ok.item())})
        say(f"[ckpt] save {save_s:.1f}s, load {load_s:.1f}s, {size:.2f} GiB, restore exact={bool(ok.item())}")

    if is0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")
        tps = result["tokens_per_s"]
        print(f"[result] {args.tag or '-'} status={status} tokens/s={tps and round(tps):} "
              f"peak={result['peak_reserved_GiB']} GiB step={step_s and round(step_s, 2)}s "
              f"25B≈{result.get('hours_for_25B')} h", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
