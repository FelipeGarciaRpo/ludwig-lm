#!/usr/bin/env python3
"""T0.2 v2: the real Ludwig 2 training step in the AFMoE format, on 2×T4.

Everything is the real thing except the data: the 1.906B model (``configs/ludwig2_afmoe.json``),
FSDP2 over the two GPUs, FP16 with loss scaling, Muon + AdamW, the chunked loss with the tied head,
the balancing losses and the expert-bias balancer, activation checkpointing. Each speed piece can be
turned off to measure what it is worth:

  --moe-impl padded_bmm|grouped_mm|eager   experts kernel (grouped_mm falls back to a loop on the T4)
  --residual fp32|fp16                     residual stream precision (fp32 = autocast + FSDP2 without input casting)
  --reduce fp16|fp32                       dtype of the gradient reduce-scatter between the GPUs
  --no-prefetch                            FSDP2 without explicit prefetching of the next layer
  --momentum-device gpu|cpu                Muon's bf16 momentum on the GPU or in pinned CPU RAM

Modes:
  speed (default)    2 warm-up + --steps timed optimizer steps on random tokens. tokens/s is reported
                     with the optimizer step amortised over a real batch of --tokens-per-step tokens.
  --train-steps N    N optimizer steps on bench v1's synthetic Markov text; logs the loss, the FP16
                     scale and the expert balance every 5 steps (SMEBU vs the sign update).
  --ckpt-test        distributed checkpoint save + load of model, optimizers, scaler and balancer.

Kaggle 2×T4:  torchrun --nproc_per_node 2 scripts/bench_afmoe.py --tag base
CPU logic:    torchrun --nproc_per_node 2 scripts/bench_afmoe.py --config configs/tiny_afmoe.json --seq 64 --micro 2 --steps 2
Appends one JSON line per run to --out (default results/t0_2v2.jsonl).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import traceback
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ludwig_lm.afmoe import (ExpertBalancer, LossWeights, LudwigAfmoe, count_params,  # noqa: E402
                             flops_per_token, load_config, shard_model)
from ludwig_lm.muon import DynamicLossScaler, build_optimizers  # noqa: E402

T4_PEAK_TFLOPS = 65.0


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "ludwig2_afmoe.json"))
    ap.add_argument("--experts", type=int, default=None)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--micro", type=int, default=8, help="sequences per GPU per micro-step")
    ap.add_argument("--accum", type=int, default=1, help="micro-steps per measured optimizer step")
    ap.add_argument("--steps", type=int, default=4, help="timed optimizer steps (after 2 warm-up)")
    ap.add_argument("--moe-impl", default="padded_bmm", choices=["padded_bmm", "grouped_mm", "eager"])
    ap.add_argument("--residual", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--reduce", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--no-prefetch", action="store_true")
    ap.add_argument("--momentum-device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--no-act-ckpt", action="store_true")
    ap.add_argument("--ce-chunk", type=int, default=1024, help="tokens per loss chunk: 1024 x 64K logits ~ 1 GB transient")
    ap.add_argument("--lr", type=float, default=1e-3, help="Muon peak LR (Trinity Nano: 1e-3)")
    ap.add_argument("--adamw-lr", type=float, default=3e-4, help="AdamW peak LR (Trinity Nano: 3e-4)")
    ap.add_argument("--balance", default="smebu", choices=["smebu", "sign"])
    ap.add_argument("--lam", type=float, default=1e-3, help="SMEBU step size")
    ap.add_argument("--gamma", type=float, default=1e-3, help="sign-update step size")
    ap.add_argument("--seq-aux", type=float, default=1e-4)
    ap.add_argument("--router-z", type=float, default=1e-4)
    ap.add_argument("--output-z", type=float, default=1e-4)
    ap.add_argument("--train-steps", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--tokens-per-step", type=int, default=524288, help="real batch, to amortise the optimizer step")
    ap.add_argument("--ckpt-test", action="store_true")
    ap.add_argument("--ckpt-dir", default="/tmp/ludwig_ckpt_test")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=str(ROOT / "results" / "t0_2v2.jsonl"))
    return ap.parse_args()


class MarkovText:
    """Bench v1's synthetic text: a Markov chain over 4,096 tokens (8 successors each); the second half
    of every sequence copies the first, so attention has work to do. Loss floor ~ bigram entropy / 2."""

    def __init__(self, seq, seed, vocab, states=4096, branches=8):
        g = torch.Generator().manual_seed(1234)                      # same chain on every rank
        states = min(states, vocab)
        self.states = states
        self.succ = torch.randint(0, states, (states, branches), generator=g)
        self.prob = torch.softmax(torch.randn(states, branches, generator=g) * 1.5, -1)
        self.seq = seq
        self.rng = torch.Generator().manual_seed(seed)

    def batch(self, b):
        x = torch.empty(b, self.seq, dtype=torch.long)
        cur = torch.randint(0, self.states, (b,), generator=self.rng)
        for t in range(self.seq):
            x[:, t] = cur
            pick = torch.multinomial(self.prob[cur], 1, generator=self.rng).squeeze(1)
            cur = self.succ[cur, pick]
        half = self.seq // 2
        x[:, half:] = x[:, : self.seq - half].clone()
        return x

    def floor(self):
        p = self.prob
        return float(-(p * p.log()).sum(-1).mean()) / 2


def main():
    args = parse()
    use_cuda = torch.cuda.is_available()
    # cpu:gloo next to cuda:nccl so Muon's momentum can live on a CPU mesh; a short timeout so that
    # one rank running out of memory makes the other fail instead of hanging
    dist.init_process_group("cpu:gloo,cuda:nccl" if use_cuda else "gloo", timeout=timedelta(minutes=10))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    is0 = rank == 0

    def say(*a):
        if is0:
            print(*a, flush=True)

    def sync():
        if use_cuda:
            torch.cuda.synchronize()
        dist.barrier()

    from torch.distributed.device_mesh import init_device_mesh
    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy
    except ImportError:  # torch < 2.6
        from torch.distributed._composable.fsdp import MixedPrecisionPolicy

    # ---------------------------------------------------------------- model
    overrides = {} if args.experts is None else {"num_experts": args.experts}
    cfg = load_config(args.config, **overrides)
    cfg._experts_implementation = args.moe_impl
    mesh = init_device_mesh(device.type, (world,))
    cpu_mesh = init_device_mesh("cpu", (world,)) if (use_cuda and args.momentum_device == "cpu") else None
    fp32_residual = args.residual == "fp32"
    half = torch.float16 if use_cuda else torch.float32
    reduce_dtype = torch.float16 if (use_cuda and args.reduce == "fp16") else torch.float32
    mp = MixedPrecisionPolicy(param_dtype=half, reduce_dtype=reduce_dtype, cast_forward_inputs=not fp32_residual)

    t_build = time.time()
    torch.manual_seed(0)
    with torch.device(device):          # built straight on each GPU (7.6 GB in FP32), then sharded
        model = LudwigAfmoe(cfg, fp32_residual=fp32_residual)
    if not args.no_act_ckpt:
        model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    shard_model(model, mesh, mp, prefetch=not args.no_prefetch)
    if use_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    counts = count_params(model)
    flops_tok = flops_per_token(cfg, counts["active"], args.seq)
    say(f"[model] AFMoE total {counts['total']/1e9:.3f}B, active {counts['active']/1e9:.3f}B | built+sharded in "
        f"{time.time()-t_build:.0f}s | moe={args.moe_impl} residual={args.residual} reduce={args.reduce} "
        f"prefetch={not args.no_prefetch} momentum={args.momentum_device} | world={world} device={device.type}")

    muon, adamw = build_optimizers(model, lr=args.lr, adamw_lr=args.adamw_lr, n_heads=cfg.num_attention_heads,
                                   head_dim=cfg.head_dim, fused_qgate=False,
                                   momentum_device="cpu" if args.momentum_device == "cpu" else None,
                                   cpu_mesh=cpu_mesh)
    params = [p for p in model.parameters() if p.requires_grad]
    scaler = DynamicLossScaler(init_scale=2.0 ** 15 if use_cuda else 1.0)
    balancer = ExpertBalancer(model, rule=args.balance, lam=args.lam, gamma=args.gamma)
    weights = LossWeights(output_z=args.output_z, seq_aux=args.seq_aux, router_z=args.router_z)
    amp_dtype = torch.float16 if use_cuda else torch.bfloat16      # bf16 only exercises the code path on CPU
    gen = torch.Generator(device="cpu").manual_seed(1234 + rank)

    def micro_step(ids):
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=fp32_residual):
            loss, mets = model(ids, weights, ce_chunk=args.ce_chunk)
        (loss * (scaler.scale / args.accum)).backward()
        balancer.accumulate(mets["chosen"])
        return mets

    def opt_step(lr_mult=1.0):
        found_inf = scaler.unscale_and_check(params)
        gnorm = None
        if not found_inf:
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
            gnorm = float(gn.full_tensor() if hasattr(gn, "full_tensor") else gn)
            for opt, base in ((muon, args.lr), (adamw, args.adamw_lr)):
                for g in opt.param_groups:
                    g["lr"] = base * lr_mult
            muon.step()
            adamw.step()
        scaler.update(found_inf)
        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        return gnorm, found_inf, balancer.step()

    def mean_over_ranks(x):
        t = torch.tensor(float(x), device=device)
        dist.all_reduce(t)
        return float(t) / world

    result = {"tag": args.tag, "world": world, "gpu": torch.cuda.get_device_name(0) if use_cuda else "cpu",
              "format": "afmoe", "experts": cfg.num_experts, "seq": args.seq, "micro": args.micro,
              "accum": args.accum, "moe_impl": args.moe_impl, "residual": args.residual, "reduce": args.reduce,
              "prefetch": not args.no_prefetch, "momentum_device": args.momentum_device,
              "act_ckpt": not args.no_act_ckpt, "balance": args.balance,
              "lam": args.lam if args.balance == "smebu" else None, "gamma": args.gamma if args.balance == "sign" else None,
              "params_total_B": round(counts["total"] / 1e9, 3), "params_active_B": round(counts["active"] / 1e9, 3),
              "torch": torch.__version__}
    status, err = "ok", None
    t_micro, t_opt, ces, curve = [], [], [], []
    try:
        if args.train_steps:
            # ---------------------------------------------------- short training on Markov text
            data = MarkovText(args.seq, seed=5000 + rank, vocab=cfg.vocab_size)
            sync()
            t0 = time.time()
            for s in range(args.train_steps):
                mult = min(1.0, (s + 1) / max(1, args.warmup))
                ce_vals = [float(micro_step(data.batch(args.micro).to(device))["ce"]) for _ in range(args.accum)]
                gnorm, found_inf, bal = opt_step(mult)
                if s % 5 == 0 or s == args.train_steps - 1:
                    point = {"step": s, "ce": mean_over_ranks(sum(ce_vals) / len(ce_vals)), "grad_norm": gnorm,
                             "scale": scaler.scale, "skipped": found_inf, **bal}
                    curve.append(point)
                    say(f"  step {s:3d}  ce={point['ce']:.3f}  load max/mean={bal['load_max_over_mean']:.2f}  "
                        f"dead={bal['dead_experts']}  |bias|max={bal['bias_abs_max']:.3f}  scale={scaler.scale:.0f}")
            sync()
            dt = time.time() - t0
            finite = [p["ce"] for p in curve if math.isfinite(p["ce"])]
            result.update({"train_steps": args.train_steps, "curve": curve, "ce_first": curve[0]["ce"],
                           "ce_last": curve[-1]["ce"], "ce_floor": round(data.floor(), 3),
                           "nan_points": len(curve) - len(finite), "loss_scale_skips": scaler.skipped,
                           "load_max_over_mean_last": curve[-1]["load_max_over_mean"],
                           "load_max_over_mean_peak": max(p["load_max_over_mean"] for p in curve),
                           "tokens_per_s": args.train_steps * args.accum * world * args.micro * args.seq / dt})
        else:
            # ---------------------------------------------------- speed on random tokens
            for i in range(2 + args.steps):
                sync()
                a = time.time()
                for _ in range(args.accum):
                    mets = micro_step(torch.randint(0, cfg.vocab_size, (args.micro, args.seq), generator=gen).to(device))
                sync()
                b = time.time()
                gnorm, found_inf, bal = opt_step()
                sync()
                c = time.time()
                if i >= 2:
                    t_micro.append((b - a) / args.accum)
                    t_opt.append(c - b)
                    ces.append(float(mets["ce"]))
                say(f"  step {i:2d}  micro {(b-a)/args.accum:6.2f}s  optimizer {c-b:5.2f}s  ce={float(mets['ce']):.3f}  "
                    f"load max/mean={bal['load_max_over_mean']:.2f}  skipped={scaler.skipped}")
    except torch.cuda.OutOfMemoryError as e:
        status, err = "oom", str(e)[:300]
    except Exception as e:  # noqa: BLE001 - every failure is reported in the results file
        status = "oom" if "out of memory" in str(e).lower() else "error"
        err = traceback.format_exc()[-1500:]

    peak = torch.tensor([torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30]
                        if use_cuda else [0.0, 0.0], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    result.update({"status": status, "error": err, "peak_alloc_GiB": round(float(peak[0]), 2),
                   "peak_reserved_GiB": round(float(peak[1]), 2)})
    if t_micro:
        tm, to = sum(t_micro) / len(t_micro), sum(t_opt) / len(t_opt)
        tok_micro = world * args.micro * args.seq
        ga_real = max(1, math.ceil(args.tokens_per_step / tok_micro))
        tps = tok_micro / (tm + to / ga_real)
        result.update({"t_micro_s": round(tm, 3), "t_opt_s": round(to, 3), "tokens_per_micro": tok_micro,
                       "ga_real": ga_real, "tokens_per_s_measured": round(tok_micro * args.accum / (tm * args.accum + to)),
                       "tokens_per_s": round(tps), "mfu": round(flops_tok * tps / (world * T4_PEAK_TFLOPS * 1e12), 4),
                       "hours_for_25B": round(25e9 / tps / 3600), "weeks_at_140h": round(25e9 / tps / 3600 / 140, 1),
                       "loss_scale_skips": scaler.skipped, "ce_last": ces[-1]})

    # ---------------------------------------------------------------- checkpoint save + load
    if args.ckpt_test and status == "ok":
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

        if is0:
            shutil.rmtree(args.ckpt_dir, ignore_errors=True)
        dist.barrier()
        probe = next(p for n, p in model.named_parameters() if "experts.gate_up_proj" in n)
        bias = balancer.blocks[0].expert_bias

        def local(t):
            return (t.to_local() if hasattr(t, "to_local") else t).detach().float().clone()

        before, bias_before, mom_before = local(probe), bias.detach().clone(), balancer.momentum.clone()

        def full_state():
            msd, osd = get_state_dict(model, [muon, adamw])
            return {"model": msd, "optim": osd, "scaler": scaler.state_dict(), "balancer": balancer.state_dict()}

        state = full_state()
        sync()
        t = time.time()
        dcp.save(state, checkpoint_id=args.ckpt_dir)
        sync()
        save_s = time.time() - t
        with torch.no_grad():                                       # corrupt, then restore
            (probe.to_local() if hasattr(probe, "to_local") else probe).add_(1.0)
            bias.add_(1.0)
            balancer.momentum.add_(1.0)
        state = full_state()
        sync()
        t = time.time()
        dcp.load(state, checkpoint_id=args.ckpt_dir)
        # dcp.load already restored everything in place. set_state_dict also casts the optimizer state to the
        # parameter dtype (Muon's bf16 momentum becomes FP32): scripts/train.py does not call it
        # (ludwig_lm/checkpoint.py). It stays here so this test is the one T0.2 v2 ran.
        set_state_dict(model, [muon, adamw], model_state_dict=state["model"], optim_state_dict=state["optim"])
        balancer.load_state_dict(state["balancer"])
        sync()
        load_s = time.time() - t
        exact = (torch.equal(before, local(probe)) and torch.equal(bias_before, bias)
                 and torch.equal(mom_before, balancer.momentum))
        ok = torch.tensor(float(exact), device=device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        size = sum(f.stat().st_size for f in Path(args.ckpt_dir).rglob("*") if f.is_file()) / 2**30 if is0 else 0
        result.update({"ckpt_save_s": round(save_s, 1), "ckpt_load_s": round(load_s, 1), "ckpt_GiB": round(size, 2),
                       "ckpt_restore_exact": bool(ok.item())})
        say(f"[ckpt] save {save_s:.1f}s, load {load_s:.1f}s, {size:.2f} GiB, restore exact={bool(ok.item())}")

    if is0:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(result) + "\n")
        print(f"[result] {args.tag or '-'} status={status} tokens/s={result.get('tokens_per_s')} "
              f"peak={result['peak_reserved_GiB']} GiB  25B≈{result.get('hours_for_25B')} h", flush=True)
        if err:
            print(err, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
