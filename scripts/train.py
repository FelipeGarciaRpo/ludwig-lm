#!/usr/bin/env python3
"""Ludwig 2's training loop: the AFMoE model on 2×T4 with FSDP2. T0.3/T2 use it now, the trunk later.

The model, sharding, precision and optimizers are the ones T0.2 v2 measured (scripts/bench_afmoe.py). The
defaults are the pieces it selected: one matmul per expert (``grouped_mm``, which is a loop on the T4), FP32
residual stream, FP16 gradient reduce-scatter, FSDP2 prefetch, and Muon's momentum in CPU RAM, which keeps
the reserved memory under the G0 limit of 13.5 GiB. What a real run adds:

  data         --shards: T1's pre-tokenized shards (ludwig_lm.shards.ShardReader; MeCo tags left out of the
               loss with --meta mask), with validation per domain; or --train/--val: parquet read by
               ludwig_lm.data.PackedReader (T0.3/T2). Both resume exactly
  schedule     WSD: linear warm-up, constant, optional 1−√ decay over the last --decay-steps steps;
               optional batch ramp (--ramp-tokens)
  evaluation   validation loss and bits per byte on fixed held-out windows, per domain with --shards
  alarms       the design's table (5.4): a loss spike above mean + 4σ is skipped, 3 within 200 steps stop
               the run; with --balance-stop, so does a broken expert balance (P30). Both stops save. Everything
               else is logged as an alarm row
  router       every step logs the router's health (P29): the smallest median gap between each token's k-th
               and (k+1)-th expert score, against SMEBU's step λ; each evaluation, per MoE layer
  checkpoints  ludwig_lm.checkpoint: the full state; each resume checks the state bit for bit and the first
               batch against its saved hash
  time limit   --max-hours: save and stop cleanly, before Kaggle's 12 h

Kaggle 2×T4: notebooks/t0_3_t2_lr_2xT4.ipynb
CPU logic:   python scripts/test_train_cpu.py
Writes one JSON row per event to --out: config, step, eval, alarm, save, resume, end.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
import traceback
from collections import deque
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ludwig_lm  # noqa: E402
from ludwig_lm import checkpoint as ckpt  # noqa: E402
from ludwig_lm.afmoe import (ExpertBalancer, LossWeights, LudwigAfmoe, count_params,  # noqa: E402
                             flops_per_token, load_config, shard_model)
from ludwig_lm.data import (PackedReader, batch_hash, eos_id_of, expand, load_tokenizer,  # noqa: E402
                            peek_batch, val_windows)
from ludwig_lm.muon import DynamicLossScaler, build_optimizers  # noqa: E402
from ludwig_lm.shards import ShardReader, TokenStream, loss_mask, val_sets  # noqa: E402

T4_PEAK_TFLOPS = 65.0
LN2 = math.log(2.0)


def parse(argv=None):
    ap = argparse.ArgumentParser(description="Ludwig 2 training loop (AFMoE, FSDP2)")
    a = ap.add_argument
    g = ap.add_argument_group("model and speed pieces (defaults: what T0.2 v2 selected)")
    g.add_argument("--config", default=str(ROOT / "configs" / "ludwig2_afmoe.json"))
    g.add_argument("--moe-impl", default="grouped_mm", choices=["grouped_mm", "padded_bmm", "eager"])
    g.add_argument("--residual", default="fp32", choices=["fp32", "fp16"])
    g.add_argument("--reduce", default="fp16", choices=["fp16", "fp32"])
    g.add_argument("--no-prefetch", action="store_true")
    g.add_argument("--momentum-device", default="cpu", choices=["cpu", "gpu"])
    g.add_argument("--ce-chunk", type=int, default=1024)
    g.add_argument("--seed", type=int, default=0, help="model initialisation")
    g = ap.add_argument_group("data")
    g.add_argument("--tokenizer", required=True, help="tokenizer.json")
    g.add_argument("--shards", default="", help="T1: the build's manifest.json (instead of --train/--val)")
    g.add_argument("--shard-roots", nargs="*", default=["/kaggle/input"],
                   help="where to look for shards not next to the manifest (one Kaggle dataset per batch)")
    g.add_argument("--meta", default="mask", choices=["mask", "keep", "strip"],
                   help="MeCo tags in the shards: out of the loss (mask), trained (keep) or removed (strip)")
    g.add_argument("--eval-domains", nargs="*", default=[], help="with --shards: these domains (default: all)")
    g.add_argument("--train", nargs="+", default=[], help="parquet files or globs")
    g.add_argument("--val", nargs="+", default=[], help="parquet files or globs, never trained on")
    g.add_argument("--text-column", default="text")
    g.add_argument("--rows-per-slice", type=int, default=1024)
    g.add_argument("--data-seed", type=int, default=0)
    g.add_argument("--eos-id", type=int, default=-1, help="-1: the tokenizer's <|endoftext|>")
    g.add_argument("--seq", type=int, default=2048)
    g.add_argument("--micro", type=int, default=8, help="sequences per GPU per micro-step")
    g.add_argument("--tokens-per-step", type=int, default=262144, help="0.25M tokens (design 5.1)")
    g.add_argument("--ramp-tokens", type=float, default=0, help="from this many tokens on, --tokens-per-step-after")
    g.add_argument("--tokens-per-step-after", type=int, default=524288)
    g = ap.add_argument_group("schedule and optimizers")
    g.add_argument("--steps", type=int, required=True, help="optimizer steps of the whole schedule")
    g.add_argument("--warmup", type=int, default=2000, help="design: 2,000 steps (Ludwig 1: 1,000 of 131K tokens)")
    g.add_argument("--decay-steps", type=int, default=0, help="1−√ decay over the last N steps (0: constant LR)")
    g.add_argument("--lr", type=float, default=1.5e-3,
                   help="Muon peak LR (Ludwig 1: 3x the base; its A/B of Oct 31 chose x3 over x1)")
    g.add_argument("--adamw-lr", type=float, default=5e-4, help="AdamW peak LR: Ludwig 1's base")
    g.add_argument("--lr-mult", type=float, default=1.0, help="multiplies both peaks (the T2 sweep)")
    g.add_argument("--weight-decay", type=float, default=0.1)
    g.add_argument("--muon-wd-lr", default="adamw", choices=["adamw", "muon"],
                   help="the LR Muon's weight decay follows. adamw (Ludwig 1): each step removes adamw_lr x wd "
                        "of every matrix; muon: lr x wd, 3x faster with these peaks")
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--balance", default="smebu", choices=["smebu", "sign"])
    g.add_argument("--lam", type=float, default=1e-2, help="SMEBU step (T0.2 v2: 0.01 balanced best)")
    g.add_argument("--lam-final", type=float, default=None,
                   help="SMEBU step from --lam-steps on (P30: 0.01 balances the start fast, then DeepSeek-V3's 0.001)")
    g.add_argument("--lam-steps", type=int, default=0, help="steps with --lam before --lam-final")
    g.add_argument("--gamma", type=float, default=1e-3, help="sign-update step")
    g.add_argument("--seq-aux", type=float, default=1e-4)
    g.add_argument("--router-z", type=float, default=0.0,
                   help="router z-loss. 0 (P30): made for softmax routers; with sigmoid scores it pushed the logits "
                        "down until SMEBU's bias decided every token, and the trunk's balance broke at step ~1250")
    g.add_argument("--output-z", type=float, default=1e-4)
    g = ap.add_argument_group("evaluation and alarms")
    g.add_argument("--eval-every", type=int, default=50, help="0: only at the start, before the decay and at the end")
    g.add_argument("--eval-windows", type=int, default=64, help="held-out windows per GPU (per domain with --shards)")
    g.add_argument("--spike-sigma", type=float, default=4.0)
    g.add_argument("--spike-window", type=int, default=200)
    g.add_argument("--spike-min-history", type=int, default=50)
    g.add_argument("--spike-stop", type=int, default=3, help="skipped spikes within --spike-window that stop the run")
    g.add_argument("--balance-stop", type=float, default=0.0,
                   help="stop and save when the busiest expert is above this many times the mean in --balance-frac "
                        "of the last --balance-window steps (0: never; P30: 3)")
    g.add_argument("--balance-window", type=int, default=50)
    g.add_argument("--balance-frac", type=float, default=0.8)
    g.add_argument("--balance-min-step", type=int, default=200, help="a run from scratch balances first")
    g = ap.add_argument_group("checkpoints and time")
    g.add_argument("--save-dir", default="", help="empty: never save")
    g.add_argument("--save-every-hours", type=float, default=0)
    g.add_argument("--save-at", type=int, nargs="*", default=[], help="also save before these steps")
    g.add_argument("--save-final", action="store_true")
    g.add_argument("--keep", type=int, default=2, help="periodic checkpoints kept (pre-decay and final stay)")
    g.add_argument("--resume", default="", help="a checkpoint directory, or 'latest' in --save-dir")
    g.add_argument("--max-hours", type=float, default=0, help="save and stop before this many hours (0: no limit)")
    g.add_argument("--pause-at-step", type=int, default=-1, help="save and stop before this step, like the time limit")
    g.add_argument("--eval-only", action="store_true",
                   help="with --resume: evaluate the checkpoint once (validation and the router per layer) and stop")
    a("--log-every", type=int, default=10, help="print every N steps (every step goes to --out)")
    a("--tag", default="run")
    a("--out", default=str(ROOT / "results" / "train.jsonl"))
    args = ap.parse_args(argv)
    if bool(args.shards) == bool(args.train or args.val):
        ap.error("give either --shards or both --train and --val")
    if not args.shards and not (args.train and args.val):
        ap.error("--train and --val go together")
    if args.eval_only and not args.resume:
        ap.error("--eval-only evaluates a checkpoint: give --resume")
    return args


def lr_factor(step: int, warmup: int, steps: int, decay: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    start = steps - decay
    if decay and step >= start:
        return 1.0 - math.sqrt((step - start) / decay)
    return 1.0


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001 - the code may come as a Kaggle dataset, without .git
        return None


_KAGGLE_OWNER = re.compile(r"(/kaggle/input/(?:notebooks|datasets|models)/)[^/\"\s]+")


def public(text: str) -> str:
    """Kaggle mounts another account's notebook output under its username (/kaggle/input/notebooks/<user>/...):
    logs and checkpoints keep the path without it."""
    return _KAGGLE_OWNER.sub(r"\1<usuario>", text)


class Log:
    def __init__(self, path, enabled: bool, tag: str):
        self.enabled, self.tag, self.f = enabled, tag, None
        if enabled:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self.f = open(path, "a")

    def row(self, kind, **kw):
        if self.f:
            self.f.write(public(json.dumps({"type": kind, "tag": self.tag, "time": round(time.time(), 1), **kw})) + "\n")
            self.f.flush()

    def say(self, msg):
        if self.enabled:
            print(public(msg), flush=True)


class Alarms:
    """The design's alarm table (5.4). Loss spikes are handled in the loop; the rest are logged once."""

    def __init__(self, log: Log):
        self.log = log
        self.count = 0
        self.inf = deque(maxlen=1000)
        self.imbalanced = self.weak = 0
        self.last = {}

    def _fire(self, step, kind, every, **kw):
        if step - self.last.get(kind, -10**9) < every:
            return
        self.last[kind] = step
        self.count += 1
        self.log.row("alarm", step=step, alarm=kind, **kw)
        self.log.say(f"[alarm] step {step}: {kind} {kw}")

    def check(self, step, found_inf, scale, bal, tps, peak_reserved):
        self.inf.append(bool(found_inf))
        if sum(self.inf) > 10:
            self._fire(step, "non_finite_gradients", 1000, last_1000=sum(self.inf))
        if scale < 1:
            self._fire(step, "loss_scale_below_1", 1000, scale=scale)
        self.imbalanced = self.imbalanced + 1 if bal["load_max_over_mean"] > 3 else 0
        if self.imbalanced >= 500:
            self._fire(step, "expert_load_max_over_mean_above_3_for_500_steps", 500)
        self.weak = self.weak + 1 if bal["load_min_over_mean"] < 0.1 else 0
        if self.weak >= 1000:
            self._fire(step, "expert_below_10pct_of_mean_for_1000_steps", 1000)
        if tps is not None and tps < 2500:
            self._fire(step, "tokens_per_s_below_2500", 200, tokens_per_s=round(tps))
        if peak_reserved > 13.5:
            self._fire(step, "peak_reserved_above_13.5_GiB_G0", 10**9, peak_reserved_GiB=peak_reserved)
        if peak_reserved > 14.3:
            self._fire(step, "peak_reserved_above_14.3_GiB", 10**9, peak_reserved_GiB=peak_reserved)


def main(argv=None):
    args = parse(argv)
    t_proc = time.time()
    use_cuda = torch.cuda.is_available()
    # cpu:gloo next to cuda:nccl so Muon's momentum can live on a CPU mesh
    dist.init_process_group("cpu:gloo,cuda:nccl" if use_cuda else "gloo", timeout=timedelta(minutes=20))
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0))) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(device)
    log = Log(args.out, rank == 0, args.tag)
    try:
        code = Trainer(args, log, rank, world, device, use_cuda, t_proc).run()
    except Exception:
        err = traceback.format_exc()
        print(public(f"[rank {rank}] {err}"), flush=True)
        log.row("end", status="error", error=err[-3000:])
        os._exit(1)        # without destroy_process_group: the other rank may be inside a collective; torchrun ends it
    dist.destroy_process_group()
    sys.exit(code)


class Trainer:
    def __init__(self, args, log, rank, world, device, use_cuda, t_proc):
        self.args, self.log, self.rank, self.world = args, log, rank, world
        self.device, self.use_cuda, self.t_proc = device, use_cuda, t_proc

    # ------------------------------------------------------------------ setup
    def build(self):
        a, device = self.args, self.device
        from torch.distributed.device_mesh import init_device_mesh
        try:
            from torch.distributed.fsdp import MixedPrecisionPolicy
        except ImportError:  # torch < 2.6
            from torch.distributed._composable.fsdp import MixedPrecisionPolicy

        cfg = self.cfg = load_config(a.config)
        cfg._experts_implementation = a.moe_impl
        self.tok = load_tokenizer(a.tokenizer)
        if self.tok.get_vocab_size() > cfg.vocab_size:
            raise ValueError(f"tokenizer has {self.tok.get_vocab_size()} tokens, the model {cfg.vocab_size}")
        self.eos = a.eos_id if a.eos_id >= 0 else eos_id_of(self.tok, cfg.eos_token_id)

        mesh = init_device_mesh(device.type, (self.world,))
        cpu_mesh = init_device_mesh("cpu", (self.world,)) if (self.use_cuda and a.momentum_device == "cpu") else None
        self.fp32_residual = a.residual == "fp32"
        half = torch.float16 if self.use_cuda else torch.float32
        reduce = torch.float16 if (self.use_cuda and a.reduce == "fp16") else torch.float32
        mp = MixedPrecisionPolicy(param_dtype=half, reduce_dtype=reduce, cast_forward_inputs=not self.fp32_residual)
        torch.manual_seed(a.seed)
        with torch.device(device):          # built straight on each GPU, then sharded
            self.model = LudwigAfmoe(cfg, fp32_residual=self.fp32_residual)
        self.model.lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        shard_model(self.model, mesh, mp, prefetch=not a.no_prefetch)
        if self.use_cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        self.counts = count_params(self.model)
        # lr_mult and the schedule scale both LRs alike, so this keeps Muon's decay at adamw_lr(t) x wd
        self.muon_wd = a.weight_decay * a.adamw_lr / a.lr if a.muon_wd_lr == "adamw" else a.weight_decay
        self.muon, self.adamw = build_optimizers(
            self.model, lr=a.lr, weight_decay=a.weight_decay, n_heads=cfg.num_attention_heads, head_dim=cfg.head_dim,
            fused_qgate=False, adamw_lr=a.adamw_lr, momentum_device="cpu" if a.momentum_device == "cpu" else None,
            cpu_mesh=cpu_mesh, muon_weight_decay=self.muon_wd)
        self.opts = [self.muon, self.adamw]
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.scaler = DynamicLossScaler(init_scale=2.0 ** 15 if self.use_cuda else 1.0)
        self.balancer = ExpertBalancer(self.model, rule=a.balance, lam=a.lam, gamma=a.gamma)
        self.weights = LossWeights(output_z=a.output_z, seq_aux=a.seq_aux, router_z=a.router_z)
        self.amp_dtype = torch.float16 if self.use_cuda else torch.bfloat16   # bf16 only exercises the CPU path

        if a.shards:
            self.stream = TokenStream(a.shards, a.shard_roots)
            man = self.stream.manifest
            sha1 = hashlib.sha1(Path(a.tokenizer).read_bytes()).hexdigest()
            if sha1 != man["tokenizer_sha1"]:
                raise ValueError(f"--tokenizer is {sha1[:12]}; the shards were built with {man['tokenizer_sha1'][:12]}")
            if self.eos != 0:
                raise ValueError(f"the shards end each document with id 0; this tokenizer's EOS is {self.eos}")
            self.mask_meta = a.meta == "mask" and man["meta"]["enabled"]
            domains = man["eval_domains"]
            if a.eval_domains:
                domains = {d: domains[d] for d in a.eval_domains}
            self.val_sets = val_sets(a.shards, self.tok, a.seq, a.eval_windows, self.rank, self.world,
                                     a.shard_roots, domains)
            self.train_files, self.val_files = [], []
        else:
            self.stream, self.mask_meta = None, False
            self.train_files, self.val_files = expand(a.train), expand(a.val)
            if set(self.train_files) & set(self.val_files):
                raise ValueError("a validation file is also a training file")
            self.val_sets = {"val": val_windows(self.val_files, self.tok, a.seq, a.eval_windows, self.rank,
                                                self.world, self.eos, a.rows_per_slice, a.text_column)}

    def make_reader(self, state):
        a = self.args
        if self.stream is not None:
            return ShardReader(self.stream, a.seq, a.micro, self.rank, self.world, state=state, meta=a.meta)
        return PackedReader(**self.reader_args(), state=state)

    def peek(self, state):
        """The batch a reader restarted from ``state`` gives first."""
        if self.stream is not None:
            return self.reader.peek(state)
        return peek_batch(**self.reader_args(), state=state)

    def reader_args(self):
        a = self.args
        return dict(files=self.train_files, tok=self.tok, seq=a.seq, micro=a.micro, rank=self.rank, world=self.world,
                    seed=a.data_seed, eos_id=self.eos, rows_per_slice=a.rows_per_slice, column=a.text_column)

    def tokens_per_step(self, tokens_seen):
        a = self.args
        tps = a.tokens_per_step_after if (a.ramp_tokens and tokens_seen >= a.ramp_tokens) else a.tokens_per_step
        per_micro = self.world * a.micro * a.seq
        if tps % per_micro:
            raise ValueError(f"--tokens-per-step {tps} is not a multiple of GPUs × micro × seq = {per_micro}")
        return tps, tps // per_micro

    # ------------------------------------------------------------------ one step
    def micro_step(self, ids, ga):
        mask = loss_mask(ids) if self.mask_meta else None
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.fp32_residual):
            loss, mets = self.model(ids, self.weights, ce_chunk=self.args.ce_chunk, loss_mask=mask)
        (loss * (self.scaler.scale / ga)).backward()
        self.balancer.accumulate(mets["chosen"])
        self.health += torch.stack([mets["router_margin"], mets["bias_decides"]])
        return torch.stack([mets["ce"], mets["z_out"], mets["seq_aux"], mets["z_router"], mets["router_entropy"]])

    def lam_at(self, step):
        a = self.args
        return a.lam_final if (a.lam_final is not None and step >= a.lam_steps) else a.lam

    def router_health(self):
        """The step's router health: the MoE layer with the smallest score gap, and the one where the bias
        decides most often (P29). λ over that gap is the balancing loop's gain: above ~5, unstable."""
        mg, dc = self.health[0], self.health[1]
        if not mg.numel():                                # no MoE layer
            return {}
        im, idc = int(mg.argmin()), int(dc.argmax())
        lam = self.balancer.lam
        return {"lam": lam, "margin_min": round(float(mg[im]), 6), "margin_min_layer": im,
                "lam_over_margin": round(lam / max(float(mg[im]), 1e-12), 2),
                "bias_decides_max": round(float(dc[idc]), 4), "bias_decides_max_layer": idc}

    def zero_grad(self):
        for opt in self.opts:
            opt.zero_grad(set_to_none=True)

    def opt_step(self, mult):
        a = self.args
        found_inf = self.scaler.unscale_and_check(self.params)
        gnorm = None
        if not found_inf:
            gn = torch.nn.utils.clip_grad_norm_(self.params, a.clip)
            gnorm = float(gn.full_tensor() if hasattr(gn, "full_tensor") else gn)
            for opt, base in ((self.muon, a.lr), (self.adamw, a.adamw_lr)):
                for group in opt.param_groups:
                    group["lr"] = base * mult
            self.muon.step()
            self.adamw.step()
        self.scaler.update(found_inf)
        self.zero_grad()
        return gnorm, found_inf, self.balancer.step()

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, step, tokens, **extra):
        a = self.args
        t = time.time()
        self.model.eval()
        names = list(self.val_sets)
        per = torch.zeros(len(names), 3, dtype=torch.float64, device=self.device)      # nats, tokens, bytes
        ent = torch.zeros((), dtype=torch.float64, device=self.device)
        n_moe, n_exp = len(self.balancer.blocks), self.cfg.num_experts
        health = torch.zeros(2, n_moe, dtype=torch.float64, device=self.device)        # margin, bias decides
        load = torch.zeros(n_moe, n_exp, dtype=torch.float64, device=self.device)      # tokens per expert
        n_mb = 0
        for j, name in enumerate(names):
            val_ids, val_bytes = self.val_sets[name]
            for i in range(0, val_ids.shape[0], a.micro):
                ids = val_ids[i:i + a.micro].to(self.device)
                with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.fp32_residual):
                    _, mets = self.model(ids, self.weights, ce_chunk=a.ce_chunk)
                n = ids.shape[0] * (ids.shape[1] - 1)
                per[j, 0] += mets["ce"].double() * n
                per[j, 1] += n
                ent += mets["router_entropy"].double()
                health += torch.stack([mets["router_margin"], mets["bias_decides"]]).double()
                for li, c in enumerate(mets["chosen"]):
                    load[li] += torch.bincount(c.reshape(-1), minlength=n_exp).double()
                n_mb += 1
            per[j, 2] += val_bytes
        self.model.train()
        if hasattr(self.model, "reshard"):
            self.model.reshard()                       # the root stays unsharded after a forward without backward
        tot = torch.cat([per.reshape(-1), (ent / max(1, n_mb)).reshape(1), (health / max(1, n_mb)).reshape(-1),
                         load.reshape(-1)])
        dist.all_reduce(tot)
        k = per.numel()
        per, ent_all = tot[:k].reshape(len(names), 3), float(tot[k])
        health = tot[k + 1:k + 1 + 2 * n_moe].reshape(2, n_moe) / self.world
        load = tot[k + 1 + 2 * n_moe:].reshape(n_moe, n_exp)
        load = load / load.mean(1, keepdim=True).clamp_min(1.0)
        bias_range = [float(b.expert_bias.max() - b.expert_bias.min()) for b in self.balancer.blocks]
        lam = self.balancer.lam
        nats, n_tok, n_bytes = (float(x) for x in per.sum(0))
        ce = nats / n_tok
        row = {"step": step, "tokens": tokens, "val_ce": round(ce, 5), "val_bpb": round(nats / LN2 / n_bytes, 5),
               "val_tokens": int(n_tok), "val_bytes": int(n_bytes), "router_entropy": round(ent_all / self.world, 4),
               "qk_gain_max": round(self.qk_gain_max(), 3), "eval_s": round(time.time() - t, 1),
               **self.balancer.window_stats(),
               "router_margin": [round(float(x), 6) for x in health[0]],
               "bias_decides": [round(float(x), 4) for x in health[1]],
               "bias_range": [round(x, 5) for x in bias_range],
               "val_load_max_over_mean": [round(float(x), 3) for x in load.max(1).values], "lam": lam, **extra}
        if names != ["val"]:
            row["val_bpb_by_domain"] = {n: round(float(per[j, 0] / LN2 / per[j, 2]), 5) for j, n in enumerate(names)}
        self.log.row("eval", **row)
        by = "".join(f"  {k} {v:.4f}" for k, v in row.get("val_bpb_by_domain", {}).items())
        self.log.say(f"[eval] step {step}: val bpb {row['val_bpb']:.4f}{by}  ce {ce:.4f}  router entropy "
                     f"{row['router_entropy']:.3f}  qk gain {row['qk_gain_max']:.2f}  dead experts "
                     f"{row['dead_experts_window']}  ({row['eval_s']:.0f} s)")
        if n_moe:
            im = min(range(n_moe), key=lambda i: row["router_margin"][i])
            self.log.say(f"[eval] router: smallest score gap {row['router_margin'][im]:.2e} in MoE layer {im} "
                         f"(λ/gap {lam / max(row['router_margin'][im], 1e-12):.1f}, bias range {bias_range[im]:.3f}); "
                         f"the bias decides for {max(row['bias_decides']):.0%} of tokens at most; busiest expert "
                         f"{max(row['val_load_max_over_mean']):.2f}× the mean")
        return row

    def peak_memory(self):
        """(allocated, reserved) GiB: the largest since the start on any GPU (G0 counts reserved)."""
        if not self.use_cuda:
            return (0.0, 0.0)
        pk = torch.tensor([torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30],
                          device=self.device)
        dist.all_reduce(pk, op=dist.ReduceOp.MAX)
        return (round(float(pk[0]), 2), round(float(pk[1]), 2))

    def qk_gain_max(self):
        """Largest |q_norm.w| × |k_norm.w| over the layers: with QK-norm it bounds the attention logits."""
        best = 0.0
        for layer in self.model.lm.model.layers:
            q, k = layer.self_attn.q_norm.weight, layer.self_attn.k_norm.weight
            q = q.full_tensor() if hasattr(q, "full_tensor") else q
            k = k.full_tensor() if hasattr(k, "full_tensor") else k
            best = max(best, float(q.abs().max() * k.abs().max()))
        return best

    # ------------------------------------------------------------------ checkpoints
    def save(self, step, tokens, reason, hist):
        a = self.args
        t = time.time()
        path = Path(a.save_dir) / f"step_{step:06d}"
        try:
            rstate, nxt = self.reader.state_dict(), []
            if not ckpt.all_ok(ckpt.local_step(lambda: nxt.append(self.peek(rstate)))):
                raise RuntimeError("could not read the batch that follows the checkpoint on every rank")
            meta = {"step": step, "tokens": tokens, "reason": reason, "tag": a.tag, "world": self.world,
                    "micro": a.micro, "seq": a.seq, "tokens_per_step": a.tokens_per_step, "lr_mult": a.lr_mult,
                    "loss_history": list(hist), "train_hours": self.hours_before + (time.time() - self.t_loop) / 3600,
                    "args": json.loads(public(json.dumps(vars(a)))), "code": self.code, "saved_at": time.time()}
            ckpt.save(path, self.model, self.opts, self.scaler, self.balancer, self.rank,
                      {"reader": rstate, "next_batch_sha": batch_hash(nxt[0])}, meta)
            if self.rank == 0:
                ckpt.rotate(a.save_dir, a.keep, self.protected)
            dt = time.time() - t
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 2**30 if self.rank == 0 else 0
            self.log.row("save", step=step, tokens=tokens, reason=reason, path=str(path), seconds=round(dt, 1),
                         GiB=round(size, 2))
            self.log.say(f"[save] step {step} ({reason}) -> {path}  {size:.2f} GiB in {dt:.0f} s")
            self.last_save = time.time()
            self.save_s = max(self.save_s, dt)
            return path
        except Exception:  # a failed save must not end a run that is otherwise fine
            err = traceback.format_exc()
            print(public(f"[rank {self.rank}] checkpoint failed:\n{err}"), flush=True)
            self.log.row("alarm", step=step, alarm="checkpoint_failed", error=err[-2000:])
            return None

    def resume(self):
        a = self.args
        path = ckpt.resolve(a.resume, a.save_dir)
        t = time.time()
        meta, extra, exact = ckpt.load(path, self.model, self.opts, self.scaler, self.balancer, self.rank)
        if meta["world"] != self.world:
            raise ValueError(f"checkpoint written by {meta['world']} GPUs, this run has {self.world}")
        ok = torch.tensor(float(exact), device=self.device)
        dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        self.expected_sha = extra["next_batch_sha"] if meta.get("micro") == a.micro else None
        # both sides without the Kaggle username: the same dataset attached from another account is not a change
        # (and a checkpoint written before the paths were redacted compares the same way)
        then, now = (json.loads(public(json.dumps(d))) for d in (meta["args"], vars(a)))
        info = {"path": str(path), "step": meta["step"], "tokens": meta["tokens"], "from_tag": meta.get("tag"),
                "reason": meta.get("reason"), "state_exact": bool(ok.item()), "seconds": round(time.time() - t, 1),
                "changed_args": {k: [then[k], v] for k, v in now.items() if k in then and then[k] != v
                                 and k not in ("resume", "out", "tag", "max_hours", "save_dir")}}
        self.log.row("resume", **info)
        self.log.say(f"[resume] {path} at step {meta['step']}: state bit-exact = {info['state_exact']} "
                     f"({info['seconds']:.0f} s)")
        return meta, extra["reader"]

    # ------------------------------------------------------------------ the loop
    def run(self) -> int:
        a = self.args
        self.build()
        self.code = {"version": ludwig_lm.__version__, "commit": git_commit()}
        self.protected = set()
        if a.decay_steps:
            self.protected.add(f"step_{a.steps - a.decay_steps:06d}")
        start, tokens, hist, self.hours_before, reader_state = 0, 0, deque(maxlen=a.spike_window), 0.0, None
        self.expected_sha, resumed = None, False
        if a.resume:
            meta, reader_state = self.resume()
            start, tokens, resumed = meta["step"], meta["tokens"], True
            hist.extend(meta.get("loss_history", []))
            self.hours_before = meta.get("train_hours", 0.0)
        self.reader = self.make_reader(reader_state)
        self.balancer.lam = self.lam_at(start)
        tps_now, ga = self.tokens_per_step(tokens)
        flops_tok = flops_per_token(self.cfg, self.counts["active"], a.seq)
        gpu = torch.cuda.get_device_name(0) if self.use_cuda else "cpu"
        if self.stream is not None:
            man = self.stream.manifest
            data = {"build_id": man["build_id"], "recipe": man["recipe"], "shards": len(man["shards"]),
                    "tokens_available": self.stream.total, "start_token": self.reader.offset, "meta": a.meta,
                    "eval_domains": {k: v for k, v in man["eval_domains"].items() if k in self.val_sets}}
        else:
            data = {"train_files": [Path(f).name for f in self.train_files],
                    "val_files": [Path(f).name for f in self.val_files], "train_slices": self.reader.src.n_slices}
        self.log.row("config", args=vars(a), world=self.world, gpu=gpu, torch=torch.__version__, code=self.code,
                     params_total=self.counts["total"], params_active=self.counts["active"], vocab=self.tok.get_vocab_size(),
                     eos_id=self.eos, grad_accum=ga, muon_weight_decay=self.muon_wd, **data,
                     val_tokens_per_rank={k: int(v[0].numel()) for k, v in self.val_sets.items()},
                     start_step=start, resumed=resumed)
        self.log.say(f"[model] AFMoE {self.counts['total']/1e9:.3f}B total, {self.counts['active']/1e9:.3f}B active | "
                     f"moe={a.moe_impl} residual={a.residual} reduce={a.reduce} momentum={a.momentum_device} | "
                     f"{self.world} × {gpu}")
        if self.stream is not None:
            what = (f"build {data['build_id']}: {data['shards']} shards, {self.stream.total / 1e9:.2f}B tokens "
                    f"(from token {self.reader.offset:,}), meta={a.meta}, val {', '.join(self.val_sets)}")
        else:
            what = (f"{len(self.train_files)} train files ({self.reader.src.n_slices} slices), "
                    f"{len(self.val_files)} val files")
        self.log.say(f"[data] {what} | vocab {self.tok.get_vocab_size()} | eos {self.eos} | "
                     f"{tps_now} tokens per step = {self.world} GPUs × {a.micro} × {a.seq} × {ga} micro-steps")

        self.t_loop = self.last_save = time.time()
        if a.eval_only:
            row = self.evaluate(start, tokens, after_resume=True, eval_only=True)
            self.reader.close()
            self.log.row("end", status="evaluated", steps_done=start, steps_this_session=0, tokens=tokens,
                         val_bpb_last=row["val_bpb"])
            self.log.say(f"[end] evaluated: step {start}, val bpb {row['val_bpb']}")
            return 0
        self.save_s = 60.0 if self.use_cuda else 1.0      # updated by the first save
        alarms = Alarms(self.log)
        spikes, spikes_total = deque(), 0
        broken = deque(maxlen=max(1, a.balance_window))   # steps with the busiest expert above --balance-stop
        steep = deque(maxlen=50)                          # steps with λ above 5× the smallest score gap
        times = deque(maxlen=20)                           # recent step times
        hour = deque()                                     # (time, tokens) over the last hour
        decay_start = a.steps - a.decay_steps if a.decay_steps else None
        last_eval, status = None, "done"
        evals = {}
        step_tps = []
        peak = (0.0, 0.0)
        step = start
        for step in range(start, a.steps):
            first = step == start
            # ---- time limit, checkpoints and evaluation happen before step `step` runs. Nothing is saved
            # before this process has taken a step: the state is the initialisation or the checkpoint it loaded
            # (and a save before any optimizer step would store the zero step that creates the optimizer state).
            can_save = bool(a.save_dir) and not first
            est = (np.mean(times) if times else 0.0) + self.save_s + 30.0
            if (a.max_hours and (time.time() - self.t_proc + est) / 3600 > a.max_hours) or step == a.pause_at_step:
                if can_save:
                    self.save(step, tokens, "time" if step != a.pause_at_step else "pause", hist)
                status = "paused"
                break
            if self.stream is not None and self.reader.tokens_left() < self.tokens_per_step(tokens)[0]:
                if can_save:                              # the next shards are not attached: stop in order
                    self.save(step, tokens, "out_of_data", hist)
                self.log.row("alarm", step=step, alarm="out_of_data", tokens_left=self.reader.tokens_left())
                self.log.say(f"[alarm] step {step}: the attached shards end here; attach the next ones")
                status = "out_of_data"
                break
            if can_save:
                if step == decay_start:
                    self.save(step, tokens, "pre_decay", hist)
                elif step in a.save_at:
                    self.save(step, tokens, "requested", hist)
                elif a.save_every_hours and time.time() - self.last_save > a.save_every_hours * 3600:
                    self.save(step, tokens, "periodic", hist)
            if (first or step == decay_start or (a.eval_every and step % a.eval_every == 0)) and last_eval != step:
                evals[step] = self.evaluate(step, tokens, after_resume=first and resumed,
                                            pre_decay=step == decay_start)
                last_eval = step

            # ---- the step
            tps_now, ga = self.tokens_per_step(tokens)
            mult = lr_factor(step, a.warmup, a.steps, a.decay_steps) * a.lr_mult
            t0 = time.time()
            acc = torch.zeros(5, device=self.device)
            self.health = torch.zeros(2, len(self.balancer.blocks), device=self.device)
            self.balancer.lam = self.lam_at(step)
            sha = None
            for m in range(ga):
                ids = self.reader.next()
                if m == 0:
                    sha = batch_hash(ids)
                    if first and self.expected_sha is not None:
                        same = torch.tensor(float(sha == self.expected_sha), device=self.device)
                        dist.all_reduce(same, op=dist.ReduceOp.MIN)
                        self.log.row("resume_check", step=step, batch_exact=bool(same.item()))
                        self.log.say(f"[resume] first batch identical to the saved hash = {bool(same.item())}")
                acc += self.micro_step(ids.to(self.device, non_blocking=True), ga)
            acc /= ga
            dist.all_reduce(acc)
            acc /= self.world
            self.health /= ga
            dist.all_reduce(self.health)
            self.health /= self.world
            ce = float(acc[0])
            spike = bool(math.isfinite(ce) and len(hist) >= a.spike_min_history
                         and ce > float(np.mean(hist)) + a.spike_sigma * max(float(np.std(hist)), 0.01))
            if spike:
                self.zero_grad()
                self.balancer.skip()
                gnorm, found_inf, bal = None, False, {"load_max_over_mean": 0.0, "load_min_over_mean": 1.0,
                                                      "dead_experts": 0, "bias_abs_max": 0.0}
                spikes.append(step)
                spikes_total += 1
                while spikes and spikes[0] <= step - a.spike_window:
                    spikes.popleft()
            else:
                gnorm, found_inf, bal = self.opt_step(mult)
                if math.isfinite(ce):
                    hist.append(ce)
            tokens += tps_now
            dt = time.time() - t0
            now = time.time()
            if not first:
                times.append(dt)
                step_tps.append(tps_now / dt)
            hour.append((now, tokens))
            while hour and hour[0][0] < now - 3600:
                hour.popleft()
            tps_hour = ((hour[-1][1] - hour[0][1]) / (hour[-1][0] - hour[0][0])
                        if len(hour) > 2 and hour[-1][0] > hour[0][0] else None)
            peak = self.peak_memory()
            row = {"step": step, "tokens": tokens, "lr_mult": round(mult, 6), "lr_muon": a.lr * mult,
                   "ce": round(ce, 5), "z_out": round(float(acc[1]), 3), "seq_aux": round(float(acc[2]), 5),
                   "z_router": round(float(acc[3]), 3), "router_entropy": round(float(acc[4]), 4),
                   "grad_norm": None if gnorm is None else round(gnorm, 4), "scale": self.scaler.scale,
                   "skipped_inf": found_inf, "skipped_spike": spike, **{k: (round(v, 4) if isinstance(v, float) else v)
                                                                        for k, v in bal.items()},
                   **self.router_health(),
                   "step_s": round(dt, 2), "tokens_per_s": round(tps_now / dt), "grad_accum": ga,
                   "peak_alloc_GiB": peak[0], "peak_reserved_GiB": peak[1], "batch_sha": sha}
            self.log.row("step", **row)
            if spike:
                self.log.row("alarm", step=step, alarm="loss_spike_skipped", ce=ce, mean=float(np.mean(hist)),
                             std=float(np.std(hist)), in_window=len(spikes))
                self.log.say(f"[alarm] step {step}: loss spike {ce:.3f} skipped ({len(spikes)} in the last "
                             f"{a.spike_window} steps)")
            if step % a.log_every == 0 or first or step == a.steps - 1:
                self.log.say(f"[train] step {step:5d}  ce {ce:.4f}  lr×{mult:.3f}  gnorm "
                             f"{'-' if gnorm is None else f'{gnorm:.3f}'}  scale {self.scaler.scale:.0f}  load max/mean "
                             f"{bal['load_max_over_mean']:.2f}  λ/gap {row.get('lam_over_margin', 0):.1f}  "
                             f"{tps_now / dt:,.0f} tok/s  peak {peak[1]:.2f} GiB")
            alarms.check(step, found_inf, self.scaler.scale, bal, tps_hour if step >= start + 20 else None, peak[1])
            steep.append(row.get("lam_over_margin", 0) > 5)
            if len(steep) == steep.maxlen and all(steep):
                alarms._fire(step, "router_gap_below_lambda_over_5_for_50_steps", 500,
                             margin_min=row["margin_min"], layer=row["margin_min_layer"], lam=row["lam"])
            if len(spikes) >= a.spike_stop:            # an orderly stop: the status is in the end row
                status = "stopped_spikes"
                step += 1
                break
            if a.balance_stop and not spike and step >= a.balance_min_step:
                broken.append(bal["load_max_over_mean"] > a.balance_stop)
                if len(broken) == broken.maxlen and sum(broken) >= a.balance_frac * broken.maxlen:
                    self.log.row("alarm", step=step, alarm="expert_balance_broke", above=sum(broken),
                                 window=broken.maxlen, threshold=a.balance_stop, margin_min=row.get("margin_min"),
                                 layer=row.get("margin_min_layer"))
                    self.log.say(f"[alarm] step {step}: the busiest expert was above {a.balance_stop}× the mean in "
                                 f"{sum(broken)} of the last {broken.maxlen} steps: stopping")
                    status = "stopped_balance"
                    step += 1
                    break
        else:
            step = a.steps
        if status in ("stopped_spikes", "stopped_balance") and a.save_dir and step > start:
            self.save(step, tokens, status, hist)        # the state where it stopped, to study or resume from
        if status == "done":
            if last_eval != step:
                evals[step] = self.evaluate(step, tokens, final=True)
            if a.save_dir and a.save_final:
                self.save(step, tokens, "final", hist)
        self.reader.close()
        peak = self.peak_memory()                         # evaluation and saving included

        hours = (time.time() - self.t_loop) / 3600
        tps_mean = float(np.mean(step_tps)) if step_tps else None
        tps_hour = ((hour[-1][1] - hour[0][1]) / (hour[-1][0] - hour[0][0])
                    if len(hour) > 2 and hour[-1][0] > hour[0][0] else tps_mean)
        pre = evals.get(decay_start) if decay_start is not None else None
        fin = evals.get(a.steps)
        summary = {"status": status, "steps_done": step, "steps_this_session": step - start, "tokens": tokens,
                   "hours_this_session": round(hours, 3), "train_hours_total": round(self.hours_before + hours, 3),
                   "tokens_per_s_mean": None if tps_mean is None else round(tps_mean),
                   "tokens_per_s_last_hour": None if tps_hour is None else round(tps_hour),
                   "mfu": None if tps_hour is None else round(flops_tok * tps_hour / (self.world * T4_PEAK_TFLOPS * 1e12), 4),
                   "hours_for_25B": None if not tps_hour else round(25e9 / tps_hour / 3600),
                   "peak_alloc_GiB": peak[0], "peak_reserved_GiB": peak[1], "loss_scale_skips": self.scaler.skipped,
                   "loss_scale_last": self.scaler.scale, "spikes_skipped": spikes_total, "spikes_in_window": len(spikes),
                   "alarms": alarms.count,
                   "val_bpb_pre_decay": pre and pre["val_bpb"], "val_bpb_final": fin and fin["val_bpb"],
                   "val_bpb_last": evals[max(evals)]["val_bpb"] if evals else None}
        self.log.row("end", **summary)
        self.log.say(f"[end] {status}: step {step}, {tokens/1e6:.1f}M tokens, {hours:.2f} h, "
                     f"{summary['tokens_per_s_last_hour']} tok/s (last hour), peak {peak[1]} GiB reserved, "
                     f"{self.scaler.skipped} FP16 skips, val bpb {summary['val_bpb_last']}")
        return 0


if __name__ == "__main__":
    main()
