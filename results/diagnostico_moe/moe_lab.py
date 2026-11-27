"""CPU lab for the expert balance (P35): a small AFMoE trained with the repo's own router, losses, Muon/AdamW
split and ExpertBalancer (ludwig_lm/afmoe.py, ludwig_lm/muon.py), on a PC without a GPU. It is a diagnosis tool,
not part of the training code: 7M parameters (width 128, 1 dense + 5 MoE layers, 44 experts, top-4, 4 shared),
4,096 tokens per step, local text (prep_data.py). ~1 step per second on 6 CPU cores.

It logs what the trunk's logs cannot show: per MoE layer and per expert, how much of the router's input is a
direction common to all tokens (rho), each expert's offset through it (w_i . x_mean), the token-specific part of
the logits, how fast both move per optimizer step, the bias that would balance the batch exactly (b*) and how
fast THAT moves, which is the speed the balancer has to follow.

    python moe_lab.py --out runs/base.jsonl [--lam 0.01 --lam-final 0.001 --lam-steps 300] [--offsets nodiff] ...

--router-init 0.0566 gives the router the trunk's logit spread at its first step (0.02 * sqrt(1024) = 0.64 with
width 1024; this model's width is 128). Every run in runs/ used it, with --steps 500 or more and --warmup 250.
--balance quantile and --offsets nodiff are this script's prototypes of the fix; --impl repo uses the repo's own
(the quantile rule, equal offsets and the bias reset of ExpertBalancer). The runs are not bit-reproducible
(threads): two runs of one recipe differ by ~0.005 in the loss.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def args_():
    ap = argparse.ArgumentParser()
    a = ap.add_argument
    a("--repo", default=str(HERE.parents[1]), help="the checkout whose ludwig_lm/ is used")
    a("--data", default=str(HERE / "data"))
    a("--out", required=True)
    a("--V", type=int, default=4096)
    a("--d", type=int, default=128)
    a("--layers", type=int, default=6)
    a("--dense", type=int, default=1)
    a("--experts", type=int, default=44)
    a("--topk", type=int, default=4)
    a("--shared", type=int, default=4)
    a("--route-scale", type=float, default=1.0)
    a("--seq", type=int, default=128)
    a("--batch", type=int, default=32)
    a("--steps", type=int, default=1000)
    a("--warmup", type=int, default=300)
    a("--lr", type=float, default=1.5e-3)
    a("--adamw-lr", type=float, default=5e-4)
    a("--router-lr-mult", type=float, default=1.0)
    a("--router-init", type=float, default=0.02)
    a("--lam", type=float, default=0.01)
    a("--lam-final", type=float, default=None)
    a("--lam-steps", type=int, default=0)
    a("--balance", default="smebu", choices=["smebu", "sign", "quantile", "none"])
    a("--q-damp", type=float, default=0.3)
    a("--offsets-rate", type=float, default=0.1,
      help="--impl repo: the part of the offsets' differences taken per step (--router-offsets-rate)")
    a("--impl", default="lab", choices=["lab", "repo"],
      help="repo: the quantile rule, the equal offsets and the bias reset of the repo's own ExpertBalancer (P35; "
           "--repo must be a checkout that has them), instead of this script's prototypes")
    a("--q-tokens", type=int, default=4096)
    a("--seq-aux", type=float, default=1e-3)
    a("--router-z", type=float, default=0.0)
    a("--router-floor", type=float, default=0.0)
    a("--router-floor-lse", type=float, default=1.5)
    a("--offsets", default="free", choices=["free", "nodiff", "none"],
      help="free: the recipe. nodiff: after each step, equalise the experts' offsets along the mean input "
           "direction (the level may still move). none: every router row orthogonal to that direction")
    a("--route-scale-final", type=float, default=None, help="route scale from --route-scale-from on")
    a("--route-scale-from", type=int, default=0)
    a("--route-scale-ramp", type=int, default=0, help="steps to go linearly from --route-scale to the final one")
    a("--route-scale-keep", action="store_true",
      help="at --route-scale-from, divide every expert's down_proj by the factor: the model computes the same "
           "function at the switch, and the routed experts then grow by training")
    a("--offsets-from", type=int, default=0, help="step from which --offsets applies (before it: free)")
    a("--balance-from", type=int, default=0, help="step from which --balance applies (before it: smebu with --lam)")
    a("--bias-reset", action="store_true", help="zero the expert bias when --offsets starts to apply")
    a("--probe-every", type=int, default=25)
    a("--val-every", type=int, default=100)
    a("--seed", type=int, default=0)
    a("--threads", type=int, default=3)
    return ap.parse_args()


@torch.no_grad()
def balanced_bias(scores, bias, k, iters=40, damp=0.5):
    """The bias that gives every expert the same share of these tokens, by damped quantile matching.

    With the others fixed, expert i takes token t iff b_i > need[t, i] = (what it has to beat) - score[t, i]; the
    load it wants is the k/E quantile of need[:, i]. One call with iters=1 is one step of the 'quantile' balancer."""
    n, e = scores.shape
    b = bias.clone().float()
    for _ in range(iters):
        tot = scores + b
        top = torch.topk(tot, k + 1, dim=1).values                 # (n, k+1), descending
        inside = tot >= top[:, k - 1:k]
        beat = torch.where(inside, top[:, k:k + 1], top[:, k - 1:k])  # the best total among the others at the edge
        need = beat - scores
        q = torch.quantile(need, k / e, dim=0)
        b = b + damp * (q - b)
        b = b - b.mean()
    return b


def main():
    a = args_()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, a.repo)
    from transformers.models.afmoe import modeling_afmoe as am

    from ludwig_lm import afmoe
    from ludwig_lm.afmoe import ExpertBalancer, LossWeights, LudwigAfmoe, moe_blocks
    from ludwig_lm.muon import Muon, split_params

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfgd = {"vocab_size": a.V, "hidden_size": a.d, "num_hidden_layers": a.layers, "num_dense_layers": a.dense,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": a.d // 4, "intermediate_size": 4 * a.d,
            "moe_intermediate_size": a.d // 2, "num_experts": a.experts, "num_experts_per_tok": a.topk,
            "num_shared_experts": a.shared, "route_scale": a.route_scale, "global_attn_every_n_layers": 4,
            "sliding_window": 2048, "max_position_embeddings": 2048,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}, "rms_norm_eps": 1e-6,
            "tie_word_embeddings": True, "hidden_act": "silu", "mup_enabled": True, "initializer_range": 0.02,
            "bos_token_id": 0, "eos_token_id": 0, "pad_token_id": 1}
    cfg_path = out.with_suffix(".config.json")
    cfg_path.write_text(json.dumps(cfgd))
    cfg = afmoe.load_config(str(cfg_path))
    torch.manual_seed(a.seed)
    model = LudwigAfmoe(cfg, fp32_residual=True)
    blocks = moe_blocks(model.lm)
    if a.router_init != 0.02:
        with torch.no_grad():
            for m in blocks:
                m.router.gate.weight.normal_(0.0, a.router_init)
    n_moe, E, k = len(blocks), a.experts, a.topk

    # ---- capture the router's input and logits (the repo's router forward, wrapped)
    cap = {"on": False, "grad": False, "items": []}
    inner = am.AfmoeTokenChoiceRouter.forward

    def router(self, hidden_states, expert_bias):
        res = inner(self, hidden_states, expert_bias)
        if cap["on"]:
            if cap["grad"]:
                res[0].retain_grad()
            cap["items"].append((hidden_states.detach().reshape(-1, hidden_states.shape[-1]).float(), res[0], res[2]))
        return res

    am.AfmoeTokenChoiceRouter.forward = router

    # ---- optimizers: the repo's split; the router in its own AdamW group
    muon_p, decay_p, no_decay_p = split_params(model)
    router_p = [m.router.gate.weight for m in blocks]
    rest = [p for p in decay_p if all(p is not r for r in router_p)]
    muon = Muon(muon_p, lr=a.lr, weight_decay=0.1 * a.adamw_lr / a.lr, n_heads=4, head_dim=a.d // 4,
                momentum_dtype=torch.float32, ns_dtype=torch.float32, fused_qgate=False)
    adamw = torch.optim.AdamW([{"params": rest, "weight_decay": 0.1}, {"params": router_p, "weight_decay": 0.1},
                               {"params": no_decay_p, "weight_decay": 0.0}], lr=a.adamw_lr, betas=(0.9, 0.95),
                              eps=1e-8)
    params = [p for p in model.parameters() if p.requires_grad]
    rule = a.balance if a.balance in ("smebu", "sign") else "smebu"
    balancer = ExpertBalancer(model, rule=rule, lam=a.lam, gamma=a.lam)
    if a.impl == "repo":            # the run starts with SMEBU and free offsets, and switches at --balance-from
        fixed = ExpertBalancer(model, rule=a.balance, lam=a.lam, gamma=a.lam, damp=a.q_damp,
                               sample=a.q_tokens, offsets="equal" if a.offsets == "nodiff" else "free",
                               offsets_rate=a.offsets_rate)
        first = balancer
    weights = LossWeights(output_z=1e-4, seq_aux=a.seq_aux, router_z=a.router_z, router_floor=a.router_floor,
                          router_floor_lse=a.router_floor_lse)
    ce_only = LossWeights(output_z=0.0, seq_aux=0.0, router_z=0.0, router_floor=0.0)

    train = np.load(Path(a.data) / "train_4096.npy")    # ids sorted by frequency: a smaller V keeps the top ones
    val = np.load(Path(a.data) / "val_4096.npy")
    if a.V < 4096:
        train, val = np.where(train >= a.V, 1, train).astype(np.int16), np.where(val >= a.V, 1, val).astype(np.int16)
    rng = np.random.default_rng(a.seed)

    def batch(src, r):
        idx = r.integers(0, len(src) - a.seq - 1, a.batch)
        return torch.from_numpy(np.stack([src[i:i + a.seq] for i in idx]).astype(np.int64))

    vr = np.random.default_rng(1234)
    val_batches = [batch(val, vr) for _ in range(8)]
    probe_ids = torch.cat(val_batches[:2])     # fixed tokens: what changes between probes is the model
    log = open(out, "w")

    def row(kind, **kw):
        log.write(json.dumps({"type": kind, **kw}) + "\n")
        log.flush()

    row("config", args=vars(a), params=sum(p.numel() for p in params))
    xbar = [None] * n_moe                      # running mean of each router's input
    prev = {}                                  # the previous probe, for speeds
    t0 = time.time()

    def lam_at(step):
        return a.lam_final if (a.lam_final is not None and step >= a.lam_steps) else a.lam

    @torch.no_grad()
    def probe(step, items, grads, w_before):
        """Per MoE layer, from this step's tokens. ``grads``: dCE/dlogits per layer (or None)."""
        res = {}

        def put(name, v):
            res.setdefault(name, []).append(round(float(v), 6))

        for li, ((x, logits, chosen), block) in enumerate(zip(items, blocks)):
            logits = logits.detach()
            w = block.router.gate.weight.detach().float()
            b = block.expert_bias.detach().float()
            xm = x.mean(0)
            dx = x - xm
            put("rho", xm.pow(2).sum() / x.pow(2).sum(1).mean())                 # common fraction of |x|^2
            off = logits.mean(0)                                                 # = w @ xm: each expert's offset
            spec = logits - off
            put("level", off.mean())
            put("off_std", off.std())
            put("spec_std", spec.std(0).mean())                                  # token-specific logit spread
            s = torch.sigmoid(logits)
            sm, sd = s.mean(0), s.std(0)
            put("score_mean_min", sm.min()); put("score_mean_med", sm.median()); put("score_mean_max", sm.max())
            put("score_sd_min", sd.min()); put("score_sd_med", sd.median()); put("score_sd_max", sd.max())
            put("bias_range", b.max() - b.min())
            put("corr_bias_score", torch.corrcoef(torch.stack([b, sm]))[0, 1] if b.std() > 0 else 0.0)
            put("lse", torch.logsumexp(logits, -1).mean())
            top = torch.topk(s, k + 1, dim=-1)
            put("gap", (top.values[:, k - 1] - top.values[:, k]).median())
            own = top.indices[:, :k].sort(-1).values
            put("decides", (own != chosen.sort(-1).values).any(-1).float().mean())
            cnt = torch.bincount(chosen.reshape(-1), minlength=E).float()
            put("load_max", cnt.max() / cnt.mean()); put("load_min", cnt.min() / cnt.mean())
            g = s.gather(1, chosen)
            g = g / g.sum(-1, keepdim=True)
            put("gate_top1", g.max(-1).values.mean())                            # 0.25 = even gates
            put("gate_eff", (1.0 / g.pow(2).sum(-1)).mean())                     # effective experts per token (max k)
            # what the chosen experts' scores look like next to the token's best raw scores
            put("chosen_score_mean", s.gather(1, chosen).mean())
            bstar = balanced_bias(s[::2], b, k, iters=25)
            put("bstar_range", bstar.max() - bstar.min())
            put("bias_err_max", (b - b.mean() - bstar).abs().max())
            key = f"bstar{li}"
            if key in prev:
                dstep = (bstar - prev[key]) / max(1, step - prev["step"])
                put("bstar_speed_max", dstep.abs().max()); put("bstar_speed_rms", dstep.pow(2).mean().sqrt())
                dso = (off - prev[f"off{li}"]) / max(1, step - prev["step"])
                put("off_speed_rms", (dso - dso.mean()).pow(2).mean().sqrt()); put("level_speed", dso.mean())
                # split the offsets' move since the last probe: the router's weights moved, or its mean input did
                n_ = max(1, step - prev["step"])
                by_w = (w - prev[f"w{li}"]) @ prev[f"xm{li}"] / n_
                by_x = w @ (xm - prev[f"xm{li}"]) / n_
                put("off_speed_by_w", (by_w - by_w.mean()).pow(2).mean().sqrt())
                put("off_speed_by_x", (by_x - by_x.mean()).pow(2).mean().sqrt())
                put("level_speed_by_w", by_w.mean()); put("level_speed_by_x", by_x.mean())
                put("xm_cos_prev", torch.nn.functional.cosine_similarity(xm, prev[f"xm{li}"], dim=0))
            prev[key], prev[f"off{li}"], prev[f"w{li}"], prev[f"xm{li}"] = bstar, off, w.clone(), xm.clone()
            put("xm_norm", xm.norm()); put("w_row_norm", w.norm(dim=1).mean())
            put("w_common_cos", ((w @ (xm / xm.norm())) / w.norm(dim=1)).abs().mean())   # |cos(w_i, x_mean)|
            if w_before is not None:                                             # what this optimizer step did
                dw = w - w_before[li]
                d_off = dw @ xm
                d_spec = (dx @ dw.T).std(0)
                put("step_off_rms", (d_off - d_off.mean()).pow(2).mean().sqrt())
                put("step_level", d_off.mean())
                put("step_spec_rms", d_spec.pow(2).mean().sqrt())
            if grads is not None:                                                # the cross-entropy's push
                gr = grads[li] * logits.shape[0]                                 # per token (CE is a mean)
                put("ce_level_push", gr.sum(1).mean())                           # > 0: descent lowers the level
                gw = gr.T @ x / x.shape[0]                                       # dCE/dW, (E, d)
                u = xm / xm.norm()
                put("ce_grad_common_frac", (gw @ u).pow(2).sum() / gw.pow(2).sum().clamp_min(1e-30))
        prev["step"] = step
        return res

    for step in range(a.steps + 1):
        mult = min(1.0, (step + 1) / a.warmup)
        balancer.lam = balancer.gamma = lam_at(step)
        rs = a.route_scale
        if a.route_scale_final is not None and step >= a.route_scale_from:
            frac = min(1.0, (step - a.route_scale_from + 1) / a.route_scale_ramp) if a.route_scale_ramp else 1.0
            rs = a.route_scale + (a.route_scale_final - a.route_scale) * frac
            if a.route_scale_keep and step == a.route_scale_from:
                with torch.no_grad():
                    for m in blocks:
                        m.experts.down_proj.div_(a.route_scale_final / a.route_scale)
        for m in blocks:
            m.router.route_scale = rs
        do_probe = bool(a.probe_every) and (step % a.probe_every == 0 or step == a.steps)
        ids = batch(train, rng)
        grads = pitems = None
        if do_probe:                                   # an extra pass on fixed tokens, cross-entropy alone
            cap.update(on=True, grad=True, items=[])
            loss, _ = model(probe_ids, ce_only, ce_chunk=1 << 20)
            cap["on"] = False
            loss.backward()
            pitems = [(x, lg.detach(), ch) for x, lg, ch in cap["items"]]
            grads = [it[1].grad.detach().clone() for it in cap["items"]]
            for opt in (muon, adamw):
                opt.zero_grad(set_to_none=True)
        if step == a.steps:
            if do_probe:
                row("probe", step=step, **probe(step, pitems, grads, None))
            break
        cap.update(on=True, grad=False, items=[])
        loss, mets = model(ids, weights, ce_chunk=1 << 20)
        cap["on"] = False
        items = cap["items"]
        loss.backward()
        if a.impl == "repo":
            balancer = fixed if step >= a.balance_from else first
            balancer.lam = balancer.gamma = lam_at(step)
            balancer.at_step = step
            balancer.accumulate(mets["chosen"], mets["router_logits"], mets["router_xsum"])
        else:
            balancer.accumulate(mets["chosen"])
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        for g_ in muon.param_groups:
            g_["lr"] = a.lr * mult
        for gi, g_ in enumerate(adamw.param_groups):
            g_["lr"] = a.adamw_lr * mult * (a.router_lr_mult if gi == 1 else 1.0)
        w_before = [m.router.gate.weight.detach().clone() for m in blocks] if do_probe else None
        muon.step()
        adamw.step()
        with torch.no_grad():
            for li, (x, _, _) in enumerate(items):
                xm = x.mean(0)
                xbar[li] = xm if xbar[li] is None else 0.9 * xbar[li] + 0.1 * xm
                if a.impl == "lab" and a.offsets != "free" and step >= a.offsets_from:
                    u = xbar[li] / xbar[li].norm()
                    w = blocks[li].router.gate.weight
                    c = w @ u
                    if a.offsets == "nodiff":
                        c = c - c.mean()
                    w.sub_(c[:, None] * u[None, :])
        for opt in (muon, adamw):
            opt.zero_grad(set_to_none=True)
        if do_probe:
            row("probe", step=step, **probe(step, pitems, grads, w_before))
        # ---- the balancer
        loads = torch.stack([torch.bincount(c.reshape(-1), minlength=E).float() for c in mets["chosen"]])
        loads = loads / loads.mean(1, keepdim=True)
        if a.impl == "repo":
            reset = a.bias_reset and step == a.balance_from
            balancer.equalise_offsets(full=reset)
            bal = balancer.step(reset=reset)
        elif a.bias_reset and a.offsets != "free" and step == a.offsets_from:
            with torch.no_grad():
                for m in blocks:
                    m.expert_bias.zero_()
                balancer.momentum.zero_()
        if a.impl == "repo":
            pass
        elif a.balance in ("smebu", "sign") or step < a.balance_from:
            bal = balancer.step()
        else:
            balancer.counts.zero_()
            if a.balance == "quantile":
                with torch.no_grad():
                    for li, (x, logits, _) in enumerate(items):
                        s = torch.sigmoid(logits.detach())
                        if s.shape[0] > a.q_tokens:
                            s = s[torch.randperm(s.shape[0])[:a.q_tokens]]
                        blocks[li].expert_bias.copy_(balanced_bias(s, blocks[li].expert_bias, k, 1, a.q_damp))
            bal = {"bias_abs_max": float(max(m.expert_bias.abs().max() for m in blocks))}
        r = {"step": step, "ce": round(float(mets["ce"]), 4), "lr_mult": round(mult, 4), "lam": balancer.lam,
             "route_scale": round(rs, 4),
             "gnorm": round(gn, 3), "seq_aux": round(float(mets["seq_aux"]), 4),
             "load_max": round(float(loads.max()), 3), "load_min": round(float(loads.min()), 3),
             "layer_load_max": [round(float(v), 3) for v in loads.max(1).values],
             "busiest": [int(v) for v in loads.argmax(1)],
             "bias_abs_max": round(bal["bias_abs_max"], 4),
             "lse": [round(float(v), 3) for v in mets["router_lse_layers"]],
             "decides": [round(float(v), 3) for v in mets["bias_decides"]],
             "margin": [round(float(v), 5) for v in mets["router_margin"]],
             "entropy": round(float(mets["router_entropy"]), 4)}
        if a.val_every and step % a.val_every == 0:
            model.eval()
            with torch.no_grad():
                r["val_ce"] = round(float(np.mean([float(model(v, ce_only, ce_chunk=1 << 20)[1]["ce"])
                                                   for v in val_batches])), 4)
            model.train()
        row("step", **r)
        if step % 50 == 0:
            print(f"step {step:5d} ce {r['ce']:.3f} val {r.get('val_ce', float('nan')):.3f} load {r['load_max']:.2f}/"
                  f"{r['load_min']:.2f} bias {r['bias_abs_max']:.3f} lse {min(r['lse']):.2f}-{max(r['lse']):.2f} "
                  f"decides {max(r['decides']):.2f} {(time.time() - t0) / (step + 1):.2f} s/step", flush=True)
    row("end", seconds=round(time.time() - t0, 1))
    if not math.isfinite(float(mets["ce"])):
        print("non-finite loss")


if __name__ == "__main__":
    main()
