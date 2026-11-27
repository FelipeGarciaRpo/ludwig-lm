"""Reference for the two pieces of the fix proposed by the diagnosis (P35), as written before the patch.
It is NOT the implementation: that is in ludwig_lm/afmoe.py (bias_targets, ExpertBalancer's quantile rule and
equalise_offsets) and scripts/train.py. This file shows the computation that moe_lab.py's prototypes tested.

Piece A, the balancer: instead of a fixed-size step, move the bias toward the value that balances the step's tokens.
Piece B, the router: after each optimizer step, keep every expert's offset along the mean router input equal.

Neither adds a tensor or changes a key of the checkpoint: the AFMoE format and llama.cpp see the same model.
"""
import torch
import torch.distributed as dist


# ----------------------------------------------------------------------------------------------- piece A
@torch.no_grad()
def bias_targets(logits: torch.Tensor, bias: torch.Tensor, k: int) -> torch.Tensor:
    """Per expert, the bias that would give it its share (k/E) of these tokens while the others keep theirs.

    logits (N, E) are the router's FP32 logits of one micro-batch, bias (E,) the layer's expert_bias.
    With the others fixed, expert i takes token t iff bias_i > need[t, i], where need is what it has to beat
    (the k-th best total among the others) minus its own score. So the bias it wants is the k/E quantile of need.
    Cost: one top-(k+1) and one kthvalue over N x E floats (2.9 MB for 16,384 tokens and 44 experts)."""
    s = torch.sigmoid(logits.float())
    tot = s + bias.float()
    top = torch.topk(tot, k + 1, dim=1).values                       # (N, k+1), descending
    inside = tot >= top[:, k - 1:k]                                   # the experts the token has now
    beat = torch.where(inside, top[:, k:k + 1], top[:, k - 1:k])      # the best total among the others at the edge
    need = beat - s
    kth = max(1, round(need.shape[0] * k / need.shape[1]))
    return torch.kthvalue(need, kth, dim=0).values                    # (E,)


class QuantileRule:
    """What ExpertBalancer does with rule='quantile'. accumulate() once per micro-step, step() once per
    optimizer step (after the optimizers, where balancer.step() is today)."""

    def __init__(self, blocks, n_exp: int, k: int, damp: float = 0.3):
        self.blocks, self.k, self.damp = blocks, k, damp
        dev = blocks[0].expert_bias.device
        self.targets = torch.zeros(len(blocks), n_exp, device=dev)
        self.micro = torch.zeros((), device=dev)

    @torch.no_grad()
    def accumulate(self, logits_per_layer):
        for i, lg in enumerate(logits_per_layer):                     # detached logits, one tensor per MoE layer
            self.targets[i] += bias_targets(lg, self.blocks[i].expert_bias, self.k)
        self.micro += 1

    @torch.no_grad()
    def step(self):
        if dist.is_available() and dist.is_initialized():             # both ranks must apply the same bias
            dist.all_reduce(self.targets)
            dist.all_reduce(self.micro)
        q = self.targets / self.micro.clamp_min(1)
        for i, block in enumerate(self.blocks):
            b = block.expert_bias
            b.add_((self.damp * (q[i] - b)).to(b.dtype))
            b.sub_(b.mean())                                          # the top-k only sees differences
        self.targets.zero_()
        self.micro.zero_()


# ----------------------------------------------------------------------------------------------- piece B
@torch.no_grad()
def equalise_offsets(gate_weight_local: torch.Tensor, x_sum: torch.Tensor, n_exp: int):
    """Remove the differences between the experts' offsets along the mean input of this layer's router.

    gate_weight_local: this rank's rows of layers[l].mlp.router.gate.weight (under FSDP2: the DTensor's
    to_local(), edited in place like Muon does). x_sum (d,): the sum over the step's tokens of the router's
    input, already all-reduced over the ranks (its scale does not matter). Call it after adamw.step().

    logit[t, i] = w_i . x_t. The part common to all tokens is w_i . x_mean: the expert's offset. Afterwards
    every expert has the same one, so the router keeps only its token-specific preferences, and the level (the
    same shift for all experts) stays free."""
    u = x_sum / x_sum.norm()
    c = gate_weight_local.float() @ u                                 # this rank's offsets, per unit of |x_mean|
    total = c.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total)
    gate_weight_local.sub_(((c - total / n_exp)[:, None] * u[None, :]).to(gate_weight_local.dtype))


# ----------------------------------------------------------------------------------------------- the probe
@torch.no_grad()
def router_probe(x: torch.Tensor, logits: torch.Tensor, bias: torch.Tensor) -> dict:
    """Per MoE layer, what the evaluations should log to see the mechanism on a real checkpoint (x: the router's
    input (N, d); logits (N, E); bias (E,)). 'rho' and 'offset_std' are the numbers behind the diagnosis."""
    xm = x.float().mean(0)
    off = logits.mean(0)                                              # = W @ x_mean
    s = torch.sigmoid(logits)
    return {"rho": float(xm.pow(2).sum() / x.float().pow(2).sum(1).mean()),      # common share of |x|^2
            "offset_std": float(off.std()),                                      # logit units, across experts
            "specific_std": float((logits - off).std(0).mean()),                 # logit units, across tokens
            "score_mean_min": float(s.mean(0).min()), "score_mean_max": float(s.mean(0).max()),
            "score_sd_min": float(s.std(0).min()), "score_sd_max": float(s.std(0).max()),
            "corr_bias_score": float(torch.corrcoef(torch.stack([bias.float(), s.mean(0)]))[0, 1])}
