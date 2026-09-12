"""
The per-node denoising step — ported and generalized from a_local_step /
b_local_step (DED_NodeA_v7.ipynb / DED_NodeB_v7.ipynb).

What changed, and what deliberately didn't:

- The algorithm itself is UNCHANGED. Same remasking rule, same EOS-margin
  gating, and critically, the SAME block-scoped transfer-count fix that
  came out of the "everything turns to <|endoftext|>" corruption bug —
  see the comment on `remaining` below. That fix is load-bearing; it is
  not being re-derived here, just carried forward verbatim.
- There is no more "A" or "B". a_local_step and b_local_step were two
  near-identical copies of the same function, differing only in variable
  suffixes — an artifact of there always being exactly two nodes, not
  anything the algorithm actually needed. Every swarm participant now
  calls this same function against its own model.
- model/tokenizer/thresholds are explicit parameters (StepConfig) instead
  of notebook globals, since this has to work as an importable module, not
  a notebook cell that can see whatever was defined in cells above it.
- The old debug_topk/debug_distribution print calls were dropped from
  inside the step function itself — they were tightly coupled to notebook
  globals (VERBOSE, a hardcoded prefix string) and are a Colab-debugging
  concern, not core algorithm logic. If step-level tracing is wanted again,
  the caller (node/generation.py) has every value needed (logits, p, x0)
  to reconstruct that without local_step needing to know about it.

What this file still owes you, honestly: it has NOT been run against a
real model. The port is mechanical (parametrize globals, rename, nothing
else touched), so the risk of a NEW bug here is low — but "low risk" is
not "verified," and this project has already been burned twice by things
that looked obviously fine until they actually ran (the topk-overflow
corruption, the websocket ping timeout). Run this for real before trusting
it.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class StepConfig:
    mask_id: int
    eos_id: int
    temperature: float
    eos_margin_threshold: float
    remask_conf_threshold: float
    remask_min_age: int
    remask_max_frac: float
    remask_reserve_frac: float


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def local_step(model, x, age, steps_remaining, block_start, block_end,
                steps_per_block, cfg: StepConfig, frozen_from=None):
    """Run this node's own model for one denoising step.

    Returns (x, age, x0, p, conf_new, transfer, remask):
      x, age    updated sequence/age tensors — this node's local view only,
                not yet reconciled with anyone else's.
      x0        this node's raw per-position prediction (argmax + Gumbel).
      p         this node's full (seq_len, vocab) softmax distribution.
                Kept around on purpose: scoring some OTHER participant's
                proposed token later, during dispute resolution, is just an
                index lookup into this — no second forward pass needed. See
                diffusion/consensus.py's docstring on why phase 2 is free.
      conf_new  this node's confidence in its own x0 pick.
      transfer  positions this node is committing this step — this becomes
                a COMMIT message's positions/tokens/confidences.
      remask    positions this node is reopening this step.
    """
    mask_index = (x == cfg.mask_id)
    logits = model(x).logits
    p = F.softmax(logits, dim=-1)

    conf_current = torch.squeeze(torch.gather(p, -1, x.unsqueeze(-1)), -1)
    x0 = torch.argmax(add_gumbel_noise(logits, cfg.temperature), dim=-1)
    conf_new = torch.squeeze(torch.gather(p, -1, x0.unsqueeze(-1)), -1)

    p_no_eos = p.clone()
    p_no_eos[..., cfg.eos_id] = 0.0
    best_non_eos_prob, _ = p_no_eos.max(dim=-1)
    eos_margin = (torch.log(p[..., cfg.eos_id].clamp_min(1e-12))
                  - torch.log(best_non_eos_prob.clamp_min(1e-12)))
    is_eos_guess = (x0 == cfg.eos_id)
    eos_not_trusted = is_eos_guess & (eos_margin < cfg.eos_margin_threshold)

    reserve_steps = max(1, round(cfg.remask_reserve_frac * steps_per_block))
    allow_remask = (steps_remaining > reserve_steps)

    remask = torch.zeros_like(mask_index)
    if allow_remask:
        eligible = (~mask_index) & (age >= cfg.remask_min_age)
        eligible[:, :block_start] = False
        eligible[:, block_end:] = False
        if frozen_from is not None:
            eligible[:, frozen_from:] = False
        low_conf = eligible & (conf_current < cfg.remask_conf_threshold)
        for j in range(low_conf.shape[0]):
            idx = torch.where(low_conf[j])[0]
            cap = max(1, int(cfg.remask_max_frac * eligible[j].sum().item()))
            if idx.numel() > cap:
                worst = torch.topk(-conf_current[j, idx], k=cap).indices
                idx = idx[worst]
            remask[j, idx] = True

    conf_new_masked = torch.where(mask_index, conf_new, -torch.inf)
    conf_new_masked[:, :block_start] = -torch.inf   # prompt / earlier blocks — never a transfer target
    conf_new_masked[:, block_end:] = -torch.inf     # future blocks — never a transfer target
    if frozen_from is not None:
        conf_new_masked[:, frozen_from:] = -torch.inf
    conf_new_masked = torch.where(mask_index & eos_not_trusted,
                                   conf_new_masked - 100.0, conf_new_masked)

    transfer = torch.zeros_like(x0, dtype=torch.bool)
    for j in range(conf_new_masked.shape[0]):
        # Scoped to THIS block only — this exact line is the fix for the
        # topk-overflow corruption bug (v6). Counting masks over the whole
        # sequence includes untouched future blocks, which inflates k past
        # the real candidate count once this block's own masks run out
        # early, and topk fills the overflow from tied -inf positions —
        # i.e. already-finished content — corrupting it. Do not revert
        # this to mask_index[j].sum() over the full sequence.
        remaining = int(mask_index[j, block_start:block_end].sum().item())
        k = -(-remaining // steps_remaining)
        k = min(k, remaining)
        if k > 0:
            _, idx = torch.topk(conf_new_masked[j], k=k)
            transfer[j, idx] = True

    x, age = x.clone(), age.clone()
    x[remask] = cfg.mask_id
    age[remask] = 0
    x[transfer] = x0[transfer]
    age[transfer] = 0
    still_settled = (~mask_index) & (~remask) & (~transfer)
    age[still_settled] += 1

    return x, age, x0, p, conf_new, transfer, remask
