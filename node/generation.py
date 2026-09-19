"""
node/generation.py — with the leader-liveness fix (continuous light polling
between syncs, not just at sync points) and its companion fix (a durable,
sync_seq-keyed inbox in SwarmNode so that polling can't lose a message
meant for a later collect_until call). See node/swarm.py for the inbox
side of this.

Also restores the per-step (debug_topk / debug_distribution) and per-sync
verbose detail that DED_NodeA_v7.ipynb / DED_NodeB_v7.ipynb had and the
2-node -> N-node port dropped. diffusion/local_step.py's docstring flagged
this explicitly: the debug printing was cut from local_step itself (it was
tied to notebook globals) with the intent that the caller reconstruct it
from the values local_step already returns — this is that reconstruction,
generalized from exactly-two-nodes (A/B) to however many participants a
round actually has.
"""
import math
import torch
from dataclasses import dataclass

from common import config, protocol
from node.swarm import SwarmNode
from diffusion.local_step import local_step, StepConfig
from diffusion.consensus import detect_disputes, resolve_disputes, merge_consensus
from diffusion.stop_detection import StopDetector, StopConfig

VERBOSE = True
DEBUG_EVERY = 8


def debug_topk(p, x, mask_index, block_start, block_end, steps_remaining,
                tokenizer, eos_id, prefix=""):
    """Top-3 candidates + EOS margin for a sample of still-masked positions
    in the current block. `p` is the already-softmaxed (seq_len, vocab)
    distribution local_step returns — no need to re-derive it from logits."""
    if not VERBOSE:
        return
    masked_pos = torch.where(mask_index[0])[0]
    masked_pos = masked_pos[(masked_pos >= block_start) & (masked_pos < block_end)]
    sample = masked_pos[:5]
    if len(sample) == 0:
        return
    print(f"\n[DEBUG {prefix}] Top predictions for masked positions (step {steps_remaining} remaining):")
    for pos in sample:
        pos = pos.item()
        top_probs, top_ids = torch.topk(p[0, pos], k=3)
        tokens = [tokenizer.decode([tid.item()]) for tid in top_ids]
        eos_prob = p[0, pos, eos_id].item()
        p_eos = p[0, pos, eos_id].clamp_min(1e-12)
        best_non_eos = p[0, pos].clone()
        best_non_eos[eos_id] = 0.0
        best_non_eos = best_non_eos.max().clamp_min(1e-12)
        margin = torch.log(p_eos) - torch.log(best_non_eos)
        print(f"  pos {pos:3d}: top tokens: {tokens[0]:12} ({top_probs[0]:.4f}), "
              f"{tokens[1]:12} ({top_probs[1]:.4f}), {tokens[2]:12} ({top_probs[2]:.4f}) | "
              f"EOS prob = {eos_prob:.4f}, margin = {margin.item():.2f}")


def debug_distribution(p, x, mask_index, block_start, block_end, steps_remaining,
                        tokenizer, eos_id, prefix=""):
    if not VERBOSE:
        return
    masked_pos = torch.where(mask_index[0])[0]
    masked_pos = masked_pos[(masked_pos >= block_start) & (masked_pos < block_end)]
    if len(masked_pos) == 0:
        return
    probs = p[0]  # (seq_len, vocab)
    entropies = -torch.sum(probs * torch.log(probs.clamp_min(1e-12)), dim=-1)
    max_probs, _ = torch.max(probs, dim=-1)
    print(f"[DEBUG {prefix}] Masked positions {block_start}:{block_end} (step {steps_remaining} rem):")
    for pos in masked_pos[:10]:
        pos = pos.item()
        eos_prob = probs[pos, eos_id].item()
        top_probs, top_ids = torch.topk(probs[pos], k=3)
        top_tokens = [tokenizer.decode([tid.item()]) for tid in top_ids]
        print(f"  pos {pos:3d}: max={max_probs[pos]:.4f}, entropy={entropies[pos]:.3f}, "
              f"top: {top_tokens[0]}({top_probs[0]:.3f}), {top_tokens[1]}({top_probs[1]:.3f}), "
              f"{top_tokens[2]}({top_probs[2]:.3f}) | EOS={eos_prob:.3f}")


@dataclass
class GenConfig:
    gen_length: int
    block_length: int
    steps: int
    sync_every: int
    step_cfg: StepConfig
    stop_cfg: StopConfig


class GenerationRunner:
    def __init__(self, swarm: SwarmNode, model, tokenizer, device):
        self.swarm = swarm
        self.id = swarm.id
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def run_round(self, prompt_ids, gen_cfg: GenConfig):
        round_state = self.swarm.round_mgr.active_round
        round_id = round_state.round_id
        participants = set(round_state.participants)
        other_participants = participants - {self.id}

        gen_start = prompt_ids.shape[1]
        gen_end = gen_start + gen_cfg.gen_length
        num_blocks = gen_cfg.gen_length // gen_cfg.block_length
        steps_per_block = gen_cfg.steps // num_blocks

        if VERBOSE:
            print(f"[{self.id}] round {round_id}: {num_blocks} blocks x "
                  f"{steps_per_block} steps/block ({gen_cfg.steps} total), "
                  f"{len(participants)} participant(s): {sorted(participants)}")

        x = torch.full((1, gen_end), gen_cfg.step_cfg.mask_id, dtype=torch.long, device=self.device)
        x[:, :gen_start] = prompt_ids

        shared_x: dict = {p: prompt_ids[0, p].item() for p in range(gen_start)}
        stop_detector = StopDetector(gen_cfg.stop_cfg)
        frozen_from = None
        sync_seq = 0  # increments once per sync, for the whole round — disambiguates
                       # this sync's messages from every other sync's in SwarmNode's inbox

        for block_idx in range(num_blocks):
            block_start = gen_start + block_idx * gen_cfg.block_length
            block_end = gen_start + (block_idx + 1) * gen_cfg.block_length

            if frozen_from is not None and block_start >= frozen_from:
                if VERBOSE:
                    print(f"[{self.id}] block {block_idx} is entirely past the "
                          f"frozen boundary — skipping.")
                break

            age = torch.zeros_like(x)
            pending_commits: dict = {}

            for step_idx in range(steps_per_block):
                # THE FIX: a cheap, non-blocking-ish poll EVERY step, not just
                # at sync points. Without this, a follower can go SYNC_EVERY
                # steps (~14s at this model's per-step cost) with zero socket
                # reads, letting round_mgr.last_leader_seen go stale and
                # falsely timing out even though the leader's heartbeats were
                # arriving the whole time — they just never got read off the
                # wire. Safe to do every step: pump_once() with a short
                # timeout costs nothing next to a ~2.9s forward pass, and any
                # gen-protocol message it happens to pick up early goes into
                # SwarmNode's durable inbox rather than being discarded, so a
                # LATER collect_until call for the same sync still finds it.
                self.swarm.pump_once(timeout_ms=10)

                if self.swarm.round_mgr.check_leader_timeout():
                    print(f"[{self.id}] round {round_id} failed mid-block — aborting generation")
                    return None

                steps_remaining = steps_per_block - step_idx
                mask_index_before = (x == gen_cfg.step_cfg.mask_id)   # for debug sampling below —
                                                                       # what WAS masked going into
                                                                       # this step, not what's left after
                x, age, x0, p, conf_new, transfer, remask = local_step(
                    self.model, x, age, steps_remaining, block_start, block_end,
                    steps_per_block, gen_cfg.step_cfg, frozen_from=frozen_from,
                )

                for pos in torch.where(transfer[0])[0].tolist():
                    pending_commits[pos] = (x0[0, pos].item(), conf_new[0, pos].item())

                if VERBOSE and step_idx % DEBUG_EVERY == 0:
                    debug_topk(p, x, mask_index_before, block_start, block_end,
                               steps_remaining, self.tokenizer, gen_cfg.step_cfg.eos_id,
                               prefix=self.id)
                    debug_distribution(p, x, mask_index_before, block_start, block_end,
                                        steps_remaining, self.tokenizer, gen_cfg.step_cfg.eos_id,
                                        prefix=self.id)

                if VERBOSE and (step_idx + 1) % DEBUG_EVERY == 0:
                    n_masked = int((x[0, block_start:block_end] == gen_cfg.step_cfg.mask_id).sum())
                    print(f"[{self.id}] block {block_idx} step {step_idx + 1}/{steps_per_block}: "
                          f"{n_masked} masks remain in block")

                is_last_step = (step_idx == steps_per_block - 1)
                if (step_idx + 1) % gen_cfg.sync_every == 0 or is_last_step:
                    shared_x, frozen_from = self._do_sync(
                        round_id, sync_seq, participants, other_participants,
                        pending_commits, p, shared_x, gen_start, gen_end,
                        stop_detector, frozen_from, block_idx, step_idx, is_last_step,
                        gen_cfg.step_cfg.mask_id,
                    )
                    sync_seq += 1
                    pending_commits = {}
                    for pos, token in shared_x.items():
                        if block_start <= pos < block_end:
                            x[0, pos] = token
                            age[0, pos] = 0

        if self.swarm.round_mgr.is_round_leader():
            self.swarm.transport.publish(self.id, protocol.GEN_DONE, {"round_id": round_id})
        return shared_x

    def _do_sync(self, round_id, sync_seq, participants, other_participants,
                 my_commits, my_p, shared_x, gen_start, gen_end,
                 stop_detector, frozen_from, block_idx, step_idx, is_last_step,
                 mask_id):
        positions = list(my_commits.keys())
        self.swarm.transport.publish(self.id, protocol.COMMIT, {
            "round_id": round_id,
            "sync_seq": sync_seq,
            "positions": positions,
            "tokens": [my_commits[p][0] for p in positions],
            "confidences": [my_commits[p][1] for p in positions],
        })
        others_raw = self.swarm.collect_until(protocol.COMMIT, round_id, sync_seq,
                                               other_participants, config.SYNC_BARRIER_TIMEOUT)

        all_commits = {self.id: my_commits}
        for node_id, payload in others_raw.items():
            all_commits[node_id] = {
                pos: (tok, conf) for pos, tok, conf in
                zip(payload["positions"], payload["tokens"], payload["confidences"])
            }

        disputes = detect_disputes(all_commits)

        resolved: dict = {}
        all_scores: dict = {}
        if disputes:
            my_scores = {
                pos: {tok: my_p[0, pos, tok].clamp_min(1e-12).log().item() for tok in tokens}
                for pos, tokens in disputes.items()
            }
            self.swarm.transport.publish(self.id, protocol.DISPUTE_SCORE, {
                "round_id": round_id,
                "sync_seq": sync_seq,
                "scores": {str(pos): {str(t): s for t, s in toks.items()}
                           for pos, toks in my_scores.items()},
            })
            others_scores_raw = self.swarm.collect_until(
                protocol.DISPUTE_SCORE, round_id, sync_seq, other_participants,
                config.SYNC_BARRIER_TIMEOUT)

            all_scores = {self.id: my_scores}
            for node_id, payload in others_scores_raw.items():
                all_scores[node_id] = {
                    int(pos): {int(tok): score for tok, score in toks.items()}
                    for pos, toks in payload["scores"].items()
                }
            resolved = resolve_disputes(all_scores)

        shared_x = merge_consensus(shared_x, all_commits, resolved)

        new_frozen_from = stop_detector.check(shared_x, gen_start, gen_end)
        if new_frozen_from is not None and new_frozen_from != frozen_from:
            print(f"[{self.id}] stop boundary confirmed at {new_frozen_from - gen_start}")

        if VERBOSE:
            tag = "  (end of block)" if is_last_step else ""
            print("\n" + "=" * 70)
            print(f"[{self.id}] BLOCK {block_idx} | STEP {step_idx + 1}{tag}  (sync_seq={sync_seq})")
            print("=" * 70)

            for node_id in sorted(all_commits.keys()):
                print(f"\nNode {node_id} committed this step:")
                node_commits = all_commits[node_id]
                if not node_commits:
                    print("  (nothing)")
                for pos in sorted(node_commits.keys()):
                    tok, conf = node_commits[pos]
                    decoded = repr(self.tokenizer.decode([tok]))
                    print(f"  Pos {pos:3d} -> {decoded:15} ({conf:.4f})")

            n_commits = sum(len(c) for c in all_commits.values())
            print(f"\nSync state: {len(all_commits)} reporter(s), {n_commits} total commits, "
                  f"{len(disputes)} disputed position(s)")

            if disputes:
                print("\nDisputes resolved this sync (weighted-avg scores, all reporters):")
                for pos in sorted(disputes.keys()):
                    winner_tok = resolved.get(pos)
                    parts = []
                    for node_id in sorted(all_scores.keys()):
                        cand_scores = all_scores[node_id].get(pos, {})
                        for tok, logprob in sorted(cand_scores.items()):
                            marker = " *" if tok == winner_tok else ""
                            decoded = repr(self.tokenizer.decode([tok]))
                            # display as prob, not log-prob — matches how resolve_disputes
                            # actually aggregates (see diffusion/consensus.py)
                            parts.append(f"{node_id}:{decoded}={math.exp(logprob):.3f}{marker}")
                    winner_str = repr(self.tokenizer.decode([winner_tok])) if winner_tok is not None else "?"
                    print(f"  Pos {pos:3d}: " + "  |  ".join(parts) + f"  -> winner: {winner_str}")
            else:
                print("  No disputes.")

            if frozen_from is not None:
                note = f"frozen from {frozen_from - gen_start}"
            else:
                note = "not frozen"
            committed_n = len([p for p in range(gen_start, gen_end) if p in shared_x])
            print(f"\nFrontier: {committed_n}/{gen_end - gen_start} committed | {note}")

            # Full-fidelity preview: every position in [gen_start, gen_end), masks and
            # eos included and VISIBLE (skip_special_tokens=False) — a position missing
            # from shared_x is still masked, not "not there yet", and should print as
            # such rather than being silently skipped.
            full_ids = [shared_x.get(p, mask_id) for p in range(gen_start, gen_end)]
            preview = self.tokenizer.decode(full_ids, skip_special_tokens=False)
            print(f"\nCurrent generation:\n{preview}\n")

        return shared_x, new_frozen_from
