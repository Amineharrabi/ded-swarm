"""
The generation loop: wires the round-locking + consensus + model-step
layers together into one running generation round, on top of a SwarmNode
(node/swarm.py) rather than talking to the transport directly — see
swarm.py's docstring for why that split exists (one poller, not two racing
ones).

HONEST STATUS: this file has NOT been run against a real model. Everything
below it in this project's dependency chain (round-locking, consensus,
stop-detection) is pure Python logic and has real passing tests with
synthetic data. This file is where those pieces meet an actual GPU-backed
model, tensors, and real network timing — none of which this sandbox can
exercise. Treat this as a structurally-complete first draft to run and
iterate on in Colab, not as verified code.

Sync-window accounting, briefly: each node accumulates its own newly
transferred positions across every step since the last sync (not just the
single most recent step — a sync window can span several steps), broadcasts
that accumulated set as one COMMIT, then resets the accumulator. This
mirrors what the old 2-node version did by recomputing a `relevant` union
at sync time; here it's tracked incrementally instead of recomputed, since
there's no longer a single peer to diff against.
"""
from dataclasses import dataclass

from common import config, protocol
from node.swarm import SwarmNode
from diffusion.local_step import local_step, StepConfig
from diffusion.consensus import detect_disputes, resolve_disputes, merge_consensus
from diffusion.stop_detection import StopDetector, StopConfig


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
        """Runs one full generation round for this node, as a participant.
        Assumes swarm.round_mgr.active_round is already set — either via
        swarm.start_round_as_leader(...) if we're the leader, or
        swarm.wait_for_round(...) if we're not."""
        round_state = self.swarm.round_mgr.active_round
        round_id = round_state.round_id
        participants = set(round_state.participants)
        other_participants = participants - {self.id}

        gen_start = prompt_ids.shape[1]
        gen_end = gen_start + gen_cfg.gen_length
        num_blocks = gen_cfg.gen_length // gen_cfg.block_length
        steps_per_block = gen_cfg.steps // num_blocks

        import torch
        x = torch.full((1, gen_end), gen_cfg.step_cfg.mask_id, dtype=torch.long, device=self.device)
        x[:, :gen_start] = prompt_ids

        # shared_x tracks the converged consensus as {absolute_position: token},
        # separately from the raw tensor `x` — makes stop-detection and
        # consensus merging position-indexed and framework-agnostic, and
        # keeps `x` as the thing actually fed back into the model each step.
        shared_x: dict[int, int] = {p: prompt_ids[0, p].item() for p in range(gen_start)}
        stop_detector = StopDetector(gen_cfg.stop_cfg)
        frozen_from = None

        for block_idx in range(num_blocks):
            block_start = gen_start + block_idx * gen_cfg.block_length
            block_end = gen_start + (block_idx + 1) * gen_cfg.block_length

            if frozen_from is not None and block_start >= frozen_from:
                break

            age = torch.zeros_like(x)
            pending_commits: dict[int, tuple[int, float]] = {}  # this node's own, since last sync

            for step_idx in range(steps_per_block):
                if self.swarm.round_mgr.check_leader_timeout():
                    print(f"[{self.id}] round {round_id} failed mid-block — aborting generation")
                    return None

                steps_remaining = steps_per_block - step_idx
                x, age, x0, p, conf_new, transfer, remask = local_step(
                    self.model, x, age, steps_remaining, block_start, block_end,
                    steps_per_block, gen_cfg.step_cfg, frozen_from=frozen_from,
                )

                for pos in torch.where(transfer[0])[0].tolist():
                    pending_commits[pos] = (x0[0, pos].item(), conf_new[0, pos].item())

                is_last_step = (step_idx == steps_per_block - 1)
                if (step_idx + 1) % gen_cfg.sync_every == 0 or is_last_step:
                    shared_x, frozen_from = self._do_sync(
                        round_id, participants, other_participants,
                        pending_commits, p, shared_x, gen_start, gen_end,
                        stop_detector, frozen_from,
                    )
                    pending_commits = {}
                    # Reconcile this node's own tensor with the converged
                    # consensus for everything synced so far, same as the
                    # old x_a = x_consensus.clone() step.
                    for pos, token in shared_x.items():
                        if block_start <= pos < block_end:
                            x[0, pos] = token
                            age[0, pos] = 0

        if self.swarm.round_mgr.is_round_leader():
            self.swarm.transport.publish(self.id, protocol.GEN_DONE, {"round_id": round_id})
        return shared_x

    def _do_sync(self, round_id, participants, other_participants,
                 my_commits, my_p, shared_x, gen_start, gen_end,
                 stop_detector, frozen_from):
        # --- phase 1: exchange COMMITs ---
        positions = list(my_commits.keys())
        self.swarm.transport.publish(self.id, protocol.COMMIT, {
            "round_id": round_id,
            "positions": positions,
            "tokens": [my_commits[p][0] for p in positions],
            "confidences": [my_commits[p][1] for p in positions],
        })
        others_raw = self.swarm.collect_until(protocol.COMMIT, round_id, other_participants,
                                               config.SYNC_BARRIER_TIMEOUT)

        all_commits = {self.id: my_commits}
        for node_id, payload in others_raw.items():
            all_commits[node_id] = {
                pos: (tok, conf) for pos, tok, conf in
                zip(payload["positions"], payload["tokens"], payload["confidences"])
            }

        disputes = detect_disputes(all_commits)

        # --- phase 2: exchange DISPUTE_SCOREs, only if needed ---
        resolved: dict[int, int] = {}
        if disputes:
            my_scores = {
                pos: {tok: my_p[0, pos, tok].clamp_min(1e-12).log().item() for tok in tokens}
                for pos, tokens in disputes.items()
            }
            self.swarm.transport.publish(self.id, protocol.DISPUTE_SCORE, {
                "round_id": round_id,
                "scores": {str(pos): {str(t): s for t, s in toks.items()}
                           for pos, toks in my_scores.items()},
            })
            others_scores_raw = self.swarm.collect_until(
                protocol.DISPUTE_SCORE, round_id, other_participants, config.SYNC_BARRIER_TIMEOUT)

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
        return shared_x, new_frozen_from
