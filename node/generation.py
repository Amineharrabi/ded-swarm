"""
node/generation.py — with the leader-liveness fix (continuous light polling
between syncs, not just at sync points) and its companion fix (a durable,
sync_seq-keyed inbox in SwarmNode so that polling can't lose a message
meant for a later collect_until call). See node/swarm.py for the inbox
side of this.
"""
from dataclasses import dataclass

from common import config, protocol
from node.swarm import SwarmNode
from diffusion.local_step import local_step, StepConfig
from diffusion.consensus import detect_disputes, resolve_disputes, merge_consensus
from diffusion.stop_detection import StopDetector, StopConfig

VERBOSE = True
DEBUG_EVERY = 8


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

        import torch
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
                x, age, x0, p, conf_new, transfer, remask = local_step(
                    self.model, x, age, steps_remaining, block_start, block_end,
                    steps_per_block, gen_cfg.step_cfg, frozen_from=frozen_from,
                )

                for pos in torch.where(transfer[0])[0].tolist():
                    pending_commits[pos] = (x0[0, pos].item(), conf_new[0, pos].item())

                if VERBOSE and (step_idx + 1) % DEBUG_EVERY == 0:
                    n_masked = int((x[0, block_start:block_end] == gen_cfg.step_cfg.mask_id).sum())
                    print(f"[{self.id}] block {block_idx} step {step_idx + 1}/{steps_per_block}: "
                          f"{n_masked} masks remain in block")

                is_last_step = (step_idx == steps_per_block - 1)
                if (step_idx + 1) % gen_cfg.sync_every == 0 or is_last_step:
                    shared_x, frozen_from = self._do_sync(
                        round_id, sync_seq, participants, other_participants,
                        pending_commits, p, shared_x, gen_start, gen_end,
                        stop_detector, frozen_from, block_idx, step_idx,
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
                 stop_detector, frozen_from, block_idx, step_idx):
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
            n_commits = sum(len(c) for c in all_commits.values())
            print(f"[{self.id}] SYNC block {block_idx} step {step_idx + 1} (sync_seq={sync_seq}): "
                  f"{len(all_commits)} reporter(s), {n_commits} total commits, "
                  f"{len(disputes)} disputed position(s)")
            preview_tokens = [shared_x.get(p) for p in range(gen_start, gen_end) if p in shared_x]
            if preview_tokens:
                preview = self.tokenizer.decode(preview_tokens, skip_special_tokens=True)
                print(f"[{self.id}] current generation:\n{preview}\n")

        return shared_x, new_frozen_from
