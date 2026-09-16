"""
Round-locked leadership: the answer to "what happens to leadership during an
active generation round."

Decision (explicit, not implicit): leadership only changes BETWEEN rounds,
never during one, regardless of what the live alive-set says. A round's
leader and participant list are frozen the instant the round starts. If a
lower-ID node rejoins mid-round, it does NOT take over — it just becomes
eligible to lead the NEXT round.

This means a round's leader dying mid-round is NOT a seamless handover.
It's a round FAILURE: followers detect the silence via ROUND_LEADER_TIMEOUT,
locally tear down the round, and go back to "no active round." Only then
does the live PeerTable-computed leader (which may now be a different node,
since the dead leader is no longer in the alive-set) get to start a fresh
round with a new round_id and a re-frozen participant list. A failed round
costs a restart, not a subtle correctness bug from half-applied state.

Known simplification, stated rather than hidden: if two nodes briefly both
believe they're the live leader (e.g. right at a leadership transition) and
both send GEN_START, a node that already has an active round ignores any
OTHER round's GEN_START until its current round ends or times out —
first-one-observed wins for that ambiguous window. This is not
Raft-grade (no term numbers, no leader-id tie-break), and is a real
candidate for hardening later if dueling initiators turn out to happen in
practice.
"""
import time
import uuid
from dataclasses import dataclass, field

from common import config


@dataclass
class RoundState:
    round_id: str
    leader_id: str
    participants: list[str]
    prompt: str
    gen_config: dict
    started_at: float
    last_leader_seen: float = field(default_factory=time.time)
    status: str = "active"   # "active" | "done" | "failed"


class RoundManager:
    def __init__(self, self_id: str):
        self.self_id = self_id
        self.active_round: RoundState | None = None

    # --- starting a round (call only if you ARE the current live leader) ---
    def start_round(self, participants: list[str], prompt: str, gen_config: dict) -> RoundState:
        round_id = f"{self.self_id}:{uuid.uuid4().hex[:8]}"
        self.active_round = RoundState(
            round_id=round_id,
            leader_id=self.self_id,
            participants=sorted(participants),
            prompt=prompt,
            gen_config=gen_config,
            started_at=time.time(),
        )
        return self.active_round

    # --- reacting to messages ---
    def on_gen_start(self, env: dict):
        if self.active_round is not None and self.active_round.status == "active":
            if env["payload"]["round_id"] != self.active_round.round_id:
                # Dueling-initiator edge case — see module docstring. First
                # active round observed wins; log and ignore the second.
                print(f"[round] ignoring competing GEN_START for "
                      f"{env['payload']['round_id']} — already in round "
                      f"{self.active_round.round_id}")
            return
        payload = env["payload"]
        self.active_round = RoundState(
            round_id=payload["round_id"],
            leader_id=env["node_id"],
            participants=sorted(payload["participants"]),
            prompt=payload["prompt"],
            gen_config=payload["gen_config"],
            started_at=env["ts"],
            last_leader_seen=env["ts"],
        )

    def on_gen_done(self, env: dict):
        if self.active_round is not None and env["payload"].get("round_id") == self.active_round.round_id:
            self.active_round.status = "done"
            self.active_round = None

    def note_leader_activity(self, node_id: str, ts: float):
        """Call this on ANY message from node_id — heartbeat, commit, whatever.
        Any traffic from the round leader counts as proof it's still alive;
        there's no separate 'round heartbeat' message type."""
        if (self.active_round is not None
                and self.active_round.status == "active"
                and node_id == self.active_round.leader_id):
            self.active_round.last_leader_seen = ts

    def check_leader_timeout(self, now: float | None = None) -> bool:
        """Returns True if this call just failed the active round. Poll this
        every loop iteration — see node/round.py's module docstring for why a
        timeout means failure, not handover."""
        now = now if now is not None else time.time()
        if self.active_round is None or self.active_round.status != "active":
            return False

        if self.active_round.leader_id == self.self_id:
            return False
        if now - self.active_round.last_leader_seen > config.ROUND_LEADER_TIMEOUT:
            print(f"[round] leader {self.active_round.leader_id} silent for "
                  f">{config.ROUND_LEADER_TIMEOUT}s — round "
                  f"{self.active_round.round_id} FAILED, not handed over")
            self.active_round.status = "failed"
            self.active_round = None
            return True
        return False

    def is_round_leader(self) -> bool:
        return self.active_round is not None and self.active_round.leader_id == self.self_id

    def is_participant(self, node_id: str) -> bool:
        return self.active_round is not None and node_id in self.active_round.participants
