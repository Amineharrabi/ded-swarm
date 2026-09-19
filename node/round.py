import time
import uuid
from dataclasses import dataclass, field

from common import config


@dataclass
class RoundState:
    round_id: str
    leader_id: str
    participants: list
    prompt: str
    gen_config: dict
    started_at: float
    last_leader_seen: float = field(default_factory=time.time)
    status: str = "active"


class RoundManager:
    def __init__(self, self_id: str):
        self.self_id = self_id
        self.active_round = None

    def start_round(self, participants, prompt, gen_config):
        round_id = f"{self.self_id}:{uuid.uuid4().hex[:8]}"
        self.active_round = RoundState(
            round_id=round_id, leader_id=self.self_id,
            participants=sorted(participants), prompt=prompt,
            gen_config=gen_config, started_at=time.time(),
        )
        return self.active_round

    def on_gen_start(self, env, adopted_at=None):
        if self.active_round is not None and self.active_round.status == "active":
            if env["payload"]["round_id"] != self.active_round.round_id:
                print(f"[round] ignoring competing GEN_START for "
                      f"{env['payload']['round_id']} — already in round "
                      f"{self.active_round.round_id}")
            return
        payload = env["payload"]
        now = adopted_at if adopted_at is not None else time.time()
        self.active_round = RoundState(
            round_id=payload["round_id"], leader_id=env["node_id"],
            participants=sorted(payload["participants"]), prompt=payload["prompt"],
            gen_config=payload["gen_config"], started_at=now,
            last_leader_seen=now,  # receive time, not the message's own embedded
                                    # ts — same reasoning as note_leader_activity
        )

    def on_gen_done(self, env):
        if self.active_round is not None and env["payload"].get("round_id") == self.active_round.round_id:
            self.active_round.status = "done"
            self.active_round = None

    def note_leader_activity(self, node_id, ts):
        if (self.active_round is not None and self.active_round.status == "active"
                and node_id == self.active_round.leader_id):
            self.active_round.last_leader_seen = ts

    def check_leader_timeout(self, now=None):
        now = now if now is not None else time.time()
        if self.active_round is None or self.active_round.status != "active":
            return False
        # A node is always alive to itself — it doesn't need its own
        # messages to round-trip through _dispatch's self-exclusion to
        # prove that. Without this, a solo leader (or any leader with no
        # OTHER participant currently sending it round-scoped traffic)
        # fails its own round exactly ROUND_LEADER_TIMEOUT seconds after
        # start, having "gone silent" on itself. Same root cause as the
        # earlier PeerTable split-brain bug: self-liveness was routed
        # through the same timeout machinery meant for external peers.
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

    def is_round_leader(self):
        return self.active_round is not None and self.active_round.leader_id == self.self_id

    def is_participant(self, node_id):
        return self.active_round is not None and node_id in self.active_round.participants
