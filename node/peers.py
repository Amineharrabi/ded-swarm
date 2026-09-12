"""
Peer liveness tracking and leader election.

Every node runs this independently, fed by the same heartbeat stream (via
the relay). There is no election protocol, no voting, no message exchange
dedicated to "who's the leader" — every node just applies the same
deterministic rule to the same shared view of who's currently alive, so
everyone converges on the same answer without needing to agree on anything
explicitly.

Rule (v1, deliberately simple): the leader is the alive node with the
lowest node_id, sorted lexicographically. Simple, deterministic, and easy
to reason about. It is NOT sticky — if a lower-ID node rejoins, leadership
moves back to it immediately. That's fine for now (no generation round is
"in flight" yet at this layer), but is a real design question to revisit
once leadership changing mid-generation has a cost — see docs/architecture.md.
"""
import time


class PeerTable:
    def __init__(self, self_id: str, timeout: float):
        self.self_id = self_id
        self.timeout = timeout
        self.last_seen: dict[str, float] = {}
        self._prev_leader: str | None = None

    def mark_alive(self, node_id: str, ts: float | None = None):
        self.last_seen[node_id] = ts if ts is not None else time.time()

    def mark_gone(self, node_id: str):
        """Immediate removal on a clean BYE — don't make peers wait out the
        full timeout to notice a graceful departure."""
        self.last_seen.pop(node_id, None)

    def alive_peers(self, now: float | None = None) -> list[str]:
        now = now if now is not None else time.time()
        alive = {
            node_id for node_id, seen in self.last_seen.items()
            if now - seen <= self.timeout
        }
        # A node always knows it's alive — it doesn't need its own heartbeat
        # to round-trip back through the relay to prove that to itself. Without
        # this, self_id's entry (seeded once at construction) ages out after
        # `timeout` seconds and a node silently drops itself from its own
        # alive set — which produces exactly the split-brain bug this fixes:
        # two survivors each thinking the OTHER one is the sole survivor.
        alive.add(self.self_id)
        return sorted(alive)

    def leader(self, now: float | None = None) -> str | None:
        alive = self.alive_peers(now)
        return alive[0] if alive else None

    def is_leader(self, now: float | None = None) -> bool:
        return self.leader(now) == self.self_id

    def leader_changed(self, now: float | None = None) -> str | None:
        """Call once per loop iteration. Returns the new leader if it changed
        since the last call, else None — so the caller can log/react only on
        actual transitions instead of every tick."""
        current = self.leader(now)
        if current != self._prev_leader:
            self._prev_leader = current
            return current
        return None
