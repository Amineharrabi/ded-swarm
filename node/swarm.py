"""
SwarmNode: the piece that was missing. node/node.py's membership loop and
node/generation.py's GenerationRunner each independently wanted to be the
only thing calling transport.poll_recv() — fine in isolation (that's how
each was tested), but wrong together: two independent pollers racing on
the same SUB socket means whichever one happens to poll first "steals" a
message the other needed, silently. A GenerationRunner mid-sync-barrier
could miss a heartbeat; a membership loop could swallow a COMMIT meant for
an active round.

Fix: exactly ONE poller, here. Every incoming message passes through
`_dispatch`, which routes it to whichever layer cares — PeerTable,
RoundManager, or (during an active round's sync barrier) a queue that
GenerationRunner reads from via `collect_until`. Membership bookkeeping
(peer liveness, leader-activity resets) happens on EVERY message
regardless of type, so a long sync wait doesn't let heartbeats go stale
just because nothing was polling for them specifically.

`node/node.py`'s standalone Node class is left as-is — it's the tested,
working membership-only demo/harness (see tests/local_swarm_demo.sh) and
there's no reason to disturb it. SwarmNode is the real integration point a
notebook running actual generation should use instead.
"""
import time

from common import config, protocol
from node.transport import Transport
from node.peers import PeerTable
from node.round import RoundManager


class SwarmNode:
    def __init__(self, self_id: str, relays: list[tuple[str, int, int]]):
        self.id = self_id
        self.transport = Transport(relays)
        self.peers = PeerTable(self_id=self_id, timeout=config.PEER_TIMEOUT)
        self.round_mgr = RoundManager(self_id=self_id)

    def _dispatch(self, env: dict) -> dict:
        node_id, msg_type, ts = env["node_id"], env["type"], env["ts"]
        if node_id != self.id:
            # ANY traffic from the round's current leader counts as proof
            # it's alive — there's no separate "round heartbeat" message.
            self.round_mgr.note_leader_activity(node_id, ts)
            if msg_type in (protocol.HEARTBEAT, protocol.HELLO):
                self.peers.mark_alive(node_id, ts)
            elif msg_type == protocol.BYE:
                self.peers.mark_gone(node_id)
        if msg_type == protocol.GEN_START:
            self.round_mgr.on_gen_start(env)
        elif msg_type == protocol.GEN_DONE:
            self.round_mgr.on_gen_done(env)
        return env

    def pump_once(self, timeout_ms: int = 100) -> dict | None:
        """The ONE place transport.poll_recv() is ever called. Everything
        else — membership loop, generation sync barriers — goes through
        this, directly or via collect_until below."""
        env = self.transport.poll_recv(timeout_ms=timeout_ms)
        if env is not None:
            self._dispatch(env)
        return env

    def collect_until(self, msg_type: str, round_id: str, expected_from: set[str],
                       timeout: float) -> dict[str, dict]:
        """Used by GenerationRunner during a sync barrier. Still routes
        every message through _dispatch (so membership/round bookkeeping
        stays live during the wait), but only RETURNS the ones matching
        msg_type + round_id that this call is actually waiting for."""
        received: dict[str, dict] = {}
        deadline = time.time() + timeout
        remaining = set(expected_from)
        while remaining and time.time() < deadline:
            env = self.pump_once(timeout_ms=100)
            if env is None:
                continue
            if env["type"] != msg_type or env["payload"].get("round_id") != round_id:
                continue
            if env["node_id"] not in remaining and env["node_id"] not in received:
                continue  # not an expected sender for this round — no vote
            received[env["node_id"]] = env["payload"]
            remaining.discard(env["node_id"])
        return received

    def send_heartbeat(self):
        self.transport.publish(self.id, protocol.HEARTBEAT)

    def start_heartbeat_thread(self) -> "threading.Thread":
        """Runs send_heartbeat() every HEARTBEAT_INTERVAL seconds in the
        background for the life of the process. Safe to run alongside the
        main thread's pump_once/collect_until calls: this only ever touches
        the PUB socket, main-thread code only ever touches the SUB socket
        (via pump_once) — ZeroMQ sockets aren't thread-safe for concurrent
        use from multiple threads, but that only matters per-socket, and
        PUB/SUB here are two separate sockets. Needed because a notebook
        cell waiting on peer discovery, or a GenerationRunner mid-round,
        can't also be the one remembering to send heartbeats on a timer.
        """
        import threading

        def _loop():
            while True:
                self.send_heartbeat()
                time.sleep(config.HEARTBEAT_INTERVAL)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        return t

    def start_round_as_leader(self, participants: list[str], prompt: str, gen_config: dict):
        assert self.round_mgr.active_round is None, "a round is already active"
        state = self.round_mgr.start_round(participants, prompt, gen_config)
        self.transport.publish(self.id, protocol.GEN_START, {
            "round_id": state.round_id,
            "participants": state.participants,
            "prompt": prompt,
            "gen_config": gen_config,
        })
        return state

    def wait_for_round(self, timeout: float | None = None):
        """Block until a round becomes active (via an incoming GEN_START
        this node adopts) or timeout elapses. Returns the RoundState or
        None on timeout."""
        deadline = None if timeout is None else time.time() + timeout
        while self.round_mgr.active_round is None:
            if deadline is not None and time.time() > deadline:
                return None
            self.pump_once(timeout_ms=200)
        return self.round_mgr.active_round

    def close(self):
        self.transport.close()
