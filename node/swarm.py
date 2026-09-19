"""
SwarmNode: the single socket poller everything else shares, now with a
durable gen-message inbox.

Why the inbox exists: check_leader_timeout() only gets fresh data when a
message is actually read off the socket, which used to only happen inside
collect_until — i.e. only at sync points. Between syncs (SYNC_EVERY steps
of local_step, ~14s at this model's per-step cost), nothing was polling at
all, so a follower could time out a perfectly healthy leader simply because
nobody read the heartbeats sitting in the socket buffer. The fix
(node/generation.py now polls every step, not just at syncs) creates a new
risk on its own: a COMMIT/DISPUTE_SCORE that arrives early — while the
receiver is still mid-block, not yet at ITS OWN collect_until call for that
sync — would get read and silently discarded by an earlier incidental poll,
and collect_until would then wrongly conclude that peer never reported.

The inbox fixes that: every pump_once() files gen-protocol messages into a
durable per-(msg_type, round_id, sync_seq) bucket, and collect_until checks
that bucket FIRST before waiting for new arrivals. sync_seq (not just
round_id) is required in the key because one round has many syncs — without
it, a stale message from an earlier sync could satisfy a later sync's wait.
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
        self._gen_inbox: dict[tuple, dict[str, dict]] = {}

    def _dispatch(self, env: dict) -> dict:
        node_id, msg_type, ts = env["node_id"], env["type"], env["ts"]
        if node_id != self.id:
            self.round_mgr.note_leader_activity(node_id, ts)
            if msg_type in (protocol.HEARTBEAT, protocol.HELLO):
                self.peers.mark_alive(node_id, ts)
            elif msg_type == protocol.BYE:
                self.peers.mark_gone(node_id)
        if msg_type == protocol.GEN_START:
            self.round_mgr.on_gen_start(env)
        elif msg_type == protocol.GEN_DONE:
            self.round_mgr.on_gen_done(env)
        elif msg_type in (protocol.COMMIT, protocol.DISPUTE_SCORE):
            payload = env["payload"]
            key = (msg_type, payload.get("round_id"), payload.get("sync_seq"))
            self._gen_inbox.setdefault(key, {})[node_id] = payload
        return env

    def pump_once(self, timeout_ms: int = 100) -> dict | None:
        env = self.transport.poll_recv(timeout_ms=timeout_ms)
        if env is not None:
            self._dispatch(env)
        return env

    def collect_until(self, msg_type: str, round_id: str, sync_seq: int,
                       expected_from: set[str], timeout: float) -> dict[str, dict]:
        key = (msg_type, round_id, sync_seq)
        received = {k: v for k, v in self._gen_inbox.get(key, {}).items()
                    if k in expected_from}
        remaining = set(expected_from) - set(received.keys())
        deadline = time.time() + timeout
        while remaining and time.time() < deadline:
            env = self.pump_once(timeout_ms=100)
            if env is None:
                continue
            payload = env["payload"]
            if (env["type"] != msg_type or payload.get("round_id") != round_id
                    or payload.get("sync_seq") != sync_seq):
                continue
            if env["node_id"] not in remaining and env["node_id"] not in received:
                continue
            received[env["node_id"]] = payload
            remaining.discard(env["node_id"])
        self._gen_inbox.pop(key, None)
        return received

    def send_heartbeat(self):
        self.transport.publish(self.id, protocol.HEARTBEAT)

    def start_heartbeat_thread(self):
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
        deadline = None if timeout is None else time.time() + timeout
        while self.round_mgr.active_round is None:
            if deadline is not None and time.time() > deadline:
                return None
            self.pump_once(timeout_ms=200)
        return self.round_mgr.active_round

    def close(self):
        self.transport.close()
