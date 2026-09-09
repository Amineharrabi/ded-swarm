"""
A runnable swarm node. Proves the membership/leadership layer end to end:
join, heartbeat, detect peers, elect a leader, detect failure, re-elect.

This does NOT yet do any diffusion work — no model, no generation. It's
deliberately just the swarm skeleton, so it can be tested and trusted on
its own (see tests/local_swarm_demo.py) before the much heavier diffusion
logic gets layered on top of it.

Run:
    python -m node.node --id node-a --relays local
    python -m node.node --id node-b --relays local
    python -m node.node --id node-c --relays local
Then Ctrl+C one of them and watch the others re-elect within ~PEER_TIMEOUT
seconds — see leader print change in the surviving nodes' output.
"""
import argparse
import signal
import sys
import time

from common import config, protocol
from node.transport import Transport
from node.peers import PeerTable


class Node:
    def __init__(self, node_id: str, relays: list[tuple[str, int, int]]):
        self.id = node_id
        self.transport = Transport(relays)
        self.peers = PeerTable(self_id=node_id, timeout=config.PEER_TIMEOUT)
        self._running = True

    def _handle_envelope(self, env: dict):
        node_id, msg_type, ts = env["node_id"], env["type"], env["ts"]
        if node_id == self.id:
            return  # ignore our own broadcasts
        if msg_type in (protocol.HEARTBEAT, protocol.HELLO):
            self.peers.mark_alive(node_id, ts)
        elif msg_type == protocol.BYE:
            self.peers.mark_gone(node_id)
            print(f"[{self.id}] {node_id} left cleanly")

    def _announce_leader_if_changed(self):
        changed = self.peers.leader_changed()
        if changed is not None:
            role = "LEADER" if changed == self.id else "follower"
            print(f"[{self.id}] leader is now: {changed}  (I am {role})  "
                  f"alive={self.peers.alive_peers()}")

    def run(self):
        self.transport.publish(self.id, protocol.HELLO)
        print(f"[{self.id}] joined swarm")
        last_heartbeat = 0.0

        def handle_sigterm(signum, frame):
            self._running = False
        signal.signal(signal.SIGTERM, handle_sigterm)

        try:
            while self._running:
                now = time.time()
                if now - last_heartbeat >= config.HEARTBEAT_INTERVAL:
                    self.transport.publish(self.id, protocol.HEARTBEAT)
                    last_heartbeat = now

                env = self.transport.poll_recv(timeout_ms=200)
                if env is not None:
                    self._handle_envelope(env)

                self._announce_leader_if_changed()
        except KeyboardInterrupt:
            pass
        finally:
            self.transport.publish(self.id, protocol.BYE)
            time.sleep(0.1)  # give the BYE a moment to actually leave the socket
            self.transport.close()
            print(f"[{self.id}] left swarm")


def _resolve_relays(spec: str) -> list[tuple[str, int, int]]:
    if spec == "local":
        return config.LOCAL_RELAYS
    if spec == "prod":
        if not config.RELAYS:
            print("common/config.py RELAYS is empty — fill in your Exoscale "
                  "relay addresses before using --relays prod", file=sys.stderr)
            sys.exit(1)
        return config.RELAYS
    raise ValueError(f"unknown --relays value: {spec!r} (use 'local' or 'prod')")


def main():
    ap = argparse.ArgumentParser(description="Run a DED swarm node")
    ap.add_argument("--id", required=True, help="unique node id")
    ap.add_argument("--relays", default="local", choices=["local", "prod"])
    args = ap.parse_args()

    node = Node(args.id, _resolve_relays(args.relays))
    node.run()


if __name__ == "__main__":
    main()
