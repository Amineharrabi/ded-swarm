"""
Transport: wraps the PUB/SUB sockets every node needs.

The key design decision, worth stating explicitly: a node connects its PUB
socket to EVERY relay's pub-port, and its SUB socket to EVERY relay's
sub-port — not just one, with failover logic if it dies. ZeroMQ sockets
support connecting to multiple endpoints natively, so:

  - publish() sends to all connected relays at once.
  - if 2 of 3 relays are up, every message still gets through both of them.
  - a subscriber connected to 2 relays receives the SAME message twice
    (once forwarded by each relay).

That duplication is harmless as long as whatever handles incoming messages
is idempotent — which peer-liveness tracking naturally is (marking a peer
"alive at time T" twice does nothing extra the second time). So relay
redundancy costs nothing here: no failure detection, no reconnect logic,
no "which relay is primary" bookkeeping. If a relay dies, traffic keeps
flowing through the others and nothing in this file even notices.
"""
import zmq

from common import protocol


class Transport:
    def __init__(self, relays: list[tuple[str, int, int]]):
        if not relays:
            raise ValueError("need at least one relay address")
        self.ctx = zmq.Context.instance()

        self.pub = self.ctx.socket(zmq.PUB)
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")  # everything — filtering happens at the app layer

        for host, pub_port, sub_port in relays:
            # Nodes are PUBLISHERS, so they connect to each relay's pub-port
            # (the relay's XSUB/frontend side).
            self.pub.connect(f"tcp://{host}:{pub_port}")
            # Nodes are SUBSCRIBERS, so they connect to each relay's sub-port
            # (the relay's XPUB/backend side).
            self.sub.connect(f"tcp://{host}:{sub_port}")

        # Give ZeroMQ's async connect + subscription propagation a moment before
        # the first publish, or early messages can be silently dropped (classic
        # PUB-SUB "slow joiner" problem). Not needed for the SUB side to start
        # receiving later messages, only relevant right after connect.
        import time
        time.sleep(0.3)

    def publish(self, node_id: str, msg_type: str, payload: dict | None = None):
        topic, body = protocol.encode(node_id, msg_type, payload)
        self.pub.send_multipart([topic, body])

    def poll_recv(self, timeout_ms: int = 200) -> dict | None:
        """Non-blocking-ish receive: returns one decoded envelope, or None if
        nothing arrived within timeout_ms. Caller loops this."""
        if self.sub.poll(timeout_ms):
            topic, body = self.sub.recv_multipart()
            return protocol.decode(topic, body)
        return None

    def close(self):
        self.pub.close()
        self.sub.close()
