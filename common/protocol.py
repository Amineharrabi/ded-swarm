"""
Shared wire protocol for every message on the swarm.

Replaces the old JSON + base64 + hand-rolled make_msg() from the 2-node
websocket version: msgpack is binary (no base64 inflation), typed, and
fast to (de)serialize. Every message is a PUB-SUB frame pair:

    [topic_bytes, msgpack_bytes]

`topic` is always the sending node's ID, encoded as bytes — this is what
lets a subscriber filter by sender if it ever wants to (e.g. "only listen
to the current coordinator"), even though the default subscription is
everything (b"").

Message types so far only cover swarm membership (heartbeat). Generation
protocol messages (commits, sync requests, N-way consensus payloads) are
deliberately not designed yet — see docs/architecture.md for why that's
being sequenced after membership/leadership are proven solid.
"""
import time
import msgpack

# --- message types -----------------------------------------------------
HEARTBEAT = "heartbeat"   # "I'm alive", sent periodically by every node
HELLO     = "hello"       # sent once on join, so peers learn about a new node immediately
                           # instead of waiting up to HEARTBEAT_INTERVAL for the first heartbeat
BYE       = "bye"         # sent once on clean shutdown, so peers don't wait out the full
                           # PEER_TIMEOUT to notice a graceful departure


def encode(node_id: str, msg_type: str, payload: dict | None = None) -> tuple[bytes, bytes]:
    """Build the (topic, body) frame pair to publish."""
    envelope = {
        "type": msg_type,
        "node_id": node_id,
        "ts": time.time(),
        "payload": payload or {},
    }
    topic = node_id.encode("utf-8")
    body = msgpack.packb(envelope, use_bin_type=True)
    return topic, body


def decode(topic: bytes, body: bytes) -> dict:
    """Inverse of encode(). Returns the envelope dict."""
    envelope = msgpack.unpackb(body, raw=False)
    return envelope
