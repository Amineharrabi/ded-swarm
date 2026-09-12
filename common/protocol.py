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

# --- generation-round message types ------------------------------------
# GEN_START — sent once, by whoever is the round's initiator (must be the
#   sender's own currently-computed PeerTable leader — see node/round.py).
#   payload: {round_id: str, participants: list[str], prompt: str,
#             gen_length: int, block_length: int, steps: int, temperature: float}
GEN_START = "gen_start"

# GEN_DONE — best-effort teardown notice from the round leader once it
#   locally detects completion. Not load-bearing for correctness: every
#   node can independently detect completion from shared consensus state
#   (the same stop-boundary logic that existed in the 2-node version), so a
#   lost GEN_DONE just means a node notices completion a moment later via
#   its own state instead of the announcement.
#   payload: {round_id: str}
GEN_DONE = "gen_done"

# COMMIT — phase 1 of a sync: a participant's own newly-settled positions
#   this sync window (its own top pick per position + its own confidence in
#   that pick). This is the direct generalization of the old pairwise
#   sync_payload, just broadcast to everyone instead of sent to one peer.
#   payload: {round_id: str, block_idx: int, sync_idx: int,
#             positions: list[int], tokens: list[int], confidences: list[float]}
COMMIT = "commit"

# DISPUTE_SCORE — phase 2, only sent when phase 1 revealed positions where
#   different participants proposed different tokens. Every participant
#   looks up ITS OWN already-computed softmax distribution (no extra model
#   forward pass) for each contested token at each contested position, and
#   reports those log-probs. True PoE needs every active model's opinion on
#   every candidate, not just the proposers' — this is what supplies that.
#   payload: {round_id: str, block_idx: int, sync_idx: int,
#             scores: dict[str, dict[str, float]]}   # {position_str: {token_str: logprob}}
#   (dict keys are stringified for msgpack map-key compatibility)
DISPUTE_SCORE = "dispute_score"


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
