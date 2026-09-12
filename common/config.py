"""
Swarm-wide configuration. One file, so relay addresses and timing constants
are never hardcoded in more than one place.

RELAYS is deliberately a list, not a single address — see node/transport.py
for why every node connects to all of them simultaneously.
"""

# Fill these in once your Exoscale relay boxes exist. Format: (host, pub_port, sub_port).
# pub_port/sub_port must match whatever --pub-port/--sub-port each relay.py was started with.
RELAYS = [
    # ("relay-1.example.com", 5555, 5556),
    # ("relay-2.example.com", 5555, 5556),
    # ("relay-3.example.com", 5555, 5556),
]

# For local testing without any real relay boxes yet (see tests/local_swarm_demo.py):
LOCAL_RELAYS = [
    ("127.0.0.1", 5555, 5556),
]

HEARTBEAT_INTERVAL = 2.0   # seconds between heartbeats
PEER_TIMEOUT       = 6.0   # seconds of silence before a peer is considered dead
                           # (3x HEARTBEAT_INTERVAL — tolerates one or two dropped
                           # heartbeats without a false-positive failure detection)

# --- generation-round settings ---
# Leadership during an active round is LOCKED at round start and does not follow
# the live alive-set (see node/round.py). If the round's leader goes silent for
# this long, the round is declared FAILED — not handed over mid-round. A fresh
# leader (per the live alive-set at that point) is free to start a new round
# once this one is torn down. This is deliberately simpler than seamless
# mid-round handover: a failed round costs you a restart, not correctness.
ROUND_LEADER_TIMEOUT = 8.0   # slightly above PEER_TIMEOUT: give a live-but-slow
                             # leader a bit more rope than a plain peer, since
                             # losing the leader is more disruptive than losing
                             # a follower

# Per-sync barrier, used for BOTH phases of a sync (collecting COMMITs, then
# collecting DISPUTE_SCOREs): how long to wait for every round participant's
# message before proceeding with whatever arrived. A stalled participant
# degrades that sync (fewer voices in the PoE sum, or its commits simply
# missing this round) rather than freezing the whole swarm waiting on it —
# see diffusion/consensus.py and node/generation.py.
SYNC_BARRIER_TIMEOUT = 2.0
