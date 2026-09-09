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
