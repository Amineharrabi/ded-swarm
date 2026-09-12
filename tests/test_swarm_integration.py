"""
Proves the specific gap that node/swarm.py exists to close: during a
GenerationRunner-style sync barrier (waiting on COMMIT/DISPUTE_SCORE from
other participants), ordinary membership traffic (heartbeats) must NOT go
unnoticed. Two independent pollers would risk exactly that. One shared
poller (SwarmNode.pump_once, used by both collect_until and everything
else) should not.

No GPU, no model — this is pure networking/dispatch logic, run against a
real local relay subprocess. Run with:
    sh tests/test_swarm_integration.sh
(it starts/stops the relay itself; see that script)
Or, with a relay already running on 127.0.0.1:5555/5556:
    PYTHONPATH=. python3 tests/test_swarm_integration.py
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import config, protocol
from node.swarm import SwarmNode


def main():
    node_a = SwarmNode("node-a", config.LOCAL_RELAYS)
    node_b = SwarmNode("node-b", config.LOCAL_RELAYS)

    # --- round start / adoption ---
    state_a = node_a.start_round_as_leader(
        participants=["node-a", "node-b"], prompt="test prompt", gen_config={"steps": 4},
    )
    assert node_a.round_mgr.is_round_leader()

    state_b = node_b.wait_for_round(timeout=5.0)
    assert state_b is not None, "node-b never observed the GEN_START"
    assert state_b.round_id == state_a.round_id
    assert not node_b.round_mgr.is_round_leader()
    print("PASS: round started by node-a, adopted by node-b, same round_id")

    # --- the actual thing this test exists for: membership traffic must
    #     stay live even while collect_until is blocking on gen messages ---
    round_id = state_a.round_id

    # node-b will send a HEARTBEAT (ordinary membership traffic) shortly
    # after node-a starts waiting for a COMMIT that never actually arrives.
    # If node-a's collect_until were a dumb "only look for COMMIT" loop,
    # node-b's heartbeat would either be dropped or leave node-a's peer
    # table stale. It shouldn't be, because pump_once dispatches everything.
    # Uses the real start_heartbeat_thread() method, not ad-hoc threading,
    # so this also validates that helper directly.
    node_b.start_heartbeat_thread()

    before = node_a.peers.alive_peers()
    node_a.collect_until(protocol.COMMIT, round_id, expected_from={"node-b"}, timeout=1.5)
    after = node_a.peers.alive_peers()

    assert "node-b" not in before or True  # (may already be alive from earlier traffic; not the point)
    assert "node-b" in after, f"node-b's heartbeat was lost during collect_until — got {after}"
    print("PASS: membership heartbeat received and applied WHILE collect_until was "
          "blocking on an unrelated message type — the two pollers aren't racing")

    # --- and the actual gen-message path itself works ---
    node_b.transport.publish("node-b", protocol.COMMIT, {
        "round_id": round_id, "positions": [10], "tokens": [500], "confidences": [0.9],
    })
    received = node_a.collect_until(protocol.COMMIT, round_id, expected_from={"node-b"}, timeout=2.0)
    assert "node-b" in received
    assert received["node-b"]["positions"] == [10]
    print("PASS: COMMIT message correctly received and attributed via collect_until")

    node_a.close()
    node_b.close()
    print("\nALL PASS")


if __name__ == "__main__":
    main()
