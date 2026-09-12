"""
Pure-logic tests for node/round.py — no sockets, no GPU, no model. Run with:
    PYTHONPATH=. python3 tests/test_round.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from node.round import RoundManager


def make_gen_start_env(node_id, round_id, participants, ts):
    return {
        "node_id": node_id,
        "type": "gen_start",
        "ts": ts,
        "payload": {
            "round_id": round_id,
            "participants": participants,
            "prompt": "test prompt",
            "gen_config": {"steps": 256},
        },
    }


def test_leader_starts_and_locks_round():
    rm = RoundManager(self_id="node-a")
    r = rm.start_round(participants=["node-a", "node-b", "node-c"], prompt="p", gen_config={})
    assert rm.is_round_leader()
    assert rm.is_participant("node-b")
    assert not rm.is_participant("node-z")
    assert r.status == "active"
    print("PASS: leader starts and locks round")


def test_follower_adopts_gen_start():
    rm = RoundManager(self_id="node-b")
    env = make_gen_start_env("node-a", "node-a:abc123", ["node-a", "node-b", "node-c"], ts=100.0)
    rm.on_gen_start(env)
    assert not rm.is_round_leader()
    assert rm.active_round.leader_id == "node-a"
    assert rm.active_round.round_id == "node-a:abc123"
    print("PASS: follower adopts leader's round")


def test_dueling_gen_start_first_wins():
    rm = RoundManager(self_id="node-c")
    env1 = make_gen_start_env("node-a", "node-a:round1", ["node-a", "node-b", "node-c"], ts=100.0)
    env2 = make_gen_start_env("node-b", "node-b:round2", ["node-a", "node-b", "node-c"], ts=100.1)
    rm.on_gen_start(env1)
    rm.on_gen_start(env2)  # should be ignored — already in an active round
    assert rm.active_round.round_id == "node-a:round1"
    assert rm.active_round.leader_id == "node-a"
    print("PASS: dueling GEN_START — first observed wins, second ignored")


def test_leader_timeout_fails_round_not_handover():
    rm = RoundManager(self_id="node-b")
    env = make_gen_start_env("node-a", "node-a:abc123", ["node-a", "node-b"], ts=100.0)
    rm.on_gen_start(env)
    assert rm.active_round is not None

    # leader goes silent — well past ROUND_LEADER_TIMEOUT
    failed = rm.check_leader_timeout(now=100.0 + 999.0)
    assert failed is True
    assert rm.active_round is None  # round is gone, NOT handed to node-b
    print("PASS: leader timeout fails the round instead of silently handing it over")


def test_leader_activity_resets_timeout_clock():
    rm = RoundManager(self_id="node-b")
    env = make_gen_start_env("node-a", "node-a:abc123", ["node-a", "node-b"], ts=100.0)
    rm.on_gen_start(env)

    # leader is quiet for a while, but not past the timeout yet
    assert rm.check_leader_timeout(now=100.0 + 5.0) is False
    # fresh activity from the leader arrives (e.g. a COMMIT message)
    rm.note_leader_activity("node-a", ts=100.0 + 5.0)
    # clock is reset — should NOT fail even though we're now well past the
    # ORIGINAL start time, because last_leader_seen moved forward
    assert rm.check_leader_timeout(now=100.0 + 5.0 + 7.0) is False
    print("PASS: any traffic from the leader resets the timeout clock")


def test_leadership_ignores_rejoin_mid_round():
    """This is the actual policy decision from the conversation: a lower-ID
    node rejoining mid-round does NOT take leadership back. RoundManager has
    no code path that would even let that happen — this test documents that
    absence is deliberate, not an oversight."""
    rm = RoundManager(self_id="node-c")
    env = make_gen_start_env("node-b", "node-b:xyz", ["node-b", "node-c"], ts=100.0)
    rm.on_gen_start(env)
    assert rm.active_round.leader_id == "node-b"
    # "node-a" (lower ID than node-b) rejoins the swarm at the PeerTable layer —
    # RoundManager has no subscription to PeerTable's live leader() at all
    # while a round is active, so there is nothing here to even react to that.
    # The only way node-a becomes leader is via a NEW round after this one
    # ends (on_gen_done) or fails (check_leader_timeout).
    assert rm.active_round.leader_id == "node-b"
    print("PASS: mid-round rejoin has no code path to steal leadership")


if __name__ == "__main__":
    test_leader_starts_and_locks_round()
    test_follower_adopts_gen_start()
    test_dueling_gen_start_first_wins()
    test_leader_timeout_fails_round_not_handover()
    test_leader_activity_resets_timeout_clock()
    test_leadership_ignores_rejoin_mid_round()
    print("\nALL PASS")
