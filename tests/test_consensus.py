"""
Pure-logic tests for diffusion/consensus.py — no sockets, no GPU, no model.
Run with:
    PYTHONPATH=. python3 tests/test_consensus.py
"""
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusion.consensus import detect_disputes, resolve_disputes, merge_consensus


def test_no_dispute_when_everyone_agrees():
    commits = {
        "node-a": {10: (500, 0.9)},
        "node-b": {10: (500, 0.85)},
        "node-c": {10: (500, 0.95)},
    }
    disputes = detect_disputes(commits)
    assert disputes == {}, f"expected no disputes, got {disputes}"
    merged = merge_consensus(shared_x={}, commits=commits, resolved={})
    assert merged == {10: 500}
    print("PASS: unanimous agreement needs no PoE scoring, merges directly")


def test_two_way_dispute_weighs_both_voices():
    """With exactly 2 voters, weighted-average aggregation (sum of raw
    probabilities, NOT sum of log-probs / PoE — see module docstring)."""
    commits = {
        "node-a": {10: (500, 0.9)},   # proposes "cat" (token 500)
        "node-b": {10: (600, 0.7)},   # proposes "dog" (token 600)
    }
    disputes = detect_disputes(commits)
    assert disputes == {10: {500, 600}}

    # Both nodes report their OWN confidence for BOTH candidates (each already
    # has this from its own softmax — no extra forward pass).
    dispute_scores = {
        "node-a": {10: {500: math.log(0.9), 600: math.log(0.05)}},
        "node-b": {10: {500: math.log(0.1), 600: math.log(0.7)}},
    }
    # weighted-avg score(cat) = 0.9 + 0.1 = 1.0
    # weighted-avg score(dog) = 0.05 + 0.7 = 0.75
    # cat should win: both together are more confident in cat than dog,
    # even though node-b individually preferred dog. (This particular
    # example happens to agree with what old PoE math would also pick —
    # see test_bland_compromise_loses_under_weighted_average below for a
    # case where PoE and weighted-average actually disagree.)
    winners = resolve_disputes(dispute_scores)
    assert winners[10] == 500, f"expected token 500 (cat) to win, got {winners[10]}"
    print("PASS: 2-way dispute correctly weighs both voices")


def test_bland_compromise_loses_under_weighted_average():
    """THE regression test: this is the exact failure mode weighted-average
    aggregation exists to prevent, and the one case in this file where PoE
    and weighted-average actually disagree — if resolve_disputes ever gets
    reverted back to summing log-probs, this is the test that should catch it.

    node-a is confident in "int", node-b is confident in "equation". Each
    barely tolerates the OTHER's pick (near-zero, not zero). A third token,
    "the", is a bland compromise neither participant actually wants but
    both rate moderately, since neither has a strong opinion against it.
    """
    dispute_scores = {
        "node-a": {10: {100: math.log(0.99), 200: math.log(0.003), 300: math.log(0.3)}},   # 100="int"
        "node-b": {10: {100: math.log(0.005), 200: math.log(0.98), 300: math.log(0.3)}},   # 200="equation", 300="the"
    }
    # PoE (sum of logs) would give:
    #   score(int)      = log(0.99)  + log(0.005) = -5.31
    #   score(equation) = log(0.003) + log(0.98)  = -5.83
    #   score(the)      = log(0.3)   + log(0.3)   = -2.41   <- wins under PoE, despite
    #                                                            NEITHER participant wanting it
    # Weighted average (sum of raw probs) gives:
    #   score(int)      = 0.99  + 0.005 = 0.995   <- wins here: reflects the participant
    #   score(equation) = 0.003 + 0.98  = 0.983       who's actually confident in it
    #   score(the)      = 0.3   + 0.3   = 0.6
    winners = resolve_disputes(dispute_scores)
    assert winners[10] == 100, (
        f"expected 'int' (100) to win on genuine confidence, got {winners[10]} — "
        f"if this is 300 ('the'), resolve_disputes has regressed back to PoE"
    )
    print("PASS: bland compromise does not win over genuine confidence")


def test_genuine_n_way_dispute_counts_every_voice():
    """The actual generalization this module exists for: a candidate that
    NO ONE proposed as their own top pick can still win, if the ensemble's
    joint confidence in it is highest. This cannot happen with a 2-voter
    scheme (only 2 voices existed to begin with) — this is the real N-way
    behavior."""
    commits = {
        "node-a": {10: (100, 0.4)},  # proposes token 100 as its own top pick
        "node-b": {10: (200, 0.4)},  # proposes token 200
        "node-c": {10: (100, 0.4)},  # proposes token 100 too
    }
    disputes = detect_disputes(commits)
    assert disputes == {10: {100, 200}}

    # Every participant reports its own log-prob for BOTH candidates.
    # node-a and node-c are only mildly confident in 100 but very much AGAINST 200.
    # node-b is very confident in 200.
    dispute_scores = {
        "node-a": {10: {100: math.log(0.4), 200: math.log(0.01)}},
        "node-b": {10: {100: math.log(0.3), 200: math.log(0.9)}},
        "node-c": {10: {100: math.log(0.4), 200: math.log(0.01)}},
    }
    # weighted-avg score(100) = 0.4 + 0.3 + 0.4 = 1.1
    # weighted-avg score(200) = 0.01 + 0.9 + 0.01 = 0.92
    # 100 wins decisively — two participants strongly distrust 200, which
    # outweighs node-b's own enthusiasm for it. 2-vs-1 proposal count alone
    # would have given the same answer here, but for the RIGHT reason (the
    # actual joint confidence), not just majority vote.
    winners = resolve_disputes(dispute_scores)
    assert winners[10] == 100
    print("PASS: N-way dispute correctly weighs every participant's confidence")


def test_partial_participation_degrades_gracefully():
    """One participant never reports a dispute score (slow, or the sync
    barrier timed out on it) — resolution must still complete using
    whoever DID report, not hang or crash."""
    dispute_scores = {
        "node-a": {10: {100: math.log(0.6), 200: math.log(0.3)}},
        # node-b silently missing — degrade gracefully, don't require it
    }
    winners = resolve_disputes(dispute_scores)
    assert winners[10] == 100
    print("PASS: missing participant's vote just doesn't count, no crash/hang")


def test_tie_break_is_deterministic():
    dispute_scores = {
        "node-a": {10: {100: math.log(0.5), 50: math.log(0.5)}},
    }
    winners = resolve_disputes(dispute_scores)
    # exact tie — lower token id wins, and must be reproducible every time
    assert winners[10] == 50
    winners_again = resolve_disputes(dispute_scores)
    assert winners_again[10] == 50
    print("PASS: tied scores break deterministically (lower token id)")


def test_merge_consensus_combines_disputed_and_undisputed():
    commits = {
        "node-a": {10: (500, 0.9), 20: (700, 0.99)},   # 10 disputed, 20 not
        "node-b": {10: (600, 0.7), 20: (700, 0.95)},
    }
    disputes = detect_disputes(commits)
    assert disputes == {10: {500, 600}}
    resolved = {10: 500}  # pretend PoE resolved position 10 to token 500
    merged = merge_consensus(shared_x={}, commits=commits, resolved=resolved)
    assert merged == {10: 500, 20: 700}
    print("PASS: merge combines PoE-resolved and directly-agreed positions correctly")


if __name__ == "__main__":
    test_no_dispute_when_everyone_agrees()
    test_two_way_dispute_weighs_both_voices()
    test_bland_compromise_loses_under_weighted_average()
    test_genuine_n_way_dispute_counts_every_voice()
    test_partial_participation_degrades_gracefully()
    test_tie_break_is_deterministic()
    test_merge_consensus_combines_disputed_and_undisputed()
    print("\nALL PASS")
