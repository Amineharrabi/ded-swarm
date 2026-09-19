"""
N-way consensus: generalizing the old pairwise PoE arbitration
(`score_x = log P_A(x) + log P_B(x)`) to N participants.

## The policy decision this makes, and why

Two options were on the table: "leader adjudicates" (one designated node
computes the answer, everyone else adopts it) vs. "everyone computes the
same answer independently" (no adjudicator at all). This module implements
the second, for a reason that follows directly from wanting a decentralized
architecture: a leader-adjudicates consensus would make the *actual
decision-making* — which token wins at a contested position — depend on
one node staying alive and responsive, even though membership and
leadership elsewhere in this project are explicitly NOT built that way.
That would be centralizing the brain while claiming to only centralize the
plumbing.

Independent computation avoids that: every participant runs the exact same
pure function (`resolve_disputes` below) over the exact same input data, so
they all arrive at the same answer without anyone being "in charge" of
producing it. This mirrors exactly how leader election already works in
this codebase (node/peers.py) — deterministic function over shared state,
not a protocol that negotiates an answer.

## Why disputes need a second phase, and why it doesn't cost extra GPU work

Faithful arbitration means every ALIVE model's opinion on a candidate
counts, not just the opinions of whichever two participants happened to
propose different tokens. With N participants, resolving a dispute between
"cat" and "dog" needs every participant's probability for BOTH "cat" and
"dog" — not just the two proposers'.

IMPORTANT — this is a WEIGHTED-AVERAGE aggregation, not PoE, and that
distinction is load-bearing, not stylistic. An earlier version of this
module (and the docstring you may be reading a stale copy of) summed
log-probabilities across participants — i.e. argmax of the PRODUCT of
their probabilities, classic Product-of-Experts. That was already tried
and deliberately abandoned in the 2-node version (DED_NodeB, the
`W_A, W_B = 0.5, 0.5; score = W_A*p_a + W_B*p_b` change) for a concrete
reason: PoE punishes genuine disagreement multiplicatively. If two
participants each strongly prefer a DIFFERENT token, each assigns the
other's pick a near-zero probability — under PoE that near-zero factor
drags BOTH real candidates' scores toward zero, and a third, blander token
that neither participant actually wanted but that both rate moderately
(because neither actively objects to it) can out-score both genuine
picks and win by default. A weighted SUM of raw probabilities doesn't
have this failure mode: one participant's near-zero term just gets
outweighed by the other's real confidence instead of nullifying it. See
test_consensus.py's `test_bland_compromise_loses_under_weighted_average`
for a worked example of exactly this failure and why the fix matters.

The good news: nobody needs an extra forward pass to get this. Every
participant already computed a full (seq_len, vocab) softmax distribution
during its own local_step this cycle — checking its own log-prob for some
OTHER participant's proposed token is just an index lookup into a tensor
it already has in memory. That's why the wire protocol splits into two
phases (see common/protocol.py's COMMIT and DISPUTE_SCORE): phase 1 reveals
which positions are even disputed, phase 2 asks specifically about those
(and only those) positions — cheap, and only paid when there's an actual
disagreement to resolve. Per the project's own benchmark history at N=2,
real disagreement has been rare (single digits per generation); phase 2
should stay a rare, small exchange even as N grows, since disagreement
rate is a property of how different the models are, not how many there are.

## The graceful-degradation choice, stated explicitly

`resolve_disputes` sums over WHOEVER reported a score for a position — not
"every participant, or we wait forever." A round's frozen participant list
means a stalled or slow node can't be silently dropped from the round (see
node/round.py), but consensus computation would be a bad place to enforce
"wait for literally everyone," because it would let a single slow node
freeze every contested position for the whole swarm. The generation loop
(node/generation.py) enforces a bounded wait (config.SYNC_BARRIER_TIMEOUT) and
then calls this module with whatever arrived — a slow participant's vote
just doesn't count for that one sync, rather than blocking everyone.
"""


def detect_disputes(commits: dict[str, dict[int, tuple[int, float]]]) -> dict[int, set[int]]:
    """commits: {node_id: {position: (proposed_token, own_confidence)}}

    Returns {position: {distinct tokens proposed at that position}} — only
    for positions where more than one distinct token was proposed. A
    position every proposer agrees on isn't a dispute; it's adopted
    directly by merge_consensus without ever needing a PoE score.
    """
    by_position: dict[int, set[int]] = {}
    for node_id, positions in commits.items():
        for pos, (token, _conf) in positions.items():
            by_position.setdefault(pos, set()).add(token)
    return {pos: tokens for pos, tokens in by_position.items() if len(tokens) > 1}


def resolve_disputes(
    dispute_scores: dict[str, dict[int, dict[int, float]]],
) -> dict[int, int]:
    """dispute_scores: {node_id: {position: {candidate_token: log_prob}}}

    Values arrive as log-probs (cheap to compute from each participant's own
    softmax, and small over the wire), but aggregation happens in PROBABILITY
    space — sum of exp(log_prob) per candidate, i.e. a weighted average
    (uniform 1/N weight; the constant factor doesn't affect the argmax below)
    — not a sum of log-probs. Summing logs would be Product-of-Experts, which
    this module deliberately does NOT do — see the module docstring for why.

    Picks the highest-scoring candidate per position. Ties broken by lowest
    token id — arbitrary but deterministic, which is the only property that
    matters here: every participant computing this must land on the same
    winner.

    Returns {position: winning_token}.
    """
    import math
    totals: dict[int, dict[int, float]] = {}
    for node_id, positions in dispute_scores.items():
        for pos, candidates in positions.items():
            bucket = totals.setdefault(pos, {})
            for token, logprob in candidates.items():
                bucket[token] = bucket.get(token, 0.0) + math.exp(logprob)

    winners = {}
    for pos, candidates in totals.items():
        # max() with a tuple key: (score, -token) so higher score wins, and
        # among equal scores the LOWER token id wins (deterministic tie-break).
        best_token = max(candidates.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        winners[pos] = best_token
    return winners


def merge_consensus(
    shared_x: dict[int, int],
    commits: dict[str, dict[int, tuple[int, float]]],
    resolved: dict[int, int],
) -> dict[int, int]:
    """Applies one sync's commits to the shared consensus state.

    - Undisputed positions (single distinct proposal across all commits):
      adopted directly, no PoE needed.
    - Disputed positions: adopted from `resolved` (the output of
      resolve_disputes).

    Returns a NEW dict — callers should treat shared_x as immutable and use
    the return value, same convention as the rest of this codebase
    (node/peers.py, etc. never mutate what a caller holds a reference to).
    """
    disputed_positions = set(resolved.keys())
    updated = dict(shared_x)

    proposals_by_position: dict[int, set[int]] = {}
    for node_id, positions in commits.items():
        for pos, (token, _conf) in positions.items():
            proposals_by_position.setdefault(pos, set()).add(token)

    for pos, tokens in proposals_by_position.items():
        if pos in disputed_positions:
            updated[pos] = resolved[pos]
        else:
            # len(tokens) == 1 by construction (detect_disputes already
            # separated out anything with >1 distinct proposal)
            updated[pos] = next(iter(tokens))

    return updated
