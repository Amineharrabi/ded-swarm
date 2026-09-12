# Architecture

## What exists at this stage

Swarm membership and leadership only. No diffusion logic yet — that's
intentional sequencing, not an oversight (see "Next steps" below).

## Topology: PUB-SUB over a stateless relay, not ROUTER-ROUTER

Every node PUBlishes its own state (heartbeat, later: generation commits)
and SUBscribes to everyone's. The relay (`relay/relay.py`) is a pure
`zmq.proxy()` between an XSUB frontend and an XPUB backend — there is no
Python application logic in its hot path. It doesn't know what a "node" or
a "leader" is. It forwards frames.

This was chosen over a ROUTER-ROUTER broker pattern because the workload
needs broadcast (every node needs visibility into every other node's
state for both leader election and eventual N-way consensus scoring), not
addressed point-to-point request/reply.

## Relay redundancy: connect to all of them, not failover

Every node's PUB and SUB sockets connect to *every* relay address in
`common/config.py` simultaneously (`node/transport.py`). If one relay
dies, traffic keeps flowing through the others — a node's message just
gets duplicated across however many relays are up, and duplicate delivery
is harmless because peer-liveness handling is idempotent (marking a peer
alive twice does nothing extra the second time). No failover logic,
no "which relay is primary" bookkeeping, no reconnect-on-relay-death code
exists anywhere, on purpose.

Run 3 relays (different providers/regions if you want real independence)
for the property "the swarm survives any single relay dying."

## Leadership: computed independently, not elected

There is no election protocol — no candidate/vote/quorum message
exchange. Every node runs the same deterministic function
(`node/peers.py: PeerTable.leader()`) over the same shared input (the
heartbeat-derived alive-set): lowest node ID among currently-alive nodes
wins. Since every node sees the same heartbeat stream (modulo relay
redundancy duplicates, which are harmless), every node computes the same
answer without needing to communicate about the computation itself.

**A bug worth documenting, not just fixing**: the first version of
`PeerTable` seeded a node's own liveness entry once at construction and
never refreshed it, since `_handle_envelope` explicitly ignores a node's
own broadcasts (`if node_id == self.id: return`). After `PEER_TIMEOUT`
seconds, a node's own entry silently expired and it dropped *itself* out
of its own alive-set — producing a real split-brain: two survivors each
electing the *other* one, neither believing itself was still alive. Fixed
by treating self-liveness as unconditional (`alive_peers()` always
includes `self.self_id`, independent of timeout bookkeeping) rather than
routing it through the same heartbeat-timeout path used for peers. This
was only caught by actually running the kill-the-leader test in
`tests/local_swarm_demo.sh` — worth keeping that test around and running
it again after any change to `peers.py`.

## Leadership is not sticky, and that's an open question

If a lower-ID node rejoins, leadership moves back to it immediately, even
mid-generation. That's fine right now because nothing is "in flight" at
this layer yet. Once a generation round has real state (blocks committed,
partial consensus in progress), an unplanned leadership handoff
mid-round has a cost — needs a real answer before the diffusion layer
lands: does a new leader resume an in-progress round, restart it, or does
leadership only change *between* rounds regardless of what the raw
alive-set says?

## Wire format: msgpack, not JSON+base64

The 2-node websocket version (`common/protocol.py`'s ancestor) used JSON
with tensors base64-encoded into strings — functional, but wasteful and
untyped. msgpack is binary, typed, and doesn't inflate payload size the
way base64 does. `common/protocol.py` is a deliberately small envelope
(`type`, `node_id`, `ts`, `payload`) — generation-specific payload shapes
(sparse position/token/confidence arrays, same idea as the old
`sync_payload` message) aren't designed yet.

## Next steps, in sequence

1. **Leadership-during-generation semantics** — **answered and implemented**
   (`node/round.py`): leadership only changes *between* rounds, never
   during one, regardless of what the live alive-set says. See the
   "Round-locked leadership" section below.
2. **Generalize the wire protocol** for generation messages — **done**
   (`common/protocol.py`: `GEN_START`, `GEN_DONE`, `COMMIT`,
   `DISPUTE_SCORE`). See "Generation wire protocol" below.
3. **Generalize consensus** from pairwise to N-way — **done**
   (`diffusion/consensus.py`), with a specific policy decision made (see
   "N-way consensus" below) rather than left open.
4. **Port the diffusion step logic** — **done, structurally, not yet
   verified on real hardware** (`diffusion/local_step.py`,
   `diffusion/stop_detection.py`, `node/generation.py`). See "What's
   ported vs. what's still unverified" below.

---

## Round-locked leadership

Implemented in `node/round.py`. A round's leader and participant list are
frozen at `start_round()` / `on_gen_start()` and never change for the life
of the round, independent of what `PeerTable.leader()` says live. If the
round's leader goes silent for `config.ROUND_LEADER_TIMEOUT`, the round is
declared **failed**, not handed over — followers tear down their local
round state and wait. Only once there's no active round does the live
`PeerTable`-computed leader get to start a fresh one. Tested in
`tests/test_round.py`, including the specific case of a lower-ID node
rejoining mid-round and confirming it has no code path to steal
leadership before the round ends.

Known simplification: dueling initiators (two nodes both briefly
believing they're leader) are resolved by "first `GEN_START` observed
wins, second ignored" — not Raft-grade term numbers with tie-breaking.
Worth hardening if this turns out to happen often in practice.

## Generation wire protocol

Four new message types in `common/protocol.py`, all still msgpack over the
same PUB-SUB relay — no new transport needed:

- `GEN_START` — round leader announces round_id, frozen participant list,
  prompt, and generation config.
- `COMMIT` — phase 1 of a sync: each participant's own newly-settled
  positions this sync window (position, token, confidence), broadcast to
  everyone. Direct generalization of the old pairwise `sync_payload`.
- `DISPUTE_SCORE` — phase 2, sent only for positions phase 1 revealed as
  contested: each participant's own log-probability for every contested
  candidate token, looked up from the softmax distribution it already
  computed this step (no extra model forward pass).
- `GEN_DONE` — best-effort teardown notice from the leader. Not load-bearing:
  every participant can independently detect completion from shared
  consensus state via `diffusion/stop_detection.py`.

## N-way consensus — the policy decision

Chose **independent deterministic computation** over **leader-adjudicates**.
Every participant runs the exact same pure function
(`diffusion/consensus.py: resolve_disputes`) over the exact same received
data and arrives at the same answer — no one is "in charge" of producing
it, which matches how leadership itself is already computed in this
codebase, and avoids quietly centralizing the actual decision-making while
claiming to only centralize the plumbing.

The real generalization from pairwise PoE (`score_x = log P_A(x) + log
P_B(x)`) to N-way (`score_x = Σ log P_i(x)` over every reporting
participant) means a candidate that NO ONE individually proposed as their
own top pick can still win, if the ensemble's joint confidence in it is
highest — this literally cannot happen with only 2 voices, and
`tests/test_consensus.py: test_genuine_n_way_dispute_counts_every_voice`
exists specifically to prove the new code does this correctly, not just
fall back to majority-vote-by-proposal-count.

Graceful degradation, stated as policy rather than left implicit: a round's
frozen participant list can't shrink mid-round (see round-locking above),
but `resolve_disputes` doesn't require hearing from literally every
participant either — it sums over whoever reported within
`config.SYNC_BARRIER_TIMEOUT` and proceeds. A stalled participant loses its
vote for that one sync; it does not freeze the round for everyone else.

## What's ported vs. what's still unverified

`diffusion/local_step.py` is a **mechanical port**: same algorithm, same
remasking rule, same EOS-margin gating, and the exact same block-scoped
transfer-count fix that came out of the original corruption bug, carried
forward verbatim rather than re-derived. Only the parametrization changed
(explicit `StepConfig` instead of notebook globals) and the two near-
duplicate `a_local_step`/`b_local_step` collapsed into one function every
participant calls identically. Risk of a *new* bug here is low because
almost nothing about the actual logic was touched.

`node/generation.py` (the loop wiring transport + round + local_step +
consensus together) is a different story: this file has **not been run
against a real model**. It couldn't be — no GPU in this environment, and
even a CPU-only PyTorch install exceeded available disk space here. Two
real bugs were still caught by manual re-reading before this was ever
handed off (not run, just *read carefully*, which is a materially weaker
guarantee than a passing test):

- A stray or non-participant message could have picked up a vote in
  consensus scoring — `_collect_until` now explicitly rejects senders that
  aren't expected participants for this round, not just filters them from
  the "still waiting on" bookkeeping.
- Looking up a participant's confidence in a near-zero-probability
  candidate token used unguarded `.log()`, which underflows to `-inf` for
  a true zero — now clamped to `1e-12` first, consistent with the pattern
  already used throughout the ported step logic.

Both were real correctness risks that "looks obviously fine" didn't catch —
consistent with this project's whole history (the topk-overflow corruption,
the websocket ping timeout) of subtle bugs only surfacing once something
actually runs. **This file needs a real run on real hardware before it's
trusted, not just a read-through.** The docstring at the top of
`node/generation.py` flags the specific spot most likely to need
iteration: `_collect_until` doesn't currently drive
`round_mgr.check_leader_timeout()` while it's waiting on a sync, so a slow
sync could in theory outlast the round-leader timeout without anyone
noticing until after the wait returns.

## What's still actually ahead

1. **Run `node/generation.py` for real** against the actual LLaDA model on
   Colab GPU hardware — this is the load-bearing next step everything else
   here was building toward, and the one piece of this whole layer that's
   still unverified.
2. **Deploy for real**: stand up the 3 Exoscale relay boxes, point
   `common/config.py: RELAYS` at their addresses, run GPU nodes from Colab
   against `--relays prod` instead of `--relays local`.
3. Wire `node/node.py`'s membership loop together with
   `node/generation.py`'s round loop into one runnable process — right now
   they exist as separate, individually-tested layers; nothing yet starts
   a round automatically when this node becomes leader and the swarm is
   ready, or hands prompts in from outside the process.
