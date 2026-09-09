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

1. **Leadership-during-generation semantics** (the open question above) —
   needs an answer before generation messages exist, not after.
2. **Generalize the wire protocol** for generation messages: N-way sparse
   commit payloads, replacing the pairwise `sync_payload`/`apply` exchange
   from the websocket version.
3. **Generalize consensus** from pairwise PoE (`score_x = log P_A(x) + log
   P_B(x)`) to N-way (`score_x = Σ log P_i(x)` over alive nodes) — the
   math generalizes cleanly, but with real disagreement expected (unlike
   today's near-identical-model near-zero-dispute reality), the resolution
   *policy* (all-vote vs. leader-adjudicates vs. something else) needs
   deciding, not just the scoring formula.
4. **Port the actual diffusion step logic** (`a_local_step`/`b_local_step`
   from the v7 notebooks) to run under this swarm layer instead of a
   fixed 2-node websocket exchange.
5. **Deploy for real**: stand up the 3 Exoscale relays, point
   `common/config.py: RELAYS` at them, run actual GPU nodes from Colab
   against `--relays prod`.
