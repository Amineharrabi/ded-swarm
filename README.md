# DED Swarm

Decentralized N-node swarm layer for the DED project 


**Status:** membership, leadership, round-locking, N-way consensus, and
stop-detection are implemented and covered by passing tests 

## What's here

```
relay/relay.py            - stateless XSUB/XPUB message relay (run on each Exoscale VPS)
common/protocol.py        - msgpack wire envelope + all message type definitions
common/config.py          - relay addresses + every timing constant, one place
node/transport.py         - PUB/SUB socket wrapper (connects to ALL relays at once)
node/peers.py             - peer liveness tracking + deterministic leader election
node/round.py             - round-locked leadership (frozen leader/participants per round)
node/node.py              - runnable swarm membership node (join/heartbeat/leader-detection)
node/generation.py        - generation loop: wires transport+round+local_step+consensus together
diffusion/local_step.py   - the per-node denoising step (ported from a_local_step/b_local_step)
diffusion/consensus.py    - N-way PoE consensus (independent deterministic computation)
diffusion/stop_detection.py - EOS-freeze boundary detection, generalized to run on every node
tests/local_swarm_demo.sh - end-to-end proof: 3 nodes, kill the leader, watch re-election
tests/test_round.py       - round-locking correctness (6 tests, pure Python)
tests/test_consensus.py   - N-way consensus correctness (6 tests, pure Python)
tests/test_stop_detection.py - stop-boundary correctness (4 tests, pure Python)
docs/architecture.md      - every design decision, why, and what's still open
```

```bash
pip install -r requirements.txt
export PYTHONPATH=.

# membership/leadership/failover, end to end:
sh tests/local_swarm_demo.sh

# pure-logic correctness suites:
python3 tests/test_round.py
python3 tests/test_consensus.py
python3 tests/test_stop_detection.py
```

`local_swarm_demo.sh` starts a local relay + 3 nodes, lets them elect a
leader, SIGKILLs the leader, and shows the survivors converging on a new
one with no disagreement - the same test that originally caught a
split-brain bug, documented in `docs/architecture.md`.

To run nodes manually against the local relay:

```bash
python -m relay.relay --pub-port 5555 --sub-port 5556 &
python -m node.node --id node-a --relays local
python -m node.node --id node-b --relays local   # in another terminal
```



## going to prod

1. Stand up 3 Exoscale relay boxes, run `relay/relay.py` on each.
2. Fill in `common/config.py: RELAYS` with their addresses.
3. Run nodes with `--relays prod` instead of `--relays local`.
4. Wire `node/node.py`'s membership loop and `node/generation.py`'s round
   loop into one process - they're separate, individually-tested layers
   right now; nothing yet auto-starts a round when this node becomes
   leader and the swarm is ready.
