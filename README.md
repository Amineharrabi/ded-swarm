# DED Swarm

Decentralized N-node swarm layer for the DED project — the successor to the 2-node
websocket architecture in `DED_NodeA_v7.ipynb` / `DED_NodeB_v7.ipynb`.



```
relay/relay.py      — stateless XSUB/XPUB message relay (run this on each Exoscale VPS)
common/protocol.py  — msgpack wire envelope
common/config.py    — relay addresses + timing constants (fill in RELAYS before prod use)
node/transport.py   — PUB/SUB socket wrapper (connects to ALL relays at once — see docs)
node/peers.py       — peer liveness tracking + deterministic leader election
node/node.py         — runnable swarm node (join/heartbeat/leader-detection/clean-leave)
tests/local_swarm_demo.sh — end-to-end proof: 3 nodes, kill the leader, watch re-election
docs/architecture.md — design decisions, and the split-brain bug this already caught once
```

```bash
pip install -r requirements.txt
sh tests/local_swarm_demo.sh
```

This starts a local relay + 3 nodes, lets them elect a leader, SIGKILLs
the leader, and shows the survivors converging on a new one. Read the
output — it's the same test that caught the split-brain bug documented
in `docs/architecture.md`.

To run nodes manually against the local relay:

```bash
python -m relay.relay --pub-port 5555 --sub-port 5556 &
python -m node.node --id node-a --relays local
python -m node.node --id node-b --relays local   # in another terminal
```

Ctrl+C one of them and watch the others re-elect within `PEER_TIMEOUT`
seconds (`common/config.py`).

## Before going to production

1. Stand up 3 Exoscale relay boxes, run `relay/relay.py` on each.
2. Fill in `common/config.py: RELAYS` with their addresses.
3. Run nodes with `--relays prod` instead of `--relays local`.

Everything else in `docs/architecture.md`'s "Next steps" is still ahead:
leadership-during-generation semantics, the generation wire protocol,
N-way consensus, and porting the actual model step logic.
