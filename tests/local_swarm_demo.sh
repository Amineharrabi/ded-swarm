#!/bin/sh
# Self-contained local swarm demo: relay + 3 nodes, kill the leader mid-run,
# prove the survivors re-elect within PEER_TIMEOUT. Everything launched in
# this one script so background processes stay alive for its duration.
set -e
cd /home/claude/ded-swarm
export PYTHONPATH=/home/claude/ded-swarm

rm -f /tmp/relay.log /tmp/node-a.log /tmp/node-b.log /tmp/node-c.log

python3 relay/relay.py --pub-port 5555 --sub-port 5556 > /tmp/relay.log 2>&1 &
RELAY_PID=$!
sleep 1

python3 -m node.node --id node-a --relays local > /tmp/node-a.log 2>&1 &
NODE_A_PID=$!
python3 -m node.node --id node-b --relays local > /tmp/node-b.log 2>&1 &
NODE_B_PID=$!
python3 -m node.node --id node-c --relays local > /tmp/node-c.log 2>&1 &
NODE_C_PID=$!

echo ">>> all started, letting them join and elect a leader (5s)..."
sleep 5

echo ">>> killing node-a (the current leader) with SIGKILL..."
kill -9 $NODE_A_PID

echo ">>> waiting for peer timeout + re-election (9s)..."
sleep 9

echo ">>> shutting down survivors cleanly..."
kill -TERM $NODE_B_PID $NODE_C_PID 2>/dev/null || true
sleep 1
kill -9 $RELAY_PID 2>/dev/null || true

echo "=== relay log ==="
cat /tmp/relay.log
echo
echo "=== node-a log (killed mid-run) ==="
cat /tmp/node-a.log
echo
echo "=== node-b log ==="
cat /tmp/node-b.log
echo
echo "=== node-c log ==="
cat /tmp/node-c.log
