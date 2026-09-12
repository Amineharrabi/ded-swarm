#!/bin/sh
set -e
cd /home/claude/ded-swarm 2>/dev/null || cd "$(dirname "$0")/.."
export PYTHONPATH=.

python3 relay/relay.py --pub-port 5555 --sub-port 5556 > /tmp/relay_swarm_test.log 2>&1 &
RELAY_PID=$!
sleep 1

python3 tests/test_swarm_integration.py
RESULT=$?

kill -9 $RELAY_PID 2>/dev/null || true
exit $RESULT
