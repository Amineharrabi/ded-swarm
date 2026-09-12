"""
The relay: pure message plumbing, nothing else.

This process holds NO state about nodes, leadership, or generation progress.
It does exactly one thing: forward every message published by any node to
every node that's subscribed. That's it. `zmq.proxy()` is a C-level loop —
there is no Python application logic in the hot path at all.

Topology:
  - PUBLISHERS (nodes broadcasting their own state) connect to --pub-port,
    which is bound as an XSUB socket (the proxy's "frontend").
  - SUBSCRIBERS (nodes listening for everyone's state) connect to
    --sub-port, which is bound as an XPUB socket (the proxy's "backend").
  - zmq.proxy(frontend, backend) pumps frames between them.

Run three of these (one per relay VPS) for redundancy. Nodes connect their
PUB and SUB sockets to all three simultaneously — see node/transport.py for
why that gives redundancy without any failover code.
"""
import argparse
import zmq


def main():
    ap = argparse.ArgumentParser(description="DED swarm relay (stateless XSUB/XPUB proxy)")
    ap.add_argument("--pub-port", type=int, default=5555,
                     help="port publishers (nodes) connect to")
    ap.add_argument("--sub-port", type=int, default=5556,
                     help="port subscribers (nodes) connect to")
    ap.add_argument("--bind-addr", default="*",
                     help="interface to bind on (default: all interfaces)")
    args = ap.parse_args()

    ctx = zmq.Context.instance()

    frontend = ctx.socket(zmq.XSUB)
    frontend.bind(f"tcp://{args.bind_addr}:{args.pub_port}")

    backend = ctx.socket(zmq.XPUB)
    backend.bind(f"tcp://{args.bind_addr}:{args.sub_port}")
    # Without this, a late-joining subscriber's subscription won't propagate
    # to publishers that already connected, and it silently misses messages
    # until something re-subscribes. Not optional for a relay nodes dial
    # into at arbitrary times.
    backend.setsockopt(zmq.XPUB_VERBOSE, 1)

    print(f"[relay] publishers connect to  tcp://<this-host>:{args.pub_port}")
    print(f"[relay] subscribers connect to tcp://<this-host>:{args.sub_port}")
    print("[relay] forwarding (Ctrl+C to stop)...")

    try:
        zmq.proxy(frontend, backend)
    except KeyboardInterrupt:
        pass
    finally:
        frontend.close()
        backend.close()
        ctx.term()


if __name__ == "__main__":
    main()
