"""
Stop-boundary detection — ported from Node B's coordinator-only logic in the
2-node version. Generalized the same way leadership and dispute resolution
were: every participant runs this identical, deterministic function over
the same shared consensus state, instead of one designated node deciding
and telling everyone. Since shared_x converges to the same value everywhere
(that's the whole point of diffusion/consensus.py), every participant's
StopDetector reaches the same stop_pos at the same time without needing to
announce it to each other — though the round leader still sends a
best-effort GEN_DONE for a clean teardown signal (see common/protocol.py).

Algorithm itself is unchanged from the original: find the frontier
(furthest committed position), walk back from it counting a contiguous
run of agreed EOS tokens, and only lock in a stop boundary once that run
is long enough, past a minimum generation length, and confirmed on two
consecutive checks (guards against a transient blip of low-confidence EOS
guesses freezing generation too early).
"""
from dataclasses import dataclass


@dataclass
class StopConfig:
    eos_id: int
    mask_id: int
    min_gen_length: int
    stop_run_min: int
    stop_confirmations: int


class StopDetector:
    def __init__(self, cfg: StopConfig):
        self.cfg = cfg
        self.stop_pos: int | None = None
        self._pending_stop_pos: int | None = None
        self._pending_confirms = 0

    def check(self, shared_x: dict[int, int], gen_start: int, gen_end: int) -> int | None:
        """shared_x: {absolute_position: token} — the full converged
        consensus state so far. Returns the current stop_pos (None if not
        yet triggered). Call this once per sync, after merge_consensus."""
        cfg = self.cfg
        committed_positions = [p for p in range(gen_start, gen_end) if p in shared_x]
        if not committed_positions:
            return self.stop_pos
        frontier = max(committed_positions) + 1

        run_start = frontier
        while (run_start > gen_start
               and shared_x.get(run_start - 1) is not None
               and shared_x[run_start - 1] == cfg.eos_id):
            run_start -= 1
        run_len = frontier - run_start

        if (frontier - gen_start) > cfg.min_gen_length and run_len >= cfg.stop_run_min:
            candidate_stop = run_start
            if candidate_stop == self._pending_stop_pos:
                self._pending_confirms += 1
            else:
                self._pending_stop_pos = candidate_stop
                self._pending_confirms = 1

            if (self._pending_confirms >= cfg.stop_confirmations
                    and (self.stop_pos is None or candidate_stop < self.stop_pos)):
                self.stop_pos = candidate_stop
        else:
            self._pending_stop_pos = None
            self._pending_confirms = 0

        return self.stop_pos
