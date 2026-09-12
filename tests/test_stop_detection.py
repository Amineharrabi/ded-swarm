"""
Pure-logic tests for diffusion/stop_detection.py. Run with:
    PYTHONPATH=. python3 tests/test_stop_detection.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diffusion.stop_detection import StopDetector, StopConfig

EOS = 999
GEN_START = 32


def make_cfg(min_gen_length=10, stop_run_min=4, stop_confirmations=2):
    return StopConfig(eos_id=EOS, mask_id=0, min_gen_length=min_gen_length,
                       stop_run_min=stop_run_min, stop_confirmations=stop_confirmations)


def real_tokens(n):
    return {GEN_START + i: 100 + i for i in range(n)}


def test_no_trigger_before_min_length():
    det = StopDetector(make_cfg(min_gen_length=20))
    shared_x = real_tokens(5)
    for i in range(5, 9):
        shared_x[GEN_START + i] = EOS
    assert det.check(shared_x, GEN_START, GEN_START + 50) is None
    print("PASS: no trigger before min_gen_length, even with a real EOS run")


def test_triggers_after_run_and_confirmations():
    det = StopDetector(make_cfg(min_gen_length=10, stop_run_min=4, stop_confirmations=2))
    shared_x = real_tokens(15)
    for i in range(15, 20):
        shared_x[GEN_START + i] = EOS
    # first check: run is long enough and past min length, but needs 2 confirmations
    assert det.check(shared_x, GEN_START, GEN_START + 50) is None
    # second identical check confirms it
    assert det.check(shared_x, GEN_START, GEN_START + 50) == GEN_START + 15
    print("PASS: triggers only after stop_run_min + stop_confirmations both satisfied")


def test_transient_short_run_does_not_falsely_confirm():
    det = StopDetector(make_cfg(min_gen_length=10, stop_run_min=4, stop_confirmations=2))
    shared_x = real_tokens(15)
    shared_x[GEN_START + 15] = EOS
    shared_x[GEN_START + 16] = EOS
    # only a 2-token EOS run — below stop_run_min=4, should never trigger
    assert det.check(shared_x, GEN_START, GEN_START + 50) is None
    assert det.check(shared_x, GEN_START, GEN_START + 50) is None
    print("PASS: a short EOS blip below stop_run_min never triggers")


def test_two_participants_converge_on_identical_result():
    """The actual property this generalization depends on: two independent
    StopDetector instances, fed the SAME shared_x, must reach the SAME
    stop_pos — no coordination between them, just the same pure function."""
    shared_x = real_tokens(15)
    for i in range(15, 20):
        shared_x[GEN_START + i] = EOS

    det_a = StopDetector(make_cfg())
    det_b = StopDetector(make_cfg())
    for _ in range(3):
        result_a = det_a.check(shared_x, GEN_START, GEN_START + 50)
        result_b = det_b.check(shared_x, GEN_START, GEN_START + 50)
        assert result_a == result_b, f"diverged: a={result_a} b={result_b}"
    assert det_a.stop_pos == GEN_START + 15
    print("PASS: two independent detectors converge on the identical stop_pos")


if __name__ == "__main__":
    test_no_trigger_before_min_length()
    test_triggers_after_run_and_confirmations()
    test_transient_short_run_does_not_falsely_confirm()
    test_two_participants_converge_on_identical_result()
    print("\nALL PASS")
