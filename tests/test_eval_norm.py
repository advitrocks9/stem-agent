"""Regression tests for _norm_math_answer.

The verbose-output case bit me on math seed 2 evaluation: a worker output of
"Final answer rounds to 80.5 hours, where 80 is integer and 0.5 represents
half" was being normalised to "0.5" because the old rule was "last number".
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import _norm_math_answer


def test_finds_number_near_answer_word():
    s = "Final answer rounds to 80.5 hours, where 80 is integer and 0.5 represents half"
    assert _norm_math_answer(s) == "80.5"


def test_handles_dollars():
    assert _norm_math_answer("The total is $24.99.") == "24.99"


def test_strips_commas():
    assert _norm_math_answer("Final: 1,234 widgets") == "1234"


def test_hhmm_wins():
    assert _norm_math_answer("They meet at 3:45 in the afternoon.") == "3:45"


def test_fraction_wins():
    assert _norm_math_answer("The probability is 3/8 in the long run.") == "3/8"


def test_trailing_zeros_normalised():
    assert _norm_math_answer("answer: 7.50") == "7.5"
    assert _norm_math_answer("$24.00") == "24"


def test_bare_number_pass_through():
    assert _norm_math_answer("42") == "42"


def test_negative_number():
    assert _norm_math_answer("answer is -3 degrees") == "-3"


def test_falls_back_to_last_number_when_no_answer_keyword():
    # The old rule still applies when there's no "answer" anchor.
    assert _norm_math_answer("First step gives 12 then we get 7") == "7"


if __name__ == "__main__":
    import traceback
    failures = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok  {name}")
            except AssertionError:
                failures += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    raise SystemExit(failures)
