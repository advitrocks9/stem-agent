"""Stem agent CLI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolve import evolve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="task_class", required=True, choices=["regex"])
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    args = ap.parse_args()

    s = evolve(args.task_class, iters=args.iters, out_dir=args.out, seed=args.seed)
    print()
    print(f"=== {args.task_class} (seed {args.seed}) ===")
    print(f"seed test:       {s['seed_test_score']:.0%}")
    print(f"specialist test: {s['specialist_test_score']:.0%}")


if __name__ == "__main__":
    main()
