"""Stem agent CLI.

  uv run python stem.py --class regex --iters 8
  uv run python stem.py --class json  --iters 8 --seed 1
  uv run python stem.py --class math  --iters 8 --no-rollback
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolve import evolve


def main() -> None:
    ap = argparse.ArgumentParser(description="Differentiate the seed into a class specialist.")
    ap.add_argument("--class", dest="task_class", required=True, choices=["regex", "json", "math"])
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0, help="random seed for meta-agent sampling")
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--no-rollback", action="store_true", help="ablate the regression rollback rule")
    args = ap.parse_args()

    s = evolve(args.task_class, iters=args.iters, out_dir=args.out,
               seed=args.seed, skip_rollback=args.no_rollback)
    print()
    print(f"=== {args.task_class} (seed {args.seed}) ===")
    print(f"seed test:       {s['seed_test_score']:.0%}")
    print(f"specialist test: {s['specialist_test_score']:.0%}")
    print(f"stop cause:      {s['stop_cause']} at iter {s['iters_run']}")
    print(f"frontier size:   {len(s['frontier'])}")
    print(f"specialist spec: {json.dumps({k: (v[:70]+'...' if isinstance(v, str) and len(v)>70 else v) for k, v in s['specialist_spec'].items()}, indent=2)}")


if __name__ == "__main__":
    main()
