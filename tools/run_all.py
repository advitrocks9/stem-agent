"""Sequential driver: evolve all classes across multiple seeds and aggregate."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evolve import evolve

CLASSES = ["regex", "json", "math", "sql"]
SEEDS = [0, 1, 2]
ITERS = 8


def main() -> None:
    rows = []
    for cls in CLASSES:
        for seed in SEEDS:
            print(f"\n=== {cls} seed={seed} ===")
            s = evolve(cls, iters=ITERS, seed=seed)
            rows.append({
                "class": cls,
                "seed": seed,
                "seed_test": s["seed_test_score"],
                "spec_test": s["specialist_test_score"],
                "stop": s["stop_cause"],
                "iters_run": s["iters_run"],
                "frontier": len(s["frontier"]),
            })

    print("\n=== summary ===")
    print(f"{'class':<8} {'seed':>4} {'seed-test':>10} {'spec-test':>10} {'iters':>6} {'frontier':>9} {'stop':<10}")
    for r in rows:
        print(f"{r['class']:<8} {r['seed']:>4} {r['seed_test']:>9.0%} {r['spec_test']:>9.0%} "
              f"{r['iters_run']:>6} {r['frontier']:>9} {r['stop']:<10}")

    print("\n=== mean +- std across seeds ===")
    for cls in CLASSES:
        seed_scores = [r["seed_test"] for r in rows if r["class"] == cls]
        spec_scores = [r["spec_test"] for r in rows if r["class"] == cls]
        if len(seed_scores) > 1:
            print(f"{cls}: seed {mean(seed_scores):.0%}+-{stdev(seed_scores):.0%}  "
                  f"specialist {mean(spec_scores):.0%}+-{stdev(spec_scores):.0%}  "
                  f"delta {mean(spec_scores)-mean(seed_scores):+.0%}")
        else:
            print(f"{cls}: seed {seed_scores[0]:.0%}  specialist {spec_scores[0]:.0%}  "
                  f"delta {spec_scores[0]-seed_scores[0]:+.0%}")

    Path("runs/multi_seed_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
