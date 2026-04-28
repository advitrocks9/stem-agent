"""Split-sensitivity sweep.

The hand-picked dev split has been the binding constraint on every result
in this repo. If the conclusions only hold on one carefully chosen split,
they aren't conclusions, they're a split artifact. This sweep runs the
full evolve loop on N alternate random splits per class, with meta-seed
fixed at 0, and reports whether the specialist's test delta stays
positive across splits.

Run: uv run python tools/split_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evolve import evolve

CLASSES = ["regex", "json", "math"]
SPLIT_SEEDS = [101, 102, 103, 104, 105]  # 5 alternate splits, far from hand-picked
META_SEED = 0
ITERS = 6  # smaller iter budget to keep runtime manageable


def main() -> None:
    rows = []
    for cls in CLASSES:
        for ss in SPLIT_SEEDS:
            saved = Path(f"runs/sensitivity/{cls}/split{ss}_seed{META_SEED}.json")
            if saved.exists():
                print(f"\n=== {cls} split={ss} (already saved, loading) ===")
                s = json.loads(saved.read_text())
            else:
                print(f"\n=== {cls} split={ss} ===")
                s = evolve(cls, iters=ITERS, seed=META_SEED, split_seed=ss,
                           out_dir=Path("runs/sensitivity"))
            rows.append({
                "class": cls,
                "split_seed": ss,
                "seed_test": s["seed_test_score"],
                "spec_test": s["specialist_test_score"],
                "delta": s["specialist_test_score"] - s["seed_test_score"],
                "stop": s["stop_cause"],
                "iters_run": s["iters_run"],
                "frontier_size": len(s["frontier"]),
            })

    print("\n=== summary ===")
    print(f"{'class':<8} {'split':>6} {'seed':>6} {'spec':>6} {'Δ':>6}  stop")
    for r in rows:
        print(f"{r['class']:<8} {r['split_seed']:>6} {r['seed_test']:>5.0%} {r['spec_test']:>5.0%} {r['delta']*100:+5.0f}  {r['stop']}")

    print("\n=== per-class aggregate (5 splits, 1 meta-seed each) ===")
    for cls in CLASSES:
        deltas = [r["delta"] for r in rows if r["class"] == cls]
        positive = sum(1 for d in deltas if d > 0)
        m = mean(deltas) * 100
        sd = stdev(deltas) * 100 if len(deltas) > 1 else 0
        print(f"{cls}: mean Δ {m:+.1f}pp ± {sd:.1f}, positive on {positive}/{len(deltas)} splits "
              f"(per-split: {[round(d*100) for d in deltas]})")

    Path("runs/sensitivity_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
