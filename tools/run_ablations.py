"""Run the architecture ablations: random-edit baseline, no-demos meta-agent,
no-feedback meta-agent. One seed per (class, mode) since the goal is whether
the gain survives, not stability across seeds.

Run: uv run python tools/run_ablations.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evolve import evolve

CLASSES = ["regex", "json", "math", "sql"]
MODES = ["random", "no_demos", "no_feedback"]
ITERS = 6
SEED = 0


def main() -> None:
    rows = []
    for cls in CLASSES:
        for mode in MODES:
            print(f"\n=== {cls} mode={mode} ===")
            s = evolve(cls, iters=ITERS, seed=SEED, propose_mode=mode,
                       out_dir=Path("runs/ablation"))
            rows.append({
                "class": cls,
                "mode": mode,
                "seed_test": s["seed_test_score"],
                "spec_test": s["specialist_test_score"],
                "delta": s["specialist_test_score"] - s["seed_test_score"],
                "stop": s["stop_cause"],
                "iters_run": s["iters_run"],
            })

    print("\n=== ablation summary ===")
    print(f"{'class':<6} {'mode':<14} {'seed':>5} {'spec':>5} {'Δ':>5}  stop")
    for r in rows:
        print(f"{r['class']:<6} {r['mode']:<14} {r['seed_test']:>4.0%} {r['spec_test']:>4.0%} "
              f"{r['delta']*100:+4.0f}  {r['stop']}")

    Path("runs/ablation_summary.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
