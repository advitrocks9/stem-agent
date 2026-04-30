"""Source-of-truth recompute for the WRITEUP headline table.

The hand-typed numbers in WRITEUP.md drifted from runs/{class}/seed*.json
because evolve.py records seed_test_score / specialist_test_score and the
table was edited by hand at later commits without re-checking.

Reads the JSONs, computes per-seed delta in percentage points, mean and
sample standard deviation across seeds, and prints a markdown row per class.
Adds the hand-tuned baseline column from runs/hand_baseline.json so the
table shows the real comparison surface.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, stdev


def main() -> None:
    classes = ["regex", "json", "math", "sql"]

    hand_path = Path("runs/hand_baseline.json")
    hand_rows = {r["class"]: r for r in json.loads(hand_path.read_text())} if hand_path.exists() else {}

    print()
    print("| class | seed test | hand strong | specialist | spec - seed | per-seed Δ |")
    print("|---|---:|---:|---:|---:|---|")

    summary = []
    for cls in classes:
        seed_pps, spec_pps, deltas = [], [], []
        for s in [0, 1, 2]:
            p = Path(f"runs/{cls}/seed{s}.json")
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            seed_pps.append(d["seed_test_score"] * 100)
            spec_pps.append(d["specialist_test_score"] * 100)
            deltas.append((d["specialist_test_score"] - d["seed_test_score"]) * 100)
        if not deltas:
            continue
        seed_m, seed_sd = mean(seed_pps), stdev(seed_pps) if len(seed_pps) > 1 else 0.0
        spec_m, spec_sd = mean(spec_pps), stdev(spec_pps) if len(spec_pps) > 1 else 0.0
        d_m, d_sd = mean(deltas), stdev(deltas) if len(deltas) > 1 else 0.0
        per_seed = ", ".join(f"{round(x):+d}" for x in deltas)

        hand_str = "n/a"
        if cls in hand_rows:
            hand_str = f"{hand_rows[cls]['hand']*100:.0f}%"

        print(f"| {cls} | {seed_m:.0f}% ± {seed_sd:.0f} | {hand_str} "
              f"| {spec_m:.0f}% ± {spec_sd:.0f} | **{d_m:+.0f} ± {d_sd:.1f}** | {per_seed} |")
        summary.append({
            "class": cls,
            "seed_mean": seed_m, "seed_sd": seed_sd,
            "hand": hand_rows.get(cls, {}).get("hand"),
            "spec_mean": spec_m, "spec_sd": spec_sd,
            "delta_mean": d_m, "delta_sd": d_sd,
            "per_seed_delta_pp": [round(x) for x in deltas],
        })

    Path("runs/headline_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nwrote runs/headline_summary.json")


if __name__ == "__main__":
    main()
