"""Cross-class transfer matrix.

Loads each class's specialist (best across saved seeds) from runs/<class>/seed*.json
and evaluates each spec on each class's test set. The diagonal cells should
be high; the off-diagonals should fall back toward the seed level if the
specialists are genuinely shaped to their class.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import SeedSpec
from eval import evaluate, load, split


def best_specialist(task_class: str) -> dict:
    """Pick the highest-dev-score specialist across all saved seeds.

    Dev not test: selecting by test would put the same numbers on both
    sides of the comparison for the diagonal, since the matrix is then
    evaluated on each class's test set.
    """
    runs = sorted(Path(f"runs/{task_class}").glob("seed*.json"))
    if not runs:
        raise SystemExit(f"no runs for class {task_class}; run stem.py first")
    best = None
    for p in runs:
        s = json.loads(p.read_text())
        if best is None or s["specialist_dev_score"] > best["specialist_dev_score"]:
            best = s
    return best["specialist_spec"]


def main() -> None:
    classes = ["regex", "json", "math", "sql"]
    specs = {}
    for c in classes:
        runs_for_c = list(Path(f"runs/{c}").glob("seed*.json"))
        if runs_for_c:
            specs[f"{c}_specialist"] = best_specialist(c)
        else:
            print(f"(no saved {c} runs, skipping that specialist)")
    specs["seed"] = SeedSpec

    test_sets = {c: split(load(c))["test"] for c in classes}

    cells = {}
    for spec_name, spec in specs.items():
        for c in classes:
            res = evaluate(spec, test_sets[c], c, label=f"{spec_name} on {c}")
            cells[(spec_name, c)] = res.score

    print()
    print("=== cross-class transfer matrix (test sets) ===")
    header = "spec".ljust(22) + "".join(f"{c:>10}" for c in classes)
    print(header)
    for spec_name in ["seed"] + [f"{c}_specialist" for c in classes]:
        row = spec_name.ljust(22) + "".join(f"{cells[(spec_name, c)]:>9.0%}" for c in classes)
        print(row)

    Path("runs/transfer.json").write_text(json.dumps({
        f"{s}@{c}": cells[(s, c)] for (s, c) in cells
    }, indent=2))
    print("\nwrote runs/transfer.json")


if __name__ == "__main__":
    main()
