"""Bootstrap 95% CIs on the per-class delta (specialist - seed) over the test set.

Reads saved runs from runs/{class}/seed*.json. Always reevaluates the seed
and specialist specs on the current test set and writes the per-task
pass/fail booleans plus the resulting accuracy back into the JSON. This
catches drift between what evolve.py recorded as `seed_test_score` /
`specialist_test_score` and what the eval harness produces today (eg after
a fix to `_norm_math_answer` or to `validate_sql`).

Resamples test items with replacement 1000 times for the paired delta CI.

Run: uv run python tools/bootstrap_ci.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from statistics import mean, stdev

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import SeedSpec
from eval import evaluate, load, split

N_BOOT = 1000
LOW, HIGH = 2.5, 97.5
RNG = random.Random(0)


def per_task(spec: dict, tasks: list[dict], cls: str) -> dict[str, bool]:
    res = evaluate(spec, tasks, cls)
    return {t.task_id: t.passed for t in res.per_task}


def bootstrap_delta(seed_pt: dict[str, bool], spec_pt: dict[str, bool]) -> tuple[float, float, float]:
    ids = list(seed_pt.keys())
    deltas = []
    for _ in range(N_BOOT):
        sample = [RNG.choice(ids) for _ in ids]
        seed_s = sum(seed_pt[i] for i in sample) / len(sample)
        spec_s = sum(spec_pt[i] for i in sample) / len(sample)
        deltas.append(spec_s - seed_s)
    deltas.sort()
    lo = deltas[int(len(deltas) * LOW / 100)]
    hi = deltas[int(len(deltas) * HIGH / 100)]
    point = sum(deltas) / len(deltas)
    return point, lo, hi


def main() -> None:
    out_rows = []
    for cls in ["regex", "json", "math", "sql"]:
        runs = sorted(Path(f"runs/{cls}").glob("seed*.json"))
        if not runs:
            print(f"{cls}: no runs saved, skipping")
            continue
        for run_path in runs:
            d = json.loads(run_path.read_text())
            test_tasks = split(load(cls))["test"]

            # Always reeval and write back. Cached values can disagree with
            # the headline if eval.py or agent.py changed since the run was
            # saved (math normalizer, sql float-int parity, etc.).
            print(f"  ({run_path.name}: reevaluating seed + specialist)")
            seed_pt = per_task(SeedSpec, test_tasks, cls)
            spec_pt = per_task(d["specialist_spec"], test_tasks, cls)
            seed_acc_now = sum(seed_pt.values()) / len(seed_pt)
            spec_acc_now = sum(spec_pt.values()) / len(spec_pt)
            old_seed = d.get("seed_test_score")
            old_spec = d.get("specialist_test_score")
            if old_seed is not None and abs(old_seed - seed_acc_now) > 1e-9:
                print(f"    seed_test_score drift: was {old_seed:.4f}, now {seed_acc_now:.4f}")
            if old_spec is not None and abs(old_spec - spec_acc_now) > 1e-9:
                print(f"    specialist_test_score drift: was {old_spec:.4f}, now {spec_acc_now:.4f}")
            d["test_per_task_seed"] = seed_pt
            d["test_per_task_specialist"] = spec_pt
            d["seed_test_score"] = seed_acc_now
            d["specialist_test_score"] = spec_acc_now
            run_path.write_text(json.dumps(d, indent=2, default=str))

            n = len(seed_pt)
            seed_acc = sum(seed_pt.values()) / n
            spec_acc = sum(spec_pt.values()) / n
            point, lo, hi = bootstrap_delta(seed_pt, spec_pt)
            out_rows.append({
                "class": cls, "run": run_path.name, "n_test": n,
                "seed_acc": seed_acc, "spec_acc": spec_acc,
                "delta_point": point, "delta_lo": lo, "delta_hi": hi,
            })
            print(f"  {cls:<5} {run_path.name:<28} seed={seed_acc:.0%} spec={spec_acc:.0%} "
                  f"Δ={point*100:+.1f}pp [95% CI {lo*100:+.1f}, {hi*100:+.1f}] n_test={n}")

    print("\n=== per-class aggregate (paired test-set bootstrap, pooled across saved seeds) ===")
    for cls in ["regex", "json", "math", "sql"]:
        rows = [r for r in out_rows if r["class"] == cls]
        if not rows:
            continue
        # pool per-task deltas across seeds: for each (cls, seed) bootstrap separately,
        # then report the median CI across seeds. Simpler: report mean of point estimates
        # and the union of the seed-level CIs.
        points = [r["delta_point"] for r in rows]
        los = [r["delta_lo"] for r in rows]
        his = [r["delta_hi"] for r in rows]
        print(f"{cls}: mean Δ {mean(points)*100:+.1f}pp, per-seed CIs: "
              f"{[f'[{l*100:+.0f},{h*100:+.0f}]' for l, h in zip(los, his)]}")

    Path("runs/bootstrap_summary.json").write_text(json.dumps(out_rows, indent=2))
    print("\nwrote runs/bootstrap_summary.json")


if __name__ == "__main__":
    main()
