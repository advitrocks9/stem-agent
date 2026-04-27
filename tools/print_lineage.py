"""ASCII printer for a saved lineage."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt_edit(edit: dict) -> str:
    if not edit:
        return "(seed)"
    parts = []
    for k, v in edit.items():
        s = str(v).replace("\n", " ")
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s}")
    return ", ".join(parts)


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]] or sorted(Path("runs").glob("*/seed*.json"))
    for p in paths:
        if not p.exists():
            print(f"missing: {p}")
            continue
        d = json.loads(p.read_text())
        print(f"=== {d['task_class']} (seed {d.get('seed_index', 0)}) ===")
        print(f"stop:            {d['stop_cause']} at iter {d['iters_run']}")
        print(f"seed test:       {d['seed_test_score']:.0%}")
        print(f"specialist test: {d['specialist_test_score']:.0%}")
        print(f"frontier size:   {len(d['frontier'])}")
        print()
        for n in d["lineage"]:
            score = f"{n['dev_score']:.0%}".rjust(5)
            tag = "S " if n["iter"] == 0 else (
                "F+" if n["on_frontier"] else
                "Ad" if n["accepted"] else
                "R " if n["rejection"] == "regression" else
                "A " if n["rejection"] in ("parse", "smoke") else
                "  "
            )
            mark = "FRONTIER" if n["on_frontier"] else (
                f"REJECT({n['rejection']})" if not n["accepted"] else "accept-dominated"
            )
            print(f"  iter{n['iter']:>2}  {n['spec_hash']}  dev={score}  {tag}  {mark}")
            if n["edit"]:
                print(f"             edit: {fmt_edit(n['edit'])}")
            if n.get("reason") and n["reason"] != "seed":
                print(f"             why : {n['reason'][:120]}")
        print()


if __name__ == "__main__":
    main()
