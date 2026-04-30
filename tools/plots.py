"""Two figures for the writeup. Trajectories of dev score across iters,
and the cross-class transfer matrix as a heatmap.

Run: uv run python tools/plots.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; uv add matplotlib first")
        return

    out_dir = Path("runs/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = ["regex", "json", "math", "sql"]
    colors = {"regex": "#1f77b4", "json": "#ff7f0e", "math": "#2ca02c", "sql": "#d62728"}

    # ---- Figure 1: dev-score trajectories ----
    # Show the *current parent's* dev score over iters (running monotone)
    # rather than the child's score, so rolled-back/apoptosis children
    # don't pull the line back to zero. Frontier additions are marked.
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for cls in classes:
        for run in sorted(Path(f"runs/{cls}").glob("seed*.json")):
            d = json.loads(run.read_text())
            running = []
            cur = 0.0
            for n in d["lineage"]:
                if n.get("accepted"):
                    cur = max(cur, n["dev_score"])
                running.append((n["iter"], cur))
            xs = [r[0] for r in running]
            ys = [r[1] for r in running]
            ax.plot(xs, ys, color=colors[cls], alpha=0.45, linewidth=1.2)
            # frontier additions: child's own score (the moment of growth)
            fx = [n["iter"] for n in d["lineage"] if n.get("on_frontier")]
            fy = [n["dev_score"] for n in d["lineage"] if n.get("on_frontier")]
            ax.scatter(fx, fy, color=colors[cls], s=24, alpha=0.85, edgecolors="white", linewidths=0.5)
    for cls in classes:
        ax.plot([], [], color=colors[cls], label=cls, linewidth=2)
    ax.set_xlabel("iter")
    ax.set_ylabel("best accepted dev score so far")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("evolution trajectories (dots: frontier additions)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "trajectories.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out_dir / 'trajectories.png'}")

    # ---- Figure 2: transfer matrix heatmap ----
    transfer_path = Path("runs/transfer.json")
    if not transfer_path.exists():
        print("runs/transfer.json missing; run tools/transfer.py first")
        return
    transfer = json.loads(transfer_path.read_text())
    # transfer.json has keys like "regex_specialist@regex"
    rows = ["seed", "regex_specialist", "json_specialist", "math_specialist", "sql_specialist"]
    cols = ["regex", "json", "math", "sql"]
    grid = []
    for r in rows:
        row = []
        for c in cols:
            key = f"{r}@{c}"
            row.append(transfer.get(key))
        grid.append(row)

    # Filter rows/cols that have any None values
    valid_rows = [i for i, r in enumerate(grid) if all(v is not None for v in r)]
    valid_cols = list(range(len(cols)))
    if not valid_rows:
        print("transfer.json has no complete rows; skipping heatmap")
        return
    grid_v = [[grid[i][j] for j in valid_cols] for i in valid_rows]
    rows_v = [rows[i] for i in valid_rows]
    cols_v = [cols[j] for j in valid_cols]

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    im = ax.imshow(grid_v, cmap="viridis", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(cols_v))); ax.set_xticklabels(cols_v)
    ax.set_yticks(range(len(rows_v))); ax.set_yticklabels(rows_v)
    for i in range(len(rows_v)):
        for j in range(len(cols_v)):
            v = grid_v[i][j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center",
                    color="white" if v < 0.65 else "black", fontsize=10)
    ax.set_title("cross-class transfer (test sets)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "transfer.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out_dir / 'transfer.png'}")

    # ---- Figure 3: Pareto frontier scatter ----
    # Each saved run contributes its frontier points as (mean_in_tokens, dev
    # score). The seed point sits at low tokens / lower score; specialists
    # spend 2-4x the tokens for 10-30pp dev-score gains. The frontier carries
    # 2-4 non-dominated points per run on average, not a single point.
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for cls in classes:
        for run in sorted(Path(f"runs/{cls}").glob("seed*.json")):
            d = json.loads(run.read_text())
            xs, ys = [], []
            for p in d["frontier"]:
                xs.append(p["tokens"])
                ys.append(p["score"] * 100)
            order = sorted(zip(xs, ys))
            ax.plot([o[0] for o in order], [o[1] for o in order],
                    color=colors[cls], alpha=0.6, linewidth=1.2, marker="o", markersize=5)
    for cls in classes:
        ax.plot([], [], color=colors[cls], label=cls, marker="o", linewidth=2)
    ax.set_xlabel("mean prompt tokens per dev eval")
    ax.set_ylabel("dev score (%)")
    ax.set_xscale("log")
    ax.set_title("Pareto frontiers across 12 saved runs")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(out_dir / "pareto_scatter.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out_dir / 'pareto_scatter.png'}")


if __name__ == "__main__":
    main()
