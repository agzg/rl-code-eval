#!/usr/bin/env python3
"""
Plot training metrics from metrics.jsonl.

Usage:
    python plot_metrics.py [metrics.jsonl]
    Default: reads ./metrics.jsonl
"""

import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Install matplotlib: uv add matplotlib", file=sys.stderr)
    sys.exit(1)


def load_metrics(path: Path) -> list[dict]:
    """Load metrics from JSONL file."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    metrics_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("metrics.jsonl")
    if not metrics_path.exists():
        print(f"File not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_metrics(metrics_path)
    if not rows:
        print("No data in metrics file", file=sys.stderr)
        sys.exit(1)

    steps = [r["step"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # --- Plot 1: Reward mean ---
    ax1 = axes[0]
    reward_means = [r.get("train/reward_mean") for r in rows]
    valid = [(s, v) for s, v in zip(steps, reward_means) if v is not None]
    if valid:
        ax1.plot(
            [x for x, _ in valid],
            [y for _, y in valid],
            "o-",
            color="tab:blue",
            linewidth=2,
            markersize=4,
        )
    ax1.set_ylabel("Train Reward Mean", fontsize=11)
    ax1.set_title("Training Reward Over Steps", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # --- Plot 2: Datums vs groups skipped ---
    ax2 = axes[1]
    num_datums = [r.get("train/num_datums", 0) for r in rows]
    groups_skipped = [r.get("train/groups_skipped", 0) for r in rows]

    x = range(len(steps))
    width = 0.35

    bars1 = ax2.bar(
        [i - width / 2 for i in x],
        num_datums,
        width,
        label="Num Datums",
        color="tab:green",
        alpha=0.8,
    )
    bars2 = ax2.bar(
        [i + width / 2 for i in x],
        groups_skipped,
        width,
        label="Groups Skipped",
        color="tab:orange",
        alpha=0.8,
    )

    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Datums vs Groups Skipped Over Steps", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(steps)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out_path = metrics_path.with_name("metrics_plot.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close()

    # --- Eval comparison (steps 10 vs 20) ---
    eval_by_step = {
        r["step"]: {"correct": r["eval/correct"], "format": r["eval/format"], "count": r["eval/count"]}
        for r in rows
        if "eval/correct" in r
    }

    steps_to_compare = [10, 20]
    present = [s for s in steps_to_compare if s in eval_by_step]
    missing = [s for s in steps_to_compare if s not in eval_by_step]

    if present:
        print("\n--- Eval comparison ---")
        for s in steps_to_compare:
            if s in eval_by_step:
                e = eval_by_step[s]
                print(f"  Step {s}: Pass@1 = {e['correct']:.3f}, Format = {e['format']:.3f} (n={e['count']})")
            else:
                print(f"  Step {s}: (no eval in this run)")
        if len(present) >= 2:
            c10 = eval_by_step[10]["correct"]
            c20 = eval_by_step.get(20)
            if c20 is not None:
                diff = c20["correct"] - c10
                print(f"\n  Change step 10→20: Pass@1 {'+' if diff >= 0 else ''}{diff:.3f}")
    else:
        print("\n--- Eval comparison ---")
        print("  No eval metrics in this run. Eval runs every eval_every steps (e.g. 10, 20).")


if __name__ == "__main__":
    main()
