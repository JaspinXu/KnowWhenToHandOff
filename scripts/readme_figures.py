"""Draw the README figures into ``docs/assets/`` from committed report tables.

Inputs: ``reports/main/main_table.csv`` and ``reports/rl/dev_curves.csv``
(written by ``make_report.py`` and ``rl_curves.py`` from run IDs). The
test_handoff precision and recall are not in a committed table; they are the
``metrics.json`` values of the runs named below (also in AGENTS.md §2 and the
model cards).

Run from the repository root (needs matplotlib, e.g. on the cluster):

    bash slurm/run.sh python scripts/readme_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets"

INK = "#14213D"
MUTED = "#6B7280"
GRID = "#E5E7EB"
COLORS = {
    "B0": "#9CA3AF",
    "B1": "#CBD5E1",
    "B2": "#2F5FD0",
    "R1": "#D9711C",
    "R2": "#0F766E",
}
NAMES = {
    "B0": "B0 prompted 4B",
    "B1": "B1 prompted 30B",
    "B2": "B2 SFT",
    "R1": "R1 GRPO, task reward",
    "R2": "R2 GRPO, RAI reward",
}

# test_handoff (98 should-transfer + 92 hard-negative tasks, 2 trials):
# B0-test_handoff-s0-20261004, B2-test_handoff-s0-20261005,
# R1-final-test_handoff-s0-20261006, R2-final-test_handoff-s0-20261007.
HANDOFF = {
    "B0": {"recall": 0.526, "over": 0.087, "f1": 0.654},
    "B2": {"recall": 0.633, "over": 0.016, "f1": 0.768},
    "R1": {"recall": 0.959, "over": 0.462, "f1": 0.802},
    "R2": {"recall": 0.750, "over": 0.103, "f1": 0.812},
}


def _interval(cell: str) -> tuple[float, float, float]:
    point, rest = cell.split(" [")
    low, high = rest.rstrip("]").split(", ")
    return float(point), float(low), float(high)


def _style(ax, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9CA3AF")
    ax.tick_params(colors=INK)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def results_figure() -> None:
    """Left: task success with CIs. Right: hand-off operating points."""
    with open(ROOT / "reports/main/main_table.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.15]}
    )
    labels = [row["Run"] for row in rows]
    values = [_interval(row["Success (pass^1)"]) for row in rows]
    points = [v[0] for v in values]
    errors = [[v[0] - v[1] for v in values], [v[2] - v[0] for v in values]]
    left.bar(
        labels,
        points,
        yerr=errors,
        capsize=4,
        color=[COLORS[label] for label in labels],
        edgecolor=INK,
        linewidth=0.6,
        error_kw={"ecolor": INK, "elinewidth": 1},
    )
    for x, y in enumerate(points):
        left.text(
            x,
            0.02,
            f"{y:.2f}",
            ha="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
    left.set_ylim(0, 0.6)
    left.set_ylabel("Task success (pass$^1$)", color=INK)
    left.set_title(
        "Official test tasks: RL ≈ SFT (95% CIs overlap)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    _style(left)

    for label, row in HANDOFF.items():
        right.scatter(
            row["over"],
            row["recall"],
            s=260,
            color=COLORS[label],
            edgecolor=INK,
            linewidth=0.8,
            zorder=3,
        )
        right.annotate(
            f"{NAMES[label]}\nF1 {row['f1']:.3f}",
            (row["over"], row["recall"]),
            textcoords="offset points",
            xytext={"B0": (14, -26), "R1": (-22, -4)}.get(label, (-14, 14)),
            ha="right" if label == "R1" else "left",
            fontsize=9,
            color=INK,
        )
    right.annotate(
        "",
        xy=(0.452, 0.95),
        xytext=(0.03, 0.64),
        arrowprops={
            "arrowstyle": "->",
            "color": COLORS["R1"],
            "lw": 1.4,
            "linestyle": "--",
        },
    )
    right.annotate(
        "",
        xy=(0.098, 0.742),
        xytext=(0.024, 0.642),
        arrowprops={"arrowstyle": "->", "color": COLORS["R2"], "lw": 1.4},
    )
    right.set_xlim(-0.02, 0.52)
    right.set_ylim(0.45, 1.0)
    right.set_xlabel(
        "Over-escalation (transfers users who did not need a human)", color=INK
    )
    right.set_ylabel("Recall (transfers users who need a human)", color=INK)
    right.set_title(
        "Hand-off operating points (test_handoff, 380 dialogues)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    _style(right, "both")
    fig.tight_layout()
    fig.savefig(OUT / "results.png", dpi=200, facecolor="white")
    plt.close(fig)


def dev_figure() -> None:
    """Dev success and transfer rate during GRPO (greedy, 60 dialogues)."""
    with open(ROOT / "reports/rl/dev_curves.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    titles = {
        "success": "Dev success",
        "transferred": "Dev transfer rate: R1 learns to escalate late",
    }
    for ax, column in zip(axes, ["success", "transferred"], strict=True):
        for run in ["R1", "R2"]:
            points = [r for r in rows if r["Run"] == run]
            ax.plot(
                [int(r["Step"]) for r in points],
                [float(r[column]) for r in points],
                marker="o",
                markersize=5,
                linewidth=2.2,
                color=COLORS[run],
                label=NAMES[run],
            )
        ax.set_title(titles[column], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("GRPO step", color=INK)
        ax.set_ylim(0.2, 0.8)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "dev_curves.png", dpi=200, facecolor="white")
    plt.close(fig)


def main() -> None:
    """Write every figure to docs/assets/."""
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10})
    results_figure()
    dev_figure()
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
