"""Draw the report figures (T4.2) from committed report tables.

Inputs: ``reports/main/main_table.csv`` and ``reports/rl/dev_curves.csv``
(written by ``scripts/make_report.py`` and ``scripts/rl_curves.py`` from run
IDs). The test_handoff calibration numbers are not in a committed table; they
are copied below from the runs' ``metrics.json`` (``test_handoff`` runs named
in the comments) and also recorded in AGENTS.md §2 and docs/experiment-log.md.

Run from the repository root (needs matplotlib):

    python report/make_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"

COLORS = {
    "B0": "#9ca3af",
    "B1": "#d1d5db",
    "B2": "#2563eb",
    "R1": "#dc2626",
    "R2": "#059669",
}

# test_handoff, neutral style, 2 trials, 380 episodes (metrics.json, overall).
# B0-test_handoff-s0-20261004, B2-test_handoff-s0-20261005,
# R1-final-test_handoff-s0-20261006, R2-final-test_handoff-s0-20261007.
HANDOFF = {
    "B0": {"precision": 0.866, "recall": 0.526, "over": 0.087},
    "B2": {"precision": 0.976, "recall": 0.633, "over": 0.016},
    "R1": {"precision": 0.689, "recall": 0.959, "over": 0.462},
    "R2": {"precision": 0.886, "recall": 0.750, "over": 0.103},
}


def _interval(cell: str) -> tuple[float, float, float]:
    point, rest = cell.split(" [")
    low, high = rest.rstrip("]").split(", ")
    return float(point), float(low), float(high)


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
    ax.set_axisbelow(True)


def success_figure() -> None:
    """pass^1 and pass^4 on test_main with 95 % bootstrap intervals."""
    with open(ROOT / "reports/main/main_table.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = [row["Run"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, column, title in zip(
        axes,
        ["Success (pass^1)", "pass^4"],
        ["Success (pass$^1$)", "pass$^4$"],
        strict=True,
    ):
        values = [_interval(row[column]) for row in rows]
        points = [v[0] for v in values]
        errors = [[v[0] - v[1] for v in values], [v[2] - v[0] for v in values]]
        ax.bar(
            labels,
            points,
            yerr=errors,
            capsize=3,
            color=[COLORS[label] for label in labels],
            edgecolor="#374151",
            linewidth=0.5,
        )
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 0.6)
        _style(ax)
    axes[0].set_ylabel("test_main (49 tasks × 4)")
    fig.tight_layout()
    fig.savefig(OUT / "success.pdf")
    plt.close(fig)


def handoff_figure() -> None:
    """Precision, recall and over-escalation on test_handoff."""
    labels = list(HANDOFF)
    metrics = [
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("over", "Over-escalation"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    width = 0.2
    for i, label in enumerate(labels):
        xs = [m + (i - 1.5) * width for m in range(len(metrics))]
        ys = [HANDOFF[label][key] for key, _ in metrics]
        ax.bar(
            xs,
            ys,
            width,
            label=label,
            color=COLORS[label],
            edgecolor="#374151",
            linewidth=0.5,
        )
        for x, y in zip(xs, ys, strict=True):
            ax.text(x, y + 0.015, f"{y:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(metrics)), [name for _, name in metrics])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("test_handoff (380 episodes)")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "handoff.pdf")
    plt.close(fig)


def dev_figure() -> None:
    """Dev success and transfer rate during GRPO (greedy, 60 episodes)."""
    with open(ROOT / "reports/rl/dev_curves.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharex=True)
    for ax, column, title in zip(
        axes,
        ["success", "transferred"],
        ["Dev success", "Dev transfer rate (all kinds)"],
        strict=True,
    ):
        for run in ["R1", "R2"]:
            points = [r for r in rows if r["Run"] == run]
            ax.plot(
                [int(r["Step"]) for r in points],
                [float(r[column]) for r in points],
                marker="o",
                markersize=3.5,
                color=COLORS[run],
                label=run,
            )
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("GRPO step")
        ax.set_ylim(0.2, 0.8)
        _style(ax)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "dev_curves.pdf")
    plt.close(fig)


def main() -> None:
    """Write every figure to report/figures/."""
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    success_figure()
    handoff_figure()
    dev_figure()
    print(f"figures written to {OUT}")


if __name__ == "__main__":
    main()
