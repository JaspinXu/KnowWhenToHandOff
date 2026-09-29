"""Reproducible tables and figures from evaluation run IDs (task T4.1).

A report *label* (``B0``, ``R2``, ...) is one checkpoint, evaluated by one or
more runs: the test protocols split one checkpoint's evaluation into
``test_main`` (neutral style), ``test_styles`` (the five other styles) and
``test_handoff`` (hand-off tasks), and the report combines them (ADR-005). The
runs of a label must be completed, evaluate the same policy revision with the
same user simulator, and not share any (task, style) pair.

Every number is recomputed from the runs' ``trajectories.jsonl`` with the
metrics of :mod:`handoff.eval.metrics` (task-level bootstrap), on fixed subsets:

- task metrics (success, pass^k, violation rate, turns): upstream tasks in the
  neutral style, the τ²-bench protocol;
- hand-off metrics (F1, over-escalation) and the citation rate: every
  neutral-style episode, upstream and hand-off tasks;
- the style table: upstream tasks in every style.

``sources.json`` records each run's provenance (git SHA, tau2 commit, model
revisions, seeds, execution kind) next to the tables.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from handoff.eval.metrics import (
    Interval,
    bootstrap,
    format_interval,
    metric_statistics,
    paired_difference,
    style_breakdown,
    task_units,
)
from handoff.rollout.schema import Trajectory, iter_trajectories
from handoff.train.monitors import CURVE_KEYS

REFERENCE_STYLE = "neutral"
TASK_METRICS = ("success", "pass^{k}", "violation_rate", "mean_turns")
HANDOFF_METRICS = ("handoff_f1", "over_escalation_rate", "citation_rate")
MAIN_COLUMNS = (
    ("success", "Success (pass^1)"),
    ("pass^{k}", "pass^{k}"),
    ("violation_rate", "Violation rate"),
    ("handoff_f1", "Hand-off F1"),
    ("over_escalation_rate", "Over-escalation"),
    ("citation_rate", "Citation rate"),
    ("mean_turns", "Turns"),
)


@dataclass(frozen=True)
class RunSource:
    """One evaluation run on disk."""

    run_id: str
    run_dir: Path

    def _json(self, name: str) -> dict[str, Any]:
        return json.loads((self.run_dir / name).read_text(encoding="utf-8"))

    def provenance(self) -> dict[str, Any]:
        """The run's immutable provenance record."""
        return self._json("provenance.json")

    def config(self) -> dict[str, Any]:
        """The run's saved configuration."""
        return self._json("config.json")

    def completion(self) -> dict[str, Any] | None:
        """The run's terminal record, if it finished."""
        path = self.run_dir / "completion.json"
        return self._json("completion.json") if path.exists() else None

    def trajectories(self) -> list[Trajectory]:
        """All trajectories of the run."""
        return list(iter_trajectories(self.run_dir / "trajectories.jsonl"))


@dataclass
class ReportLabel:
    """One checkpoint in a report: its runs and their merged trajectories."""

    label: str
    sources: list[RunSource]
    trajectories: list[Trajectory] = field(default_factory=list)

    @property
    def run_ids(self) -> str:
        """Run IDs joined with ``+`` (as given on the command line)."""
        return "+".join(source.run_id for source in self.sources)


def load_label(
    label: str, sources: Sequence[RunSource], *, allow_synthetic: bool = False
) -> ReportLabel:
    """Validate that ``sources`` evaluate one checkpoint and merge their episodes."""
    if not sources:
        raise ValueError(f"{label}: no runs")
    revisions: set[tuple[str, str]] = set()
    seen: dict[tuple[str, str, str], str] = {}
    merged: list[Trajectory] = []
    for source in sources:
        completion = source.completion()
        if completion is None or completion.get("status") != "completed":
            raise ValueError(f"{source.run_id} did not complete; refusing to report it")
        provenance = source.provenance()
        if provenance.get("execution_kind") != "real" and not allow_synthetic:
            raise ValueError(
                f"{source.run_id} is a {provenance.get('execution_kind')} run, not a "
                "real one (use --allow-synthetic for pipeline tests)"
            )
        models = provenance.get("model_revisions") or {}
        revisions.add((str(models.get("policy")), str(models.get("user_sim"))))
        for trajectory in source.trajectories():
            key = (trajectory.domain, trajectory.task_id, trajectory.persona)
            if key in seen and seen[key] != source.run_id:
                raise ValueError(
                    f"{label}: {'/'.join(key)} is in both {seen[key]} and "
                    f"{source.run_id}; a label must not count an episode twice"
                )
            seen[key] = source.run_id
            merged.append(trajectory)
    if len(revisions) > 1:
        raise ValueError(
            f"{label}: runs evaluate different policies or simulators ({revisions})"
        )
    return ReportLabel(label, list(sources), merged)


def task_subset(trajectories: Sequence[Trajectory]) -> list[Trajectory]:
    """Upstream tasks in the neutral style (all episodes if there are none)."""
    subset = [
        t for t in trajectories if t.kind == "upstream" and t.persona == REFERENCE_STYLE
    ]
    return subset or list(trajectories)


def neutral_subset(trajectories: Sequence[Trajectory]) -> list[Trajectory]:
    """Every neutral-style episode (all episodes if there are none)."""
    subset = [t for t in trajectories if t.persona == REFERENCE_STYLE]
    return subset or list(trajectories)


def label_metrics(
    data: ReportLabel, *, k: int = 4, n_resamples: int = 1000, seed: int = 0
) -> dict[str, Interval]:
    """The main-table metrics of one label, keyed as in :data:`MAIN_COLUMNS`."""
    statistics = metric_statistics(k)
    result: dict[str, Interval] = {}
    for names, subset in (
        (TASK_METRICS, task_subset(data.trajectories)),
        (HANDOFF_METRICS, neutral_subset(data.trajectories)),
    ):
        units = task_units(subset)
        for name in names:
            key = name.format(k=k)
            result[name] = bootstrap(
                units, statistics[key], n_resamples=n_resamples, seed=seed
            )
    return result


def main_table_rows(
    labels: Sequence[ReportLabel],
    metrics: Mapping[str, Mapping[str, Interval]],
    *,
    k: int = 4,
) -> list[list[str]]:
    """Header plus one row per label with the headline metrics and 95% CIs."""
    rows = [["Run", "Run ID", *[title.format(k=k) for _, title in MAIN_COLUMNS]]]
    for data in labels:
        values = metrics[data.label]
        rows.append(
            [
                data.label,
                data.run_ids,
                *[format_interval(values[key].as_dict()) for key, _ in MAIN_COLUMNS],
            ]
        )
    return rows


def paired_rows(
    labels: Sequence[ReportLabel], baseline_label: str, *, seed: int = 0
) -> list[list[str]]:
    """Paired success differences (task bootstrap) of every label vs a baseline."""
    by_label = {data.label: data for data in labels}
    if baseline_label not in by_label:
        raise ValueError(f"baseline {baseline_label!r} is not among the runs")
    baseline = task_subset(by_label[baseline_label].trajectories)
    rows = [["Run", f"Δ success vs {baseline_label} [95% CI]", "Shared tasks"]]
    for data in labels:
        if data.label == baseline_label:
            continue
        interval = paired_difference(
            task_subset(data.trajectories), baseline, seed=seed
        )
        rows.append(
            [data.label, format_interval(interval.as_dict()), str(interval.n_units)]
        )
    return rows


def style_rows(
    labels: Sequence[ReportLabel], *, n_resamples: int = 1000, seed: int = 0
) -> list[list[str]]:
    """Per-style success, gap and worst style on upstream tasks (≥ 2 styles)."""
    breakdowns: dict[str, dict[str, Interval]] = {}
    styles: list[str] = []
    for data in labels:
        upstream = [t for t in data.trajectories if t.kind == "upstream"]
        if len({t.persona for t in upstream}) < 2:
            continue
        breakdown = style_breakdown(upstream, n_resamples=n_resamples, seed=seed)
        breakdowns[data.label] = breakdown
        for key in breakdown:
            if key.startswith("success/") and key[8:] not in styles:
                styles.append(key[8:])
    rows = [["Run", *styles, "Gap", "Worst style"]]
    for data in labels:
        breakdown = breakdowns.get(data.label)
        if breakdown is None:
            continue
        cell = {key: interval.as_dict() for key, interval in breakdown.items()}
        rows.append(
            [
                data.label,
                *[format_interval(cell.get(f"success/{s}")) for s in styles],
                format_interval(cell.get("style_gap")),
                format_interval(cell.get("worst_style_success")),
            ]
        )
    return rows


def to_markdown(rows: Sequence[Sequence[str]]) -> str:
    """Render rows (first row = header) as a Markdown table."""
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines) + "\n"


def to_csv(rows: Sequence[Sequence[str]]) -> str:
    """Render rows as CSV text."""
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


def to_latex(rows: Sequence[Sequence[str]], caption: str) -> str:
    """Render rows as a LaTeX ``tabular`` inside a ``table`` environment."""

    def escape(cell: str) -> str:
        return (
            cell.replace("\\", "\\textbackslash{}")
            .replace("_", "\\_")
            .replace("%", "\\%")
            .replace("^", "\\^{}")
            .replace("&", "\\&")
            .replace("#", "\\#")
        )

    header, *body = rows
    spec = "l" * len(header)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        f"\\caption{{{escape(caption)}}}",
        f"\\begin{{tabular}}{{{spec}}}",
        "\\hline",
        " & ".join(escape(c) for c in header) + " \\\\",
        "\\hline",
        *[" & ".join(escape(c) for c in row) + " \\\\" for row in body],
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def write_table(out_dir: Path, name: str, rows: Sequence[Sequence[str]], caption: str):
    """Write one table as Markdown, CSV and LaTeX."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.md").write_text(to_markdown(rows), encoding="utf-8")
    (out_dir / f"{name}.csv").write_text(to_csv(rows), encoding="utf-8")
    (out_dir / f"{name}.tex").write_text(to_latex(rows, caption), encoding="utf-8")


def sources_record(labels: Sequence[ReportLabel]) -> list[dict[str, Any]]:
    """Provenance of every run used in the report, grouped by label."""
    records = []
    for data in labels:
        for source in data.sources:
            provenance = source.provenance()
            config = source.config()
            records.append(
                {
                    "label": data.label,
                    "run_id": source.run_id,
                    "eval_name": config.get("eval_name"),
                    "execution_kind": provenance.get("execution_kind"),
                    "git_sha": provenance.get("git_sha"),
                    "git_dirty": provenance.get("git_dirty"),
                    "tau2_sha": provenance.get("tau2_sha"),
                    "model_revisions": provenance.get("model_revisions"),
                    "seeds": provenance.get("seeds"),
                    "config_sha256": provenance.get("config_sha256"),
                }
            )
    return records


def plot_success(metrics: Mapping[str, Mapping[str, Interval]], path: Path) -> None:
    """Bar chart of success with 95% CIs (requires matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    intervals = {label: values["success"] for label, values in metrics.items()}
    labels = [label for label, i in intervals.items() if i.point is not None]
    points = [float(intervals[label].point) for label in labels]
    lower = [
        p - (p if intervals[label].low is None else intervals[label].low)
        for label, p in zip(labels, points, strict=True)
    ]
    upper = [
        (p if intervals[label].high is None else intervals[label].high) - p
        for label, p in zip(labels, points, strict=True)
    ]
    figure, axis = plt.subplots(figsize=(1.2 * len(labels) + 2, 3.2))
    axis.bar(labels, points, yerr=[lower, upper], capsize=4, color="#4B6CB7")
    axis.set_ylabel("Success (pass^1)")
    axis.set_ylim(0, 1)
    axis.set_title("Task success with 95% bootstrap CIs")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_tradeoff(metrics: Mapping[str, Mapping[str, Interval]], path: Path) -> None:
    """Success versus over-escalation, one point per label (λ sweep, RQ2)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.6, 3.6))
    for label, values in metrics.items():
        x = values["over_escalation_rate"].point
        y = values["success"].point
        if x is None or y is None:
            continue
        axis.scatter([x], [y], s=36)
        axis.annotate(label, (x, y), textcoords="offset points", xytext=(4, 4))
    axis.set_xlabel("Over-escalation rate")
    axis.set_ylabel("Success (pass^1)")
    axis.set_title("Hand-off calibration vs task success")
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def curve_rows(
    curves: Mapping[str, Mapping[str, Sequence[Mapping[str, float]]]],
    phases: Sequence[str] = ("train", "dev"),
) -> list[list[str]]:
    """RL learning curves as rows: one per (label, phase, step), terms as columns.

    ``curves`` maps a label to ``{"train": rows, "dev": rows}`` as returned by
    :func:`handoff.train.monitors.step_curve`.
    """
    rows = [["Run", "Phase", "Step", "Episodes", *CURVE_KEYS]]
    for label, by_phase in curves.items():
        for phase in phases:
            for row in by_phase.get(phase, []):
                rows.append(
                    [
                        label,
                        phase,
                        str(int(row["step"])),
                        str(int(row["n_episodes"])),
                        *[
                            "" if key not in row else f"{row[key]:.3f}"
                            for key in CURVE_KEYS
                        ],
                    ]
                )
    return rows


def plot_learning_curves(
    curves: Mapping[str, Mapping[str, Sequence[Mapping[str, float]]]], path: Path
) -> None:
    """Train reward, dev success and train penalties per step (needs matplotlib)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = (
        ("train", ("total", "task"), "Train reward (mean per step)"),
        ("dev", ("success", "total"), "Dev validation"),
        (
            "train",
            ("violation_penalty", "fp_penalty", "fn_penalty", "format_penalty"),
            "Train penalties",
        ),
    )
    figure, axes = plt.subplots(1, len(panels), figsize=(13, 3.6))
    for axis, (phase, keys, title) in zip(axes, panels, strict=True):
        for label, by_phase in curves.items():
            rows = by_phase.get(phase, [])
            for key in keys:
                points = [(row["step"], row[key]) for row in rows if key in row]
                if points:
                    steps, values = zip(*points, strict=True)
                    axis.plot(steps, values, marker=".", label=f"{label} {key}")
        axis.set_title(title)
        axis.set_xlabel("Training step")
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
