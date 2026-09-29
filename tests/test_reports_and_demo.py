"""Report tables, review packets, the HTML viewer and run summaries."""

import json

import pytest
from fakes import (
    build_all,
    call,
    conversation,
    result,
    say,
    synthetic_task,
    trajectory,
)

from handoff.demo.viewer import render_html
from handoff.envs.registry import TaskRegistry
from handoff.eval.analysis import (
    dialogue_markdown,
    is_failure,
    per_task_changes,
    stratified_sample,
    task_markdown,
)
from handoff.eval.report import (
    RunSource,
    label_metrics,
    load_label,
    main_table_rows,
    paired_rows,
    sources_record,
    style_rows,
    to_csv,
    to_latex,
    to_markdown,
    write_table,
)
from handoff.eval.runner import (
    EvalResult,
    EvalSettings,
    compute_run_metrics,
    infra_breakdown,
    render_summary,
    select_refs,
)
from handoff.rewards.composer import RewardConfig
from handoff.rollout.schema import Violation, write_trajectories


def handoff_dialogue(**overrides):
    messages = conversation(
        say("user", "I need wheelchair help."),
        say("assistant", "Per our policy I cannot arrange that; transferring you."),
        say("assistant", None, call("t", "transfer_to_human_agents", summary="s")),
        result("t", "Transfer successful"),
        say("assistant", "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON."),
    )
    values = {
        "messages": messages,
        "transferred": True,
        "transfer_turn": 3,
        "should_transfer": True,
        "kind": "out_of_scope",
        "violations": [Violation("off_reference_write", 2)],
    }
    values.update(overrides)
    return trajectory(**values)


def make_run(
    tmp_path,
    run_id,
    outcomes,
    personas=("neutral",),
    *,
    kind="upstream",
    policy="p" * 40,
    execution_kind="real",
    status="completed",
    should_transfer=False,
    transferred=False,
):
    """A finished evaluation run on disk: one outcome series per task."""
    run_dir = tmp_path / run_id
    run_dir.mkdir()
    data = [
        trajectory(
            task_id=str(t) if kind == "upstream" else f"{t}__{kind}",
            source_task_id=str(t),
            trial=i,
            persona=p,
            kind=kind,
            task_reward=float(o),
            should_transfer=should_transfer,
            transferred=transferred,
        )
        for t, series in enumerate(outcomes)
        for p in personas
        for i, o in enumerate(series)
    ]
    write_trajectories(run_dir / "trajectories.jsonl", data)
    metrics = compute_run_metrics(data, reward_config=RewardConfig(), n_resamples=50)
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "git_sha": "a" * 40,
                "git_dirty": False,
                "execution_kind": execution_kind,
                "tau2_sha": "b" * 40,
                "model_revisions": {"policy": policy, "user_sim": "u" * 40},
                "seeds": {"run": 0},
            }
        )
    )
    (run_dir / "config.json").write_text(json.dumps({"eval_name": run_id}))
    (run_dir / "completion.json").write_text(json.dumps({"status": status}))
    return RunSource(run_id, run_dir)


def test_run_metrics_blocks_and_summary(tmp_path):
    source = make_run(
        tmp_path, "B0", [[1, 1, 0, 0], [1, 1, 1, 1]], personas=("neutral", "terse")
    )
    metrics = json.loads((source.run_dir / "metrics.json").read_text())
    assert metrics["n_episodes"] == 16
    assert metrics["overall"]["success"]["point"] == pytest.approx(0.75)
    assert set(metrics["styles"]) >= {"success/neutral", "success/terse", "style_gap"}
    assert metrics["reward"]["mean"]["total"] == pytest.approx(0.75)
    summary = render_summary(metrics)
    assert "| success | 0.750" in summary and "Style metric" in summary


def test_tables_in_three_formats(tmp_path):
    labels = [
        load_label("B0", [make_run(tmp_path, "B0-s0-20261001", [[1, 0, 0, 0]])]),
        load_label("B2", [make_run(tmp_path, "B2-s0-20261001", [[1, 1, 1, 0]])]),
    ]
    metrics = {data.label: label_metrics(data, n_resamples=50) for data in labels}
    rows = main_table_rows(labels, metrics)
    assert rows[0][:4] == ["Run", "Run ID", "Success (pass^1)", "pass^4"]
    assert rows[2][1] == "B2-s0-20261001" and rows[2][2].startswith("0.750")
    paired = paired_rows(labels, "B0")
    assert paired[1][0] == "B2" and paired[1][1].startswith("0.500")
    with pytest.raises(ValueError):
        paired_rows(labels, "missing")
    assert to_markdown([["a", "b"], ["1", "2"]]).splitlines()[1] == "| --- | --- |"
    assert to_csv([["a", "b"], ["1", "x,y"]]) == 'a,b\n1,"x,y"\n'
    latex = to_latex([["pass^4", "a_b"], ["1%", "2"]], "Main")
    assert "pass\\^{}4" in latex and "a\\_b" in latex and "1\\%" in latex
    write_table(tmp_path / "out", "main", rows, "Main")
    assert {p.suffix for p in (tmp_path / "out").iterdir()} == {".md", ".csv", ".tex"}
    record = sources_record(labels)[0]
    assert record["git_sha"] == "a" * 40 and record["execution_kind"] == "real"
    assert record["git_dirty"] is False
    assert len(style_rows(labels)) == 1  # header only: one style per label


def test_label_combines_a_checkpoints_test_protocols(tmp_path):
    """test_main (neutral) + test_styles (other styles) + test_handoff."""
    main = make_run(tmp_path, "R2-test_main", [[1, 1, 1, 1], [1, 1, 1, 1]])
    styles = make_run(
        tmp_path, "R2-test_styles", [[0, 0], [1, 0]], personas=("terse", "verbose")
    )
    handoff = make_run(
        tmp_path,
        "R2-test_handoff",
        [[1, 1], [1, 1]],
        kind="explicit_request",
        should_transfer=True,
        transferred=False,
    )
    data = load_label("R2", [main, styles, handoff])
    assert data.run_ids == "R2-test_main+R2-test_styles+R2-test_handoff"
    metrics = label_metrics(data, n_resamples=50)
    # task metrics: upstream tasks in the neutral style only (test_main)
    assert metrics["success"].point == pytest.approx(1.0)
    assert metrics["pass^{k}"].point == pytest.approx(1.0)
    # hand-off metrics: every neutral episode; the hand-off positives were missed
    assert metrics["handoff_f1"].point == pytest.approx(0.0)
    rows = style_rows([data], n_resamples=50)
    assert rows[0] == ["Run", "neutral", "terse", "verbose", "Gap", "Worst style"]
    assert rows[1][1].startswith("1.000") and rows[1][2].startswith("0.250")
    assert rows[1][4].startswith("0.750")  # gap: neutral 1.0 vs terse 0.25


def test_label_refuses_mixed_or_unfinished_runs(tmp_path):
    first = make_run(tmp_path, "a", [[1]])
    with pytest.raises(ValueError, match="twice"):
        load_label("X", [first, make_run(tmp_path, "b", [[0]])])
    with pytest.raises(ValueError, match="different policies"):
        load_label(
            "X",
            [first, make_run(tmp_path, "c", [[0]], personas=("terse",), policy="q")],
        )
    with pytest.raises(ValueError, match="did not complete"):
        load_label("X", [make_run(tmp_path, "d", [[1]], status="failed")])
    synthetic = make_run(tmp_path, "e", [[1]], execution_kind="synthetic_test")
    with pytest.raises(ValueError, match="synthetic_test"):
        load_label("X", [synthetic])
    assert load_label("X", [synthetic], allow_synthetic=True).trajectories


def test_review_packets_and_failure_selection():
    good = trajectory()
    bad = handoff_dialogue(should_transfer=False)
    assert not is_failure(good) and is_failure(bad)
    text = dialogue_markdown(handoff_dialogue())
    assert "handoff@3:cited" in text and "off_reference_write" in text
    assert "- [ ] Hand-off decision correct" in text
    task_text = task_markdown(
        {
            "id": "1__oos0",
            "user_scenario": {"instructions": {"reason_for_call": "x"}},
            "evaluation_criteria": {
                "reward_basis": ["DB", "ACTION"],
                "actions": [{"name": "transfer_to_human_agents"}],
            },
        },
        {
            "domain": "airline",
            "kind": "out_of_scope",
            "template_id": "oos_meal",
            "split": "test",
            "source_task_id": "1",
            "should_transfer": True,
        },
    )
    assert "transfer_to_human_agents" in task_text
    picked = stratified_sample(list(range(20)), lambda i: i % 2, 6, seed=1)
    assert len(picked) == 6 and sum(i % 2 for i in picked) == 3
    assert stratified_sample(list(range(20)), lambda i: i % 2, 6, seed=1) == picked
    changes = per_task_changes(
        [trajectory(task_id="1"), trajectory(task_id="2", task_reward=0.0)],
        [trajectory(task_id="1", task_reward=0.0), trajectory(task_id="2")],
    )
    assert changes == {"improved": ["airline/1"], "regressed": ["airline/2"]}


def test_html_viewer_highlights_handoff_citation_and_violation():
    page = render_html([handoff_dialogue()], "Demo <script>")
    assert page.startswith("<!doctype html>")
    assert "Demo &lt;script&gt;" in page
    assert "b-handoff" in page and "<mark>policy</mark>" in page
    assert "off_reference_write" in page
    assert "prefers-color-scheme:dark" in page


def test_eval_settings_guard_the_test_split():
    base = {
        "domains": ["airline"],
        "splits": ["test"],
        "kinds": ["upstream"],
        "personas": ["neutral"],
        "num_trials": 4,
        "seed": 300,
    }
    with pytest.raises(ValueError, match="allow_test"):
        EvalSettings.from_mapping("B0-s0-20261008", base)
    settings = EvalSettings.from_mapping("B0-s0-20261008", {**base, "allow_test": True})
    assert settings.domains == ("airline",)
    with pytest.raises(ValueError):
        EvalSettings.from_mapping(
            "B0-s0-20261008", {**base, "allow_test": True, "personas": []}
        )


def test_infra_failure_threshold_and_breakdown(tmp_path):
    base = {
        "domains": ["airline"],
        "splits": ["dev"],
        "kinds": ["upstream"],
        "personas": ["neutral"],
        "num_trials": 1,
        "seed": 0,
    }
    assert (
        EvalSettings.from_mapping("B0-s0-20261008", base).max_infra_failure_share == 0
    )
    with pytest.raises(ValueError):
        EvalSettings.from_mapping(
            "B0-s0-20261008", {**base, "max_infra_failure_share": 1.0}
        )
    assert EvalResult(tmp_path, 99, 1).infra_failure_share == pytest.approx(0.01)
    assert EvalResult(tmp_path, 0, 0).infra_failure_share == 0.0
    breakdown = infra_breakdown(
        [
            {"persona": "terse", "kind": "upstream", "error_type": "Timeout"},
            {"persona": "terse", "kind": "upstream", "error_type": "APIError"},
        ]
    )
    assert breakdown["by_persona"] == {"terse": 2}
    assert breakdown["by_error_type"] == {"APIError": 1, "Timeout": 1}


def test_handoff_kinds_need_the_handoff_manifest():
    tasks = [synthetic_task(str(i), transfer=(i == 3)) for i in range(12)]
    splits = {"train": [str(i) for i in range(8)], "test": ["8", "9", "10", "11"]}
    _, labels, manifest, _, handoff = build_all((tasks, splits))
    settings = EvalSettings.from_mapping(
        "B0-dev-s0-20261008",
        {
            "domains": ["airline"],
            "splits": ["dev"],
            "kinds": ["upstream", "explicit_request"],
            "personas": ["neutral"],
            "num_trials": 1,
            "seed": 0,
        },
    )
    kinds = {
        r.kind for r in select_refs(TaskRegistry(manifest, labels, handoff), settings)
    }
    assert kinds == {"upstream", "explicit_request"}
    with pytest.raises(ValueError, match="hand-off manifest"):
        select_refs(TaskRegistry(manifest, labels, None), settings)
