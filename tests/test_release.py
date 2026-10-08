"""Release packaging: which runs and files go out, and path scrubbing."""

import json

from handoff.release import hostnames, plan, release_runs, run_index, scrub, write


def _run(runs, run_id, kind="real", status="completed", rollouts=False):
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    provenance = {"execution_kind": kind, "hardware": {"hostname": "xnode7"}}
    (run_dir / "provenance.json").write_text(json.dumps(provenance))
    if status is not None:
        (run_dir / "completion.json").write_text(json.dumps({"status": status}))
    metrics = {"overall": {"success": {"point": 0.5}, "handoff_f1": {"point": 0.8}}}
    (run_dir / "metrics.json").write_text(json.dumps(metrics))
    (run_dir / "config.json").write_text(json.dumps({"runs": str(runs)}))
    (run_dir / "checkpoints").mkdir()
    (run_dir / "checkpoints" / "model.safetensors").write_bytes(b"weights")
    if rollouts:
        (run_dir / "rollouts").mkdir()
        (run_dir / "rollouts" / "xnode7-4242.jsonl").write_text('{"host": "xnode7"}\n')
    return run_dir


def test_release_selects_completed_real_runs_with_known_prefixes(tmp_path):
    runs = tmp_path / "runs"
    _run(runs, "B2-test_main-s0-20261005")
    _run(runs, "R1-s1-20261006", status=None, rollouts=True)  # training run
    _run(runs, "SPIKE-s1-20261005")
    _run(runs, "R2-final-dev-s0-20261007", status="incomplete")
    _run(runs, "B0-smoke-s0-20261004", kind="synthetic_test")
    names = [path.name for path in release_runs(runs)]
    assert names == ["B2-test_main-s0-20261005", "R1-s1-20261006"]


def test_plan_skips_weights_and_write_scrubs_paths(tmp_path):
    repo = tmp_path / "repo"
    (repo / "data/handoff_tasks/v1").mkdir(parents=True)
    (repo / "data/handoff_tasks/v1/airline_tasks.json").write_text("[]")
    (repo / "data/manifests").mkdir(parents=True)
    (repo / "data/manifests/splits_v1.json").write_text("{}")
    runs = tmp_path / "scratch" / "runs"
    _run(runs, "R1-s1-20261006", status=None, rollouts=True)
    sft = tmp_path / "scratch" / "datasets" / "b2_v1"
    sft.mkdir(parents=True)
    (sft / "manifest.json").write_text("{}")
    (sft / "records.jsonl").write_text("{}\n")  # token arrays: not released
    items = plan(repo, runs, tmp_path / "scratch" / "datasets", ["b2_v1"])
    targets = [item.target for item in items]
    assert "tasks/v1/airline_tasks.json" in targets
    assert "tasks/manifests/splits_v1.json" in targets
    assert "runs/R1-s1-20261006/rollouts/part-000.jsonl" in targets
    assert "sft/b2_v1/manifest.json" in targets
    assert not any("checkpoints" in t or "records.jsonl" in t for t in targets)

    out = tmp_path / "out"
    assert hostnames(items) == {"xnode7"}
    replacements = [(str(tmp_path / "scratch"), "<scratch>"), ("xnode7", "<host>")]
    digests = write(items, out, replacements, extra={"runs.jsonl": "{}\n"})
    config = (out / "runs/R1-s1-20261006/config.json").read_text()
    assert "<scratch>" in config
    assert str(tmp_path) not in config
    assert json.dumps(str(tmp_path))[1:-1] not in config
    assert set(digests) == set(targets) | {"runs.jsonl"}
    assert (out / "SHA256SUMS").read_text().count("\n") == len(targets) + 1
    published = "".join(p.read_text() for p in out.rglob("*") if p.is_file())
    assert "xnode7" not in published
    rows = run_index(items)
    assert rows == [
        {"run_id": "R1-s1-20261006", "files": 4, "success": 0.5, "handoff_f1": 0.8}
    ]


def test_scrub_replaces_longest_path_first():
    text = "/home/u/handoff/runs and /home/u/x"
    pairs = [("/home/u", "~"), ("/home/u/handoff", "<scratch>")]
    assert scrub(text, pairs) == "<scratch>/runs and ~/x"


def test_kaggle_metadata_strips_card_front_matter():
    from handoff.release import kaggle_metadata

    card = (
        "---\nlicense: mit\n---\n\n# Title\n\nBody text.\n\n"
        "## Loading\n\nload_dataset(...)\n\n## Licence\n\nMIT\n"
    )
    meta = kaggle_metadata("someone", card)
    assert meta["id"] == "someone/knowwhentohandoff-data"
    assert meta["licenses"] == [{"name": "other"}]
    description = str(meta["description"])
    assert description.startswith("Body text.")
    assert 'Path("/kaggle/input").rglob("runs.jsonl")' in description
    assert "load_dataset" not in description and "## Licence" in description
    assert meta["expectedUpdateFrequency"] == "never"
    assert "license: mit" not in str(meta["description"])
    assert "Licence: MIT" in str(meta["description"])
    assert 6 <= len(str(meta["title"])) <= 50
    assert 20 <= len(str(meta["subtitle"])) <= 80


def test_kaggle_kernel_metadata_attaches_the_dataset_privately():
    from handoff.release import (
        KAGGLE_NOTEBOOK,
        KAGGLE_NOTEBOOK_TITLE,
        KAGGLE_SLUG,
        kaggle_kernel_metadata,
    )

    meta = kaggle_kernel_metadata("someone", "kaggle_quickstart.ipynb")
    assert meta["id"] == f"someone/{KAGGLE_NOTEBOOK}"
    assert meta["dataset_sources"] == [f"someone/{KAGGLE_SLUG}"]
    assert meta["is_private"] is True and meta["enable_internet"] is False
    # Kaggle derives the notebook's slug from its title; they must agree.
    assert KAGGLE_NOTEBOOK_TITLE.lower().replace(" ", "-") == KAGGLE_NOTEBOOK


def _release_fixture(root):
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    (root / "tasks" / "manifests").mkdir(parents=True)
    (root / "tasks" / "v1").mkdir(parents=True)
    for source, target in [
        ("data/manifests/handoff_tasks_v1.json", "tasks/manifests"),
        ("data/handoff_tasks/v1/airline_tasks.json", "tasks/v1"),
    ]:
        (root / target / Path(source).name).write_bytes((repo / source).read_bytes())
    rows = []
    message = {"role": "assistant", "content": None, "tool_calls": [{"name": "x"}]}
    for model in ["B2", "R1", "R2", "R1-step40"]:  # step40: an intermediate checkpoint
        run_id = f"{model}-final-test_handoff-s0-20261007".replace(
            "step40-final", "step40"
        )
        (root / "runs" / run_id).mkdir(parents=True)
        lines = [
            {
                "task_id": "0__hn0",
                "kind": kind,
                "transferred": kind != "hard_negative" or model == "R1",
                "should_transfer": kind != "hard_negative",
                "messages": [{"role": "user", "content": "hi"}, message],
            }
            for kind in ["explicit_request", "out_of_scope", "hard_negative"]
        ]
        text = "".join(json.dumps(line) + "\n" for line in lines)
        (root / "runs" / run_id / "trajectories.jsonl").write_text(text)
        # No over-escalation column: the notebook must cope with missing metrics.
        rows.append({"run_id": run_id, "success": 0.3, "handoff_f1": 0.8})
    for model in ["R1", "R2"]:
        run_id = f"{model}-s1-20261006"
        (root / "runs" / run_id / "rollouts").mkdir(parents=True)
        episodes = [
            {
                "time": t,
                "validate": t % 50 == 0,
                "should_transfer": t % 2,
                "transferred": t % 3 == 0,
                "infra_failure": 0.0,
            }
            for t in range(256)
        ]
        text = "".join(json.dumps(e) + "\n" for e in episodes)
        (root / "runs" / run_id / "rollouts" / "part-000.jsonl").write_text(text)
        rows.append({"run_id": run_id})
    (root / "runs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_kaggle_notebook_runs_on_the_release_layout(tmp_path, monkeypatch):
    import pytest

    pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    from pathlib import Path

    matplotlib.use("Agg")
    path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_quickstart.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    _release_fixture(tmp_path)
    monkeypatch.setenv("HANDOFF_RELEASE", str(tmp_path))
    namespace: dict = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), path.name, "exec"), namespace)
    assert set(namespace["evals"]) == {"B2", "R1", "R2"}
    assert len(namespace["evals"]["R1"]) == 3  # the step-40 run is left out


def test_kaggle_notebook_is_clean():
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_quickstart.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        assert cell.get("outputs", []) == []  # Kaggle runs it; no stale outputs
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))


def test_kaggle_markdown_avoids_autolinks_and_soft_breaks():
    from handoff.release import kaggle_markdown

    card = (
        "Code: <https://example.org/x>. Kaggle mirror: <https://kaggle.com/y>.\n"
        "Second line\nof a paragraph.\n\n> quote\n> continued\n\n"
        "- item\n  continued\n- next\n\n| a |\n| b |\n\n```python\nx = 1\ny = 2\n```\n"
    )
    out = kaggle_markdown(card)
    assert "<https" not in out and "kaggle.com/y" not in out
    assert "[https://example.org/x](https://example.org/x). Second line of a" in out
    assert "> quote continued" in out
    assert "- item continued\n- next" in out
    assert "| a |\n| b |" in out and "x = 1\ny = 2" in out
    assert kaggle_markdown("see `<host>`") == "see `<host>`"
