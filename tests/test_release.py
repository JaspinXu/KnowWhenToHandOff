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
    assert "/kaggle/input/knowwhentohandoff-data" in description
    assert "load_dataset" not in description and "## Licence" in description
    assert meta["expectedUpdateFrequency"] == "never"
    assert "license: mit" not in str(meta["description"])
    assert "Licence: MIT" in str(meta["description"])
    assert 6 <= len(str(meta["title"])) <= 50
    assert 20 <= len(str(meta["subtitle"])) <= 80
