"""The CPU half of the cluster pipeline, script by script, on the toy domain.

Runs the real command-line entry points in-process: build the frozen task sets,
evaluate a (scripted) policy, then write the report tables, a review packet and
the demo page from that run. Only the model servers and the benchmark data are
replaced (see ``toy_tau2``). Skipped when tau2 is not installed.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
pytest.importorskip("tau2")

import tau2.user.user_simulator as user_simulator  # noqa: E402
import tau2.utils.llm_utils as llm_utils  # noqa: E402
import toy_tau2  # noqa: E402
from fakes import COMMIT  # noqa: E402

import handoff.envs.handoff_tasks as handoff_tasks  # noqa: E402
import handoff.envs.tau2_adapter as tau2_adapter  # noqa: E402
from handoff.sim.endpoints import Endpoint, write_endpoint  # noqa: E402

pytestmark = pytest.mark.enable_socket
DOMAIN = toy_tau2.DOMAIN
ROOT = Path(__file__).resolve().parents[1]


def run_script(name, argv, monkeypatch):
    """Import ``scripts/<name>.py`` fresh and run its ``main`` with ``argv``."""
    spec = importlib.util.spec_from_file_location(
        f"script_{name}", ROOT / "scripts" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [name, *argv])
    return module.main()


@pytest.fixture
def toy_cluster(tmp_path, monkeypatch):
    """Scratch, endpoints, toy tau2 data and scripted models."""
    toy_tau2.register_toy_domain()
    data = tmp_path / "tau2data"
    domain_dir = data / "tau2" / "domains" / DOMAIN
    domain_dir.mkdir(parents=True)
    (domain_dir / "tasks.json").write_text(json.dumps(toy_tau2.UPSTREAM_TASKS))
    (domain_dir / "split_tasks.json").write_text(json.dumps(toy_tau2.OFFICIAL_SPLITS))
    (domain_dir / "policy.md").write_text(toy_tau2.POLICY)
    monkeypatch.setattr(tau2_adapter, "tau2_data_dir", lambda: data)
    monkeypatch.setitem(
        handoff_tasks.OUT_OF_SCOPE_TEMPLATES,
        DOMAIN,
        handoff_tasks.OUT_OF_SCOPE_TEMPLATES["airline"],
    )
    monkeypatch.setattr(llm_utils, "completion", toy_tau2.fake_completion)
    guidelines = tmp_path / "guidelines.md"
    guidelines.write_text("Simulate the user.<PERSONA_GUIDELINES>", encoding="utf-8")
    monkeypatch.setattr(user_simulator, "GLOBAL_USER_SIM_GUIDELINES_PATH", guidelines)
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("HANDOFF_SCRATCH", str(scratch))
    # The toy data are not a tau2 checkout; pretend they are the pinned commit.
    monkeypatch.setattr(tau2_adapter, "tau2_commit", lambda: COMMIT)
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    endpoints = scratch / "state" / "endpoints"
    for role, name in (
        ("policy", toy_tau2.POLICY_MODEL),
        ("usersim", toy_tau2.USER_MODEL),
    ):
        write_endpoint(
            endpoints / f"{role}.json",
            Endpoint(
                role=role,
                base_url="http://toy.invalid:1/v1",
                served_model_name=name,
                model_revision="c" * 40,
            ),
        )
    manifests = tmp_path / "manifests"
    paths = [
        f"paths.manifests_dir={manifests}",
        f"paths.handoff_tasks_dir={tmp_path / 'handoff_tasks'}",
    ]
    return {"tmp": tmp_path, "scratch": scratch, "endpoints": endpoints, "paths": paths}


def test_task_sets_evaluation_and_reports(toy_cluster, monkeypatch, capsys):
    paths = toy_cluster["paths"]
    counts = "{train:{explicit_request:1,out_of_scope:1,hard_negative:1}}"
    code = run_script(
        "build_task_sets",
        [f"domains=[{DOMAIN}]", f"+handoff_tasks.counts.{DOMAIN}={counts}", *paths],
        monkeypatch,
    )
    assert code == 0
    manifests = toy_cluster["tmp"] / "manifests"
    splits = json.loads((manifests / "splits_v1.json").read_text())
    assert splits["tau2_commit"] == COMMIT
    handoff = json.loads((manifests / "handoff_tasks_v1.json").read_text())
    kinds = {t["kind"] for t in handoff["domains"][DOMAIN]["tasks"]}
    assert kinds == {"explicit_request", "out_of_scope", "hard_negative"}
    with pytest.raises(SystemExit):  # frozen files are never overwritten silently
        run_script("build_task_sets", [f"domains=[{DOMAIN}]", *paths], monkeypatch)

    endpoints = toy_cluster["endpoints"]
    code = run_script(
        "run_eval",
        [
            "eval=smoke",
            f"eval.domains=[{DOMAIN}]",
            "eval.kinds=[upstream,explicit_request,out_of_scope,hard_negative]",
            "eval.max_tasks_per_domain=null",
            "eval.max_retries=0",
            "eval.bootstrap_resamples=10",
            "experiment=SMOKE",
            "tags=[toy]",
            "execution_kind=synthetic_test",
            "policy.label=toy",
            f"policy.endpoint_file={endpoints / 'policy.json'}",
            f"user_sim.endpoint_file={endpoints / 'usersim.json'}",
            *paths,
        ],
        monkeypatch,
    )
    assert code == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    run_dir = toy_cluster["scratch"] / "runs" / summary["run_id"]
    assert summary["episodes"] > 0 and summary["infra_failures"] == 0
    assert (
        json.loads((run_dir / "completion.json").read_text())["status"] == "completed"
    )
    provenance = json.loads((run_dir / "provenance.json").read_text())
    assert provenance["tau2_sha"] == COMMIT
    assert provenance["model_revisions"]["policy"] == "c" * 40

    reports = toy_cluster["tmp"] / "reports"
    run_id = summary["run_id"]
    assert (
        run_script(
            "make_report",
            [
                "--name",
                "toy",
                "--runs",
                f"B0={run_id}",
                "--allow-synthetic",
                "--resamples",
                "20",
                "--out-dir",
                str(reports / "toy"),
            ],
            monkeypatch,
        )
        == 0
    )
    assert (reports / "toy" / "main_table.md").exists()
    assert (reports / "toy" / "sources.json").exists()
    assert (
        run_script(
            "sample_review",
            [
                "failures",
                "--run",
                run_id,
                "--n",
                "2",
                "--out-dir",
                str(reports / "review"),
            ],
            monkeypatch,
        )
        == 0
    )
    assert (
        run_script(
            "sample_review",
            [
                "tasks",
                "--n",
                "3",
                "--out-dir",
                str(reports / "review"),
                "--paths",
                *paths,
            ],
            monkeypatch,
        )
        == 0
    )
    assert len(list((reports / "review").glob("*.md"))) == 2
    page = reports / "demo.html"
    assert (
        run_script("render_demo", ["--run", run_id, "--out", str(page)], monkeypatch)
        == 0
    )
    assert page.read_text().startswith("<!doctype html>")


def test_lost_test_episodes_resume_under_the_one_evaluation_rule(
    toy_cluster, monkeypatch, capsys
):
    """A test run that loses an episode is incomplete (exit 3); a second test run
    of the checkpoint is refused; resuming re-runs only the lost episode."""
    import litellm

    splits = {"train": ["0", "1", "2", "3"], "test": ["4", "5"]}
    monkeypatch.setattr(toy_tau2, "OFFICIAL_SPLITS", splits)
    domain_dir = toy_cluster["tmp"] / "tau2data" / "tau2" / "domains" / DOMAIN
    (domain_dir / "split_tasks.json").write_text(json.dumps(splits))
    paths = toy_cluster["paths"]
    counts = "{train:{explicit_request:1,out_of_scope:0,hard_negative:0}}"
    assert (
        run_script(
            "build_task_sets",
            [f"domains=[{DOMAIN}]", f"+handoff_tasks.counts.{DOMAIN}={counts}", *paths],
            monkeypatch,
        )
        == 0
    )
    user_calls = []
    outage = {"on": True}

    def flaky(**kwargs):
        if kwargs["model"] == f"openai/{toy_tau2.USER_MODEL}":
            system = [m for m in kwargs["messages"] if m["role"] == "system"]
            task_id = toy_tau2._marker(system)
            user_calls.append(task_id)
            if outage["on"] and task_id == "4":
                raise litellm.APIConnectionError(
                    message="simulator down", llm_provider="openai", model="x"
                )
        return toy_tau2.fake_completion(**kwargs)

    monkeypatch.setattr(llm_utils, "completion", flaky)
    endpoints = toy_cluster["endpoints"]
    common = [
        "execution_kind=synthetic_test",
        f"policy.endpoint_file={endpoints / 'policy.json'}",
        f"user_sim.endpoint_file={endpoints / 'usersim.json'}",
        *paths,
    ]
    evaluation = [
        "eval=test_main",
        f"eval.domains=[{DOMAIN}]",
        "eval.num_trials=1",
        "eval.max_retries=0",
        "eval.bootstrap_resamples=10",
        "experiment=B0",
        "policy.label=toy",
    ]
    capsys.readouterr()
    assert run_script("run_eval", [*evaluation, *common], monkeypatch) == 3
    output = capsys.readouterr().out
    assert "INCOMPLETE" in output
    summary = json.loads(
        next(line for line in output.splitlines() if line.startswith('{"run_id"'))
    )
    run_id = summary["run_id"]
    run_dir = toy_cluster["scratch"] / "runs" / run_id
    completion = json.loads((run_dir / "completion.json").read_text())
    assert completion["status"] == "failed"
    assert completion["error_code"] == "INFRA_FAILURES"
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert metrics["infra_failures"]["by_persona"] == {"neutral": 1}
    ledger = toy_cluster["scratch"] / "state" / "test_ledger.jsonl"
    entries = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [e["run_id"] for e in entries] == [run_id]
    assert entries[0]["tasks"] == [f"{DOMAIN}/4", f"{DOMAIN}/5"]

    # a second test evaluation of the same checkpoint is refused, forced or not
    with pytest.raises(SystemExit):
        run_script("run_eval", [*evaluation, *common], monkeypatch)
    assert "evaluated once per checkpoint" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run_script(
            "run_eval", [*evaluation, "force_test_rerun=true", *common], monkeypatch
        )
    assert "test_rerun_reason" in capsys.readouterr().err

    outage["on"] = False
    user_calls.clear()
    assert (
        run_script("run_eval", [f"resume_run_id={run_id}", *common], monkeypatch) == 0
    )
    assert set(user_calls) == {"4"}  # only the lost episode ran again
    completion = json.loads((run_dir / "completion.json").read_text())
    assert completion["status"] == "completed"
    assert (run_dir / "completion.attempt1.json").exists()
    attempts = (run_dir / "attempts.jsonl").read_text().splitlines()
    assert [json.loads(a)["resumed"] for a in attempts] == [False, True]
    trajectories = (run_dir / "trajectories.jsonl").read_text().splitlines()
    assert len(trajectories) == 2
    assert len(ledger.read_text().splitlines()) == 1
    with pytest.raises(SystemExit):  # a completed run is never resumed
        run_script("run_eval", [f"resume_run_id={run_id}", *common], monkeypatch)
    assert "already completed" in capsys.readouterr().err


def test_evaluation_refuses_tau2_data_that_changed_after_the_freeze(
    toy_cluster, monkeypatch, capsys
):
    paths = toy_cluster["paths"]
    counts = "{train:{explicit_request:1,out_of_scope:0,hard_negative:0}}"
    assert (
        run_script(
            "build_task_sets",
            [f"domains=[{DOMAIN}]", f"+handoff_tasks.counts.{DOMAIN}={counts}", *paths],
            monkeypatch,
        )
        == 0
    )
    policy = toy_cluster["tmp"] / "tau2data" / "tau2" / "domains" / DOMAIN / "policy.md"
    policy.write_text(toy_tau2.POLICY + "\n- A new rule.\n")
    endpoints = toy_cluster["endpoints"]
    capsys.readouterr()
    with pytest.raises(SystemExit):
        run_script(
            "run_eval",
            [
                "eval=smoke",
                f"eval.domains=[{DOMAIN}]",
                "experiment=SMOKE",
                "execution_kind=synthetic_test",
                "policy.label=toy",
                f"policy.endpoint_file={endpoints / 'policy.json'}",
                f"user_sim.endpoint_file={endpoints / 'usersim.json'}",
                *paths,
            ],
            monkeypatch,
        )
    assert "differ from the split manifest (policy)" in capsys.readouterr().err
    assert not (toy_cluster["scratch"] / "runs").exists() or not any(
        (toy_cluster["scratch"] / "runs").iterdir()
    )


def test_teacher_data_and_sft_pipeline(toy_cluster, monkeypatch, capsys):
    """TEACH rollouts → filtered token records → B2 checkpoint, all on CPU."""
    for module in ("torch", "transformers", "tokenizers", "peft"):
        pytest.importorskip(module)
    import tiny_hf

    paths = toy_cluster["paths"]
    counts = "{train:{explicit_request:1,out_of_scope:0,hard_negative:1}}"
    assert (
        run_script(
            "build_task_sets",
            [f"domains=[{DOMAIN}]", f"+handoff_tasks.counts.{DOMAIN}={counts}", *paths],
            monkeypatch,
        )
        == 0
    )
    endpoints = toy_cluster["endpoints"]
    assert (
        run_script(
            "run_eval",
            [
                "eval=teacher_train",
                f"eval.domains=[{DOMAIN}]",
                "eval.personas=[neutral,terse]",
                "eval.num_trials=1",
                "eval.max_concurrency=4",
                "eval.max_retries=0",
                "eval.bootstrap_resamples=10",
                "experiment=TEACH",
                "execution_kind=synthetic_test",
                "policy.label=toy-teacher",
                f"policy.endpoint_file={endpoints / 'policy.json'}",
                f"user_sim.endpoint_file={endpoints / 'usersim.json'}",
                *paths,
            ],
            monkeypatch,
        )
        == 0
    )
    teacher_run = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["run_id"]

    scratch = toy_cluster["scratch"]
    model_dir = tiny_hf.save_tiny_model(toy_cluster["tmp"] / "tiny")
    lock = scratch / "state" / "models.lock.json"
    lock.write_text(
        json.dumps({"qwen3_4b": {"local_path": str(model_dir), "revision": "d" * 40}})
    )
    assert (
        run_script(
            "build_sft_data",
            [f"teacher_runs=[{teacher_run}]", "dataset_name=toy_b2", *paths],
            monkeypatch,
        )
        == 0
    )
    dataset = scratch / "datasets" / "toy_b2"
    manifest = json.loads((dataset / "manifest.json").read_text())
    kept = manifest["statistics"]["n_kept_episodes"]
    assert kept >= 2 and manifest["statistics"]["splits"] == ["train"]
    capsys.readouterr()

    assert (
        run_script(
            "train_sft",
            [
                "dataset_name=toy_b2",
                "execution_kind=synthetic_test",
                "sft.bf16=false",
                "sft.max_length=4096",
                "sft.num_train_epochs=1",
                "sft.gradient_accumulation_steps=1",
                *paths,
            ],
            monkeypatch,
        )
        == 0
    )
    checkpoint = (
        capsys.readouterr().out.strip().splitlines()[-1].split("checkpoint: ")[-1]
    )
    assert (Path(checkpoint) / "config.json").exists()
    run_dir = Path(checkpoint).parent
    assert (
        json.loads((run_dir / "completion.json").read_text())["status"] == "completed"
    )


def test_rl_launch_dry_run(toy_cluster, monkeypatch, capsys):
    """RL data, agent-loop settings and a verl command validated against verl."""
    for module in ("verl", "pandas", "pyarrow", "transformers"):
        pytest.importorskip(module)
    import tiny_hf

    paths = toy_cluster["paths"]
    counts = "{train:{explicit_request:1,out_of_scope:1,hard_negative:1}}"
    assert (
        run_script(
            "build_task_sets",
            [f"domains=[{DOMAIN}]", f"+handoff_tasks.counts.{DOMAIN}={counts}", *paths],
            monkeypatch,
        )
        == 0
    )
    capsys.readouterr()
    init = tiny_hf.save_tiny_model(toy_cluster["tmp"] / "b2")
    endpoints = toy_cluster["endpoints"]
    assert (
        run_script(
            "train_rl",
            [
                "rl=grpo",
                "reward=r2",
                "experiment=R2",
                "tags=[toy]",
                "execution_kind=synthetic_test",
                "dry_run=true",
                f"policy.init_path={init}",
                f"user_sim.endpoint_file={endpoints / 'usersim.json'}",
                f"data.domains=[{DOMAIN}]",
                "data.train_personas=[neutral,terse]",
                *paths,
            ],
            monkeypatch,
        )
        == 0
    )
    command = capsys.readouterr().out.strip().splitlines()[-1]
    assert "handoff.train.verl_main" in command and "tau2_agent" in command
    run_dir = next((toy_cluster["scratch"] / "runs").glob("R2-toy-*"))
    summary = json.loads((run_dir / "data" / "summary.json").read_text())
    assert summary["train"]["split"] == {"train": summary["train"]["n_rows"]}
    assert summary["val"]["split"] == {"dev": summary["val"]["n_rows"]}
    assert summary["train"]["persona"].keys() == {"neutral", "terse"}
    import pandas as pd

    rows = pd.read_parquet(run_dir / "data" / "train.parquet")
    task = json.loads(rows.iloc[0]["extra_info"]["task_json"])
    assert task["id"].endswith(("::neutral", "::terse"))
    settings = json.loads((run_dir / "agent_loop_settings.json").read_text())
    assert settings["reward"]["lambda_fp"] > 0 and settings["run_id"] == run_dir.name
    assert (
        json.loads((run_dir / "completion.json").read_text())["error_code"] == "DRY_RUN"
    )


def test_interactive_demo_scores_a_typed_conversation(toy_cluster, monkeypatch, capsys):
    """scripts/demo_chat.py with typed user turns (toy task 0)."""
    typed = iter(
        ["Hi, I want to cancel my order o1. [toy:0]", "Yes, please cancel it.", "/quit"]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(typed))
    endpoints = toy_cluster["endpoints"]
    code = run_script(
        "demo_chat",
        [
            "--domain",
            DOMAIN,
            "--task-id",
            "0",
            "--endpoint-file",
            str(endpoints / "policy.json"),
        ],
        monkeypatch,
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "tool: cancel_order" in out
    assert "task reward: 1.0" in out and "transferred: False" in out
    assert "violation" not in out
