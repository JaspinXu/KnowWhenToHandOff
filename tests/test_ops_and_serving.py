"""Endpoints, server launching, run IDs, fingerprints, ledgers and logs."""

import json
import os
import stat
import sys
from datetime import date
from pathlib import Path

import pytest

from handoff.ops.provenance import RUN_ID_PATTERN
from handoff.ops.runs import (
    LedgerScope,
    append_experiment_log,
    checkpoint_fingerprint,
    ledger_conflicts,
    make_run_id,
    model_identity,
    record_in_ledger,
    safe_name,
)
from handoff.ops.slurm import parse_duration
from handoff.sim import endpoints as endpoints_module
from handoff.sim.endpoints import (
    Endpoint,
    ensure_api_key,
    litellm_args,
    load_api_key_into_env,
    read_endpoint,
    server_identity,
    wait_until_ready,
    write_endpoint,
)
from handoff.sim.serving import (
    ServeSpec,
    endpoint_for,
    run_server,
    serve_spec_from_config,
)


def endpoint(**overrides):
    values = {
        "role": "usersim",
        "base_url": "http://node1:8001/v1",
        "served_model_name": "qwen3-30b-a3b",
        "model_revision": "a" * 40,
    }
    values.update(overrides)
    return Endpoint(**values)


def test_endpoint_round_trip_and_validation(tmp_path):
    path = tmp_path / "e" / "usersim.json"
    write_endpoint(path, endpoint())
    assert read_endpoint(path) == endpoint()
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert endpoint().litellm_model == "openai/qwen3-30b-a3b"
    with pytest.raises(ValueError):
        endpoint(base_url="ftp://x")
    with pytest.raises(ValueError):
        endpoint(base_url="http://user:pw@node/v1")
    with pytest.raises(ValueError):
        endpoint(served_model_name="bad name")


def test_api_key_is_private_and_stable(tmp_path, monkeypatch):
    path = tmp_path / "secrets" / "key"
    key = ensure_api_key(path)
    assert ensure_api_key(path) == key and len(key) == 48
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    load_api_key_into_env(path)
    assert os.environ["OPENAI_API_KEY"] == key
    # A key inherited from the login shell must never reach the local servers.
    monkeypatch.setenv("OPENAI_API_KEY", "personal")
    load_api_key_into_env(path)
    assert os.environ["OPENAI_API_KEY"] == key
    load_api_key_into_env(tmp_path / "missing")
    assert os.environ["OPENAI_API_KEY"] == key
    monkeypatch.delenv("OPENAI_API_KEY")
    load_api_key_into_env(None)
    assert os.environ["OPENAI_API_KEY"] == "EMPTY"


def test_litellm_args_never_contain_the_key():
    args = litellm_args(endpoint(), temperature=0.0, max_tokens=64, extra_body={"x": 1})
    assert args == {
        "api_base": "http://node1:8001/v1",
        "temperature": 0.0,
        "max_tokens": 64,
        "extra_body": {"x": 1},
    }


def test_litellm_args_send_the_models_chat_template_kwargs():
    thinking_off = endpoint(chat_template_kwargs={"enable_thinking": False})
    assert litellm_args(thinking_off, temperature=0.0)["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    args = litellm_args(
        thinking_off,
        temperature=0.0,
        extra_body={"x": 1, "chat_template_kwargs": {"enable_thinking": True}},
    )
    assert args["extra_body"] == {
        "x": 1,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    assert "extra_body" not in litellm_args(endpoint(), temperature=0.0)


def test_server_identity_ignores_address_memory_and_path():
    first = endpoint(
        max_model_len=32768,
        server_args=[
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            "/a",
            "--host",
            "0.0.0.0",
            "--port",
            "8001",
            "--gpu-memory-utilization",
            "0.9",
            "--max-model-len",
            "32768",
            "--tool-call-parser",
            "hermes",
        ],
    )
    moved = endpoint(
        base_url="http://node9:9/v1",
        max_model_len=32768,
        server_args=[
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            "/b",
            "--host",
            "0.0.0.0",
            "--port",
            "9",
            "--gpu-memory-utilization",
            "0.5",
            "--max-model-len",
            "32768",
            "--tool-call-parser",
            "hermes",
        ],
    )
    identity = server_identity(first)
    assert identity == server_identity(moved)
    assert identity["server_flags"] == [
        "--max-model-len",
        "32768",
        "--tool-call-parser",
        "hermes",
    ]
    changed = endpoint(max_model_len=32768, server_args=["--tool-call-parser", "x"])
    assert server_identity(changed) != identity


def test_wait_until_ready_polls_models(monkeypatch):
    responses = iter(
        [
            OSError("refused"),
            {"data": [{"id": "other"}]},
            {"data": [{"id": "qwen3-30b-a3b"}]},
        ]
    )

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(self.payload).encode()

    def fake_urlopen(request, timeout):
        item = next(responses)
        if isinstance(item, Exception):
            raise item
        assert request.get_header("Authorization") == "Bearer k"
        return Response(item)

    monkeypatch.setattr(endpoints_module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(endpoints_module.time, "sleep", lambda s: None)
    wait_until_ready(endpoint(), api_key="k", timeout_s=60, interval_s=0)
    monkeypatch.setattr(
        endpoints_module.urllib.request,
        "urlopen",
        lambda r, timeout: (_ for _ in ()).throw(OSError("down")),
    )
    with pytest.raises(TimeoutError):
        wait_until_ready(endpoint(), api_key=None, timeout_s=0.01, interval_s=0)


def test_serve_spec_command_and_lock_resolution(tmp_path):
    model = {
        "name": "qwen3_4b",
        "repository": "Qwen/Qwen3-4B-Instruct-2507",
        "served_name": "qwen3-4b",
        "tool_call_parser": "hermes",
        "max_model_len": 32768,
        "serve_args": ["--enforce-eager"],
    }
    lock = {"qwen3_4b": {"local_path": "/scratch/hf/qwen", "revision": "b" * 40}}
    spec = serve_spec_from_config(
        model, role="policy", port=8123, models_lock=lock, seed=3
    )
    command = spec.command()
    assert command[command.index("--model") + 1] == "/scratch/hf/qwen"
    assert command[command.index("--tool-call-parser") + 1] == "hermes"
    assert command[command.index("--generation-config") + 1] == "vllm"
    assert command[-1] == "--enforce-eager" and "--api-key" not in command
    assert spec.revision == "b" * 40
    published = endpoint_for(
        serve_spec_from_config(
            {**model, "chat_template_kwargs": {"enable_thinking": False}},
            role="policy",
            port=8123,
            models_lock=lock,
        ),
        host="node3",
    )
    assert published.chat_template_kwargs == {"enable_thinking": False}
    assert published.max_model_len == 32768
    assert published.server_args[:2] == ["-m", "vllm.entrypoints.openai.api_server"]
    with pytest.raises(ValueError, match="models lock"):
        serve_spec_from_config(
            {**model, "name": "missing"}, role="policy", port=1, models_lock={}
        )
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    local = serve_spec_from_config(
        {**model, "local_path": str(checkpoint)}, role="policy", port=1, models_lock={}
    )
    assert local.revision.startswith("sha256:") and local.model_path == str(checkpoint)


def test_run_server_publishes_and_removes_endpoint(tmp_path, monkeypatch):
    import handoff.sim.serving as serving

    published = []
    monkeypatch.setattr(
        serving, "wait_until_ready", lambda *a, **k: published.append(1)
    )
    spec = ServeSpec("policy", "/m", "m", "c" * 40, "repo", 9000)
    target = tmp_path / "endpoints" / "policy.json"
    code = run_server(
        spec,
        endpoint_file=target,
        api_key="secret",
        command=[
            sys.executable,
            "-c",
            "import time, os; time.sleep(1.0); "
            "assert os.environ['VLLM_API_KEY'] == 'secret'",
        ],
    )
    assert code == 0 and published == [1]
    assert not target.exists()
    assert endpoint_for(spec, host="node7").base_url == "http://node7:9000/v1"


def test_run_ids_follow_readme_naming():
    day = date(2026, 10, 24)
    assert make_run_id("R2", seed=1, tags=("fp05",), day=day) == "R2-fp05-s1-20261024"
    assert (
        make_run_id("B0", seed=0, tags=("test_main",), day=day)
        == "B0-test_main-s0-20261024"
    )
    assert RUN_ID_PATTERN.fullmatch("SMOKE-s0-20261001")
    assert RUN_ID_PATTERN.fullmatch("TEACH-teacher_train-s0-20261012")
    with pytest.raises(ValueError):
        make_run_id("X9", seed=0, day=day)


def test_same_day_run_ids_do_not_collide(tmp_path):
    day = date(2026, 10, 24)
    first = make_run_id("R2", seed=1, tags=("fp05",), day=day, runs_dir=tmp_path)
    assert first == "R2-fp05-s1-20261024"
    (tmp_path / first).mkdir()
    second = make_run_id("R2", seed=1, tags=("fp05",), day=day, runs_dir=tmp_path)
    assert second == "R2-fp05-try2-s1-20261024"
    (tmp_path / second).mkdir()
    third = make_run_id("R2", seed=1, tags=("fp05",), day=day, runs_dir=tmp_path)
    assert third == "R2-fp05-try3-s1-20261024" and RUN_ID_PATTERN.fullmatch(third)


def test_checkpoint_fingerprint_changes_with_weights(tmp_path):
    (tmp_path / "config.json").write_text('{"a": 1}')
    (tmp_path / "model.safetensors").write_bytes(b"12345")
    first = checkpoint_fingerprint(tmp_path)
    (tmp_path / "model.safetensors").write_bytes(b"123456")
    assert checkpoint_fingerprint(tmp_path) != first
    assert first.startswith("sha256:") and len(first) == 71
    with pytest.raises(ValueError):
        checkpoint_fingerprint(tmp_path / "missing")
    assert model_identity({"local_path": str(tmp_path)}, {})[0] == str(tmp_path)


def test_checkpoints_with_same_shapes_get_different_fingerprints(tmp_path):
    """Two RL steps of one model: same file names and sizes, different weights."""
    fingerprints = []
    for name, weights in (("step10", b"\x01" * 64), ("step20", b"\x02" * 64)):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "config.json").write_text('{"model_type": "qwen3"}')
        (directory / "model.safetensors").write_bytes(weights)
        fingerprints.append(checkpoint_fingerprint(directory))
    assert fingerprints[0] != fingerprints[1]


def test_checkpoint_fingerprint_is_cached_until_a_file_changes(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"abc")
    first = checkpoint_fingerprint(tmp_path)
    cache = tmp_path / ".handoff_fingerprint.json"
    record = json.loads(cache.read_text())
    record["fingerprint"] = "sha256:" + "0" * 64
    cache.write_text(json.dumps(record))
    assert checkpoint_fingerprint(tmp_path) == "sha256:" + "0" * 64  # cache hit
    weights.write_bytes(b"abd")
    os.utime(weights, ns=(1, 1))
    assert checkpoint_fingerprint(tmp_path) not in (first, "sha256:" + "0" * 64)


def scope(tasks=("airline/1", "airline/2"), personas=("neutral",), **overrides):
    values = {
        "policy_revision": "p",
        "user_sim": "qwen3-30b-a3b@u",
        "tasks": tuple(tasks),
        "personas": tuple(personas),
    }
    values.update(overrides)
    return LedgerScope(**values)


def test_ledger_refuses_overlapping_test_runs_whatever_their_name(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    assert ledger_conflicts(ledger, scope(), run_id="B0-a") == []
    assert record_in_ledger(ledger, scope(), run_id="B0-a", eval_name="test_main")
    # the same tasks in other styles, other tasks, another checkpoint or another
    # simulator are separate evaluations
    styles = scope(personas=("terse", "verbose"))
    assert ledger_conflicts(ledger, styles, run_id="B0-b") == []
    assert ledger_conflicts(ledger, scope(tasks=("retail/9",)), run_id="B0-c") == []
    assert ledger_conflicts(ledger, scope(policy_revision="q"), run_id="R2-a") == []
    other_sim = scope(user_sim="other-sim@m")
    assert ledger_conflicts(ledger, other_sim, run_id="B0-d") == []
    # a renamed protocol that reruns part of test_main conflicts
    partial = scope(tasks=("airline/2", "airline/3"), personas=("neutral", "terse"))
    conflicts = ledger_conflicts(ledger, partial, run_id="B0-e")
    assert [c["run_id"] for c in conflicts] == ["B0-a"]
    # resuming the same run is not a second evaluation, and records once
    assert ledger_conflicts(ledger, scope(), run_id="B0-a") == []
    assert not record_in_ledger(ledger, scope(), run_id="B0-a", eval_name="test_main")
    assert record_in_ledger(
        ledger, partial, run_id="B0-e", eval_name="x", forced_reason="server crash"
    )
    records = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [r["run_id"] for r in records] == ["B0-a", "B0-e"]
    assert records[1]["forced_reason"] == "server crash"


def test_legacy_ledger_records_without_scope_overlap_everything(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"policy_revision": "p", "run_id": "old"}) + "\n")
    assert [c["run_id"] for c in ledger_conflicts(ledger, scope(), run_id="new")] == [
        "old"
    ]


def test_slurm_durations():
    assert parse_duration("1-02:03:04") == 93784
    assert parse_duration("02:03:04") == 7384
    assert parse_duration("03:04") == 184
    assert parse_duration("5") == 300
    assert parse_duration("2-12") == 2 * 86400 + 12 * 3600
    assert parse_duration("UNLIMITED") is None
    assert parse_duration("INVALID") is None


def test_experiment_log_append(tmp_path):
    log = tmp_path / "log.md"
    with pytest.raises(ValueError):
        append_experiment_log(log, run_id="r", purpose="p", headline="h", decision="d")
    log.write_text(
        "| Date | Run ID | Purpose | Headline metrics (95% CI) | Decision |\n"
        "| --- | --- | --- | --- | --- |\n"
    )
    append_experiment_log(
        log,
        run_id="B0-s0-20261008",
        purpose="audit",
        headline="success 0.4",
        decision="keep",
    )
    assert log.read_text().splitlines()[-1].endswith("| keep |")
    with pytest.raises(ValueError):
        append_experiment_log(
            log, run_id="r", purpose="a|b", headline="h", decision="d"
        )
    assert safe_name("R2 fp=0.5/x") == "R2-fp-0.5-x"


def test_git_state_ignores_docs_and_reports_only(tmp_path):
    import subprocess

    from handoff.ops.runs import git_state

    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "experiment-log.md").write_text("| a |\n")
    (tmp_path / "code.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    sha, dirty = git_state(tmp_path)
    assert len(sha) == 40 and not dirty
    (tmp_path / "docs" / "experiment-log.md").write_text("| a |\n| b |\n")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "table.md").write_text("new\n")
    assert git_state(tmp_path)[1] is False
    (tmp_path / "data.json").write_text("{}\n")
    assert git_state(tmp_path)[1] is True
    (tmp_path / "data.json").unlink()
    (tmp_path / "code.py").write_text("x = 2\n")
    assert git_state(tmp_path)[1] is True


@pytest.mark.skipif(os.name != "posix", reason="Slurm tools and PIDs are POSIX")
def test_model_server_wait_skips_stale_endpoints(tmp_path, monkeypatch, capsys):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "model_server", root / "scripts" / "model_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    squeue = bin_dir / "squeue"
    squeue.write_text("#!/bin/sh\necho 'Invalid job id specified' >&2\nexit 1\n")
    squeue.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HANDOFF_SCRATCH", str(tmp_path / "scratch"))
    endpoint_file = tmp_path / "usersim.json"
    write_endpoint(endpoint_file, endpoint(slurm_job_id="123"))
    base = ["model_server", "wait", "--endpoint-file", str(endpoint_file)]
    monkeypatch.setattr(sys, "argv", [*base, "--timeout", "0.3", "--interval", "0.05"])
    with pytest.raises(SystemExit):
        module.main()
    assert "stale endpoint of ended job 123" in capsys.readouterr().out
    monkeypatch.setattr(sys, "argv", [*base, "--pid", "999999999", "--timeout", "5"])
    with pytest.raises(SystemExit):
        module.main()
    assert "exited before it was ready" in capsys.readouterr().err


@pytest.mark.skipif(os.name != "posix", reason="Slurm tools are POSIX")
def test_model_server_wait_checks_the_server_outlives_the_client(
    tmp_path, monkeypatch, capsys
):
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "model_server", root / "scripts" / "model_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    squeue = bin_dir / "squeue"
    # job 123 (the simulator) has 5 hours left, job 456 (the client) a day
    squeue.write_text(
        "#!/bin/sh\n"
        'case "$3" in 123) t=05:00:00;; *) t=1-00:00:00;; esac\n'
        'if [ "$5" = "%T" ]; then echo RUNNING; else echo "$t"; fi\n'
    )
    squeue.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("HANDOFF_SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.setattr(module, "wait_until_ready", lambda *a, **k: None)
    endpoint_file = tmp_path / "usersim.json"
    write_endpoint(endpoint_file, endpoint(slurm_job_id="123"))
    base = ["model_server", "wait", "--endpoint-file", str(endpoint_file)]
    monkeypatch.setattr(sys, "argv", [*base, "--outlive-job", "456"])
    with pytest.raises(SystemExit):
        module.main()
    assert "ends in 5.0 h" in capsys.readouterr().err
    monkeypatch.setattr(sys, "argv", [*base, "--outlive-job", "456", "--warn-only"])
    assert module.main() == 0
    assert "WARNING" in capsys.readouterr().out


def test_export_merges_saved_steps_once_and_keeps_metadata(
    tmp_path, monkeypatch, capsys
):
    import importlib.util

    import handoff.train.export as export

    calls = []

    def fake_merger(command, check):
        calls.append(command)
        work = Path(command[command.index("--target_dir") + 1])
        work.mkdir(parents=True)
        (work / "model.safetensors").write_bytes(b"w")
        (work / "config.json").write_text("{}")

    monkeypatch.setattr(export.subprocess, "run", fake_merger)
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "export_checkpoint", root / "scripts" / "export_checkpoint.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scratch = tmp_path / "scratch"
    monkeypatch.setenv("HANDOFF_SCRATCH", str(scratch))
    run = scratch / "runs" / "R1-s1-20261020"
    for step in (10, 20):
        hf = run / "checkpoints" / f"global_step_{step}" / "actor" / "huggingface"
        hf.mkdir(parents=True)
        (hf / "config.json").write_text('{"from": "verl"}')
        (hf / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["export", "R1-s1-20261020"])
    assert module.main() == 0
    target = Path(capsys.readouterr().out.strip())
    assert target == run / "export" / "global_step_20"
    assert {p.name for p in target.iterdir()} == {
        "model.safetensors",
        "config.json",
        "tokenizer.json",
        "handoff_export.json",
    }
    assert json.loads((target / "config.json").read_text()) == {}  # merger's own
    assert len(calls) == 1 and "verl.model_merger" in calls[0]
    assert module.main() == 0  # already exported: nothing to do
    assert len(calls) == 1
    capsys.readouterr()
    monkeypatch.setattr(sys, "argv", ["export", "R1-s1-20261020", "--step", "all"])
    assert module.main() == 0
    assert len(calls) == 2 and (run / "export" / "global_step_10").is_dir()
    # a step whose resumable checkpoint was rotated away keeps its export
    import shutil

    shutil.rmtree(run / "checkpoints" / "global_step_10" / "actor")
    monkeypatch.setattr(sys, "argv", ["export", "R1-s1-20261020", "--step", "10"])
    capsys.readouterr()
    assert module.main() == 0
    assert capsys.readouterr().out.strip().endswith("global_step_10")
    monkeypatch.setattr(sys, "argv", ["export", "R1-s1-20261020", "--step", "30"])
    with pytest.raises(SystemExit):
        module.main()
    # with only exports left, "latest" is the highest step (100 after 20)
    shutil.rmtree(run / "checkpoints")
    (run / "export" / "global_step_100").mkdir()
    (run / "export" / "global_step_100" / "handoff_export.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["export", "R1-s1-20261020", "--step", "latest"])
    capsys.readouterr()
    assert module.main() == 0
    assert capsys.readouterr().out.strip().endswith("global_step_100")


@pytest.mark.skipif(os.name != "posix", reason="tau2 runs on the Linux cluster")
def test_tau2_commit_comes_from_a_clean_checkout(tmp_path):
    import subprocess

    from handoff.envs.tau2_adapter import checkout_commit

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-q")
    git("config", "user.email", "t@example.com")
    git("config", "user.name", "t")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "code.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-q", "-m", "init")
    head = git("rev-parse", "HEAD")
    assert checkout_commit(tmp_path) == head
    # a sub-directory is not the checkout's top level (e.g. a venv inside a repo)
    assert checkout_commit(tmp_path / "src") is None
    (tmp_path / "untracked.txt").write_text("results\n")
    assert checkout_commit(tmp_path) == head  # untracked files are fine
    (tmp_path / "src" / "code.py").write_text("x = 2\n")
    with pytest.raises(RuntimeError, match="modified tracked files"):
        checkout_commit(tmp_path)
    assert checkout_commit(tmp_path / "not-a-repo-at-all") is None
