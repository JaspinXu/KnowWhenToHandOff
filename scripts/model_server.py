"""Serve a model with vLLM and publish its endpoint file; or wait for one.

Subcommands:

``serve``  start vLLM for a model config (``--model qwen3_4b``) or a trained
           checkpoint (``--local-path DIR --served-name NAME``), write the
           endpoint file once the server answers, remove it on exit.
``wait``   block until an endpoint file exists and its server answers. Endpoint
           files left behind by a Slurm job that has ended are ignored, and
           ``--pid`` stops waiting as soon as a local server process dies.
           ``--outlive-job JOB`` fails (or with ``--warn-only`` warns) when the
           server's Slurm job will end before job ``JOB``.
``show``   print an endpoint file.

The shared API key lives in ``<scratch>/state/secrets/vllm_api_key`` and is
passed to vLLM through the environment, never on the command line.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf

from handoff.cli import CliError, compose_config, fail, repo_root, require_scratch
from handoff.ops.runs import read_models_lock
from handoff.ops.slurm import job_alive, time_left
from handoff.sim.endpoints import ensure_api_key, read_endpoint, wait_until_ready
from handoff.sim.serving import run_server, serve_spec_from_config

ROOT = repo_root(__file__)


def _model_config(args: argparse.Namespace) -> dict:
    if args.local_path:
        if not args.served_name:
            raise CliError("--local-path needs --served-name")
        base = OmegaConf.load(
            ROOT / "configs" / "model" / f"{args.model or 'custom'}.yaml"
        )
        model = OmegaConf.to_container(base, resolve=True)
        model.update(
            {
                "local_path": args.local_path,
                "served_name": args.served_name,
                "name": args.served_name,
            }
        )
        return model
    path = ROOT / "configs" / "model" / f"{args.model}.yaml"
    if not path.exists():
        raise CliError(f"unknown model config {args.model!r}")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def _serve(args: argparse.Namespace, paths) -> int:
    model = _model_config(args)
    if args.tool_call_parser is not None:
        model["tool_call_parser"] = args.tool_call_parser or None
    spec = serve_spec_from_config(
        model,
        role=args.role,
        port=args.port,
        models_lock=read_models_lock(Path(paths.models_lock)),
        server={
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            **({"max_model_len": args.max_model_len} if args.max_model_len else {}),
        },
        seed=args.seed,
    )
    print(json.dumps({"serving": asdict(spec)}, indent=2), flush=True)
    key = ensure_api_key(Path(paths.api_key_file))
    return run_server(
        spec,
        endpoint_file=Path(args.endpoint_file),
        api_key=key,
        ready_timeout_s=args.ready_timeout,
    )


def _check_outlives(server_job: str, client_job: str, *, warn_only: bool) -> None:
    """Refuse (or warn) when the server's job ends before the client's job."""
    server_left = time_left(server_job)
    client_left = time_left(client_job)
    if server_left is None or client_left is None or server_left >= client_left:
        return
    message = (
        f"the server job {server_job} ends in {server_left / 3600:.1f} h, before "
        f"this job ({client_left / 3600:.1f} h left)"
    )
    if warn_only:
        print(f"WARNING: {message}; restart it in time (the run waits for it)")
        return
    raise CliError(
        f"{message}; restart it first (slurm/submit.sh serve) or set ALLOW_SHORT_SIM=1"
    )


def _process_alive(pid: int | None) -> bool:
    # os.kill(pid, 0) only probes on POSIX; on Windows it would kill the process.
    if pid is None or os.name != "posix":
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait(args: argparse.Namespace, paths) -> int:
    deadline = time.monotonic() + args.timeout
    path = Path(args.endpoint_file)
    stale_reported = False
    while True:
        if not _process_alive(args.pid):
            raise CliError(f"server process {args.pid} exited before it was ready")
        if path.exists():
            endpoint = read_endpoint(path)
            if job_alive(endpoint.slurm_job_id) is not False:
                break
            if not stale_reported:
                print(f"ignoring stale endpoint of ended job {endpoint.slurm_job_id}")
                stale_reported = True
        if time.monotonic() > deadline:
            raise CliError(f"no live endpoint at {path} after {args.timeout}s")
        time.sleep(args.interval)
    key_file = Path(paths.api_key_file)
    key = key_file.read_text(encoding="utf-8").strip() if key_file.exists() else None
    wait_until_ready(
        endpoint,
        api_key=key,
        timeout_s=max(deadline - time.monotonic(), 60),
        interval_s=args.interval,
    )
    if args.outlive_job:
        _check_outlives(
            endpoint.slurm_job_id, args.outlive_job, warn_only=args.warn_only
        )
    print(f"ready: {endpoint.role} {endpoint.served_model_name} at {endpoint.base_url}")
    return 0


def main() -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", nargs="*", default=[], help="paths overrides")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="start a vLLM server")
    serve.add_argument("--model", default=None, help="model config name")
    serve.add_argument("--local-path", default=None, help="checkpoint directory")
    serve.add_argument("--served-name", default=None)
    serve.add_argument(
        "--role", required=True, choices=["policy", "usersim", "teacher"]
    )
    serve.add_argument("--port", type=int, required=True)
    serve.add_argument("--endpoint-file", required=True)
    serve.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    serve.add_argument("--tensor-parallel-size", type=int, default=1)
    serve.add_argument("--max-model-len", type=int, default=None)
    serve.add_argument("--tool-call-parser", default=None, help="'' disables")
    serve.add_argument("--seed", type=int, default=0)
    serve.add_argument("--ready-timeout", type=float, default=3600.0)
    wait = commands.add_parser("wait", help="wait for an endpoint to be ready")
    wait.add_argument("--endpoint-file", required=True)
    wait.add_argument("--timeout", type=float, default=3600.0)
    wait.add_argument("--pid", type=int, default=None, help="local server process")
    wait.add_argument("--interval", type=float, default=10.0, help="poll seconds")
    wait.add_argument(
        "--outlive-job",
        default=None,
        help="Slurm job that needs the server: fail if the server's job ends first",
    )
    wait.add_argument(
        "--warn-only", action="store_true", help="with --outlive-job: only warn"
    )
    show = commands.add_parser("show", help="print an endpoint file")
    show.add_argument("--endpoint-file", required=True)
    args = parser.parse_args()
    try:
        config = compose_config(ROOT, "base", args.paths)
        require_scratch(config)
        if args.command == "serve":
            if not (args.model or args.local_path):
                raise CliError("serve needs --model or --local-path")
            return _serve(args, config.paths)
        if args.command == "wait":
            return _wait(args, config.paths)
        print(json.dumps(asdict(read_endpoint(Path(args.endpoint_file))), indent=2))
        return 0
    except (CliError, TimeoutError, ValueError, OSError) as error:
        fail(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
