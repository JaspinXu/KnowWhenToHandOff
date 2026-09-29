"""Launch a vLLM OpenAI-compatible server and publish its endpoint file.

Used by the Slurm serving jobs and by evaluation jobs that host the policy under
test. The server command is built from a resolved model config; the API key is
passed through the ``VLLM_API_KEY`` environment variable so it never appears in
the process list, logs or endpoint files.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from handoff.ops.runs import model_identity
from handoff.sim.endpoints import Endpoint, now_iso, wait_until_ready, write_endpoint

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServeSpec:
    """Everything needed to start one model server."""

    role: str
    model_path: str
    served_name: str
    revision: str
    repository: str
    port: int
    tool_call_parser: str | None = None
    max_model_len: int = 32768
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    dtype: str = "bfloat16"
    seed: int = 0
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    python: str = sys.executable

    def command(self) -> list[str]:
        """The vLLM server command line (no secrets)."""
        command = [
            self.python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_path,
            "--served-model-name",
            self.served_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
            "--max-model-len",
            str(self.max_model_len),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
            "--tensor-parallel-size",
            str(self.tensor_parallel_size),
            "--seed",
            str(self.seed),
            "--generation-config",
            "vllm",
            "--middleware",
            "handoff.sim.vllm_auth.require_api_key",
        ]
        if self.tool_call_parser:
            command += ["--enable-auto-tool-choice", "--tool-call-parser"]
            command.append(self.tool_call_parser)
        return command + list(self.extra_args)


def serve_spec_from_config(
    model: Mapping[str, Any],
    *,
    role: str,
    port: int,
    models_lock: Mapping[str, Any],
    server: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> ServeSpec:
    """Resolve a model config (and optional server overrides) into a ServeSpec."""
    server = dict(server or {})
    path, revision = model_identity(model, models_lock)
    served_name = str(model.get("served_name") or model.get("name"))
    return ServeSpec(
        role=role,
        model_path=path,
        served_name=served_name,
        revision=revision,
        repository=str(model.get("repository") or ""),
        port=int(port),
        tool_call_parser=model.get("tool_call_parser"),
        max_model_len=int(
            server.get("max_model_len", model.get("max_model_len", 32768))
        ),
        gpu_memory_utilization=float(server.get("gpu_memory_utilization", 0.9)),
        tensor_parallel_size=int(server.get("tensor_parallel_size", 1)),
        dtype=str(server.get("dtype", "bfloat16")),
        seed=seed,
        extra_args=tuple(str(a) for a in model.get("serve_args") or ()),
        chat_template_kwargs=dict(model.get("chat_template_kwargs") or {}),
    )


def endpoint_for(spec: ServeSpec, host: str | None = None) -> Endpoint:
    """Endpoint record for a spec running on this host."""
    host = host or socket.gethostname()
    return Endpoint(
        role=spec.role,
        base_url=f"http://{host}:{spec.port}/v1",
        served_model_name=spec.served_name,
        model_path=spec.model_path,
        model_repo=spec.repository,
        model_revision=spec.revision,
        slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
        started_at=now_iso(),
        max_model_len=spec.max_model_len,
        chat_template_kwargs=dict(spec.chat_template_kwargs),
        server_args=spec.command()[1:],
    )


def run_server(
    spec: ServeSpec,
    *,
    endpoint_file: Path,
    api_key: str,
    ready_timeout_s: float = 3600.0,
    command: Sequence[str] | None = None,
) -> int:
    """Start the server, publish the endpoint when ready, and wait for exit.

    The endpoint file is removed when the server stops, so clients never find a
    stale endpoint. SIGTERM and SIGINT are forwarded to the server process.
    """
    environment = dict(os.environ, VLLM_API_KEY=api_key)
    process = subprocess.Popen(list(command or spec.command()), env=environment)
    endpoint = endpoint_for(spec)

    def forward(signum: int, _frame: Any) -> None:
        process.send_signal(signum)

    previous = {
        signum: signal.signal(signum, forward)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }

    def publish() -> None:
        try:
            wait_until_ready(endpoint, api_key=api_key, timeout_s=ready_timeout_s)
        except TimeoutError as error:
            logger.error("%s", error)
            process.terminate()
            return
        if process.poll() is None:
            write_endpoint(endpoint_file, endpoint)
            logger.info("endpoint published: %s", endpoint.base_url)

    watcher = threading.Thread(target=publish, daemon=True)
    watcher.start()
    try:
        return process.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        if endpoint_file.exists():
            try:
                current = endpoint_file.read_text(encoding="utf-8")
                if endpoint.base_url in current:
                    endpoint_file.unlink()
            except OSError:
                pass
