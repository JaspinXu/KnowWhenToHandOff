"""Model-server endpoint records and LiteLLM settings for local vLLM servers.

A serving job writes an endpoint file (JSON, no secrets) that other jobs read to
find the server. All servers of one user share one API key stored in a private
file; drivers export it as ``OPENAI_API_KEY`` so the key never appears in run
configs or tau2 result files.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ENDPOINT_SCHEMA_VERSION = 1
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class Endpoint:
    """Where an OpenAI-compatible model server listens and what it serves.

    ``chat_template_kwargs`` come from the model config (for example
    ``enable_thinking: false``); vLLM 0.12 has no server-side default for them,
    so clients send them with every request. ``server_args`` is the server's
    command line (no secrets) and ``max_model_len`` its context limit, both
    recorded in run provenance.
    """

    role: str
    base_url: str
    served_model_name: str
    model_path: str = ""
    model_repo: str = ""
    model_revision: str = ""
    slurm_job_id: str = ""
    started_at: str = ""
    max_model_len: int | None = None
    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    server_args: list[str] = field(default_factory=list)
    schema_version: int = ENDPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must be an http(s) URL")
        if "@" in self.base_url.split("//", 1)[1].split("/", 1)[0]:
            raise ValueError("base_url must not embed credentials")
        if not _NAME.fullmatch(self.served_model_name):
            raise ValueError("invalid served model name")

    @property
    def litellm_model(self) -> str:
        """LiteLLM model string routing to an OpenAI-compatible server."""
        return f"openai/{self.served_model_name}"


# Server flags that change between restarts without changing any output.
_VOLATILE_FLAGS = {"--host", "--port", "--gpu-memory-utilization", "--model"}


def server_identity(endpoint: Endpoint) -> dict[str, Any]:
    """What determines a server's outputs, for run provenance.

    The served model and revision, the context limit, the chat-template
    arguments and the vLLM flags, except the address, the memory share and the
    local model path (the revision identifies the weights).
    """
    args = list(endpoint.server_args)
    if args[:1] == ["-m"]:
        args = args[2:]
    flags: list[str] = []
    index = 0
    while index < len(args):
        if args[index] in _VOLATILE_FLAGS:
            index += 2
            continue
        flags.append(args[index])
        index += 1
    return {
        "served_model_name": endpoint.served_model_name,
        "model_revision": endpoint.model_revision,
        "max_model_len": endpoint.max_model_len,
        "chat_template_kwargs": dict(endpoint.chat_template_kwargs),
        "server_flags": flags,
    }


def write_endpoint(path: Path, endpoint: Endpoint) -> None:
    """Atomically write an endpoint file readable only by the owner."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(asdict(endpoint), stream, indent=2)
        stream.write("\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def read_endpoint(path: Path) -> Endpoint:
    """Read and validate an endpoint file."""
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("schema_version") != ENDPOINT_SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported endpoint schema")
    return Endpoint(**data)


def ensure_api_key(path: Path) -> str:
    """Return the shared server API key, creating it (mode 600) if missing."""
    if path.exists():
        key = path.read_text(encoding="utf-8").strip()
        if key:
            return key
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_hex(24)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(key + "\n")
    return key


def load_api_key_into_env(path: Path | None) -> None:
    """Export the shared server key as ``OPENAI_API_KEY``.

    Every LiteLLM call of this project goes to one of the user's own vLLM
    servers, so the key file wins over a key inherited from the login shell
    (which those servers would reject, and which must not be sent to them).
    Without a key file an existing variable is kept, else a placeholder is set.
    """
    if path is not None and path.exists():
        os.environ["OPENAI_API_KEY"] = path.read_text(encoding="utf-8").strip()
    elif not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = "EMPTY"


def litellm_args(
    endpoint: Endpoint,
    *,
    temperature: float,
    max_tokens: int | None = None,
    extra_body: Mapping[str, Any] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """LiteLLM keyword arguments for chat completions against ``endpoint``.

    The endpoint's ``chat_template_kwargs`` are sent in ``extra_body`` (values
    given in ``extra_body`` win). The API key is deliberately absent: LiteLLM
    reads ``OPENAI_API_KEY``.
    """
    args: dict[str, Any] = {"api_base": endpoint.base_url, "temperature": temperature}
    if max_tokens is not None:
        args["max_tokens"] = int(max_tokens)
    body = dict(extra_body or {})
    if endpoint.chat_template_kwargs:
        body["chat_template_kwargs"] = {
            **endpoint.chat_template_kwargs,
            **dict(body.get("chat_template_kwargs") or {}),
        }
    if body:
        args["extra_body"] = body
    if timeout_s is not None:
        args["timeout"] = float(timeout_s)
    return args


def register_local_models(served_names: Iterable[str]) -> None:
    """Register locally served models with LiteLLM at zero cost.

    LiteLLM cannot price unknown models, and tau2 logs an error for every call it
    cannot price. Local vLLM servers cost nothing per token, so say so.
    """
    import litellm

    litellm.suppress_debug_info = True
    entry = {
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
        "litellm_provider": "openai",
        "mode": "chat",
    }
    litellm.register_model(
        {key: dict(entry) for name in served_names for key in (name, f"openai/{name}")}
    )


def probe(endpoint: Endpoint, *, api_key: str | None, timeout_s: float = 30.0) -> str:
    """One ``GET /models`` check: ``""`` if the model is served, else the reason."""
    url = endpoint.base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url)
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = json.load(response)
    except (urllib.error.URLError, OSError, ValueError) as error:
        return type(error).__name__
    names = {item.get("id") for item in data.get("data", [])}
    if endpoint.served_model_name in names:
        return ""
    return f"model not listed (got {sorted(n for n in names if n)})"


def wait_until_ready(
    endpoint: Endpoint,
    *,
    api_key: str | None,
    timeout_s: float = 1800.0,
    interval_s: float = 10.0,
) -> None:
    """Poll ``GET /models`` until the served model is listed or time runs out."""
    deadline = time.monotonic() + timeout_s
    last_error = "no response"
    while time.monotonic() < deadline:
        last_error = probe(endpoint, api_key=api_key)
        if not last_error:
            return
        time.sleep(interval_s)
    raise TimeoutError(
        f"{endpoint.role} server at {endpoint.base_url} not ready: {last_error}"
    )


def now_iso() -> str:
    """UTC timestamp for endpoint records."""
    return datetime.now(UTC).isoformat(timespec="seconds")
