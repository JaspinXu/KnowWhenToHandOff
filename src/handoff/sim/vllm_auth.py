"""vLLM middleware: the shared API key on every route except the health check.

vLLM's own ``--api-key`` check covers the ``/v1`` routes only; ``/pause``,
``/resume``, ``/invocations`` and ``/inference/v1/generate`` would stay open to
anyone on the cluster network. The model servers load this function with
``--middleware handoff.sim.vllm_auth.require_api_key`` (see
:meth:`handoff.sim.serving.ServeSpec.command`); the key comes from
``VLLM_API_KEY``, as for vLLM itself.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Awaitable, Callable
from typing import Any

OPEN_PATHS = frozenset({"/health"})


async def require_api_key(
    request: Any, call_next: Callable[[Any], Awaitable[Any]]
) -> Any:
    """Reject requests without ``Authorization: Bearer $VLLM_API_KEY``."""
    key = os.environ.get("VLLM_API_KEY")
    if key and request.url.path not in OPEN_PATHS:
        supplied = request.headers.get("authorization", "")
        if not hmac.compare_digest(supplied.encode(), f"Bearer {key}".encode()):
            from starlette.responses import JSONResponse

            return JSONResponse({"error": "Unauthorized"}, status_code=401)
    return await call_next(request)
