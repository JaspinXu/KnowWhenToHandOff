"""Guard reward function configured in verl for the tau2 agent loop.

``Tau2AgentLoop`` sets ``reward_score`` on every output, so verl never needs a
reward function. verl still requires one to be configured; this one fails
loudly if it is ever called, which would mean an episode bypassed the agent loop
and would silently train on a wrong reward.
"""

from __future__ import annotations

from typing import Any


def compute_score(*args: Any, **kwargs: Any) -> float:
    """Refuse to compute a reward outside the agent loop."""
    raise RuntimeError(
        "handoff rewards are computed inside Tau2AgentLoop; a verl reward function "
        "was called, so a sample bypassed the agent loop (check data.agent_name)"
    )
