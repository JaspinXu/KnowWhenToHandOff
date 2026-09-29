"""A tiny tau2 domain, scripted users and a scripted policy for integration tests.

Nothing here reads tau2 data files or calls a model: the domain lives in memory,
LiteLLM's ``completion`` is replaced by :func:`fake_completion`, and every
conversation follows a fixed script, so the evaluation driver (tau2 runner) and
the training driver (token-in/token-out) can be compared turn by turn.

Import only after ``pytest.importorskip("tau2")``.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import litellm
from fakes import FakeChatTokenizer
from pydantic import BaseModel
from tau2.environment.db import DB
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool

from handoff.envs.handoff_tasks import (
    EXPLICIT_REQUEST_TEMPLATES,
    make_positive_variant,
)

DOMAIN = "handoff_toy"
POLICY_MODEL = "toy-policy"
USER_MODEL = "toy-user"
POLICY = (
    "# Toy store policy\n\n"
    "- Look up an order before changing it.\n"
    "- Cancel a pending order only after the user explicitly confirms (yes).\n"
    "- Transfer to a human agent only if the user asks for one or the request "
    "is outside this policy."
)
MARKER = re.compile(r"\[toy:([^\]]+)\]")


class Order(BaseModel):
    """One order of the toy store."""

    order_id: str
    user_id: str
    status: str


class ToyDB(DB):
    """Orders indexed by ID."""

    orders: dict[str, Order]


def fresh_db() -> ToyDB:
    """The initial database of every episode."""
    return ToyDB(
        orders={
            order_id: Order(order_id=order_id, user_id="u1", status="pending")
            for order_id in ("o1", "o2", "o3")
        }
    )


class ToyTools(ToolKitBase):
    """Tools of the toy store."""

    db: ToyDB

    def __init__(self, db: ToyDB) -> None:
        super().__init__(db)

    @is_tool(ToolType.READ)
    def get_order(self, order_id: str) -> Order:
        """
        Look up an order.

        Args:
            order_id: The order ID, such as 'o1'.

        Returns:
            The order.
        """
        if order_id not in self.db.orders:
            raise ValueError(f"Order {order_id} not found")
        return self.db.orders[order_id]

    @is_tool(ToolType.WRITE)
    def cancel_order(self, order_id: str) -> Order:
        """
        Cancel a pending order.

        Args:
            order_id: The order ID, such as 'o1'.

        Returns:
            The cancelled order.
        """
        order = self.get_order(order_id)
        if order.status != "pending":
            raise ValueError(f"Order {order_id} is not pending")
        order.status = "cancelled"
        return order

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.

        Args:
            summary: A summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        return "Transfer successful"


def get_environment(db: ToyDB | None = None, solo_mode: bool = False, **_: Any):
    """tau2 environment constructor for the toy domain."""
    return Environment(
        domain_name=DOMAIN, policy=POLICY, tools=ToyTools(db or fresh_db())
    )


def _action(name: str, **arguments: Any) -> dict[str, Any]:
    return {
        "action_id": f"{name}_{'_'.join(map(str, arguments.values()))}",
        "requestor": "assistant",
        "name": name,
        "arguments": arguments,
        "info": None,
        "compare_args": None,
    }


def _task(task_id: str, reason: str, actions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": task_id,
        "description": {"purpose": reason, "relevant_policies": None, "notes": None},
        "user_scenario": {
            "persona": None,
            "instructions": {
                "domain": DOMAIN,
                "reason_for_call": reason,
                "known_info": f"You are user u1. [toy:{task_id}]",
                "unknown_info": None,
                "task_instructions": "Follow the script.",
            },
        },
        "ticket": None,
        "initial_state": None,
        "evaluation_criteria": {
            "actions": actions,
            "env_assertions": None,
            "communicate_info": [],
            "nl_assertions": None,
            "reward_basis": ["DB", "ACTION"],
        },
        "issues": None,
        "required_documents": None,
        "user_tools": None,
    }


UPSTREAM_TASKS = [
    _task("0", "Cancel order o1.", [_action("cancel_order", order_id="o1")]),
    _task("1", "Cancel order o2.", [_action("cancel_order", order_id="o2")]),
    _task("2", "Ask where order o1 is.", []),
    _task("3", "Ask about order o3.", []),
    _task("4", "Ask about order o2.", []),
    _task("5", "Check order o1.", []),
]
OFFICIAL_SPLITS = {"train": ["0", "1", "2", "3", "4", "5"], "test": []}


def derived_tasks() -> list[dict[str, Any]]:
    """An explicit-request variant of task 0 built by the real generator."""
    task = make_positive_variant(
        UPSTREAM_TASKS[0],
        domain=DOMAIN,
        kind="explicit_request",
        template=EXPLICIT_REQUEST_TEMPLATES[0],
        index=0,
    )
    instructions = task["user_scenario"]["instructions"]
    instructions["known_info"] = f"You are user u1. [toy:{task['id']}]"
    return [task]


def get_tasks(task_split_name: str | None = None) -> list[Any]:
    """tau2 task loader for the toy domain."""
    from tau2.data_model.tasks import Task

    tasks = [Task.model_validate(t) for t in UPSTREAM_TASKS]
    if task_split_name is None:
        return tasks
    wanted = set(OFFICIAL_SPLITS[task_split_name])
    return [task for task in tasks if task.id in wanted]


def register_toy_domain() -> None:
    """Register the toy domain with tau2's registry (idempotent)."""
    from tau2.registry import registry

    if DOMAIN not in registry.get_domains():
        registry.register_domain(get_environment, DOMAIN)
        registry.register_tasks(
            get_tasks, DOMAIN, get_task_splits=lambda: dict(OFFICIAL_SPLITS)
        )


def _call(name: str, **arguments: Any) -> str:
    payload = json.dumps({"name": name, "arguments": arguments})
    return f"<tool_call>\n{payload}\n</tool_call>"


# What the policy generates, turn by turn (raw Hermes text). A ``None`` entry
# means "cut at the token limit": the text before it is generated without an
# end-of-turn token.
AGENT_SCRIPTS: dict[str, list[Any]] = {
    "0": [
        _call("get_order", order_id="o1"),
        "Order o1 is pending. Shall I cancel it? Please confirm (yes/no).",
        _call("cancel_order", order_id="o1"),
        "Order o1 is cancelled. Per our policy, refunds take 5 days.",
    ],
    # Writes without confirmation and mixes text with a tool call.
    "1": [
        "Cancelling it now.\n" + _call("cancel_order", order_id="o2"),
        "Done.",
    ],
    # The generation is cut at the token limit.
    "2": [("Let me look that up for you, it may take", None)],
    # An empty message.
    "3": [""],
    # Unparsable tool-call markup is shown to the user as text.
    "4": [
        '<tool_call>{"name": "get_order", "arguments": {"order_id": o2}}</tool_call>'
    ],
    # Arguments that are not an object: vLLM passes them on, tau2 rejects them.
    "5": ['<tool_call>\n{"name": "get_order", "arguments": "o1"}\n</tool_call>'],
    "0__xr0": [
        _call("transfer_to_human_agents", summary="The user asks for a human agent."),
        "YOU ARE BEING TRANSFERRED TO A HUMAN AGENT. PLEASE HOLD ON.",
    ],
}

USER_SCRIPTS: dict[str, list[str]] = {
    "0": [
        "Hi, I want to cancel my order o1. [toy:0]",
        "Yes, please cancel it.",
        "Thanks! ###STOP###",
    ],
    "1": ["Cancel order o2 please. [toy:1]", "###STOP###"],
    "2": ["Where is my order o1? [toy:2]"],
    "3": ["Tell me about order o3. [toy:3]"],
    "4": ["What about order o2? [toy:4]", "Never mind. ###STOP###"],
    "5": ["Check order o1. [toy:5]"],
    "0__xr0": [
        "I want to talk to a human agent, please. [toy:0__xr0]",
        "###TRANSFER###",
    ],
}


def agent_turn(task_id: str, index: int) -> tuple[str, bool]:
    """Raw generated text of the policy's ``index``-th turn and whether it ended."""
    entry = AGENT_SCRIPTS[task_id][index]
    if isinstance(entry, tuple):
        return entry[0], False
    return entry, True


def vllm_hermes(text: str) -> tuple[str | None, list[dict[str, Any]]]:
    """What vLLM's ``hermes`` tool parser returns for a generation.

    Written independently of :mod:`handoff.rollout.hermes` (after vLLM's
    ``Hermes2ProToolParser.extract_tool_calls``) so the tests compare the
    training parser against the serving behavior rather than against itself.
    """
    if "<tool_call>" not in text:
        return text, []
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
    try:
        raw = [json.loads(a if a else b) for a, b in pattern.findall(text)]
        calls = [
            {
                "id": f"chatcmpl-tool-{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": call["name"],
                    "arguments": json.dumps(call["arguments"], ensure_ascii=False),
                },
            }
            for call in raw
        ]
    except Exception:  # noqa: BLE001 - vLLM falls back to plain text on any error
        return text, []
    content = text[: text.find("<tool_call>")]
    return (content or None), calls


def completion_tokens(text: str, finished: bool) -> int:
    """Tokens of a generation under the fake tokenizer (plus end-of-turn)."""
    return len(FakeChatTokenizer().encode(text)) + (1 if finished else 0)


def _marker(messages: Sequence[Mapping[str, Any]]) -> str:
    for message in messages:
        found = MARKER.search(str(message.get("content") or ""))
        if found:
            return found.group(1)
    raise AssertionError("no toy marker in the conversation")


def _response(
    model: str,
    content: str | None,
    tool_calls: list[dict[str, Any]],
    finish_reason: str,
    n_tokens: int,
) -> litellm.ModelResponse:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return litellm.ModelResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        model=model,
        choices=[{"index": 0, "finish_reason": finish_reason, "message": message}],
        usage={
            "prompt_tokens": 1,
            "completion_tokens": n_tokens,
            "total_tokens": n_tokens + 1,
        },
    )


def fake_completion(*, model: str, messages: list[dict[str, Any]], **_: Any):
    """Stand-in for ``litellm.completion`` serving the scripted policy and users."""
    if model == f"openai/{POLICY_MODEL}":
        users = [m for m in messages if m["role"] == "user"]
        task_id = _marker(users[:1])
        index = sum(1 for m in messages if m["role"] == "assistant") - 1
        text, finished = agent_turn(task_id, index)
        content, calls = vllm_hermes(text) if finished else (text, [])
        finish = "stop" if finished else "length"
        return _response(
            POLICY_MODEL, content, calls, finish, completion_tokens(text, finished)
        )
    if model == f"openai/{USER_MODEL}":
        task_id = _marker([m for m in messages if m["role"] == "system"])
        index = sum(1 for m in messages if m["role"] == "assistant")
        reply = USER_SCRIPTS[task_id][index]
        return _response(USER_MODEL, reply, [], "stop", 5)
    raise AssertionError(f"unexpected model {model}")


class ScriptedPolicy:
    """Token-in/token-out generator replaying one task's agent script."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self.tokenizer = FakeChatTokenizer()
        self.index = 0
        self.prompts: list[list[int]] = []

    async def generate(self, prompt_ids: list[int], *, max_tokens: int):
        """Return the next scripted turn as token IDs."""
        from handoff.rollout.episode import GenerationOutput

        self.prompts.append(list(prompt_ids))
        text, finished = agent_turn(self.task_id, self.index)
        self.index += 1
        ids = self.tokenizer.encode(text)
        if finished:
            ids.append(self.tokenizer.eos_token_id)
        return GenerationOutput(ids, [-0.1] * len(ids))
