"""Test doubles: a Qwen-style chat tokenizer and trajectory builders (CPU only)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from handoff.envs.handoff_tasks import generate_handoff_tasks
from handoff.envs.labels import build_label_file, labels_by_task
from handoff.envs.registry import build_handoff_manifest, build_split_manifest
from handoff.rollout.schema import DEFAULT_GREETING, Trajectory

SPECIAL = ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<tool_call>", "</tool_call>")
_SPECIAL_IDS = {token: 10 + index for index, token in enumerate(SPECIAL)}
_OFFSET = 1000


class FakeChatTokenizer:
    """Character-level tokenizer with a Qwen2.5/Qwen3-Instruct style template.

    Every character maps to its own ID, so encoding text in pieces equals
    encoding it at once; special tokens are single IDs. ``drop_old_tool_results``
    makes the template deliberately non-prefix-stable for negative tests.
    """

    eos_token = "<|im_end|>"
    pad_token = "<|endoftext|>"

    def __init__(self, *, drop_old_tool_results: bool = False) -> None:
        self.drop_old_tool_results = drop_old_tool_results

    @property
    def eos_token_id(self) -> int:
        return _SPECIAL_IDS[self.eos_token]

    @property
    def pad_token_id(self) -> int:
        return _SPECIAL_IDS[self.pad_token]

    def convert_tokens_to_ids(self, token: str) -> int:
        return _SPECIAL_IDS[token]

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        index = 0
        while index < len(text):
            for token in SPECIAL:
                if text.startswith(token, index):
                    ids.append(_SPECIAL_IDS[token])
                    index += len(token)
                    break
            else:
                ids.append(_OFFSET + ord(text[index]))
                index += 1
        return ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        reverse = {v: k for k, v in _SPECIAL_IDS.items()}
        parts = []
        for token_id in ids:
            if token_id in reverse:
                if not skip_special_tokens:
                    parts.append(reverse[token_id])
            else:
                parts.append(chr(token_id - _OFFSET))
        return "".join(parts)

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, Any]],
        *,
        tools: Sequence[Mapping[str, Any]] | None = None,
        add_generation_prompt: bool = False,
        tokenize: bool = False,
        **_: Any,
    ) -> str | list[int]:
        messages = list(conversation)
        text = ""
        if tools and (not messages or messages[0].get("role") != "system"):
            text += "<|im_start|>system\n" + self._tools_block(tools) + "<|im_end|>\n"
        last_user = max(
            (i for i, m in enumerate(messages) if m.get("role") == "user"), default=-1
        )
        for index, message in enumerate(messages):
            role = message.get("role")
            content = message.get("content") or ""
            if role == "system":
                block = content + (("\n\n" + self._tools_block(tools)) if tools else "")
                text += f"<|im_start|>system\n{block}<|im_end|>\n"
            elif role == "user":
                text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                text += f"<|im_start|>assistant\n{content}"
                for n, call in enumerate(message.get("tool_calls") or []):
                    function = call.get("function", call)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if content or n:
                        text += "\n"
                    text += (
                        "<tool_call>\n"
                        + json.dumps({"name": function["name"], "arguments": arguments})
                        + "\n</tool_call>"
                    )
                text += "<|im_end|>\n"
            elif role == "tool":
                previous = messages[index - 1].get("role") if index else None
                following = (
                    messages[index + 1].get("role")
                    if index + 1 < len(messages)
                    else None
                )
                if previous != "tool":
                    text += "<|im_start|>user"
                shown = content
                if self.drop_old_tool_results and index < last_user:
                    shown = "(omitted)"
                text += f"\n<tool_response>\n{shown}\n</tool_response>"
                if following != "tool":
                    text += "<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n"
        return self.encode(text) if tokenize else text

    @staticmethod
    def _tools_block(tools: Sequence[Mapping[str, Any]] | None) -> str:
        lines = ["# Tools", "<tools>"]
        lines += [json.dumps(tool, sort_keys=True) for tool in tools or []]
        lines.append("</tools>")
        return "\n".join(lines)


def call(call_id: str, name: str, **arguments: Any) -> dict[str, Any]:
    """A normalized tool call."""
    return {"id": call_id, "name": name, "arguments": arguments}


def result(call_id: str, content: str = "ok", *, error: bool = False) -> dict[str, Any]:
    """A normalized tool result (requestor assistant)."""
    return {
        "role": "tool",
        "content": content,
        "tool_call_id": call_id,
        "requestor": "assistant",
        "error": error,
    }


def say(role: str, content: str | None, *calls: dict[str, Any]) -> dict[str, Any]:
    """A normalized user or assistant message."""
    return {"role": role, "content": content, "tool_calls": list(calls) or None}


def conversation(*messages: dict[str, Any]) -> list[dict[str, Any]]:
    """Prefix the tau2 greeting and number the turns."""
    items = [say("assistant", DEFAULT_GREETING), *messages]
    return [{**message, "turn": index} for index, message in enumerate(items)]


def trajectory(**overrides: Any) -> Trajectory:
    """A valid trajectory with sensible defaults for metric tests."""
    values: dict[str, Any] = {
        "run_id": "TEST-s0-20261001",
        "task_id": "1",
        "source_task_id": "1",
        "domain": "airline",
        "split": "test",
        "kind": "upstream",
        "persona": "neutral",
        "trial": 0,
        "seed": 0,
        "policy_ckpt": "policy@x",
        "user_sim": "user@y",
        "messages": conversation(say("user", "hi"), say("assistant", "hello")),
        "termination_reason": "user_stop",
        "task_reward": 1.0,
        "reward_basis": ["DB"],
        "db_match": True,
        "transferred": False,
        "transfer_turn": None,
        "should_transfer": False,
    }
    values.update(overrides)
    return Trajectory(**values)


COMMIT = "f" * 40


def synthetic_task(task_id: str, transfer: bool = False) -> dict:
    actions = [
        {
            "action_id": "a",
            "requestor": "assistant",
            "name": "cancel_reservation",
            "arguments": {"reservation_id": "R1"},
            "info": None,
            "compare_args": None,
        }
    ]
    if transfer:
        actions.append(
            {
                "action_id": "b",
                "requestor": "assistant",
                "name": "transfer_to_human_agents",
                "arguments": {"summary": "s"},
                "info": None,
                "compare_args": None,
            }
        )
    return {
        "id": task_id,
        "description": {"purpose": "p", "relevant_policies": None, "notes": None},
        "user_scenario": {
            "persona": None,
            "instructions": {
                "domain": "airline",
                "reason_for_call": f"Cancel reservation for task {task_id}.",
                "known_info": "You are Ada Lovelace, user id ada_1.",
                "unknown_info": "You do not remember the flight number.",
                "task_instructions": "Ask to cancel the reservation.",
            },
        },
        "ticket": None,
        "initial_state": {
            "initialization_data": None,
            "initialization_actions": None,
            "message_history": None,
        },
        "evaluation_criteria": {
            "actions": actions,
            "env_assertions": None,
            "communicate_info": [],
            "nl_assertions": None,
            "reward_basis": ["DB", "COMMUNICATE"],
        },
        "issues": None,
        "required_documents": None,
        "user_tools": None,
    }


def build_all(domain_data, counts=None):
    tasks, splits = domain_data
    label_file = build_label_file({"airline": tasks}, tau2_commit=COMMIT)
    labels = labels_by_task(label_file)
    manifest = build_split_manifest(
        {
            "airline": {
                "official_splits": splits,
                "labels": {t: v for (_, t), v in labels.items()},
                "file_sha256": {"tasks": "a" * 64},
            }
        },
        dev_fraction=0.25,
        seed=7,
        tau2_commit=COMMIT,
        labels_sha256=label_file["labels_sha256"],
    )
    split_of = {
        tid: name
        for name, ids in manifest["domains"]["airline"]["splits"].items()
        for tid in ids
    }
    counts = counts or {
        "train": {"explicit_request": 1, "out_of_scope": 2, "hard_negative": 2},
        "dev": {"explicit_request": 1, "out_of_scope": 1, "hard_negative": 1},
        "test": {"explicit_request": 1, "out_of_scope": 1, "hard_negative": 1},
    }
    pairs = generate_handoff_tasks(
        "airline",
        tasks,
        split_of=split_of,
        labels={t: v for (_, t), v in labels.items()},
        counts=counts,
        seed=11,
    )
    handoff = build_handoff_manifest(
        {"airline": [ref for _, ref in pairs]},
        generator_version="handoff_tasks_v1",
        split_manifest_sha256=manifest["manifest_sha256"],
        tasks_file_sha256={"airline": "b" * 64},
        config={"counts": counts},
    )
    return label_file, labels, manifest, pairs, handoff
