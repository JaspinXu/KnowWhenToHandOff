"""Trajectory schema round trips and the shared trajectory builder."""

import json

import pytest
from fakes import call, conversation, result, say, trajectory

from handoff.rewards.rules import GoldAction
from handoff.rollout.schema import (
    FormatError,
    Trajectory,
    Violation,
    assistant_messages,
    find_transfer,
    iter_trajectories,
    tool_calls_with_results,
    write_trajectories,
)
from handoff.rollout.scoring import EpisodeContext, build_trajectory, with_turns


def test_round_trip_through_jsonl(tmp_path):
    original = trajectory(
        violations=[Violation("off_reference_write", 3, "x")],
        format_errors=[FormatError("unknown_tool", 2)],
        extra={"k": 1},
    )
    path = tmp_path / "t.jsonl"
    assert write_trajectories(path, [original, original]) == 2
    loaded = list(iter_trajectories(path))
    assert loaded == [original, original]
    assert loaded[0].episode_id == "airline/1/neutral/0"


def test_from_dict_rejects_unknown_fields_and_versions():
    data = trajectory().to_dict()
    with pytest.raises(ValueError):
        Trajectory.from_dict({**data, "surprise": 1})
    with pytest.raises(ValueError):
        Trajectory.from_dict({**data, "schema_version": 99})


def test_corrupt_line_reports_location(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps(trajectory().to_dict()) + '\n{"x": 1}\n')
    with pytest.raises(ValueError, match=":2:"):
        list(iter_trajectories(path))


def test_greeting_is_not_a_policy_turn_and_transfer_detection():
    messages = conversation(
        say("user", "human please"),
        say("assistant", None, call("t", "transfer_to_human_agents", summary="s")),
        result("t", "Transfer successful"),
        say("assistant", "Transferring."),
    )
    assert len(assistant_messages(messages)) == 2
    assert find_transfer(messages) == 2
    pairs = tool_calls_with_results(messages)
    assert pairs[0][1]["content"] == "Transfer successful"
    # tau2's action check counts the call whatever the tool returned, and so
    # does the hand-off label (no disagreement between reward and label).
    failed = [*messages[:3], result("t", "Error", error=True)]
    failed = [{**m, "turn": i} for i, m in enumerate(failed)]
    assert find_transfer(failed) == 2
    assert find_transfer(messages[:2]) is None


def context(**overrides):
    values = {
        "run_id": "TEST-s0-20261001",
        "domain": "retail",
        "task_id": "7__xr0",
        "source_task_id": "7",
        "split": "train",
        "kind": "explicit_request",
        "persona": "terse",
        "should_transfer": True,
        "trial": 1,
        "seed": 42,
        "policy_ckpt": "p",
        "user_sim": "u",
        "gold_actions": (GoldAction("transfer_to_human_agents", {}, ()),),
        "known_tools": frozenset({"transfer_to_human_agents", "find_user_id_by_email"}),
    }
    values.update(overrides)
    return EpisodeContext(**values)


def test_build_trajectory_applies_rules_labels_and_counts():
    messages = conversation(
        say("user", "I want a human agent."),
        say("assistant", None, call("t", "transfer_to_human_agents", summary="s")),
        result("t", "Transfer successful"),
        say("assistant", "Per our policy you are being transferred."),
    )
    for message in messages:
        message.pop("turn")
    built = build_trajectory(
        context(),
        messages=messages,
        termination_reason="user_stop",
        task_reward=1.0,
        reward_basis=["DB", "ACTION"],
        db_match=True,
    )
    assert built.transferred and built.transfer_turn == 2
    assert built.violations == []
    assert built.n_turns == 2 and built.n_tool_calls == 1
    assert built.should_transfer and built.kind == "explicit_request"
    assert [m["turn"] for m in built.messages] == [0, 1, 2, 3, 4]


def test_driver_errors_merge_and_protocol_error_not_double_counted():
    built = build_trajectory(
        context(),
        messages=conversation(say("user", "hi")),
        termination_reason="agent_error",
        task_reward=0.0,
        driver_format_errors=[FormatError("empty_message", 2)],
    )
    assert [e.kind for e in built.format_errors] == ["empty_message"]


def test_context_validation_and_turn_normalisation():
    with pytest.raises(ValueError):
        context(kind="weird")
    with pytest.raises(ValueError):
        context(split="val")
    assert with_turns([{"role": "user"}, {"role": "user", "turn": 7}]) == [
        {"role": "user", "turn": 0},
        {"role": "user", "turn": 7},
    ]
