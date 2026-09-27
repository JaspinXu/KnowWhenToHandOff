"""Preserve sampled IDs across a multi-turn tool conversation."""

import pytest

from handoff.rollout.tokens import TokenAccumulator


def test_sampled_tokens_and_masks_are_preserved_across_turns() -> None:
    buffer = TokenAccumulator([10, 20], max_response_tokens=20)
    buffer.append("assistant", [300, 301])
    buffer.append("tool", [400, 401])
    buffer.append("tool", [402])
    buffer.append("assistant", [9000])
    buffer.append("user", [500, 501])
    buffer.append("assistant", [600])
    assert buffer.as_output() == {
        "prompt_ids": [10, 20],
        "response_ids": [300, 301, 400, 401, 402, 9000, 500, 501, 600],
        "response_mask": [1, 1, 0, 0, 0, 1, 0, 0, 1],
    }


def test_inputs_and_outputs_do_not_alias_internal_buffers() -> None:
    prompt = [1]
    generated = [2]
    buffer = TokenAccumulator(prompt, max_response_tokens=3)
    buffer.append("assistant", generated)
    prompt[0] = 99
    generated.append(100)
    output = buffer.as_output()
    output["prompt_ids"].append(12)
    output["response_mask"].clear()
    output["response_ids"][0] = 11
    assert buffer.as_output() == {
        "prompt_ids": [1],
        "response_ids": [2],
        "response_mask": [1],
    }


def test_capacity_counts_context_and_rejects_overflow_atomically() -> None:
    buffer = TokenAccumulator([1, 2, 3], max_response_tokens=3)
    buffer.append("assistant", [4, 5])
    before = buffer.as_output()
    with pytest.raises(ValueError, match="capacity"):
        buffer.append("tool", [6, 7])
    assert buffer.as_output() == before
    buffer.append("user", [8])
    buffer.append("assistant", [])
    assert buffer.as_output()["response_ids"] == [4, 5, 8]
    with pytest.raises(ValueError, match="capacity"):
        buffer.append("assistant", [9])


@pytest.mark.parametrize("ids", [[True], [-1], [1.5], ["3"], "text", b"abc", [1, None]])
def test_invalid_ids_never_mutate_buffer(ids: object) -> None:
    buffer = TokenAccumulator([1], max_response_tokens=5)
    with pytest.raises((TypeError, ValueError)):
        buffer.append("assistant", ids)
    assert buffer.as_output()["response_ids"] == []


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_capacity_must_be_a_positive_integer(limit: object) -> None:
    with pytest.raises(ValueError, match="max_response_tokens"):
        TokenAccumulator([1], max_response_tokens=limit)


@pytest.mark.parametrize("role", ["system", "policy", "", None])
def test_rejects_unknown_role(role: object) -> None:
    buffer = TokenAccumulator([1], max_response_tokens=3)
    with pytest.raises(ValueError, match="role"):
        buffer.append(role, [2])


def test_empty_prompt_is_rejected_and_empty_segments_are_noops() -> None:
    with pytest.raises(ValueError, match="prompt"):
        TokenAccumulator([], max_response_tokens=3)
    buffer = TokenAccumulator([0], max_response_tokens=3)
    for role in ("assistant", "user", "tool"):
        buffer.append(role, [])
    assert buffer.as_output() == {
        "prompt_ids": [0],
        "response_ids": [],
        "response_mask": [],
    }
