"""Verify assistant-only causal labels without a trainer, tokenizer or GPU."""

import copy

import pytest


def settings(**updates: int) -> dict[str, int]:
    return {
        "max_length": 16,
        "pad_token_id": 0,
        "ignore_index": -100,
        "pad_to_multiple_of": 1,
        **updates,
    }


def sample() -> dict:
    # prompt [10, 11]; assistant [20, 21]; tool [30, 31]; user [40]; assistant [22]
    return {
        "split": "train",
        "prompt_ids": [10, 11],
        "response_ids": [20, 21, 30, 31, 40, 22],
        "response_mask": [1, 1, 0, 0, 0, 1],
    }


def build(samples: list[dict], **overrides: int) -> dict:
    from handoff.train.sft_batch import build_sft_batch, sft_config_from_mapping

    return build_sft_batch(
        samples, config=sft_config_from_mapping(settings(**overrides))
    )


def test_multiturn_rollout_has_only_policy_targets_and_one_causal_shift() -> None:
    batch = build([sample()])
    assert batch == {
        "input_ids": [[10, 11, 20, 21, 30, 31, 40, 22]],
        "attention_mask": [[1, 1, 1, 1, 1, 1, 1, 1]],
        "labels": [[-100, -100, 20, 21, -100, -100, -100, 22]],
    }
    # A causal LM compares logits at position t to labels[t+1]. The collator
    # must not pre-shift labels or train on observation tokens.
    targets = [
        (index - 1, token)
        for index, token in enumerate(batch["labels"][0])
        if token != -100
    ]
    assert targets == [(1, 20), (2, 21), (6, 22)]


def test_right_padding_preserves_real_eos_even_when_it_equals_pad() -> None:
    short = {
        "split": "train",
        "prompt_ids": [0, 11],
        "response_ids": [0],
        "response_mask": [1],
    }
    batch = build([sample(), short], pad_token_id=0, pad_to_multiple_of=4)
    assert batch["input_ids"][1] == [0, 11, 0, 0, 0, 0, 0, 0]
    assert batch["attention_mask"][1] == [1, 1, 1, 0, 0, 0, 0, 0]
    assert batch["labels"][1] == [-100, -100, 0, -100, -100, -100, -100, -100]
    assert batch["input_ids"][0][-1] == batch["labels"][0][-1] == 22


def test_pad_alignment_can_extend_longest_row_without_truncation() -> None:
    batch = build(
        [sample()], pad_to_multiple_of=5, max_length=10, pad_token_id=7, ignore_index=-1
    )
    assert batch["input_ids"] == [[10, 11, 20, 21, 30, 31, 40, 22, 7, 7]]
    assert batch["attention_mask"] == [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]
    assert batch["labels"] == [[-1, -1, 20, 21, -1, -1, -1, 22, -1, -1]]


def test_source_arrays_and_repeated_rows_are_independent() -> None:
    source = sample()
    original = copy.deepcopy(source)
    batch = build([source, source])
    assert source == original
    batch["input_ids"][0][2] = 999
    batch["labels"][0][2] = 999
    batch["attention_mask"][0][0] = 0
    assert batch["input_ids"][1][2] == batch["labels"][1][2] == 20
    assert batch["attention_mask"][1][0] == 1
    assert source == original


@pytest.mark.parametrize(
    "split", ["dev", "test", "handoff-test", "validation", "Train", None, True]
)
def test_rejects_any_non_train_row_in_a_mixed_batch(split: object) -> None:
    bad = sample()
    bad["split"] = split
    sources = [sample(), bad]
    original = copy.deepcopy(sources)
    with pytest.raises(ValueError, match="train"):
        build(sources)
    assert sources == original


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_split",
        "unknown_field",
        "empty_prompt",
        "empty_response",
        "no_policy",
        "short_mask",
        "mask_bool",
        "mask_two",
        "negative_token",
        "bool_token",
        "text_ids",
        "float_token",
    ],
)
def test_rejects_malformed_samples_instead_of_guessing(mutation: str) -> None:
    item = sample()
    if mutation == "missing_split":
        del item["split"]
    elif mutation == "unknown_field":
        item["reponse_ids"] = [1]
    elif mutation == "empty_prompt":
        item["prompt_ids"] = []
    elif mutation == "empty_response":
        item["response_ids"], item["response_mask"] = [], []
    elif mutation == "no_policy":
        item["response_mask"] = [0] * len(item["response_ids"])
    elif mutation == "short_mask":
        item["response_mask"].pop()
    elif mutation == "mask_bool":
        item["response_mask"][0] = True
    elif mutation == "mask_two":
        item["response_mask"][0] = 2
    elif mutation == "negative_token":
        item["response_ids"][0] = -1
    elif mutation == "bool_token":
        item["prompt_ids"][0] = True
    elif mutation == "text_ids":
        item["prompt_ids"] = "10,11"
    elif mutation == "float_token":
        item["response_ids"][0] = 2.5
    with pytest.raises(ValueError):
        build([item])


@pytest.mark.parametrize(
    "overrides", [{"max_length": 7}, {"max_length": 8, "pad_to_multiple_of": 5}]
)
def test_rejects_length_and_padding_overflow(overrides: dict) -> None:
    source = sample()
    original = copy.deepcopy(source)
    with pytest.raises(ValueError, match="length"):
        build([source], **overrides)
    assert source == original


@pytest.mark.parametrize("samples", [[], None, {}, "tokens", [None]])
def test_rejects_empty_or_invalid_batch(samples: object) -> None:
    with pytest.raises(ValueError):
        build(samples)


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_length", 0),
        ("max_length", True),
        ("pad_token_id", -1),
        ("pad_token_id", None),
        ("pad_token_id", "???"),
        ("ignore_index", 0),
        ("ignore_index", False),
        ("pad_to_multiple_of", 0),
        ("pad_to_multiple_of", 1.5),
    ],
)
def test_rejects_invalid_config(field: str, value: object) -> None:
    from handoff.train.sft_batch import SFTBatchConfig, sft_config_from_mapping

    config = settings()
    config[field] = value
    with pytest.raises(ValueError):
        sft_config_from_mapping(config)
    with pytest.raises(ValueError):
        SFTBatchConfig(**config)


def test_config_is_frozen_and_rejects_extra_fields() -> None:
    from dataclasses import FrozenInstanceError

    from handoff.train.sft_batch import sft_config_from_mapping

    raw = settings()
    config = sft_config_from_mapping(raw)
    raw["ignore_index"] = 0
    assert config.ignore_index == -100
    with pytest.raises(FrozenInstanceError):
        config.ignore_index = 0
    with pytest.raises(ValueError):
        sft_config_from_mapping({**settings(), "packing": True})
