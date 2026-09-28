"""Build unpacked assistant-only SFT labels from preserved rollout token IDs."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_CONFIG_FIELDS = {"max_length", "pad_token_id", "ignore_index", "pad_to_multiple_of"}
_SAMPLE_FIELDS = {"split", "prompt_ids", "response_ids", "response_mask"}


@dataclass(frozen=True)
class SFTBatchConfig:
    """Explicit sequence, padding and loss-label settings for one batch builder."""

    max_length: int
    pad_token_id: int
    ignore_index: int
    pad_to_multiple_of: int

    def __post_init__(self) -> None:
        for name in _CONFIG_FIELDS:
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        if self.max_length < 1 or self.pad_to_multiple_of < 1:
            raise ValueError("length and padding multiple must be positive")
        if self.pad_token_id < 0:
            raise ValueError("pad_token_id must be nonnegative")
        if self.ignore_index >= 0:
            raise ValueError(
                "ignore_index must be negative and distinct from token IDs"
            )


def sft_config_from_mapping(config: Mapping[str, object]) -> SFTBatchConfig:
    """Validate a fully resolved Hydra mapping without reading a tokenizer."""
    if not isinstance(config, Mapping) or set(config) != _CONFIG_FIELDS:
        raise ValueError("SFT config has missing or unknown fields")
    return SFTBatchConfig(**config)


def _ids(value: object, name: str) -> list[int]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of integer IDs")
    if any(type(token) is not int or token < 0 for token in value):
        raise ValueError(f"{name} must contain nonnegative integer IDs")
    return list(value)


def _prepare_row(
    sample: Mapping[str, object], config: SFTBatchConfig
) -> tuple[list[int], list[int]]:
    if not isinstance(sample, Mapping) or set(sample) != _SAMPLE_FIELDS:
        raise ValueError("SFT sample has missing or unknown fields")
    if sample["split"] != "train":
        raise ValueError("SFT preparation accepts only declared train samples")
    prompt = _ids(sample["prompt_ids"], "prompt_ids")
    response = _ids(sample["response_ids"], "response_ids")
    mask = _ids(sample["response_mask"], "response_mask")
    if not prompt or not response:
        raise ValueError("prompt and response must not be empty")
    if len(mask) != len(response) or any(value not in (0, 1) for value in mask):
        raise ValueError("response_mask must contain one binary value per response ID")
    if not any(mask):
        raise ValueError("SFT sample must contain at least one policy target")
    if len(prompt) + len(response) > config.max_length:
        raise ValueError("sample exceeds configured max_length; truncation is disabled")
    inputs = prompt + response
    labels = [config.ignore_index] * len(prompt) + [
        token if owned else config.ignore_index
        for token, owned in zip(response, mask, strict=True)
    ]
    return inputs, labels


def build_sft_batch(
    samples: Sequence[Mapping[str, object]], *, config: SFTBatchConfig
) -> dict[str, list[list[int]]]:
    """Return right-padded input IDs, attention masks and unshifted causal labels.

    All samples must be caller-declared train rollouts with unpadded token arrays.
    Registry membership, teacher eligibility and tokenizer/vocabulary consistency
    must be established by the data pipeline before calling this function. No
    sample is filtered, truncated, packed or retokenized here. Labels assume the
    future causal-LM backend shifts targets exactly once inside its loss function.
    """
    if not isinstance(config, SFTBatchConfig):
        raise ValueError("config must be a validated SFTBatchConfig")
    if (
        isinstance(samples, str | bytes | bytearray)
        or not isinstance(samples, Sequence)
        or not samples
    ):
        raise ValueError("samples must be a nonempty sequence")
    rows = [_prepare_row(sample, config) for sample in samples]
    longest = max(len(inputs) for inputs, _ in rows)
    multiple = config.pad_to_multiple_of
    width = ((longest + multiple - 1) // multiple) * multiple
    if width > config.max_length:
        raise ValueError("padding alignment exceeds max_length")
    batch: dict[str, list[list[int]]] = {
        "input_ids": [],
        "attention_mask": [],
        "labels": [],
    }
    for inputs, labels in rows:
        padding = width - len(inputs)
        batch["input_ids"].append(inputs + [config.pad_token_id] * padding)
        batch["attention_mask"].append([1] * len(inputs) + [0] * padding)
        batch["labels"].append(labels + [config.ignore_index] * padding)
    return batch
