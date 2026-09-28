# Assistant-only SFT batches

This T2.2 prerequisite connects `TokenAccumulator` output to dense token batches
using the standard library. It does not load a model, select teacher trajectories,
write a dataset, run an optimizer or produce a checkpoint. No new dependencies
or scientific definitions are introduced.

For file-based preparation, the [offline exporter](sft-export.md) now streams
tokenized JSONL through this builder and writes batches with a hashed manifest.
It still does not select or certify eligible teacher data.

## Input and output

Each sample contains exactly `split`, `prompt_ids`, `response_ids` and
`response_mask`. Token arrays must be unpadded sequences of nonnegative integer
IDs; booleans are rejected. Masks must have one integer 0/1 per response token.
Every row must contain a nonempty prompt and at least one policy target.

Only `split="train"` is accepted. Missing, dev, test or other split declarations
fail the whole batch; rows are never silently filtered. **This checks the caller's
declaration, not official membership or task lineage.** The future data pipeline
must obtain the split from its approved registry and retain source provenance.
It must also verify successful, violation-free, correctly transferred teacher
episodes before preparing their tokens. That eligibility logic is not implemented
here, and no split assignment or should-transfer rule is changed.

`build_sft_batch` returns `input_ids`, `attention_mask` and `labels` as ordinary
lists of rows in input order. Prompt tokens and all response tokens remain in
the input context. Labels copy policy token IDs at their original positions and
use `ignore_index` for prompt, user and tool positions. The collator does not
shift labels: the eventual causal loss must compare logits at `t` to targets at
`t+1` exactly once. No actual backend loss/gradient computation is tested here.

Padding is on the right, rounded to `pad_to_multiple_of`. Added padding has
attention 0 and ignored labels; real tokens have attention 1 even when their ID
equals `pad_token_id`. This preserves a real EOS token's target if EOS and PAD
share an ID. Samples are never packed together, truncated, reordered or
retokenized, and no BOS/EOS/header tokens are added. Oversize input or rounding
beyond `max_length` is an error rather than an implicit training-data change.

## Configuration and use

Compose `configs/sft_batch.yaml`; settings live in `configs/train/sft_batch.yaml`.
`pad_token_id` is mandatory (`???`) and must come from the actual policy tokenizer.
The checked-in 4096-token cap and padding multiple 8 are explicit starting
settings for local preparation, not a frozen experiment configuration or a claim
about a model's context window. The loss backend must honor the same configured
negative ignore index (the template uses -100). Vocabulary size, context-window
compatibility and tokenizer/model revision matching belong to the future adapter.

```python
from handoff.rollout.tokens import TokenAccumulator
from handoff.train.sft_batch import build_sft_batch, sft_config_from_mapping

# resolved_train_config is the resolved Hydra `train` section.
config = sft_config_from_mapping(resolved_train_config)
tokens = TokenAccumulator([10, 11], max_response_tokens=12)
tokens.append("assistant", [20, 21])
tokens.append("tool", [30])
tokens.append("user", [40])
tokens.append("assistant", [22])

# This example is synthetic. Actual code must use the registry-derived split.
sample = {"split": "train", **tokens.as_output()}
batch = build_sft_batch([sample], config=config)
assert batch["labels"][0][2:7] == [20, 21, config.ignore_index, config.ignore_index, 22]
```

Configuration is immutable; batches and rows do not alias input arrays or each
other. Preparation is deterministic and consumes no random state. A future
training driver must still record the actual source/model/tokenizer revisions,
resolved configuration and sampling seeds through the provenance utilities.

## Verification and remaining work

```bash
uv run --locked pytest -q tests/test_sft_batch.py tests/test_tokens.py
```

Tests cover multiple policy/context turns, causal target positions, EOS/PAD
collisions, unequal row lengths, padding alignment, invalid masks and IDs,
declared split protection, copy independence and Hydra composition from another
working directory. They require no GPU, trainer, tokenizer or network.

The SFT dataset builder, teacher eligibility checks, tensor/device conversion,
backend loss integration, optimizer, training launcher and B2 evaluation remain
unimplemented. T2.1 and T2.2 are not marked complete.
