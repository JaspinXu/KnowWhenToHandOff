# Configuration

Hydra configuration is kept in this checkout. The foundation root config contains
only an explicit seed. Model, environment, reward, training and evaluation groups
are introduced by their owning tasks; none is silently selected here.

When composing from Python, use an absolute config directory with
`hydra.initialize_config_dir(version_base="1.3", config_dir=...)`. Scripts resolve
the directory from their own location, not the shell's current directory.

Configuration templates contain no credentials. Store credentials in the ignored
`.env` or the cluster environment. Do not copy the complete process environment
into experiment records.

`personas.yaml` composes the six draft communication-style cards from
`persona/styles_v1.yaml`. This config does not select tasks, splits or a simulator.
See [persona injection](../docs/personas.md) for usage and the pending dialogue audit.

`sft_batch.yaml` composes `train/sft_batch.yaml` for CPU token-batch preparation.
Supply the actual tokenizer's `train.pad_token_id` before resolving it. This does
not launch training or choose a checkpoint; see [SFT batches](../docs/sft-batches.md).
