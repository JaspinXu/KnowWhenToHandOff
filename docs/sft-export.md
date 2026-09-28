# Offline SFT batch export

`scripts/prepare_sft.py` connects tokenized JSONL files to the existing assistant-only
batch builder. It uses the standard library and the existing Hydra dependency.
No model, network client, optimizer, dataset selection or experiment is run.

## Command

Run `uv run python scripts/prepare_sft.py --help` from this checkout.
Provide an existing JSONL input, a new directory on scratch outside the checkout,
and the pad ID from the actual policy tokenizer. For example, after setting the
three variables to your real paths/tokenizer value in PowerShell:

```powershell
uv run python scripts/prepare_sft.py "input_path=$tokenJsonl" "output_dir=$sftOutput" "train.pad_token_id=$policyPadId" batch_size=8
```

The command reads `configs/prepare_sft.yaml`, which includes `train/sft_batch.yaml`.
Other overrides include `train.max_length`, `train.ignore_index`,
`train.pad_to_multiple_of` and `max_record_bytes`. Input/output paths resolve from
the invoking directory; Hydra configuration resolves from the script's location.
Quote Hydra values when paths contain spaces or grammar characters.

The output directory must not already exist. The CLI also refuses paths inside
the repository after resolving links. Library callers must independently choose
scratch paths; the generic exporter does not discover repository locations.

## Input

One JSON object per line, using exactly the existing SFT sample fields:

```json
{"split":"train","prompt_ids":[10,11],"response_ids":[20,30,21],"response_mask":[1,0,1]}
```

This example is synthetic, not a tokenized real conversation. Real input must
already have approved membership and teacher eligibility established by the
future data pipeline. The command checks the declared `train` value; it cannot
verify official split membership, task lineage, successful completion, compliance,
correct transfer or tokenizer provenance. Retain the upstream task/run manifests
and row mapping separately; these four token fields do not contain that evidence.

Blank lines, duplicate keys, malformed JSON, nonstandard numeric constants,
invalid token arrays/masks, nontrain declarations and oversized samples fail.
No row is silently dropped or repaired. `max_record_bytes` counts bytes including
line endings; the reader requests at most that bound plus one detection byte.
The final record may omit its newline. CRLF inputs are accepted and hashed exactly.

## Bundle and provenance

- `batches.jsonl`: one JSON object per batch, with `input_ids`, `attention_mask`
  and `labels`. The final smaller batch is retained. Rows preserve source order;
  labels and positional padding follow [the batch contract](sft-batches.md).
- `config.json`: exact resolved batching/export settings, excluding paths and
  process environment. It contains no credentials.
- `manifest.json`: artifact kind, schema version, creation time, code SHA and
  dirty flag, SHA-256 of the exact input bytes and both output files, record/batch
  counts, real input-token count (excluding padding) and supervised-token count.

Only a bounded batch is held in memory. Output uses UTF-8 and LF. No reshuffle or
random sampling occurs, so batch/config bytes are deterministic for identical
input and configuration; the manifest's creation time naturally changes.
Counts describe this transformation, not model performance or training results.
The CLI obtains Git metadata from this checkout. A dirty flag records incomplete
code provenance; that bundle cannot be reproduced from its commit SHA alone.

The manifest is written last. Treat a directory without a complete, valid manifest
as unfinished. Normal exceptions remove only files created by the attempt, while
preserving the input and unrelated files. Abrupt process termination or a cleanup
I/O failure may leave an incomplete directory; inspect it and choose a new output
path rather than overwriting it. This is not a transactional filesystem or a
malicious-tampering defense. Consumers must check the stored hashes before reuse.

## Verification and boundaries

```bash
uv run --locked pytest -q tests/test_sft_export.py tests/test_sft_export_cli.py
```

Tests execute the real CLI from another directory with synthetic local files,
and exercise late failures, byte bounds, exact hashes/counts, non-overwrite and
write/close cleanup. No sockets, model weights or GPU are needed.

This completes an offline export command, not T2.1 teacher dataset construction
or T2.2 training. Official membership/eligibility checks, tokenizer/task provenance,
backend tensor/loss integration, optimizer/checkpoints and B2 evaluation remain open.
