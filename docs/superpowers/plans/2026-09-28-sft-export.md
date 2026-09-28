# Offline SFT Batch Export Plan

> Execute inline with test-driven-development and one independent final review.

**Goal:** Make existing token batch preparation runnable from JSONL files without
installing a trainer or changing the training-data selection rules.

**Spec:** README T2.2, `docs/sft-batches.md`, AGENTS provenance/artifact rules.
The owner permits independent code; ADR-006 and ADR-007 remain deferred.

## Contract

- `train/sft_export.py::export_sft_batches(source, destination, *, batch_config,
  batch_size, max_record_bytes, git_sha, git_dirty) -> dict` streams one JSON object
  per input line. Records use the existing four SFT sample fields exactly.
- Read at most `max_record_bytes + 1` bytes per record and hold only one batch.
  Blank lines, duplicate JSON keys, invalid JSON and invalid train samples fail
  explicitly; no sampling, filtering, truncation or dataset selection is performed.
- Write `batches.jsonl`, `config.json`, and `manifest.json` into a newly created
  directory. Manifest is written last with source/output/config hashes, counts,
  git SHA/dirty flag, and `kind=offline_sft_batch_export`. It is not a run result.
- Refuse existing destinations. On normal exceptions clean only files created
  by this invocation, including close/flush failures; preserve the input and any
  unrelated concurrent file. Hard termination can leave a manifest-less directory.
- `scripts/prepare_sft.py` is a thin Hydra CLI. Resolve paths from the invoking cwd,
  configs from the script location. Refuse output inside the repository checkout;
  use scratch. Print counts and safe errors, never input records or raw exceptions.
- No verified membership/eligibility claim: the caller must supply approved,
  eligible tokenized data later. Actual trainer and tokenizer integration remain
  unimplemented. Dirty-source bundles explicitly record that limitation.

## Steps

- [x] Write failing exporter tests: multiple batches and final remainder, exact
  hashes/counts/token preservation, empty/blank/oversize/duplicate-key/malformed
  inputs, late nontrain failure cleanup, no overwrite, write/close failures.
- [x] Implement exporter with the existing SFTBatchConfig/build_sft_batch API.
- [x] Write failing CLI tests from an unrelated cwd, no implicit tokenizer ID,
  secret-safe errors, scratch restriction, existing-output preservation.
- [x] Implement Hydra config and thin CLI; run CPU/lint/format/build checks.
- [x] Independent review and important fixes, then update bilingual usage/status.
- [x] Commit T2.2 code with reproduction command; do not tick training completion.

Validation: 274 CPU tests passed, including 33 export/CLI cases independently
rerun in review. Ruff, formatting and wheel/sdist builds passed. Independent
review found no actionable issues. Actual training and data eligibility remain
outside this completed prerequisite.
