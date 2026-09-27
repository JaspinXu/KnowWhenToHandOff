# Independent Runtime Utilities

> Execute natively with test-driven-development. Owner requested continued
> independent implementation while ADR-006 remains deferred.

**Goal:** Record reproducible runs without storing credentials, and preserve
sampled token IDs with correct policy/context masks before GPU integration.

**Spec:** AGENTS sections 5 and 7; README component contracts and T3.1 token rules.
This introduces no benchmark pin, split, reward, metric or Trajectory changes.
Existing dependencies suffice; only Python standard library is used.

## Units and verification

### T0.3 supporting run provenance

- Create `src/handoff/ops/provenance.py` and `tests/test_provenance.py`.
- API: `create_run_directory(root, *, run_id, config, provenance) -> Path`.
  Inputs contain an explicit Git SHA, benchmark SHA, policy/simulator model SHAs,
  named nonnegative integer seeds, hardware metadata, purpose and execution kind
  (`real` or `synthetic_test`). No revision is chosen by this utility.
- Output: exclusively created directory with sanitized resolved config,
  provenance plus config hash, and an explicit initialized state. No experiment
  is considered completed merely because this directory exists.
- API: `finish_run(run_dir, *, status, error_code=None) -> None` writes an
  exclusive terminal record (`completed`, `failed`, `interrupted`).
  Failure text is a bounded identifier, not arbitrary exception content.
- Validate before creating output; redact known credential keys recursively,
  reject unresolved interpolation/nonfinite JSON values, never copy process env.
- Test immutable provenance, exact config hash, secret redaction, missing
  revision/seed errors, non-overwrite, corrupted source records, terminal state
  validation and cleanup of failed initialization.

### T3.1 supporting token bookkeeping

- Create `src/handoff/rollout/tokens.py` and `tests/test_tokens.py`.
- API: `TokenAccumulator(prompt_ids, *, max_response_tokens)`,
  `append(role, token_ids)`, `as_output() -> dict[str, list[int]]`.
- Output keys exactly match the README token fields: `prompt_ids`,
  `response_ids`, `response_mask`. Assistant generation (including tool-call
  text) has mask 1; user/tool observations have mask 0.
- Inputs are actual nonnegative integer IDs, never text. Copy input buffers.
  Reject unsupported roles, booleans/non-integer IDs and overflow before
  mutation. Never silently truncate or tokenize messages.
- Test multiple turns, adjacent tool observations, exact token identity, masks,
  empty segments, invalid inputs, capacity edges and independent output copies.
- This is not a verl adapter or a reward parity test. Those integrations remain
  pending; do not tick T3.1 as complete.

## Workflow

- [x] Observe initial failing contract tests.
- [x] Implement provenance; validate focused tests.
- [x] Implement tokens; validate focused tests.
- [x] Run full CPU/lint/format/build checks and independent review.
- [x] Commit bounded task steps and update bilingual implementation status.
