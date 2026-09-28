# Model Interaction Implementation Plan

> Execute inline with test-driven-development and one independent final review.

**Goal:** Implement the actual model I/O boundary required by baselines and
simulators, while preserving all deferred scientific decisions.

**Spec:** README T0.3/T0.5, AGENTS async/no-secrets rules, ADR-007 dependency proposal.
No changes to Trajectory, RewardBreakdown, TaskRegistry or ADR-006.

## Task 1: Dependency-free completion parser (T0.5)

Files: `src/handoff/agents/completions.py`, `tests/test_completions.py`.
`parse_completion(raw: bytes, *, allowed_tools: Collection[str],
max_tool_calls: int) -> Completion` returns immutable text/tool/usage records.
`Completion.to_message()` returns an independent assistant message suitable for
the next request; `ToolCall.parsed_arguments()` returns a new JSON object.

- [x] Write/observe failing tests for text, multiple calls and exact argument
  preservation; unknown tools, duplicated IDs/JSON keys, nonfinite or non-object
  arguments, malformed envelope, truncation, empty turns, and invalid usage.
- [x] Implement strict JSON parsing and sanitized failure codes. Never execute a
  tool or infer a policy violation, task reward, or RL token IDs from chat text.
- [x] Verify focused tests, then full CPU gates.

## Task 2: Async client and service smoke (T0.3/T0.5)

**Deferred by owner on 28 September: implement only Task 1's pure parser.**
Requires owner acceptance of ADR-007 before adding HTTPX. Do not implement a
replacement HTTP stack to work around that dependency decision.

- [ ] After approval, lock HTTPX 0.28.1 and add `agents/client.py`, client Hydra
  config, `sim/service_smoke.py` and thin `scripts/smoke_model.py`.
- [ ] Write failing transport tests for exact seeded request data, safe endpoint
  validation, bounded concurrency, whole-request timeouts, cancellation, model
  mismatch, bounded bytes, HTTP/redirect failures and secret-free diagnostics.
- [ ] Implement async context lifecycle and stream-bounded nonstreaming responses.
  No automatic retries, implicit proxies, redirect following or external defaults.
- [ ] Smoke generates an automatic tool call, validates/locally returns a harmless
  synthetic tool result and requests a final text turn. A failed probe must exit
  nonzero. No benchmark reward or throughput result is inferred.
- [ ] Test real CLI validation and in-process end-to-end smoke with fake transport;
  all sockets remain disabled. Real endpoint validation stays pending.

## Review focus and completion

Untrusted model replies must not execute unknown tools; parser failures must not
leak content; cancellation must release concurrency slots; response bounds must
apply before buffering; tool messages must preserve IDs and original arguments.

- [x] Independent read-only review; reproduce/fix important findings once.
- [x] Update code-readiness, usage, both READMEs and AGENTS commands.
- [x] Full pytest, Ruff, build, pre-commit and diff checks; task-prefixed commits.

Task 1 evidence: 43 expected missing-module failures, then 43 focused passes.
Review found a Unicode serialization defect; seven added regression cases failed
before the fix. Full suite after fixes: 200 passed. Exception wording was narrowed
to code-only diagnostics rather than claiming Python exceptions retain no input.
Task 2 remains deferred, and no dependency files were changed.
