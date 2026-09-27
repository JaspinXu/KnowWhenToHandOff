# ADR-006: Pre-implementation benchmark and reward clarifications

- Status: **Deferred by owner — not accepted**
- Date: 2026-09-27
- Related tasks: T0.2, T1.1, T1.2, T1.5, T3.2
- Reserved ADR-001 through ADR-005 retain their README purposes.

Owner response on 27 September: do not adopt these definition changes yet; implement independent infrastructure first. No part of this draft changes the active benchmark, splits, rewards, or evaluation protocol.

## Context

The [initial audit](../repository-audit-2026-09-27.md) found differences between
the design, upstream behavior, and the available execution environment. AGENTS
requires asking before changes to evaluation, splits, dependencies, or shared
schemas. This document proposes resolutions; it changes no operational contract.

## Alternatives

1. **Recommended: retain the design and resolve the concrete ambiguities.** Pin
   the inspected upstream revision, keep official outcome rewards intact, and
   distinguish policy violations from reference-trajectory deviations. This
   supports honest comparisons but requires updating dataset counts and the
   English/Chinese reward descriptions after approval.
2. **Find and pin an older tau2 release.** Potentially closer to the draft's task
   counts, but it requires a fresh compatibility audit and can retain task bugs.
   An older count alone is not a reason to select a benchmark revision.
3. **Build only a simulated demonstration now.** Useful for interface prototyping,
   but it does not fulfill the requested RL study and must not be presented as
   a completed project. Not recommended as the main deliverable.

## Proposed decisions

### D1. Initial upstream dependency

Use `tau2` pinned to commit
`b7ea9074c1cba482b30687fecdb5c8425fd6f619` for T0.2, including the matching upstream
data revision. Verify packaging and imports before accepting the pin. Keep
downloaded data outside git, point `TAU2_DATA_DIR` to that exact snapshot, and
record hashes. Do not fetch `main` dynamically at run time. If compatibility fails,
return to this decision rather than silently substituting a revision.

### D2. Preserve official task reward; separate compliance

Respect each task's deterministic `reward_basis` for official task success.
Do not make reference-action matching an extra official success requirement.
Treat off-reference writes as an audit diagnostic, not automatically as policy
violations. A compliance penalty requires a documented policy clause and a
deterministic check with positive and negative fixtures.

Training rejects tasks whose reward requires `NL_ASSERTION`; training evaluation
uses `EvaluationType.ALL` after that validation. No training path calls the
judge, including diagnostic-only calls. Tests replace the judge with a raising
fake to verify this boundary. Synthetic hand-off tasks receive their own approved
deterministic criteria, without modifying upstream tasks.

### D3. Protect source-task lineage

T0.2 smoke uses five explicitly selected official training tasks. Later T1.1
carves dev only from train and freezes the manifest before selection runs. Every
derived task carries its upstream domain, source task ID, upstream SHA, and split.
All variants of a source task stay in the same split. Do not use different variant
IDs to bypass this rule. Exact sampling rules and seeds belong in the T1.1 proposal.

### D4. Clarify model scope and trials

Interpret the <=8B constraint as applying to the policy being trained. Larger
models may be used for the teacher, simulator, and B1 reference within confirmed
compute. Preserve four neutral trials and two trials per non-neutral style;
reuse neutral results for the neutral style reference rather than double-counting.
Report pass^4 only for groups with at least four trials. This protocol detail
must be accepted in ADR-005 before reportable evaluations.

### D5. Defer unapproved shared-schema details

T0.1 creates no `Trajectory`, `RewardBreakdown`, or `TaskRegistry` API. Before
T1.5, propose explicit fields and semantics for tool-format failures, termination,
refusal/citation evidence, tool counts, and provenance. Before T3.1, specify
per-turn generated IDs and masks. The eventual schema proposal must update every
consumer together; this ADR does not approve a schema that has not been written.

## Consequences and approval scope

Accepting D1–D4 permits the associated implementation proposals in their original
task order and synchronized README corrections. It does not mark any experiment,
human review, metric freeze, or shared-schema change complete. Heavy trainer and
serving dependencies still require their task-specific ADRs and resource checks.

T0.1 can proceed independently once its small dependency set and plan are
approved. No accepted decision or experiment result is recorded by this draft.
