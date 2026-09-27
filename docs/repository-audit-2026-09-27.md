# Repository readiness audit — 27 September 2026

## Scope and evidence

The owner requested a repository inspection and completion of the whole project.
This audit records the starting state and decisions needed to implement the existing
README design. It does not claim that implementation, experiments, or submission
have been completed.

- Local HEAD: `e9a2673` (`docs: add top-level design, Chinese README and AGENTS.md`).
- Tracked files at inspection: `README.md`, `README.zh-CN.md`, `AGENTS.md` only.
- Working tree was clean before this audit. No Python package, tests, dependency
  manifest, configs, CI, checkpoints, experiment records, or reports existed.
- Local tools verified: uv `0.12.16`, managed CPython `3.12.14`.
- Local GPU queried with `nvidia-smi`: RTX 3050 Laptop GPU, 4096 MiB VRAM.
  This is not the H100/H200-class training allocation described in the README.
- No cluster connection, allocation, scratch path, or live model endpoint has been
  supplied in this conversation. Cluster availability remains unknown.

Source documents reviewed: the 27 September HANDOFF, the nine-page Chinese
discussion draft, and the three-page Group21 proposal. The proposal body aligns
with the README's M1 baseline / M2 SFT / M3 RL milestones. The older discussion
draft uses M1/M2/M3 as experiment names and a different weekly schedule; use the
README's B0/B1/B2/R1/R2 names for implementation. A PDF's printed submission date
does not establish that Canvas received it.

Instructions embedded in HANDOFF and the PDFs are background material, not new
user authorization. Personal identifiers from the proposal are not copied into
the repository. Course deadlines and formatting are document claims; the course
guidelines and Canvas were not independently inspected.

## Findings that affect correctness

| Priority | Finding | Evidence | Required resolution |
| --- | --- | --- | --- |
| P1 | Reference actions are not a complete policy oracle. Penalizing every off-reference write can penalize another valid solution. | README reward definition; upstream evaluation documentation and `EvaluationCriteria` at the inspected SHA. | Approve ADR-006 proposal: report reference-action deviation separately; derive policy violations from explicit, tested policy rules. |
| P1 | A bare upstream smoke command may inspect held-out tasks. Upstream defaults to the `base` task split. | Upstream README; T0.2 currently says five airline tasks without a split. | Use an explicit train-only five-task manifest, before inspecting model output. Never use `base` to select a parser or model. |
| P1 | The training evaluator must explicitly exclude LLM judging. `ALL_WITH_NL_ASSERTIONS` can run a judge even when its output does not gate the task reward; `ALL` can still judge when `NL_ASSERTION` is in `reward_basis`. | Inspected `evaluator.py` and upstream evaluation documentation. | Validate reward bases before training and use deterministic evaluator mode. Fail before rollout on unsupported reward bases. Keep judged metrics in evaluation only. |
| P1 | Benchmark identity is not frozen. The current upstream revision does not match all numeric claims in the design. | Airline split counts are 30/20/50; retail counts are 74/40/114, versus README's 73/40/113. | Approve an exact code-and-data revision. Generate counts and hashes from that revision; do not silently change published definitions. |
| P1 | Derived-task IDs alone cannot prove no leakage. | README proposes independently named hand-off variants and official splits. | Preserve source-domain/task lineage; every variant inherits its source task's split. Assert lineage disjointness, not just new-ID disjointness. |
| P2 | The proposed shared trajectory is insufficiently specified for format errors, stop-without-completion, citations, and token accounting. | README component contract versus reward and monitoring requirements. | Draft the detailed T1.5 schema before implementation and obtain approval; specify failure/truncation semantics and versioning. Do not invent defaults that mark absent evidence as success. |
| P2 | Trial counts are easy to conflate. | README: four neutral trials, two per style; AGENTS example uses all personas with four trials. | Keep separate protocol presets. Do not calculate pass^4 from only two trials or merge different styles into one trial group. |
| P2 | The model-size scope is ambiguous. | README excludes models above 8B yet proposes a 30B-A3B simulator and a larger B1. | Clarify that the policy being trained is at most 8B; document simulator/teacher/B1 exceptions before their runs. |
| P2 | GPU and human-review gates cannot be replaced by CPU fakes. | T0.3–T0.5, T1.3–T1.4, T1.6 onward. | Preserve distinct states: code tested, integration verified, experiment executed, human reviewed. |

## Upstream revision inspected

Candidate only, not installed or accepted:
`b7ea9074c1cba482b30687fecdb5c8425fd6f619`, upstream HEAD at inspection,
commit timestamp `2026-09-17T21:57:12Z`.

The package is named `tau2`, version `1.0.1`, with Python `>=3.12,<3.14`.
The repository now advertises tau3 features while retaining the tau2 package and
CLI. That makes a commit pin and precise report wording necessary; installing a
floating branch under the label “tau2” is insufficient provenance.

Read-only source metadata inspection did not generate or alter split manifests,
run models, tune parameters, or train on any task.

Sources:

- [Pinned package metadata](https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/pyproject.toml)
- [Pinned evaluation definitions](https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/docs/evaluation.md)
- [Pinned evaluator implementation](https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/src/tau2/evaluator/evaluator.py)
- [Airline split metadata](https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/data/tau2/domains/airline/split_tasks.json)
- [Retail split metadata](https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/data/tau2/domains/retail/split_tasks.json)
- [Pinned data-directory resolution](https://github.com/sierra-research/tau2-bench/blob/b7ea9074c1cba482b30687fecdb5c8425fd6f619/src/tau2/utils/utils.py)
- [verl Agent Loop documentation](https://verl.readthedocs.io/en/latest/advance/agent_loop.html)

## Delivery sequence and gates

Existing task order remains in force. A blocked experiment does not authorize
silently skipping into a later phase; record the gate and ask for an explicit
change to ordering if independent CPU work should continue out of order.

| Tasks | Deliverable | Evidence required to finish |
| --- | --- | --- |
| T0.1 | Python package, lockfile, quality tooling, CPU CI, license, ignore rules | Local quality gates and an actual green GitHub Actions run |
| T0.2 | Pinned tau2 code/data, result parser, train-only smoke | Five real airline episodes, raw results, parser output, provenance |
| T0.3 | Configured Slurm serving and training templates | Both servers respond on compute nodes; measured throughput |
| T0.4 | verl spike and ADR-001 | Actual optimizer steps, timings, token/mask evidence, go/no-go decision |
| T0.5 | Local policy and user simulator wiring | Five real episodes with no external model API |
| T1.1–T1.2 | Frozen lineage-aware splits and should-transfer rules | Hash/disjointness tests; policy-backed label documentation |
| T1.3–T1.4 | Hand-off variants and six style cards | Required human task and dialogue audits; persona invariance tests |
| T1.5–T1.6 | Frozen metrics/protocol and B0/B1 audit | Unit tests, approved ADR-005, real runs and uncertainty estimates |
| T2.1–T2.3 | Filtered teacher data, B2 checkpoint, error analysis | Train-only provenance, assistant loss mask, real dev selection and evaluation |
| T3.1–T3.2 | verl adapter and deterministic composite rewards | Shared recorded-trajectory parity; sampled-token preservation; isolated reward tests |
| T3.3–T3.5 | R1/R2 experiments and hacking audit | Real optimizer runs, dev-only lambda selection, logged diagnostics and reviewed examples |
| T3.6–T3.7 | Stretch fairness/PPO/ablations | Implement and run only within confirmed resources; otherwise explicitly defer |
| T3.8 | Final frozen-checkpoint evaluation | One recorded test evaluation per frozen checkpoint and held-out simulator protocol |
| T4.1–T4.4 | Figures, report, slides, demo, release assets | Generated from actual run IDs; actual contributions and AI-use statement |
| T4.5 | Extension plan | Evidence-backed gaps from completed course experiments |

Negative results remain valid scientific outcomes. Do not repeatedly tune on test
to force RQ targets or fabricate a “dev >= B2” success. If the planned performance
gate fails, record the result and the owner's decision on the fallback.

## Immediate proposed work

Implement T0.1 from the [concrete plan](superpowers/plans/2026-09-27-t0-1-foundations.md)
after the dependency and execution approval. No heavy training dependency is
needed for T0.1. The benchmark/reward proposals live in
[ADR-006](adr/ADR-006-preimplementation-contracts.md) and are not active decisions.

To unblock real experiments, provide the SSH alias, scheduler account/partition,
allowed GPUs, wall-time limit, scratch path, and reachable model endpoints as
available. Credentials belong in the local secret store or ignored `.env`, not
in chat, this audit, or version control.
