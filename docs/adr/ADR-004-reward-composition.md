# ADR-004: Reward composition

- Status: **Proposed** — confirm before the R2 run (18 Oct 2026)
- Related tasks: T3.2–T3.4; definitions of labels and rules in
  [labels.md](../labels.md), metrics in [metrics.md](../metrics.md)

## Decision

Every episode's reward is computed by `handoff.rewards.composer.RewardComposer`
from the shared `Trajectory` record, identically in evaluation and training:

$$
R = R_{\text{task}} - \lambda_v \sum_r w_r N_r - \lambda_{FP}\,\mathbf{1}[\text{transfer} \wedge \neg y]
  - \lambda_{FN}\,\mathbf{1}[\neg\text{transfer} \wedge y] - \lambda_f N_{\text{format}}
$$

- $R_{\text{task}} \in \{0, 1\}$: τ²-bench's evaluator (`EvaluationType.ALL`) on
  the task's own `reward_basis`. Tasks whose reward would call tau2's LLM judge
  (`NL_ASSERTION` in the basis with a non-empty `nl_assertions` list) are left
  out of the splits by the task-set build and rejected before any rollout; a
  basis with `NL_ASSERTION` but no assertions scores 1.0 without a judge
  (ADR-005). Early stops (turn cut at the token limit, empty message,
  non-object tool arguments, context overflow) end the episode with reward 0 in
  both drivers.
- $N_r$: violations of rule $r$ (five deterministic rules, see labels.md),
  weighted by `violation_weights` (default 1 each). `off_reference_write` is the
  README's "write actions outside the gold action set"; its weight can be set to
  0 in the reward config if it proves to punish valid alternative solutions (the
  concern raised in ADR-006 D2) without code changes.
- $y$: the should-transfer label of the task; "transfer" means a
  `transfer_to_human_agents` call, whatever the tool returned (as in τ²-bench's
  action check, so reward and label agree).
- $N_{\text{format}}$: malformed actions (unparsed tool-call markup, unknown tool,
  rejected arguments, empty message, truncated generation, non-object arguments).
- Policy citations are **measured, not rewarded** (README highlight): rewarding
  a phrase is easy to game.

Every term is logged separately (verl `reward_extra_info`, `metrics.json`).

## Settings (configs/reward)

| Preset | λ_v | λ_FP | λ_FN | λ_f | Run |
| --- | --- | --- | --- | --- | --- |
| `task_only` | 0 | 0 | 0 | 0 | R1 (total = task reward) |
| `r2` | 0.25 | 0.5 | 1.0 | 0.1 | R2 |

Rationale: one violation (0.25) costs a quarter of a solved task, so solving
with a slip still beats failing; a wrong transfer on a solvable task costs half
a task (FP), a missed necessary transfer a full task (FN), because the latter
leaves a customer without help. The FP term exists because violation penalties
alone make "transfer everything" safe (README).

## No λ sweep

To keep the study small, R2 uses the one setting above, chosen from the
rationale rather than tuned; there is no grid and no reward ablation. If R2
clearly over- or under-escalates on dev (hand-off F1 and over-escalation rate
against R1), one adjusted setting may be tried, recorded here with its reason
and run IDs. Test is run only for the final checkpoints (AGENTS hard rule 1).
