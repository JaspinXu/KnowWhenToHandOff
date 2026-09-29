# Metric definitions

Implemented in `src/handoff/eval/metrics.py` (pure functions over `Trajectory`
records) and written to `runs/<run_id>/metrics.json` by every evaluation. These
definitions are part of the evaluation protocol ([ADR-005](adr/ADR-005-evaluation-protocol.md))
and are frozen at M1; `tests/test_metrics.py` pins them down (pass^k against
brute force, bootstrap shape, undefined cases).

## Units, intervals and missing values

- **Episode**: one conversation (task × style × trial). **Group**: the trials of
  one task in one style. **Bootstrap unit**: one task (all its styles and trials).
- **95 % intervals**: percentile bootstrap over tasks, 1,000 resamples, fixed
  seed (the run seed). `metrics.json` stores `point`, `low`, `high`, `n_units`
  and `n_resamples`, the number of resamples on which the statistic was defined
  (for example, precision needs at least one transfer in the resample).
- **Undefined** values (for example precision when nothing was transferred) are
  `null`, never 0.
- **Complete runs only.** Episodes lost to infrastructure are listed in
  `infra_failures.jsonl` and counted in `n_infra_failures`,
  `infra_failure_share` and `infra_failures` (by style, kind and error). A run
  with more lost episodes than `eval.max_infra_failure_share` (0 for every
  protocol except teacher rollouts, 0.05) does not complete and is resumed until
  it does; reports accept completed runs only.

## Headline metrics (`overall`, and per domain / kind / split when several)

| Name | Definition |
| --- | --- |
| `success` (pass^1) | Mean over groups of the group's success fraction; success = τ²-bench task reward 1 |
| `pass^k` | Mean over groups of C(c, k) / C(n, k), c successes in n ≥ k trials (unbiased estimator); `null` if some group has fewer than k trials |
| `violation_rate` | Share of episodes with ≥ 1 violation of any rule ([labels.md](labels.md)) |
| `violation_rate_rules_only` | Same, ignoring `off_reference_write` |
| `rule/<rule>` | Share of episodes with ≥ 1 violation of that rule |
| `format_error_rate` | Share of episodes with ≥ 1 format error |
| `transfer_rate` | Share of episodes in which the agent called `transfer_to_human_agents` (whatever the tool returned, as in τ²-bench's action check) |
| `handoff_precision` | TP / (TP + FP) over episodes (label y vs. transferred) |
| `handoff_recall` | TP / (TP + FN) |
| `handoff_f1` | 2TP / (2TP + FP + FN), which equals 2PR / (P + R) when both exist; 0 for an agent that never transfers although some episode should; `null` only without positives and transfers |
| `over_escalation_rate` | Transfers among episodes with y = 0 |
| `under_escalation_rate` | No transfer among episodes with y = 1 |
| `citation_rate` | Share of refusal and hand-off events whose text names the policy (rule match, below) |
| `user_stop_failure_rate` | Share of episodes the user ended without success and without a transfer |
| `mean_turns`, `mean_tool_calls`, `mean_tokens` | Cost per episode: assistant turns, tool calls, completion tokens |

## Style breakdown (`styles`, when a run has several styles)

- `success/<style>`: pass^1 restricted to that style.
- `style_gap`: max − min of the per-style success values (≥ 2 styles).
- `worst_style_success`: min of the per-style success values.

The main test protocol runs neutral only; `test_styles` runs the five other
styles, and the report combines both (the neutral reference is not
double-counted).

## Report tables (`scripts/make_report.py`)

A report label is one checkpoint and may join several of its runs
(`LABEL=RUN+RUN+...`, typically `test_main`, `test_styles` and `test_handoff`);
the runs must be completed real runs of the same policy and simulator revisions
and must not share a (task, style) pair. Every value is recomputed from the
joined trajectories:

| Table | Episodes |
| --- | --- |
| Main: success, pass^k, violation rate, turns | Upstream tasks, neutral style (the τ²-bench protocol) |
| Main: hand-off F1, over-escalation, citation rate | Every neutral-style episode, upstream and hand-off tasks |
| Styles: success per style, style gap, worst style | Upstream tasks, every style |
| Paired differences | Upstream tasks, neutral style, tasks shared with the baseline |

## Citation events

A **hand-off event** covers the agent's last message to the user before the
transfer and its first message after it. A **refusal event** is any other agent
message containing refusal language ("unfortunately", "unable to", "cannot",
"not eligible", ...). An event is **cited** if its text names the policy
("policy", "per our", "according to our/the", "our rules/terms/guidelines/
conditions", ...). The patterns are in `src/handoff/eval/citations.py`.

The rule match is spot-checked in the manual dialogue review (packet `audit`);
no LLM judge is used.

## Paired comparisons and reward terms

- **Paired difference** (report tables): bootstrap over the tasks both runs share,
  resampling the same task indices for both runs.
- **Reward terms** (`reward.mean` in `metrics.json`): the mean of every
  `RewardBreakdown` term under the run's reward config, for comparing with
  training curves.

## Reward-hacking watch-list (`hacking`, AGENTS §7)

`transfer_rate_on_negatives` (= over-escalation), `mean_messages`, `mean_turns`,
`format_error_rate`, `user_stop_failure_rate`. `scripts/monitor_rl.py` computes
the same values from the last 512 training episodes and flags any that exceeds
max(2 × the B2 value, 0.02).
