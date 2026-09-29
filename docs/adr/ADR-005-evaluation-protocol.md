# ADR-005: Evaluation protocol

- Status: **Proposed — to be frozen at M1 (11 Oct 2026).** After acceptance any
  change to a split, trial count, simulator, metric or setting below needs a new
  ADR (AGENTS hard rule 4).
- Related tasks: T0.2, T1.1–T1.6, T3.6; supersedes ADR-006 (see its status line)

## Benchmark

- τ²-bench **v1.0.1**, commit `fc0055dc4e0a316c3f83133267fbd6faaa770992`,
  domains airline and retail, installed from a git checkout (tau2 is not on
  PyPI). Its task data equals commit `b7ea907` audited in ADR-006. Every run
  records the commit; `run_eval.py` refuses to run if the installed commit differs
  from the one in the split manifest.
- Upstream code is never edited; the project registers its agent under
  `handoff_agent:<prompt variant>` and wraps everything else.

## Splits (data/manifests, built by `scripts/build_task_sets.py`)

| Split | Airline | Retail | Source |
| --- | --- | --- | --- |
| Train | official train minus dev | official train minus dev | `split_tasks.json` (30 / 74 tasks) |
| Dev | ≈ 20 % of official train | ≈ 20 % of official train | stratified by should-transfer label, seed 20261011 |
| Test | 20 tasks | 40 tasks | official test |

Hand-off tasks (explicit request, out of scope, hard negative) are generated per
source task and **inherit the source task's split**; no source task contributes
to two splits. Counts per split and kind: `configs/handoff_tasks/v1.yaml`.
Labels, splits and derived tasks are hashed in the manifests and committed.

## Episodes

| Setting | Value |
| --- | --- |
| Agent | tau2 `LLMAgent` with the shared system prompt (`handoff_v1`), vLLM `hermes` parser |
| Agent sampling | temperature 0, max 2,048 tokens per turn (teacher: 0.6) |
| User simulator | ADR-003; temperature 0.7, max 1,024 tokens per turn |
| Limits | 200 steps (tau2 messages), 10 tool errors, 30 min wall clock |
| Seeds | run seed 300 (teacher 1234); tau2 derives one seed per trial |
| Scoring | `EvaluationType.ALL` on the task's reward basis (no NL assertions) |
| Early stops | a turn cut at the token limit, an empty message or non-object tool arguments ends the episode (`context_window_exceeded` / `agent_error`, reward 0), the same as in training; so does a conversation that no longer fits the server's 32,768-token context with a full turn (no format error) |
| End of a turn | vLLM stops on every end-of-turn token of the model's generation config (Qwen: `<|im_end|>`, `<|endoftext|>`); training uses the same set |
| Infrastructure failures | an episode that still fails after 3 retries (server error, timeout) is listed and counted (`n_infra_failures`, by style, kind and error); a run with lost episodes does not complete and is resumed in place until it has none (teacher rollouts tolerate 5 %) |

## Protocols (configs/eval)

| Preset | Split / kinds | Styles | Trials | Purpose |
| --- | --- | --- | --- | --- |
| `smoke` | 5 train tasks (airline) | neutral | 1 | stack check |
| `dev` | dev, all kinds | neutral | 4 | checkpoint decisions (B2 gate, R1 vs R2) |
| `test_main` | test, upstream | neutral | 4 | main table |
| `test_styles` | test, upstream | 5 non-neutral styles | 2 | style gap (neutral from `test_main`) |
| `test_handoff` | test, hand-off kinds | neutral | 2 | hand-off calibration |
| `teacher_train` | train, all kinds | 6 styles | 2 | SFT data (T2.1) |

Test protocols run **once per frozen checkpoint** and simulator; the test ledger
(`state/test_ledger.jsonl`) records each test run's policy revision, simulator,
test tasks and styles, and refuses a new run that overlaps an earlier one in
both a task and a style (whatever the preset is called), unless forced with a
documented reason. Resuming an unfinished run is not a new evaluation. Reports
join a checkpoint's `test_main`, `test_styles` and `test_handoff` runs
([metrics](../metrics.md#report-tables-scriptsmake_reportpy)). All selection
uses dev.

## Metrics

Definitions are in [metrics.md](../metrics.md): pass^1 and pass^k (unbiased
estimator per task × style), violation rates, hand-off precision/recall/F1,
over- and under-escalation, format-error rate, citation rate (rule-based), style gap and worst-style success,
cost. 95 % percentile bootstrap intervals over tasks (1,000 resamples, seeded);
comparisons between runs are paired on shared tasks.

## Human checks

- 30+ hand-off tasks hand-checked before M1 (review packet `tasks`).
- 10 dialogues per style reviewed (packet `styles`, T1.4).
- 20 dialogues per final checkpoint reviewed (packet `audit`, T3.5/T3.6).
