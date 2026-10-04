# Experiment log

One row per run that produced numbers (AGENTS §7). Append rows with
`python scripts/append_experiment_log.py <run_id> --purpose "..." --decision "..."`;
the headline metrics come from the run's `metrics.json` with 95 % CIs, so every
number here traces back to a run ID (hard rule 7). Runs live in
`$HANDOFF_SCRATCH/runs/<run_id>/`.

| Date | Run ID | Purpose | Headline metrics (95% CI) | Decision |
| --- | --- | --- | --- | --- |
| 2026-10-04 | B0-dev-s0-20261004 | B0 baseline audit (M1) | success 0.50 [0.38, 0.60]; pass^4 0.32 [0.20, 0.43]; violation_rate 0.26 [0.18, 0.34]; handoff_f1 0.69 [0.55, 0.80] | baseline for RQ1-RQ3 |
| 2026-10-04 | B0-test_main-s0-20261004 | B0 baseline audit (M1) | success 0.31 [0.21, 0.41]; pass^4 0.10 [0.02, 0.18]; violation_rate 0.51 [0.43, 0.59]; handoff_f1 0.22 [0.00, 0.51] | baseline for RQ1-RQ3 |
| 2026-10-04 | B0-test_styles-s0-20261004 | B0 baseline audit (M1) | success 0.33 [0.24, 0.43]; violation_rate 0.53 [0.46, 0.60]; handoff_f1 0.28 [0.00, 0.49] | baseline for RQ1-RQ3 |
| 2026-10-04 | B0-test_handoff-s0-20261004 | B0 baseline audit (M1) | success 0.42 [0.36, 0.48]; violation_rate 0.31 [0.26, 0.36]; handoff_f1 0.65 [0.57, 0.73] | baseline for RQ1-RQ3 |
