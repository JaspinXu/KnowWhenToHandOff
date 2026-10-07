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
| 2026-10-04 | B1-test_main-s0-20261004 | B1 reference ceiling (30B prompted) | success 0.40 [0.29, 0.51]; pass^4 0.20 [0.10, 0.33]; violation_rate 0.53 [0.44, 0.62]; handoff_f1 0.40 [0.00, 0.70] | ceiling for RQ1; same model as the simulator |
| 2026-10-05 | B2-dev-s0-20261005 | B2 SFT evaluation (M2) | success 0.50 [0.40, 0.60]; pass^4 0.25 [0.15, 0.35]; violation_rate 0.23 [0.16, 0.32]; handoff_f1 0.74 [0.61, 0.85] | gate passed (dev 0.504 vs B0 0.496); init for R1/R2 |
| 2026-10-05 | B2-test_main-s0-20261005 | B2 SFT evaluation (M2) | success 0.33 [0.22, 0.44]; pass^4 0.10 [0.02, 0.20]; violation_rate 0.47 [0.37, 0.58]; handoff_f1 0.11 [0.00, 0.30] | gate passed (dev 0.504 vs B0 0.496); init for R1/R2 |
| 2026-10-05 | B2-test_styles-s0-20261005 | B2 SFT evaluation (M2) | success 0.36 [0.27, 0.47]; violation_rate 0.46 [0.36, 0.55]; handoff_f1 0.18 [0.00, 0.43] | gate passed (dev 0.504 vs B0 0.496); init for R1/R2 |
| 2026-10-05 | B2-test_handoff-s0-20261005 | B2 SFT evaluation (M2) | success 0.47 [0.40, 0.53]; violation_rate 0.34 [0.28, 0.41]; handoff_f1 0.77 [0.70, 0.83] | gate passed (dev 0.504 vs B0 0.496); init for R1/R2 |
| 2026-10-07 | R1-final-dev-s0-20261006 | R1 GRPO task-only, step 60 (M3) | success 0.68 [0.56, 0.77]; pass^4 0.52 [0.38, 0.63]; violation_rate 0.16 [0.09, 0.23]; handoff_f1 0.82 [0.72, 0.90] | learns to over-escalate (RQ4); R2 tests the FP/FN penalties |
| 2026-10-07 | R1-final-test_main-s0-20261006 | R1 GRPO task-only, step 60 (M3) | success 0.35 [0.23, 0.47]; pass^4 0.22 [0.10, 0.35]; violation_rate 0.33 [0.23, 0.42]; handoff_f1 0.18 [0.00, 0.37] | learns to over-escalate (RQ4); R2 tests the FP/FN penalties |
| 2026-10-07 | R1-final-test_styles-s0-20261006 | R1 GRPO task-only, step 60 (M3) | success 0.34 [0.24, 0.45]; violation_rate 0.34 [0.25, 0.42]; handoff_f1 0.22 [0.00, 0.41] | learns to over-escalate (RQ4); R2 tests the FP/FN penalties |
| 2026-10-07 | R1-final-test_handoff-s0-20261006 | R1 GRPO task-only, step 60 (M3) | success 0.63 [0.57, 0.70]; violation_rate 0.21 [0.16, 0.26]; handoff_f1 0.80 [0.74, 0.85] | learns to over-escalate (RQ4); R2 tests the FP/FN penalties |
