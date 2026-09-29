# Experiment log

One row per run that produced numbers (AGENTS §7). Append rows with
`python scripts/append_experiment_log.py <run_id> --purpose "..." --decision "..."`;
the headline metrics come from the run's `metrics.json` with 95 % CIs, so every
number here traces back to a run ID (hard rule 7). Runs live in
`$HANDOFF_SCRATCH/runs/<run_id>/`.

| Date | Run ID | Purpose | Headline metrics (95% CI) | Decision |
| --- | --- | --- | --- | --- |
