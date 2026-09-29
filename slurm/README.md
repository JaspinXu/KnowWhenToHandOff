# Slurm jobs

All cluster work goes through one wrapper that reads `slurm/cluster.env`
(copy `cluster.env.example`; the file is gitignored):

```bash
bash slurm/submit.sh <job> [KEY=VALUE ...] [-- extra sbatch flags]
```

`KEY=VALUE` pairs are exported to the job; `EXTRA` is split like a shell
command line, so a quoted value stays one Hydra override
(`EXTRA="tags=[final] test_rerun_reason='a b'"`). Config names (`EVAL`, `RL`,
`REWARD`) and `INIT` are checked before anything is queued. Logs go to
`logs/handoff-<job>-<jobid>.out` in this checkout (`HANDOFF_LOGS` overrides it).
The full experiment sequence is in the [cluster runbook](../docs/cluster-runbook.md).

Light commands that only read run records run on the login node inside the
environment: `bash slurm/run.sh python scripts/monitor_rl.py ...` (also
`rl_curves.py`, `make_report.py`, `append_experiment_log.py`,
`model_server.py show`).

| Job | Resources | Runs | Main variables |
| --- | --- | --- | --- |
| `setup` | 1 GPU (any) | `jobs/setup_env.sbatch`: venv, pinned stack, tau2 checkout, CPU test suite | — |
| `download` | CPU | `scripts/download_models.py` → models lock | `MODELS` |
| `tasks` | CPU | `scripts/build_task_sets.py` + hand-off review packet | `EXTRA`, `REVIEW_N` |
| `check-tokens` | CPU | `scripts/check_tokenization.py` | `MODEL`, `TOKENIZER`, `TRAJECTORIES`, `MAX_PROMPT_TOKENS` |
| `serve` | `GPU_TYPE_SIM` × `NGPUS` | `scripts/model_server.py serve` (long-running) | `ROLE`, `MODEL`, `ENDPOINT_NAME`, `LOCAL_PATH`+`SERVED_NAME`, `MAX_MODEL_LEN` |
| `eval` | 1 × `GPU_TYPE_BIG` (CPU if `POLICY_ENDPOINT` is set) | policy server + `scripts/run_eval.py`; exit 3 = incomplete, resume it | `EVAL`, `EXPERIMENT`, `POLICY_MODEL`, `POLICY_PATH`, `POLICY_LABEL`, `POLICY_ENDPOINT`, `USERSIM_ENDPOINT`, `RESUME`, `ALLOW_SHORT_SIM`, `EXTRA` |
| `teacher` | as `eval` | `eval` with `EVAL=teacher_train EXPERIMENT=TEACH` | as `eval` |
| `sft-data` | CPU | `scripts/build_sft_data.py` | `TEACHER_RUNS`, `DATASET`, `EXTRA` |
| `sft` | `GPU_TYPE_BIG` × `NGPUS` | `scripts/train_sft.py` (torchrun + FSDP if `NGPUS>1`) | `DATASET`, `EXTRA` |
| `rl` | `GPU_TYPE_BIG` × `NGPUS` | `scripts/train_rl.py` → verl | `RL`, `REWARD`, `EXPERIMENT`, `INIT`, `RESUME`, `EXTRA` |
| `cpu` | CPU | any command in the environment | `CMD` |

Model servers publish `$HANDOFF_SCRATCH/state/endpoints/<name>.json` (address,
served name, model revision, context length, chat-template arguments, server
flags, Slurm job ID; never the key). The shared API key is in
`state/secrets/vllm_api_key` (mode 600) and reaches vLLM and LiteLLM only
through environment variables (it replaces any `OPENAI_API_KEY` of your shell);
every server route except `/health` requires it.
Clients wait for the endpoint; files left by ended jobs are ignored, and an
evaluation refuses a simulator whose job ends before its own. Cancel idle
servers with `scancel` when a campaign ends.

Checks before every change to these scripts (`bash -n` checks one file per
call):

```bash
for f in slurm/*.sh slurm/lib/common.sh slurm/jobs/*.sbatch; do bash -n "$f" || exit 1; done
shellcheck -x -P slurm slurm/*.sh slurm/lib/common.sh slurm/jobs/*.sbatch
```
