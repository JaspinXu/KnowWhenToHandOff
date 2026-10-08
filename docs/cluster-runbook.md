# Cluster runbook

How to run the whole study on a Slurm cluster (written for the NUS SoC compute
cluster, but only `slurm/cluster.env` is cluster-specific). Every step is one
`bash slurm/submit.sh <job> KEY=VALUE ...` call from the login node; nothing
heavy runs on the login node and nothing is downloaded on your laptop. Light
commands that only read run records (monitoring, reports, the experiment log)
run on the login node inside the environment with `bash slurm/run.sh <command>`.

Run IDs look like `B0-test_main-s0-20261008` (`<experiment>-<tags>-s<seed>-<date>`;
a second run of the same experiment on the same day gets a `try2` tag). Every
run writes `runs/<run_id>/` on scratch with `config.json`, `provenance.json`
(git SHA, tau2 commit, model revisions, seeds, hardware), `attempts.jsonl` (job
IDs and server addresses of each attempt) and, once it ends, `completion.json`.
Job logs go to `logs/<job name>-<job id>.out` in this checkout.

> [!IMPORTANT]
> **Real runs need a clean checkout.** A run whose code, configs or data differ
> from the committed version is refused (`git_dirty must be false`). Changes
> under `docs/` and `reports/` are allowed. Commit (and push) before submitting.

## 0. Resources

On the NUS SoC cluster the `normal` QoS allows each user one H100 96 GB and two A100 80 GB at a time; `gpu-long` (3 days) has 2 nodes with an A100 80 GB and 5 with H100 96 GB, and the only H200 node is in the 3-hour `gpu` partition. The plan keeps each GPU on one job:

| GPU | Holds | Notes |
| --- | --- | --- |
| A100 80 GB (`GPU_TYPE_SIM=a100-80`) | User simulator Qwen3-30B-A3B-Instruct-2507 (bf16, ~61 GB) during a campaign | Only two such nodes in `gpu-long`: submit `serve` early and `scancel` it between campaigns (SoC cancels jobs idle for > 30 min) |
| H100 96 GB (`GPU_TYPE_BIG=h100-96`) | One of: the policy server of an evaluation, SFT, or RL (policy + vLLM rollout) | Jobs queue one after another; one per user, so `NGPUS=2` is not possible (RL falls back to `rl.use_lora=true` if memory is short) |

`HANDOFF_SCRATCH` must be on storage that **every compute node sees** (the
simulator job publishes its address there; models, runs and the environment
live there too). The setup job refuses a filesystem that looks node-local; check
a candidate with `df -T <path>` on two nodes. Budget: environment ~15 GB, models
~70 GB (Qwen3-4B, Qwen3-30B-A3B), each RL run ~100 GB (its latest resumable
checkpoint, ~50 GB, plus an 8 GB export per saved step), evaluation runs well
under 1 GB each.

The whole plan is five runs: B0, B1 (optional), B2, R1 and R2, one seed each.
There are no sweeps or ablations; §9 lists the job order.

## 1. One-time setup

```bash
git clone https://github.com/JaspinXu/KnowWhenToHandOff.git && cd KnowWhenToHandOff
cp slurm/cluster.env.example slurm/cluster.env   # then edit it
sinfo -o "%P %l %G"                                # partitions, time limits, GPU types
bash slurm/submit.sh setup                         # venv, tau2 @ v1.0.1, pinned stack, tests
tail -f logs/handoff-setup-*.out
```

Edit in `slurm/cluster.env`: `HANDOFF_SCRATCH`, `SLURM_ACCOUNT`, the partitions
(`sinfo`), the GPU type names (`GPU_TYPE_SIM`, `GPU_TYPE_BIG`; `--gres=gpu:<type>:N`
is used) and the time limits allowed by each partition. Source it in your shell
(`source slurm/cluster.env`) so `$HANDOFF_SCRATCH` works in the commands below.

The setup job needs internet access (PyPI, GitHub). It checks that scratch is
shared, glibc (>= 2.31, for the vLLM wheels) and, if a GPU is visible, the
NVIDIA driver (>= 525 for the CUDA 12.8 wheels; setup itself runs on a CPU
node), installs exactly `requirements/cluster-gpu.txt`, installs
tau2 (commit `fc0055d`, v1.0.1) and this package, then runs the whole CPU test
suite, including the tests that need tau2, vLLM and verl. **All tests must
pass** (`N passed`); a failure here means the pinned stack does not match what
the code expects. Rerunning `setup` resets the environment to the lock.

## 2. Models and frozen task sets (T0.2, T1.1–T1.3)

```bash
bash slurm/submit.sh download MODELS="qwen3_4b qwen3_30b_a3b"
bash slurm/submit.sh tasks
```

`download` resolves each model's current Hugging Face commit, stores the
snapshot under `$HF_HOME` and records repository, commit SHA and path in
`$HANDOFF_SCRATCH/state/models.lock.json`. Later jobs refuse models missing from
the lock.

`tasks` builds the should-transfer labels, the frozen train/dev/test split
(dev = 20 % of official train, stratified by label, seed 20261011) and the
hand-off task set, checks that every task works with every communication style,
records the hashes of tau2's task, split, policy and database files, and writes
a review packet. Then, on the login node:

```bash
less reports/review/handoff_tasks.md      # hand-check >= 30 tasks (T1.3 DoD)
git add data/manifests data/handoff_tasks
git commit -m "T1.1-T1.3: freeze splits, labels and hand-off tasks"
git push
```

The files are frozen from now on; `build_task_sets.py` refuses to overwrite them
(regenerating needs an ADR), and every evaluation and RL run checks that tau2's
files still have the recorded hashes.

## 3. Tokenizer check (T3.1 prerequisite)

```bash
bash slurm/submit.sh check-tokens MODEL=qwen3_4b
```

It must print `PASS`. It checks that token-by-token accumulation during RL equals
a full encoding for both domains (with the model config's chat-template
arguments), and that the initial RL prompt (policy + tool schemas + greeting)
leaves at least 1,024 tokens for the first user message within
`episode.max_prompt_tokens` (8,192; pass `MAX_PROMPT_TOKENS=` if you change it).
`TRAJECTORIES=<trajectories.jsonl>` re-checks recorded conversations too.

## 4. User simulator server

```bash
bash slurm/submit.sh serve                    # ROLE=usersim MODEL=qwen3_30b_a3b on GPU_TYPE_SIM
bash slurm/run.sh python scripts/model_server.py show --endpoint-file "$HANDOFF_SCRATCH/state/endpoints/usersim.json"
```

The server writes `state/endpoints/usersim.json` once it answers. Its API key is
in `state/secrets/`, readable only by you, and every route except `/health`
requires it (vLLM's own check covers `/v1` only; `handoff.sim.vllm_auth` closes
the rest), since the server listens on the cluster network. Jobs that need it
wait for it; endpoint files of ended jobs are ignored.

The serve job has a time limit (`TIME_SERVE`, 3 days). An evaluation refuses to
start when the simulator's job would end before its own (restart `serve` first,
or pass `ALLOW_SHORT_SIM=1`). RL jobs only warn: when the simulator goes away
mid-run, every affected episode waits for a restarted simulator (its new
endpoint file is picked up automatically) for up to 30 minutes
(`user_sim.outage_timeout_s`); if none comes back, the RL job stops and you
resume it (§7). `scancel` the simulator when a campaign is over.

## 5. Smoke test and baselines (T0.2, T0.5, T1.6 → M1)

```bash
# five train tasks, one trial: the whole evaluation path
bash slurm/submit.sh eval EVAL=smoke EXPERIMENT=SMOKE POLICY_MODEL=qwen3_4b
```

Check `runs/SMOKE-smoke-s0-<date>/summary.md`: no infrastructure failures, few
format errors. Its `duration_s` values give the episodes-per-hour estimate (T0.3).

B0 (Qwen3-4B prompted), development split first, then the frozen test protocol:

```bash
bash slurm/submit.sh eval EVAL=dev          EXPERIMENT=B0 POLICY_MODEL=qwen3_4b
bash slurm/submit.sh eval EVAL=test_main    EXPERIMENT=B0 POLICY_MODEL=qwen3_4b
bash slurm/submit.sh eval EVAL=test_styles  EXPERIMENT=B0 POLICY_MODEL=qwen3_4b
bash slurm/submit.sh eval EVAL=test_handoff EXPERIMENT=B0 POLICY_MODEL=qwen3_4b
```

B1 (optional; the 30B model as the policy) reuses the simulator server, so it
runs on a CPU node. Its `test_main` run is enough:

```bash
usersim="$HANDOFF_SCRATCH/state/endpoints/usersim.json"
bash slurm/submit.sh eval EVAL=test_main EXPERIMENT=B1 POLICY_ENDPOINT="$usersim" POLICY_LABEL=qwen3_30b_a3b
```

Read ten B0 `test_styles` dialogues per style
(`bash slurm/submit.sh cpu CMD="python scripts/sample_review.py styles --run B0-test_styles-s0-<date>"`)
to check the style cards (T1.4); this only reads dialogues, it selects nothing.

**Complete runs only.** An evaluation that loses episodes to infrastructure
failures (after 3 retries each) ends with exit status 3 and `INCOMPLETE` in its
log, and is not reported. Fix the cause (usually the simulator), then resume it
in the same directory; only the missing episodes run again:

```bash
bash slurm/submit.sh eval RESUME=B0-test_main-s0-<date> POLICY_MODEL=qwen3_4b   # same POLICY_* as before
```

A resumed run must see the same commit, tau2 data, manifests and model
revisions (servers may be new). If a task fails deterministically, accept a
small loss explicitly with `EXTRA="accept_infra_failure_share=0.01"` (recorded
in `attempts.jsonl`).

The test split is evaluated **once per frozen checkpoint**: the test ledger
(`state/test_ledger.jsonl`) refuses any new test run of the same policy revision
and simulator that shares a test task and a style with an earlier one, whatever
its preset is called (resuming the same run is fine). `test_main`,
`test_styles` and `test_handoff` cover disjoint tasks or styles, so each runs
once per checkpoint. Select checkpoints on `dev` only.

After each run, record it (the headline comes from `metrics.json` with CIs):

```bash
bash slurm/run.sh python scripts/append_experiment_log.py B0-test_main-s0-<date> --purpose "B0 audit" --decision "..."
```

**M1 (11 Oct):** accept [ADR-005](adr/ADR-005-evaluation-protocol.md) (frozen
protocol) and [ADR-002](adr/ADR-002-base-policy.md), and commit the log.

## 6. Imitation warm start B2 (T2.1–T2.3 → M2)

Teacher rollouts: the 30B model on the **train** split and the hand-off train
set, all six styles, two sampled trials (config `eval=teacher_train`).

```bash
bash slurm/submit.sh teacher POLICY_ENDPOINT="$usersim" POLICY_LABEL=qwen3_30b_a3b
bash slurm/submit.sh sft-data TEACHER_RUNS=TEACH-teacher_train-s0-<date> DATASET=b2_v1
bash slurm/submit.sh sft DATASET=b2_v1                 # 1x H100 96 GB
```

`sft-data` keeps only successful, violation-free, format-clean episodes with the
right transfer decision, refuses anything outside the train split, and writes
`datasets/b2_v1/manifest.json` with statistics (T2.1 DoD). `sft` trains with loss
on assistant tokens only and writes `runs/B2-s0-<date>/final` (a Hugging Face
checkpoint; LoRA adapters are merged, and LoRA needs a single GPU).

Evaluate B2 like any checkpoint:

```bash
b2="$HANDOFF_SCRATCH/runs/B2-s0-<date>/final"
bash slurm/submit.sh check-tokens TOKENIZER="$b2"
bash slurm/submit.sh eval EVAL=dev EXPERIMENT=B2 POLICY_PATH="$b2" POLICY_LABEL=b2
bash slurm/submit.sh cpu CMD="python scripts/sample_review.py compare --run B2-dev-s0-<date> --baseline B0-dev-s0-<date>"
bash slurm/submit.sh cpu CMD="python scripts/sample_review.py failures --run B2-dev-s0-<date> --n 10"
```

Then the test protocol for B2 (`test_main`, `test_styles`, `test_handoff`) as in §5.

## 7. Online RL (T3.3–T3.4 → M3)

First the T0.4 spike ([ADR-001](adr/ADR-001-trainer-and-gpu-stack.md)): three
GRPO steps from B2, to measure step time and memory before the real runs
(record them in ADR-001):

```bash
bash slurm/submit.sh rl REWARD=task_only EXPERIMENT=SPIKE INIT="$b2" \
  EXTRA="rl.verl.trainer.total_training_steps=3 rl.verl.trainer.val_before_train=false"
```

Then the two real runs. Each trains from B2 with GRPO (group size 8, 60 steps,
dev validation every 10 steps). The user simulator must be running.

```bash
bash slurm/submit.sh rl REWARD=task_only EXPERIMENT=R1 INIT="$b2"
bash slurm/submit.sh rl REWARD=r2        EXPERIMENT=R2 INIT="$b2"
```

Before a long run, try `EXTRA="dry_run=true"` (builds the data and validates
the verl command without training; it gets its own `dry` run ID).

While it runs (per-episode records, with timestamps, are in
`runs/<id>/rollouts/`; verl's per-step dumps with every reward term are in
`runs/<id>/rollout_dumps/` and `validation_dumps/`):

```bash
bash slurm/run.sh python scripts/monitor_rl.py --run R2-s1-<date> --baseline B2-dev-s0-<date> --self-baseline 128
bash slurm/run.sh python scripts/rl_curves.py --name rl --runs R1=R1-s1-<date> R2=R2-s1-<date>
```

`rl_curves.py` writes the learning curves (mean reward terms per training step
and per dev validation) to `reports/rl/` as tables and a figure; commit the
final version with the report (T3.2–T3.4).

Exit status 3 means a reward-hacking indicator is more than 2× both the run's
own first two steps (`--self-baseline 128`, sampled from B2 at the training
temperature) and the B2 evaluation (without the option, B2 alone, which
under-counts transfers because it is greedy: R1 step 1 had 0.22 on negatives
vs 0.04 in B2 dev). Then stop the run with `scancel` (AGENTS §7); 4 means more than 5 % of recent
episodes were lost to infrastructure (check the simulator job).

A run stopped by its time limit, a simulator outage or a crash resumes from its
last checkpoint with the same code and settings (check out the run's commit if
you moved on): `bash slurm/submit.sh rl RESUME=R2-s1-<date>` (no `EXTRA`).

Every saved step (every 10 steps) is exported right away to a bf16 Hugging Face
directory, `runs/<id>/export/global_step_<N>`, and verl keeps only the latest
resumable checkpoint. By default the final step is the checkpoint; the dev
validation curve (`reports/rl/dev_curves.md`) shows whether an earlier step is
clearly better.
Evaluate it on dev, then run the final test protocol (T3.6):

```bash
r2="$HANDOFF_SCRATCH/runs/R2-s1-<date>/export/global_step_60"
bash slurm/submit.sh eval EVAL=dev EXPERIMENT=R2 POLICY_PATH="$r2" POLICY_LABEL=r2-final EXTRA="tags=[final]"
for e in test_main test_styles test_handoff; do
  bash slurm/submit.sh eval EVAL=$e EXPERIMENT=R2 POLICY_PATH="$r2" POLICY_LABEL=r2-final EXTRA="tags=[final]"
done
```

R1 the same way (`EXPERIMENT=R1`, `POLICY_LABEL=r1-final`).

`scripts/export_checkpoint.py <run id> --step latest|all|<N>` prints an export
and creates any that is missing (for example after a job was killed mid-save).

## 8. Reports, audits and demo (T3.5, T4.1, T4.3)

A report label is one checkpoint; its test runs are joined with `+`, so the
style table covers all six styles (neutral from `test_main`) and hand-off
metrics include the hand-off test set:

```bash
t="test_main-s0"; s="test_styles-s0"; h="test_handoff-s0"   # plus each run's date
bash slurm/run.sh python scripts/make_report.py --name main --baseline B2 --runs \
  B0=B0-$t-<date>+B0-$s-<date>+B0-$h-<date> \
  B1=B1-$t-<date> \
  B2=B2-$t-<date>+B2-$s-<date>+B2-$h-<date> \
  R1=R1-final-$t-<date>+R1-final-$s-<date>+R1-final-$h-<date> \
  R2=R2-final-$t-<date>+R2-final-$s-<date>+R2-final-$h-<date>
bash slurm/submit.sh cpu CMD="python scripts/sample_review.py audit --run <final test run> --n 20"
bash slurm/run.sh python scripts/render_demo.py --run <test_handoff run> \
  --episodes <episode ids> --out reports/demo/<name>.html --jsonl reports/demo/<name>.jsonl
git add reports/main reports/rl reports/demo docs/experiment-log.md && git commit -m "T4.1: report tables"
```

`reports/<name>/` gets Markdown, CSV and LaTeX tables, PNG figures and
`sources.json` (run IDs, git SHAs, execution kind and seeds behind every
number); task metrics use upstream tasks in the neutral style, hand-off metrics
every neutral episode ([metrics](metrics.md)). Only completed real runs are
accepted. Review packets under `reports/review/` hold full dialogues and are not
committed. `scripts/demo_chat.py` is an interactive terminal demo against a
served policy (run it inside an interactive GPU allocation with the policy
server started).

## 9. The whole plan at a glance

| Order | Job (GPU) | Runs |
| --- | --- | --- |
| 1 | `setup`, `download`, `tasks` (CPU) | — |
| 2 | `serve` (A100, stays up) | user simulator |
| 3 | `check-tokens`, `eval smoke` (H100) | SMOKE |
| 4 | `eval` dev + 3 test presets (H100) | B0 |
| 5 | `eval test_main` via the simulator server (CPU) | B1 (optional) |
| 6 | `teacher` (CPU), `sft-data` (CPU), `sft` (H100) | TEACH, B2 |
| 7 | `eval` dev + 3 test presets (H100) | B2 |
| 8 | `rl` spike, then R1 and R2 (H100, one after the other) | SPIKE, R1, R2 |
| 9 | `eval` dev + 3 test presets for each (H100) | R1, R2 |
| 10 | `make_report`, review packets, demo (login node / CPU) | — |

If time runs short, drop B1 first, then lower `rl.verl.trainer.total_training_steps`.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `git_dirty must be false for a reproducible real run` | Commit or stash changes outside `docs/` and `reports/` |
| Setup: `... is on a '...' filesystem, which is usually local to one node` | Choose a shared `HANDOFF_SCRATCH` (`df -T` on two nodes); set `HANDOFF_ALLOW_LOCAL_SCRATCH=1` only if it is shared after all |
| `installed tau2 ... differs from the manifest's ...` or `tau2 at ... is at commit ...` | The environment was built from another tau2 commit; rerun `setup` |
| `tau2 ... data differ from the split manifest` | `TAU2_DATA_DIR` or the tau2 checkout changed since `tasks`; rerun `setup` (never edit tau2's files) |
| `... is not in the models lock` | Run the `download` job for that model config |
| Evaluation waits for the simulator forever | `squeue -u $USER`; restart `serve`; check its log for CUDA OOM |
| `the server job ... ends in ... h, before this job` | Restart `serve` so it outlives the evaluation, or pass `ALLOW_SHORT_SIM=1` |
| Evaluation exits with status 3 (`INCOMPLETE`) | Episodes were lost (`infra_failures.jsonl`): fix the simulator, then `RESUME=<run id>` |
| RL: `user simulator ... unreachable` and the job stopped | Restart `serve`, then `bash slurm/submit.sh rl RESUME=<run id>` |
| RL: `initial prompt has N tokens` | Raise `episode.max_prompt_tokens` (and rerun `check-tokens` with `MAX_PROMPT_TOKENS=`) |
| RL: CUDA OOM in training | `EXTRA="rl.verl.actor_rollout_ref.rollout.gpu_memory_utilization=0.35 rl.verl.actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384"` or `rl.use_lora=true` |
| `test split requested without allow_test` | Use a `test_*` eval preset; test runs are deliberate |
| `this checkpoint was already evaluated on these test tasks and styles` | The test ledger protects the test split. Resume an unfinished run instead; if a rerun is justified, pass `EXTRA="force_test_rerun=true test_rerun_reason='why it is needed'"` (no commas) and document it in an ADR |
| `cannot parse EXTRA` | Unbalanced quotes; values with spaces go in single quotes inside the double-quoted `EXTRA` |
