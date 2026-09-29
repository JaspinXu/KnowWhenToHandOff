#!/usr/bin/env bash
# T3.3-T3.4: GRPO with tau2 episodes (verl). Needs a running user simulator.
#   REWARD=task_only (R1) | r2 (R2)   EXPERIMENT=R1|R2   INIT=<B2 checkpoint dir>
#   EXTRA="dry_run=true ..."  (extra Hydra overrides; quotes are kept)
#   RESUME=<run_id> resumes a run from its latest checkpoint (no EXTRA: the
#   run's saved command is reused).
source "${HANDOFF_REPO}/slurm/lib/common.sh"
job_banner
activate_venv
parse_extra
if [[ -n "${RESUME:-}" ]]; then
  if [[ -n "${EXTRA:-}" ]]; then
    echo "RESUME reuses the run's saved settings; drop EXTRA" >&2
    exit 2
  fi
else
  : "${INIT:?set INIT=<policy checkpoint dir>}"
  if [[ ! -d "${INIT}" ]]; then echo "INIT ${INIT} is not a directory" >&2; exit 2; fi
  require_config rl "${RL:-grpo}"
  require_config reward "${REWARD:-task_only}"
fi
usersim="${USERSIM_ENDPOINT:-${HANDOFF_SCRATCH}/state/endpoints/usersim.json}"
# Training waits out a simulator restart (user_sim.outage_timeout_s), so a
# simulator that ends first is only a warning here.
python scripts/model_server.py wait --endpoint-file "${usersim}" \
  --timeout "${WAIT_USERSIM:-900}" --outlive-job "${SLURM_JOB_ID:-}" --warn-only
export RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR:-/tmp}/ray-${USER}}"
if [[ -n "${RESUME:-}" ]]; then
  exec python scripts/train_rl.py "resume_run_id=${RESUME}"
fi
exec python scripts/train_rl.py "rl=${RL:-grpo}" "reward=${REWARD:-task_only}" \
  "experiment=${EXPERIMENT:-R1}" "policy.init_path=${INIT}" \
  "user_sim.endpoint_file=${usersim}" \
  "rl.verl.trainer.n_gpus_per_node=${NGPUS:-1}" "${EXTRA_ARGS[@]}"
