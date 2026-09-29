#!/usr/bin/env bash
# Evaluate a policy. Starts the policy server on this job's GPU unless
# POLICY_ENDPOINT points to a running server (e.g. B1 on the 30B server).
#   EVAL=smoke|dev|test_main|test_styles|test_handoff|teacher_train
#   EXPERIMENT=SMOKE|B0|B1|B2|R1|R2|TEACH
#   POLICY_MODEL=qwen3_4b   [POLICY_PATH=<checkpoint dir> POLICY_LABEL=b2]
#   USERSIM_ENDPOINT=<endpoint json> (default: state/endpoints/usersim.json)
#   RESUME=<run id>  continue an incomplete run (same POLICY_MODEL / POLICY_PATH)
#   ALLOW_SHORT_SIM=1  start although the simulator's job ends before this one
#   EXTRA="tags=[final] seed=1"  (more run_eval.py overrides; quotes are kept)
# Exit status 3: the run lost episodes to infrastructure failures; resume it.
source "${HANDOFF_REPO}/slurm/lib/common.sh"
job_banner
activate_venv
parse_extra
if [[ -z "${RESUME:-}" ]]; then require_config eval "${EVAL:-smoke}"; fi
if [[ -n "${POLICY_PATH:-}" && ! -d "${POLICY_PATH}" ]]; then
  echo "POLICY_PATH ${POLICY_PATH} is not a directory" >&2
  exit 2
fi
endpoints="${HANDOFF_SCRATCH}/state/endpoints"
usersim="${USERSIM_ENDPOINT:-${endpoints}/usersim.json}"
label="${POLICY_LABEL:-${POLICY_MODEL:-qwen3_4b}}"
outlive=()
if [[ -z "${ALLOW_SHORT_SIM:-}" && -n "${SLURM_JOB_ID:-}" ]]; then
  outlive=(--outlive-job "${SLURM_JOB_ID}")
fi
python scripts/model_server.py wait --endpoint-file "${usersim}" \
  --timeout "${WAIT_USERSIM:-900}" "${outlive[@]}"
if [[ -n "${POLICY_ENDPOINT:-}" ]]; then
  policy_endpoint="${POLICY_ENDPOINT}"
else
  policy_endpoint="${endpoints}/policy-${SLURM_JOB_ID:-local}.json"
  args=(--model "${POLICY_MODEL:-qwen3_4b}" --role policy --port "$(free_port)"
        --endpoint-file "${policy_endpoint}" --gpu-memory-utilization 0.9)
  if [[ -n "${POLICY_PATH:-}" ]]; then
    args+=(--local-path "${POLICY_PATH}" --served-name "${label}")
  fi
  python scripts/model_server.py serve "${args[@]}" &
  server_pid=$!
  trap 'kill "${server_pid}" 2>/dev/null || true; wait "${server_pid}" 2>/dev/null || true' EXIT
fi
python scripts/model_server.py wait --endpoint-file "${policy_endpoint}" --timeout 3600 \
  ${server_pid:+--pid "${server_pid}"}
if [[ -n "${RESUME:-}" ]]; then
  run=("resume_run_id=${RESUME}")
else
  run=("eval=${EVAL:-smoke}" "experiment=${EXPERIMENT:-SMOKE}" "policy.label=${label}")
fi
python scripts/run_eval.py "${run[@]}" "policy.endpoint_file=${policy_endpoint}" \
  "user_sim.endpoint_file=${usersim}" "${EXTRA_ARGS[@]}"
