#!/usr/bin/env bash
# Long-running model server (user simulator by default).
#   ROLE=usersim MODEL=qwen3_30b_a3b                      (default)
#   ROLE=teacher MODEL=qwen3_30b_a3b ENDPOINT_NAME=teacher
# Clients find it through $HANDOFF_SCRATCH/state/endpoints/<ENDPOINT_NAME>.json
source "${HANDOFF_REPO}/slurm/lib/common.sh"
job_banner
activate_venv
role="${ROLE:-usersim}"
name="${ENDPOINT_NAME:-${role}}"
port="${PORT:-$(free_port)}"
args=(--model "${MODEL:-qwen3_30b_a3b}" --role "${role}" --port "${port}"
      --endpoint-file "${HANDOFF_SCRATCH}/state/endpoints/${name}.json"
      --gpu-memory-utilization "${GPU_MEM_UTIL:-0.9}"
      --tensor-parallel-size "${NGPUS:-1}")
if [[ -n "${LOCAL_PATH:-}" ]]; then
  args+=(--local-path "${LOCAL_PATH}" --served-name "${SERVED_NAME:?SERVED_NAME}")
fi
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then args+=(--max-model-len "${MAX_MODEL_LEN}"); fi
exec python scripts/model_server.py serve "${args[@]}"
