#!/usr/bin/env bash
# T2.1: filter teacher rollouts and tokenize them for SFT (CPU only).
#   TEACHER_RUNS="TEACH-teacher_train-s0-20261012[,TEACH-...]"  DATASET=b2_v1
#   EXTRA="filter.max_per_task_persona=3" (more build_sft_data.py overrides)
source "${HANDOFF_REPO}/slurm/lib/common.sh"
job_banner
activate_venv
parse_extra
: "${TEACHER_RUNS:?set TEACHER_RUNS=<comma-separated TEACH run IDs>}"
: "${DATASET:?set DATASET=<new dataset name>}"
python scripts/build_sft_data.py "teacher_runs=[${TEACHER_RUNS}]" \
  "dataset_name=${DATASET}" "${EXTRA_ARGS[@]}"
