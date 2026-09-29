#!/usr/bin/env bash
# Download model snapshots to $HF_HOME and record revisions in the models lock.
# MODELS="qwen3_4b qwen3_30b_a3b"  (model config names)
source "${HANDOFF_REPO}/slurm/lib/common.sh"
job_banner
activate_venv
# shellcheck disable=SC2086
python scripts/download_models.py ${MODELS:-qwen3_4b qwen3_30b_a3b}
