#!/usr/bin/env bash
# Any light project command on a CPU node, inside the environment, e.g.
#   CMD="python scripts/make_report.py --name main --runs B0=... B2=..."
source "${HANDOFF_REPO}/slurm/lib/common.sh"
job_banner
activate_venv
: "${CMD:?set CMD=<command to run in the repository>}"
log "running: ${CMD}"
bash -c "${CMD}"
