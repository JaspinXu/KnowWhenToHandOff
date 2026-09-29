#!/usr/bin/env bash
# Run a light project command on the login node, inside the cluster environment:
#
#   bash slurm/run.sh python scripts/monitor_rl.py --run <RL run id> --baseline <id>
#   bash slurm/run.sh python scripts/make_report.py --name main --runs ...
#
# Only for commands that read run records and write small files (monitoring,
# reports, the experiment log). Anything heavier goes through slurm/submit.sh
# (the `cpu` job runs any command on a compute node).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HANDOFF_REPO="$(cd "${here}/.." && pwd)"
export HANDOFF_REPO
if [[ $# -eq 0 ]]; then
  sed -n '2,9p' "$0"
  exit 2
fi
# shellcheck source=slurm/lib/common.sh
source "${here}/lib/common.sh"
activate_venv
exec "$@"
