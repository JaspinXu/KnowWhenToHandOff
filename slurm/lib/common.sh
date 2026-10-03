# shellcheck shell=bash
# Sourced at the top of every job in slurm/jobs/. Loads cluster.env, prepares
# scratch, defines helpers and moves to the repository checkout.
set -euo pipefail
: "${HANDOFF_REPO:?HANDOFF_REPO is not set; submit jobs with slurm/submit.sh}"
# shellcheck disable=SC1090
source "${HANDOFF_CLUSTER_ENV:-${HANDOFF_REPO}/slurm/cluster.env}"
umask 077
mkdir -p "${HANDOFF_SCRATCH}"/{logs,runs,datasets,state/endpoints,state/secrets}
mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
# LiteLLM would otherwise fetch its model price list at import time; compute
# nodes may have no internet access.
export LITELLM_LOCAL_MODEL_COST_MAP=True
export HANDOFF_SCRATCH HF_HOME TAU2_DATA_DIR TAU2_COMMIT
# Inside a job, keep temporary files (and Ray's sockets, via RAY_TMPDIR) on the
# node-local disk; a TMPDIR inherited from ~/.bashrc may point at NFS.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export TMPDIR="/tmp/${USER}-handoff-${SLURM_JOB_ID}"
  mkdir -p "${TMPDIR}"
fi
cd "${HANDOFF_REPO}"

log() { echo "[$(date -Is)] $*"; }

activate_venv() {
  if [[ ! -x "${HANDOFF_VENV}/bin/python" ]]; then
    echo "No environment at ${HANDOFF_VENV}; run: bash slurm/submit.sh setup" >&2
    exit 2
  fi
  # shellcheck disable=SC1091
  source "${HANDOFF_VENV}/bin/activate"
}

free_port() {
  python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])'
}

# Split $EXTRA into the array EXTRA_ARGS like a shell would, so a quoted value
# with spaces stays one argument: EXTRA="tags=[final] test_rerun_reason='a b'".
parse_extra() {
  # shellcheck disable=SC2034  # used by the job scripts that source this file
  EXTRA_ARGS=()
  if [[ -z "${EXTRA:-}" ]]; then return 0; fi
  local split
  if ! split="$(python3 -c 'import shlex, sys; print("\n".join(shlex.split(sys.argv[1])))' "${EXTRA}")"; then
    echo "cannot parse EXTRA (unbalanced quotes?): ${EXTRA}" >&2
    exit 2
  fi
  # shellcheck disable=SC2034
  if [[ -n "${split}" ]]; then mapfile -t EXTRA_ARGS <<< "${split}"; fi
}

# Fail before any GPU time is spent when a config name has a typo.
require_config() {  # require_config <group> <name>
  if [[ "$2" == _* || ! -f "${HANDOFF_REPO}/configs/$1/$2.yaml" ]]; then
    echo "unknown $1 config '$2' (see configs/$1/)" >&2
    exit 2
  fi
}

job_banner() {
  log "job ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-no-slurm}) on $(hostname)"
  log "repo $(git -C "${HANDOFF_REPO}" rev-parse --short HEAD) dirty=$(git -C "${HANDOFF_REPO}" status --porcelain | wc -l)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  fi
}
