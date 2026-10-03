#!/usr/bin/env bash
# Create the cluster environment: uv, Python 3.12 venv, pinned GPU stack
# (torch 2.9.0 + vLLM 0.12.0 + verl 0.8.0, see ADR-001), tau2 at the pinned
# commit (editable, with its data) and this package. Then run sanity checks.
source "${HANDOFF_REPO}/slurm/lib/common.sh"
job_banner

fstype="$(stat -f -c %T "${HANDOFF_SCRATCH}" 2>/dev/null || echo unknown)"
log "scratch ${HANDOFF_SCRATCH} is on a '${fstype}' filesystem"
case "${fstype}" in
  nfs*|lustre|gpfs|beegfs|ceph*|fuse*|wekafs|panfs|smb*|cifs|autofs) ;;
  *)
    echo "HANDOFF_SCRATCH (${HANDOFF_SCRATCH}) is on a '${fstype}' filesystem, which" \
         "is usually local to one node. Every job (simulator, evaluation, training)" \
         "must see the same environment, models and endpoint files: choose a shared" \
         "path in slurm/cluster.env (check with: df -T <path> on two nodes), or set" \
         "HANDOFF_ALLOW_LOCAL_SCRATCH=1 if it is shared after all." >&2
    if [[ -z "${HANDOFF_ALLOW_LOCAL_SCRATCH:-}" ]]; then exit 3; fi ;;
esac

# vLLM 0.12.0 wheels need glibc >= 2.31.
glibc="$(ldd --version 2>/dev/null | head -n1 | grep -oE '[0-9]+\.[0-9]+$' || echo 0.0)"
if ! printf '%s\n' 2.31 "${glibc}" | sort -V -C; then
  echo "glibc ${glibc} is older than 2.31, which the pinned vLLM wheels need." >&2
  exit 3
fi

# Setup runs on a CPU node; check the driver only if a GPU happens to be visible.
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)"
major="${driver%%.*}"
if [[ "${major}" =~ ^[0-9]+$ ]]; then
  if (( major < 525 )); then
    echo "NVIDIA driver '${driver}' is too old for the CUDA 12.8 wheels (need >= 525)." >&2
    exit 3
  fi
  log "driver ${driver} is compatible with the cu128 stack"
else
  log "no GPU visible: installing only (every GPU job prints its GPU and driver)"
fi

if ! command -v uv >/dev/null 2>&1; then
  log "installing uv into ~/.local"
  # Ubuntu 24.04 marks the system Python as externally managed (PEP 668);
  # --user only writes to ~/.local, so the system packages stay untouched.
  python3 -m pip install --user --break-system-packages --quiet "uv==0.12.16"
  export PATH="${HOME}/.local/bin:${PATH}"
fi
uv --version

if [[ ! -x "${HANDOFF_VENV}/bin/python" ]]; then
  uv venv --python "${HANDOFF_PYTHON}" "${HANDOFF_VENV}"
fi
export VIRTUAL_ENV="${HANDOFF_VENV}"

if [[ ! -d "${TAU2_REPO_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${TAU2_REPO_DIR}")"
  git clone https://github.com/sierra-research/tau2-bench.git "${TAU2_REPO_DIR}"
fi
git -C "${TAU2_REPO_DIR}" fetch --quiet origin
git -C "${TAU2_REPO_DIR}" checkout --quiet "${TAU2_COMMIT}"
log "tau2 at $(git -C "${TAU2_REPO_DIR}" rev-parse HEAD)"

# sync installs exactly the lock (and removes anything else, e.g. from an
# earlier setup); tau2, this package and the test runner follow without deps.
uv pip sync --python "${HANDOFF_VENV}/bin/python" requirements/cluster-gpu.txt
uv pip install --python "${HANDOFF_VENV}/bin/python" --no-deps -e "${TAU2_REPO_DIR}"
uv pip install --python "${HANDOFF_VENV}/bin/python" --no-deps -e "${HANDOFF_REPO}"
uv pip install --python "${HANDOFF_VENV}/bin/python" --no-deps \
  "pytest==9.1.1" "pytest-socket==0.8.1" "iniconfig==2.3.0" "pluggy==1.6.0"

activate_venv
python - <<'PY'
import importlib.metadata as md
import torch
for name in ("torch", "vllm", "verl", "transformers", "tau2", "litellm", "handoff"):
    print(f"{name:14s} {md.version(name)}")
print("cuda available:", torch.cuda.is_available(), torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
python -c "from tau2.registry import registry; print('tau2 domains', registry.get_domains())"
# CPU tests, including the tau2, vLLM, verl and Hugging Face integration tests
# that are skipped on a laptop.
python -m pytest -q -rs -p no:cacheprovider tests
uv pip freeze --python "${HANDOFF_VENV}/bin/python" > "${HANDOFF_SCRATCH}/state/env.freeze.txt"
log "environment ready: ${HANDOFF_VENV} (freeze in state/env.freeze.txt)"
