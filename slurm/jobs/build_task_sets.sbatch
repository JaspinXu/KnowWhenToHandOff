#!/usr/bin/env bash
# T1.1-T1.3: labels, frozen splits and the hand-off task set (CPU only).
# Afterwards commit data/manifests and data/handoff_tasks from the checkout.
source "${HANDOFF_REPO}/slurm/lib/common.sh"
job_banner
activate_venv
parse_extra
python scripts/build_task_sets.py "${EXTRA_ARGS[@]}"
python scripts/sample_review.py tasks --n "${REVIEW_N:-40}"
git -C "${HANDOFF_REPO}" status --short data reports
