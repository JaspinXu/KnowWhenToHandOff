"""Upload the release to Hugging Face as private repositories (docs/release.md).

Needs ``hf auth login`` once (your own write token) and the dataset folder from
``scripts/export_release.py``. Run it as a CPU job, not on the login node.

Example:
    python scripts/upload_release.py --user JaspinXu
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {  # repository suffix: (folder under runs/, card)
    "R2": ("R2-s1-20261006/export/global_step_60", "docs/model-cards/R2.md"),
    "R1": ("R1-s1-20261006/export/global_step_60", "docs/model-cards/R1.md"),
    "SFT": ("B2-s0-20261005/final", "docs/model-cards/B2.md"),
}
# Local bookkeeping next to the weights; the export record holds a cluster path.
BOOKKEEPING = [".handoff_fingerprint.json", "handoff_export.json"]


def main() -> int:
    """Create the repositories (private) and upload folders and cards."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="Hugging Face user or org")
    parser.add_argument("--only", nargs="*", default=[*MODELS, "data"])
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove bookkeeping files from the model repositories and squash "
        "their history (repairs uploads made before they were excluded)",
    )
    args = parser.parse_args()
    from huggingface_hub import HfApi

    scratch = Path(os.environ["HANDOFF_SCRATCH"])
    api = HfApi()
    if args.clean:
        for name in MODELS:
            if name not in args.only:
                continue
            repo = f"{args.user}/KnowWhenToHandOff-Qwen3-4B-{name}"
            present = set(api.list_repo_files(repo))
            for path in BOOKKEEPING:
                if path in present:
                    api.delete_file(path, repo_id=repo, commit_message=f"Remove {path}")
            api.super_squash_history(repo_id=repo, repo_type="model")
            print(f"cleaned https://huggingface.co/{repo}")
        return 0
    for name, (folder, card) in MODELS.items():
        if name not in args.only:
            continue
        source = scratch / "runs" / folder
        if not (source / "config.json").is_file():
            raise SystemExit(f"missing model folder {source}")
        repo = f"{args.user}/KnowWhenToHandOff-Qwen3-4B-{name}"
        api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
        api.upload_folder(
            repo_id=repo,
            folder_path=source,
            repo_type="model",
            ignore_patterns=BOOKKEEPING,
        )
        api.upload_file(
            path_or_fileobj=ROOT / card, path_in_repo="README.md", repo_id=repo
        )
        print(f"https://huggingface.co/{repo}")
    if "data" in args.only:
        data = scratch / "release" / "data"
        if not (data / "SHA256SUMS").is_file():
            raise SystemExit(f"missing {data}; run scripts/export_release.py first")
        repo = f"{args.user}/KnowWhenToHandOff-data"
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=repo, folder_path=data, repo_type="dataset")
        api.upload_file(
            path_or_fileobj=ROOT / "docs/dataset-card.md",
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
        )
        print(f"https://huggingface.co/datasets/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
