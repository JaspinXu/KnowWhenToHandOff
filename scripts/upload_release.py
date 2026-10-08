"""Upload the release to Hugging Face as private repositories (docs/release.md).

Needs ``hf auth login`` once (your own write token) and the dataset folder from
``scripts/export_release.py``. Run it as a CPU job, not on the login node.

Example:
    python scripts/upload_release.py --user JaspinXu
    python scripts/upload_release.py --user JaspinXu --collection
    python scripts/upload_release.py --user JaspinXu --publish   # release day
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
# Local bookkeeping next to the weights; the export and SFT records hold cluster
# paths.
BOOKKEEPING = [".handoff_fingerprint.json", "handoff_export.json", "handoff_sft.json"]
COLLECTION = "KnowWhenToHandOff"
COLLECTION_ABOUT = (
    "When should a tool-using customer-service agent hand off to a human? "
    "Qwen3-4B SFT and GRPO models with all data."
)
NOTES = {  # collection item notes, in display order
    "R2": "Recommended. GRPO with the Responsible-AI reward: hand-off F1 0.812, "
    "over-escalation 0.103.",
    "R1": "GRPO with the task reward only. Learns to over-escalate (0.462); kept "
    "for comparison.",
    "SFT": "Imitation of Qwen3-30B-A3B teacher dialogues; the starting point of "
    "R1 and R2.",
    "data": "Hand-off task set, every evaluation conversation and every GRPO "
    "training episode.",
}


def repo_ids(user: str) -> list[tuple[str, str, str]]:
    """(key, repository ID, type) of every release repository."""
    models = [
        (name, f"{user}/KnowWhenToHandOff-Qwen3-4B-{name}", "model") for name in MODELS
    ]
    return [*models, ("data", f"{user}/KnowWhenToHandOff-data", "dataset")]


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
    parser.add_argument(
        "--squash",
        action="store_true",
        help="after uploading, squash each repository's history to one commit",
    )
    parser.add_argument(
        "--cards-only",
        action="store_true",
        help="re-upload only the cards (README.md) of the selected repositories",
    )
    parser.add_argument(
        "--collection",
        action="store_true",
        help=f"create (or refresh) the private collection {COLLECTION!r} holding "
        "the four repositories",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="make the four repositories and the collection public",
    )
    args = parser.parse_args()
    from huggingface_hub import HfApi

    api = HfApi()
    if args.collection or args.publish:
        collection = api.create_collection(
            title=COLLECTION,
            namespace=args.user,
            description=COLLECTION_ABOUT,
            private=not args.publish,
            exists_ok=True,
        )
        present = {item.item_id for item in collection.items}
        for key, repo, kind in repo_ids(args.user):
            if args.publish:
                api.update_repo_settings(repo, private=False, repo_type=kind)
                print(f"public {repo}")
            if repo not in present:
                api.add_collection_item(
                    collection.slug, repo, kind, note=NOTES[key], exists_ok=True
                )
        if args.publish:
            api.update_collection_metadata(collection.slug, private=False)
        print(f"https://huggingface.co/collections/{collection.slug}")
        return 0
    scratch = Path(os.environ["HANDOFF_SCRATCH"])
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
    if args.cards_only:
        for name, (_, card) in MODELS.items():
            if name in args.only:
                repo = f"{args.user}/KnowWhenToHandOff-Qwen3-4B-{name}"
                api.upload_file(
                    path_or_fileobj=ROOT / card,
                    path_in_repo="README.md",
                    repo_id=repo,
                    commit_message="Update model card",
                )
                print(f"card https://huggingface.co/{repo}")
        if "data" in args.only:
            repo = f"{args.user}/KnowWhenToHandOff-data"
            api.upload_file(
                path_or_fileobj=ROOT / "docs/dataset-card.md",
                path_in_repo="README.md",
                repo_id=repo,
                repo_type="dataset",
                commit_message="Update dataset card",
            )
            print(f"card https://huggingface.co/datasets/{repo}")
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
        if args.squash:
            api.super_squash_history(repo_id=repo, repo_type="model")
        print(f"https://huggingface.co/{repo}")
    if "data" in args.only:
        data = scratch / "release" / "data"
        if not (data / "SHA256SUMS").is_file():
            raise SystemExit(f"missing {data}; run scripts/export_release.py first")
        repo = f"{args.user}/KnowWhenToHandOff-data"
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(  # also deletes remote data files no longer in the export
            repo_id=repo,
            folder_path=data,
            repo_type="dataset",
            delete_patterns=["runs/**", "tasks/**", "sft/**", "*.jsonl", "SHA256SUMS"],
        )
        api.upload_file(
            path_or_fileobj=ROOT / "docs/dataset-card.md",
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
        )
        if args.squash:
            api.super_squash_history(repo_id=repo, repo_type="dataset")
        print(f"https://huggingface.co/datasets/{repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
