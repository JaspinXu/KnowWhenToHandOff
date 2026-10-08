"""Mirror the dataset release to Kaggle as a private dataset (docs/release.md).

Kaggle hosts the dataset only; model weights stay on Hugging Face. Needs the
Kaggle CLI (its own small environment, see docs/release.md), your API token in
``~/.kaggle/access_token`` (or a legacy ``~/.kaggle/kaggle.json``), and the
folder from ``scripts/export_release.py``. Run it as a CPU job.

The folder is staged next to the export with hard links (no extra disk), minus
the Hugging Face card, plus ``dataset-metadata.json``; sub-folders are uploaded
as zip archives, which Kaggle unpacks.

Example:
    python scripts/upload_kaggle.py --user <kaggle-handle> --dry-run
    python scripts/upload_kaggle.py --user <kaggle-handle> \
        --kaggle "$HANDOFF_SCRATCH/kaggle-venv/bin/kaggle"
    python scripts/upload_kaggle.py --user <kaggle-handle> --version "Fix run index"
    python scripts/upload_kaggle.py --user <kaggle-handle> --metadata-only
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from handoff.release import KAGGLE_SLUG, kaggle_metadata

ROOT = Path(__file__).resolve().parents[1]
COVER = ROOT / "docs" / "assets" / "kaggle-cover.png"


def main() -> int:
    """Stage the folder with Kaggle metadata and create (or version) the dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--user", required=True, help="Kaggle handle (from your profile URL)"
    )
    parser.add_argument("--kaggle", default="kaggle", help="path to the Kaggle CLI")
    parser.add_argument(
        "--version", metavar="MESSAGE", help="upload a new version instead of creating"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="stage and print metadata only"
    )
    parser.add_argument(
        "--delete-old-versions",
        action="store_true",
        help="with --version: delete the dataset's earlier versions on Kaggle",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="update title, description and cover image of the existing dataset",
    )
    args = parser.parse_args()

    release = Path(os.environ["HANDOFF_SCRATCH"]) / "release"
    card = (ROOT / "docs" / "dataset-card.md").read_text(encoding="utf-8")
    if args.metadata_only:
        meta_dir = release / "kaggle-metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        meta = kaggle_metadata(args.user, card)
        if COVER.is_file():
            shutil.copyfile(COVER, meta_dir / "dataset-cover-image.png")
            meta["image"] = "dataset-cover-image.png"
        (meta_dir / "dataset-metadata.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        dataset = f"{args.user}/{KAGGLE_SLUG}"
        command = [args.kaggle, "datasets", "metadata", dataset, "-p", str(meta_dir)]
        subprocess.run([*command, "--update"], check=True)
        print(f"updated metadata of https://www.kaggle.com/datasets/{dataset}")
        return 0
    data, stage = release / "data", release / "kaggle"
    if not (data / "SHA256SUMS").is_file():
        raise SystemExit(f"missing {data}; run scripts/export_release.py first")
    if stage.exists():
        shutil.rmtree(stage)

    def top_card(folder: str, names: list[str]) -> list[str]:
        # Top level only: the Hugging Face card (the description replaces it) and
        # SHA256SUMS (Kaggle would preview it first). Sub-folder READMEs stay.
        return ["README.md", "SHA256SUMS"] if Path(folder) == data else []

    shutil.copytree(data, stage, copy_function=os.link, ignore=top_card)
    meta = kaggle_metadata(args.user, card)
    if COVER.is_file():  # picked up automatically on create, not uploaded as data
        shutil.copyfile(COVER, stage / "dataset-cover-image.png")
    (stage / "dataset-metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"staged {stage}: {sorted(p.name for p in stage.iterdir())}")
    print(json.dumps({k: v for k, v in meta.items() if k != "description"}, indent=2))
    if args.dry_run:
        return 0

    if args.version:
        command = [
            args.kaggle,
            "datasets",
            "version",
            "-p",
            str(stage),
            "-m",
            args.version,
            "-r",
            "zip",
            "-t",
            *(["-d"] if args.delete_old_versions else []),
        ]
    else:
        command = [
            args.kaggle,
            "datasets",
            "create",
            "-p",
            str(stage),
            "-r",
            "zip",
            "-t",
        ]
    subprocess.run(command, check=True)
    url = f"https://www.kaggle.com/datasets/{args.user}/{KAGGLE_SLUG}"
    print(f"{url} (private until you make it public)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
