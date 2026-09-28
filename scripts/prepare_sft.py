"""Export offline SFT token batches; no model, optimizer or dataset selection."""

import argparse
import subprocess
from pathlib import Path
from typing import NoReturn

from hydra import compose, initialize_config_dir
from hydra.errors import HydraException
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException

from handoff.train.sft_batch import sft_config_from_mapping
from handoff.train.sft_export import export_sft_batches

ROOT = Path(__file__).resolve().parents[1]


class _SafeArgumentParser(argparse.ArgumentParser):
    """Do not echo rejected user-supplied values."""

    def error(self, message: str) -> NoReturn:
        self.exit(2, "Invalid command-line arguments; use --help.\n")


def main() -> int:
    """Compose preparation settings, record Git metadata and export to scratch."""
    parser = _SafeArgumentParser(description=__doc__)
    parser.add_argument("overrides", nargs="*", help="Hydra key=value overrides")
    args = parser.parse_args()
    try:
        with initialize_config_dir(
            version_base="1.3", config_dir=str(ROOT / "configs")
        ):
            config = compose(config_name="prepare_sft", overrides=args.overrides)
        if set(config) != {
            "train",
            "input_path",
            "output_dir",
            "batch_size",
            "max_record_bytes",
        }:
            raise ValueError("unexpected preparation settings")
        if OmegaConf.missing_keys(config):
            parser.exit(
                2,
                "Unresolved configuration; supply input_path, output_dir and "
                "train.pad_token_id.\n",
            )
        resolved = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
        if not isinstance(resolved, dict):
            raise ValueError("expected config mapping")
        for name in ("input_path", "output_dir"):
            if not isinstance(resolved[name], str) or not resolved[name].strip():
                raise ValueError("expected nonempty file paths")
        destination = Path(resolved["output_dir"]).resolve()
        if destination.is_relative_to(ROOT):
            parser.exit(
                2, "Choose an output directory on scratch outside the checkout.\n"
            )
        batch_config = sft_config_from_mapping(resolved["train"])
        git_sha = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.PIPE,
            timeout=10,
        ).strip()
        git_dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(ROOT), "status", "--porcelain"],
                stderr=subprocess.PIPE,
                timeout=10,
            )
        )
        manifest = export_sft_batches(
            Path(resolved["input_path"]).resolve(),
            destination,
            batch_config=batch_config,
            batch_size=resolved["batch_size"],
            max_record_bytes=resolved["max_record_bytes"],
            git_sha=git_sha,
            git_dirty=git_dirty,
        )
        print(
            f"Prepared {manifest['record_count']} records in "
            f"{manifest['batch_count']} batches (not a training run)."
        )
    except FileExistsError:
        parser.exit(2, "Output directory exists; choose a new scratch directory.\n")
    except (
        ValueError,
        OSError,
        HydraException,
        OmegaConfBaseException,
        subprocess.SubprocessError,
    ):
        parser.exit(
            2,
            "Cannot prepare batches: check configuration, input records, "
            "paths and Git.\n",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
