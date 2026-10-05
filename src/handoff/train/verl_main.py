"""verl GRPO entry point that exports every saved checkpoint.

Usage is identical to ``python -m verl.trainer.main_ppo`` (same Hydra config
and overrides). Inside verl's ``TaskRunner`` actor process (installed through
the ``task_runner_class`` hook that ``run_ppo`` provides), every saved actor
checkpoint is exported to a servable Hugging Face directory
(``<run_dir>/export/global_step_N``) right after it is written, before verl's
checkpoint rotation (``trainer.max_actor_ckpt_to_keep``) can delete it. Steps
saved without an export (a job killed mid-save) are exported at start-up.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
import ray
import verl.trainer.main_ppo as verl_main_ppo
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device

from handoff.train.export import export_pending, export_step
from handoff.train.verl_compat import install_flash_attn_fallback

VERL_CONFIG_DIR = str(Path(verl_main_ppo.__file__).resolve().parent / "config")
logger = logging.getLogger(__name__)


def _install_checkpoint_export(config) -> None:
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    checkpoints = Path(str(config.trainer.default_local_dir))
    exports = checkpoints.parent / "export"
    for target in export_pending(checkpoints, exports):
        logger.info("exported %s", target)
    original = RayPPOTrainer._save_checkpoint
    if getattr(original, "handoff_export", False):
        return

    def save_and_export(self) -> None:
        original(self)
        step = f"global_step_{self.global_steps}"
        target = export_step(checkpoints / step, exports / step)
        logger.info("exported %s", target)

    save_and_export.handoff_export = True
    RayPPOTrainer._save_checkpoint = save_and_export


class HandoffTaskRunner(TaskRunner):
    """verl TaskRunner that exports saved checkpoints."""

    def run(self, config):
        """Install the export hook and padding fallback here, then train.

        Both patches must live in this (TaskRunner) process: it saves the
        checkpoints and runs verl's padding conversions. Ray workers are left
        alone; importing torch/verl before Ray assigns their GPUs breaks CUDA.
        """
        _install_checkpoint_export(config)
        install_flash_attn_fallback()
        return super().run(config)


@hydra.main(config_path=VERL_CONFIG_DIR, config_name="ppo_trainer", version_base=None)
def main(config) -> None:
    """Same flow as verl's ``main_ppo.main`` with a custom task runner."""
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(HandoffTaskRunner))


if __name__ == "__main__":
    main()
