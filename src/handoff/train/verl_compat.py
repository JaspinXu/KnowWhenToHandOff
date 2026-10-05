"""Run verl 0.8 without flash-attn (the cluster venv uses SDPA attention).

verl's padding helpers (``verl.utils.attention_utils``) import
``flash_attn.bert_padding`` on CUDA even with ``use_remove_padding=false``,
for example when the trainer recomputes old log-probs. flash-attn has no wheel
for the pinned torch, so when it is missing these helpers are pointed at verl's
own pure-PyTorch copies of the same functions (``verl.utils.npu_flash_attn_utils``,
copied from flash-attn's ``bert_padding``). Attention itself stays SDPA.

The patch must exist in every Ray worker process, so it is installed as Ray's
``worker_process_setup_hook`` as well as in the driver.
"""

from __future__ import annotations

import importlib.util
from typing import Any

WORKER_SETUP_HOOK = "handoff.train.verl_compat.install_flash_attn_fallback"


def install_flash_attn_fallback() -> bool:
    """Patch verl's padding helpers when flash-attn is absent; True if patched."""
    if importlib.util.find_spec("flash_attn") is not None:
        return False
    import verl.utils.attention_utils as attention_utils
    import verl.utils.npu_flash_attn_utils as torch_padding

    def torch_attention_functions():
        return (
            torch_padding.index_first_axis,
            torch_padding.pad_input,
            torch_padding.rearrange,
            torch_padding.unpad_input,
        )

    attention_utils._get_attention_functions = torch_attention_functions
    return True


def add_worker_setup_hook(config: Any) -> None:
    """Ask Ray to run :func:`install_flash_attn_fallback` in every worker."""
    from omegaconf import open_dict

    with open_dict(config):
        ray_init = config.ray_kwargs.ray_init
        if ray_init.get("runtime_env") is None:
            ray_init.runtime_env = {}
        ray_init.runtime_env.worker_process_setup_hook = WORKER_SETUP_HOOK
