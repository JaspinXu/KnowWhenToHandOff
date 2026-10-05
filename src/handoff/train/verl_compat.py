"""Run verl 0.8 without flash-attn (the cluster venv uses SDPA attention).

verl's padding helpers (``verl.utils.attention_utils``) import
``flash_attn.bert_padding`` on CUDA even with ``use_remove_padding=false``,
for example when the trainer recomputes old log-probs. flash-attn has no wheel
for the pinned torch, so when it is missing these helpers are pointed at verl's
own pure-PyTorch copies of the same functions (``verl.utils.npu_flash_attn_utils``,
copied from flash-attn's ``bert_padding``). Attention itself stays SDPA.

Only verl's trainer (the TaskRunner actor) calls these helpers, in
``RayPPOTrainer`` when it converts batches to and from the no-padding layout;
the FSDP workers use the padded path (``use_remove_padding=false``). So the
patch is installed in the TaskRunner (``handoff.train.verl_main``). It must not
run as a Ray worker setup hook: importing torch before Ray sets a worker's
``CUDA_VISIBLE_DEVICES`` leaves CUDA with stale devices ("Invalid device id").
"""

from __future__ import annotations

import importlib.util


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
