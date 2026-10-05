"""flash-attn fallback for verl's padding helpers (no verl or GPU needed)."""

import importlib
import importlib.util
import sys
import types

import pytest
from omegaconf import OmegaConf

from handoff.train import verl_compat


def _fake_verl(monkeypatch):
    modules = {
        name: types.ModuleType(name)
        for name in (
            "verl",
            "verl.utils",
            "verl.utils.attention_utils",
            "verl.utils.npu_flash_attn_utils",
        )
    }
    torch_padding = modules["verl.utils.npu_flash_attn_utils"]
    for name in ("index_first_axis", "pad_input", "rearrange", "unpad_input"):
        setattr(torch_padding, name, object())
    attention = modules["verl.utils.attention_utils"]

    def needs_flash_attn():
        importlib.import_module("flash_attn.bert_padding")

    attention._get_attention_functions = needs_flash_attn
    modules["verl"].utils = modules["verl.utils"]
    modules["verl.utils"].attention_utils = attention
    modules["verl.utils"].npu_flash_attn_utils = torch_padding
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return attention, torch_padding


def test_fallback_points_verl_at_torch_padding(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    attention, torch_padding = _fake_verl(monkeypatch)
    assert verl_compat.install_flash_attn_fallback() is True
    assert attention._get_attention_functions() == (
        torch_padding.index_first_axis,
        torch_padding.pad_input,
        torch_padding.rearrange,
        torch_padding.unpad_input,
    )


def test_fallback_is_a_noop_with_flash_attn(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    attention, _ = _fake_verl(monkeypatch)
    original = attention._get_attention_functions
    assert verl_compat.install_flash_attn_fallback() is False
    assert attention._get_attention_functions is original


@pytest.mark.parametrize("ray_init", [{"num_cpus": None}, {"runtime_env": {"x": 1}}])
def test_worker_setup_hook_is_added_to_struct_config(ray_init):
    config = OmegaConf.create({"ray_kwargs": {"ray_init": ray_init}})
    OmegaConf.set_struct(config, True)
    verl_compat.add_worker_setup_hook(config)
    runtime_env = config.ray_kwargs.ray_init.runtime_env
    assert runtime_env.worker_process_setup_hook == verl_compat.WORKER_SETUP_HOOK
    assert runtime_env.get("x", 1) == 1
    module, _, function = verl_compat.WORKER_SETUP_HOOK.rpartition(".")
    assert getattr(importlib.import_module(module), function)
