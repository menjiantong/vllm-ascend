# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import builtins
import importlib.util
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_engram_patch_is_noop_without_upstream_config(monkeypatch):
    original_import = builtins.__import__

    def without_engram(name, *args, **kwargs):
        if name == "vllm.config.engram":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_engram)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    root = Path(__file__).resolve().parents[4]
    namespace = runpy.run_path(str(root / "vllm_ascend/patch/platform/patch_engram_config.py"))
    assert "verify_model_config" not in namespace


@pytest.mark.parametrize("model", [None, "unsupported", "empty"])
def test_device_patch_retains_upstream_model_checks(model):
    from vllm.config import EngramConfig, VllmConfig

    if model is not None:
        model = SimpleNamespace(
            architecture="DeepseekV41ForCausalLM" if model == "empty" else "UnsupportedModel",
            hf_text_config=SimpleNamespace(engram_layer_ids=[] if model == "empty" else [1]),
        )
    with pytest.raises(ValueError, match="non-empty n-gram"):
        EngramConfig().verify_model_config(model)
    assert VllmConfig._resolve_and_verify_engram_config.__module__ == "vllm.config.vllm"


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"embedding_across_dp": True}, "embedding_across_dp"),
        ({"pipeline_parallel_size": 2}, "PP=PCP=DCP"),
        ({"prefill_context_parallel_size": 2}, "PP=PCP=DCP"),
        ({"decode_context_parallel_size": 2}, "PP=PCP=DCP"),
        ({"tensor_parallel_size": 16}, "TP=1/2/4/8"),
        ({"enable_elastic_ep": True}, "TP=1/2/4/8"),
        ({"architecture": "Qwen4ExpForCausalLM"}, "DeepSeek V4.1"),
        ({"load_format": "pt"}, "indexed safetensors"),
        ({"load_format": "dummy", "dp_shared_memory": True}, "load_format"),
    ],
)
def test_platform_retains_ascend_engram_limits(overrides, error):
    from vllm.config import EngramConfig

    from vllm_ascend.platform import _validate_engram_config

    config = SimpleNamespace(
        model_config=SimpleNamespace(architecture=overrides.get("architecture", "DeepseekV41ForCausalLM")),
        speculative_config=None,
        engram_config=EngramConfig(
            cpu_offload=True,
            dp_shared_memory=overrides.get("dp_shared_memory", False),
            embedding_across_dp=overrides.get("embedding_across_dp", False),
        ),
        parallel_config=SimpleNamespace(
            **{
                name: overrides.get(name, default)
                for name, default in dict(
                    tensor_parallel_size=4,
                    data_parallel_size=4,
                    pipeline_parallel_size=1,
                    prefill_context_parallel_size=1,
                    decode_context_parallel_size=1,
                    enable_elastic_ep=False,
                ).items()
            }
        ),
        load_config=SimpleNamespace(load_format=overrides.get("load_format", "safetensors")),
    )
    with pytest.raises(ValueError, match=error):
        _validate_engram_config(config)


def test_platform_leaves_non_engram_models_alone():
    from vllm_ascend.platform import _validate_engram_config

    config = SimpleNamespace(
        engram_config=None,
        model_config=SimpleNamespace(architecture="OtherModel"),
        speculative_config=None,
    )
    _validate_engram_config(config)
    assert config.engram_config is None
