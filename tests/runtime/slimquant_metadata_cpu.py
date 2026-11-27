import ast
from dataclasses import dataclass, field
import os
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional
from types import MappingProxyType
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ["SGLANG_SOURCE_ROOT"]) / "python/sglang/srt"


def definition(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    unit = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
        ast.alias(name="annotations")], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), "exec"), namespace)
    return namespace[name]


class BaseConfig:
    pass


class MethodBase:
    pass


class Linear(torch.nn.Module):
    pass


class MoE(torch.nn.Module):
    pass


class UnquantizedLinear:
    pass


class UnquantizedMoE:
    pass


scope = dict(torch=torch, os=os, QuantizationConfig=BaseConfig, LinearBase=Linear,
             FusedMoEMethodBase=MethodBase, W8a8GetCacheJSON=lambda: None,
             get_bool_env_var=lambda key: False)
config_path = ROOT / "hcu_sglang/layers/quantization/slimquant_w4a8.py"
Config = definition(config_path, "SlimQuantW4A8Int8Config", scope)
Config.__module__ = "hcu_sglang.layers.quantization.slimquant_w4a8"
definition(config_path, "SlimQuantW4A8Int8MoEMethod", scope)
scope["SlimQuantW4A8Int8LinearMethod"] = definition(
    config_path, "SlimQuantW4A8Int8LinearMethod",
    dict(scope, LinearMethodBase=MethodBase))
Native = definition(SOURCE / "layers/quantization/w8a8_int8.py",
                    "W8A8Int8MoEMethod", dict(FusedMoEMethodBase=MethodBase))
utils_scope = dict(re=re, MappingProxyType=MappingProxyType)
for name in ("_is_equal_or_regex_match", "check_equal_or_regex_match", "should_ignore_layer"):
    definition(SOURCE / "layers/quantization/compressed_tensors/utils.py", name, utils_scope)
mapper_scope = dict(dataclass=dataclass, field=field, WeightsMapping=dict)
Mapper = definition(SOURCE / "models/utils.py", "WeightsMapper", mapper_scope)
HIPC = definition(ROOT / "hcu_sglang/layers/quantization/slimquant_deepep.py",
                  "SlimQuantDeepEPHIPCMethod", scope)
definitions = {
    "sglang.srt.layers.moe.fused_moe_triton": {"FusedMoE": MoE},
    "sglang.srt.layers.quantization.unquant": {
        "UnquantizedLinearMethod": UnquantizedLinear,
        "UnquantizedFusedMoEMethod": UnquantizedMoE},
    "sglang.srt.layers.quantization.compressed_tensors.utils": utils_scope,
    "sglang.srt.layers.quantization.w8a8_int8": {"W8A8Int8MoEMethod": Native},
    "hcu_sglang.layers.quantization.slimquant_w4a8": {"SlimQuantW4A8Int8Config": Config},
    "hcu_sglang.layers.quantization.slimquant_deepep": {"SlimQuantDeepEPHIPCMethod": HIPC},
    "hcu_sglang.utils.detection": {"is_hcu": lambda: True},
    "sglang.srt.layers.moe": {"get_moe_a2a_backend": lambda: SimpleNamespace(is_deepep=lambda: True)},
}
modules = {}
for name, attrs in definitions.items():
    module = ModuleType(name)
    module.__dict__.update(attrs)
    modules[name] = module
predicate = definition(ROOT / "hcu_sglang/layers/moe/deepep_int8.py",
                       "_is_w8a8_int8_deepgemm_quant", dict(
                           W8A8Int8Config=type("OtherConfig", (), {}),
                           SlimQuantCompressedTensorsMarlinConfig=type("Compressed", (), {})))
converter = definition(ROOT / "hcu_sglang/layers/quantization/registry.py",
                       "around_get_all_weights", {})

with patch.dict(sys.modules, modules):
    legacy = Config.from_config({})
    assert isinstance(legacy.get_quant_method(Linear(), "linear"), UnquantizedLinear)
    assert type(legacy.get_quant_method(MoE(), "experts")).__name__ == "SlimQuantW4A8Int8MoEMethod"
    config = Config.from_config({"w4_dir": True, "moe_int8_layers": [45],
                               "ignore": ["model.visual", "model.layers.45.self_attn.qkv_proj"]})
    mapper = Mapper(orig_to_new_substr={"model.layers.45": "model.decoder"},
                    orig_to_new_prefix={"model.visual": "visual"})
    config.apply_weight_name_mapper(mapper)
    assert config.moe_int8_prefixes == ["model.decoder.mlp.experts"]
    assert isinstance(config.get_quant_method(Linear(), "visual.blocks.0.attn.qkv"), UnquantizedLinear)
    assert isinstance(config.get_quant_method(Linear(), "model.decoder.self_attn.qkv_proj"), UnquantizedLinear)
    assert isinstance(config.get_quant_method(Linear(), "model.layers.0.self_attn.q_proj"),
                      scope["SlimQuantW4A8Int8LinearMethod"])
    draft = MoE()
    draft.layer_id = 0
    draft.quant_method = config.get_quant_method(draft, "model.decoder.mlp.experts")
    assert isinstance(draft.quant_method, Native) and draft.quant_method.quant_config is config
    assert predicate(draft, config)
    ordinary = MoE()
    ordinary.layer_id = 44
    ordinary.quant_method = config.get_quant_method(ordinary, "model.layers.44.mlp.experts")
    assert not predicate(ordinary, config)
    scope["get_bool_env_var"] = lambda key: True
    assert isinstance(config.get_quant_method(ordinary, "model.layers.44.mlp.experts"), HIPC)
    qweight = torch.tensor([[0x12, 0xF0]], dtype=torch.uint8)
    scales = torch.tensor([[32.0]])
    weights = [("expert.qweight", qweight), ("expert.scales", scales), ("other", scales)]
    converted = dict(converter(lambda *args: iter(weights), None, None,
                               SimpleNamespace(quant_config=config)))
    assert torch.equal(converted["expert.weight"], torch.tensor([[0x21, 0x0F]], dtype=torch.int8))
    assert torch.equal(converted["expert.weight_scale"], torch.tensor([[2.0]]))
    assert converted["other"] is scales
    assert dict(converter(lambda *args: iter(weights), None, None,
                          SimpleNamespace(quant_config=legacy)))["expert.qweight"] is qweight
    for bad in (torch.ones(2, 2, dtype=torch.int8), torch.ones(2, dtype=torch.uint8)):
        try:
            list(config.convert_weights([("bad.qweight", bad)]))
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid qweight accepted")

print("PASS: metadata, real upstream mapper/native INT8 constructor/ignore rules, "
      "MTP predicate, HIPC selection, loader conversion and legacy fallback; GPU pending")
