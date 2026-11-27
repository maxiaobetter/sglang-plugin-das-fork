import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "hcu_sglang"


class BaseConfig:
    pass


class MoELayer:
    pass


class NativeInt8Method:
    def __init__(self, config):
        self.quant_config = config


def selected_definition(path, name, namespace):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


Config = selected_definition(
    PKG / "layers/quantization/slimquant_config.py", "SlimQuantW4A8Int8Config",
    {"QuantizationConfig": BaseConfig},
)
Predicate = selected_definition(
    PKG / "layers/moe/deepep_int8.py", "_is_w8a8_int8_deepgemm_quant",
    {"Optional": __import__("typing").Optional, "QuantizationConfig": BaseConfig,
     "W8A8Int8Config": type("W8A8Config", (), {}),
     "SlimQuantCompressedTensorsMarlinConfig": type("CompressedConfig", (), {})},
)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        definitions = {
            "sglang.srt.layers.linear": {"LinearBase": type("Linear", (), {})},
            "sglang.srt.layers.moe.fused_moe_triton": {"FusedMoE": MoELayer},
            "sglang.srt.layers.quantization.compressed_tensors.utils": {
                "should_ignore_layer": lambda *args: False,
            },
            "sglang.srt.layers.quantization.unquant": {
                "UnquantizedFusedMoEMethod": object, "UnquantizedLinearMethod": object,
            },
            "sglang.srt.layers.quantization.w8a8_int8": {
                "W8A8Int8MoEMethod": NativeInt8Method,
            },
            "hcu_sglang.layers.quantization.slimquant_config": {
                "SlimQuantW4A8Int8Config": Config,
            },
        }
        modules = {}
        for name, attrs in definitions.items():
            module = ModuleType(name)
            module.__dict__.update(attrs)
            modules[name] = module
        self.patch = patch.dict(sys.modules, modules)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_checkpoint_selected_mtp_layer_is_int8(self):
        config = Config({"moe_int8_layers": [45]})
        layer = MoELayer()
        layer.layer_id = 45
        layer.quant_method = config.get_quant_method(layer, "model.layers.45.mlp.experts")
        self.assertIsInstance(layer.quant_method, NativeInt8Method)
        self.assertTrue(Predicate(layer, config))

    def test_mapped_mtp_prefix_is_int8(self):
        config = Config({"moe_int8_layers": [45]})
        config.moe_int8_prefixes = ["model.decoder.mlp.experts"]
        layer = MoELayer()
        layer.layer_id = 0
        layer.quant_method = config.get_quant_method(layer, "model.decoder.mlp.experts")
        self.assertTrue(Predicate(layer, config))

    def test_ordinary_int4_layer_is_not_reclassified(self):
        config = Config({"moe_int8_layers": [45]})
        config.get_int4_moe_method = lambda: object()
        layer = MoELayer()
        layer.layer_id = 44
        layer.quant_method = config.get_quant_method(layer, "model.layers.44.mlp.experts")
        self.assertFalse(Predicate(layer, config))

    def test_other_config_with_method_is_not_admitted(self):
        layer = SimpleNamespace(quant_method=NativeInt8Method(BaseConfig()))
        self.assertFalse(Predicate(layer, BaseConfig()))


if __name__ == "__main__":
    unittest.main()
