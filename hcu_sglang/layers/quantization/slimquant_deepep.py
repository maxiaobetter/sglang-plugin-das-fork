import torch

from hcu_sglang.layers.quantization.slimquant_w4a8 import SlimQuantW4A8Int8MoEMethod
from sglang.srt.layers.quantization.base_config import FusedMoEMethodBase


class SlimQuantDeepEPHIPCMethod(SlimQuantW4A8Int8MoEMethod, FusedMoEMethodBase):
    hcu_slimquant_deepep_hipc = True

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **attrs):
        if params_dtype != torch.bfloat16:
            raise ValueError("SlimQuant DeepEP HIPC requires bfloat16 output")
        return super().create_weights(
            layer, num_experts, hidden_size, intermediate_size_per_partition,
            params_dtype, **attrs,
        )

    def process_weights_after_loading(self, layer):
        if getattr(layer, "_hcu_slimquant_deepep_weights_ready", False):
            return
        from deepgemm import pack_w4a8_moe_hipc_weight

        packed = [pack_w4a8_moe_hipc_weight(getattr(layer, name).data)
                  for name in ("w13_weight", "w2_weight")]
        for name, weight in zip(("w13_weight", "w2_weight"), packed):
            setattr(layer, name, torch.nn.Parameter(weight, requires_grad=False))
        layer._hcu_slimquant_deepep_weights_ready = True

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config
        self.runner = None


def configure_layer(layer):
    from hcu_sglang.layers.moe.deepep_w4a8 import (
        _hcu_deepep_dispatcher,
        configure_hipc,
    )

    mode = _hcu_deepep_dispatcher(layer).deepep_mode
    layer.deprecate_flag = False
    layer.deepep_mode = mode
    configure_hipc(layer, masked=mode.enable_low_latency(), contiguous=mode.enable_normal())
