from hcu_sglang.utils.detection import is_hcu

_GLM_ARCHS = {"GlmMoeDsaForCausalLM", "GlmMoeDsaForCausalLMNextN"}


def model_matches(config):
    return bool(_GLM_ARCHS & set(getattr(config, "architectures", None) or ()))


def register_hooks():
    if not is_hcu():
        return
    from hcu_sglang.layers.attention.current_dsa.dcp import around_mla_dcp_phase
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.models.deepseek_common.attention_forward_methods.forward_mla."
        "DeepseekMLAForwardMixin.is_dcp_mla_decode_phase",
        around_mla_dcp_phase,
        HookType.AROUND,
    )
