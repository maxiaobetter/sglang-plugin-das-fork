import copy
import logging
import os
import threading
from collections import OrderedDict

logger = logging.getLogger(__name__)


def around_initialize(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    manager = self.tokenizer_manager
    config = getattr(getattr(manager, "model_config", None), "hf_config", None)
    tokenizer = manager.tokenizer
    if getattr(config, "model_type", None) != "glm5_next":
        return result
    if not getattr(tokenizer, "is_fast", False):
        return result
    self._hcu_chat_tokenizer_seed = copy.deepcopy(tokenizer)
    self._hcu_chat_tokenizer_pid = os.getpid()
    self._hcu_chat_tokenizer_lock = threading.Lock()
    self._hcu_chat_tokenizer_local = threading.local()
    logger.info("GLM5 tokenizer thread isolation initialized pid=%s", os.getpid())
    return result


def around_render(original, self, *args, **kwargs):
    if not hasattr(self, "_hcu_chat_tokenizer_local"):
        return original(self, *args, **kwargs)
    if self._hcu_chat_tokenizer_pid != os.getpid():
        self._hcu_chat_tokenizer_pid = os.getpid()
        self._hcu_chat_tokenizer_lock = threading.Lock()
        self._hcu_chat_tokenizer_local = threading.local()
    local = self._hcu_chat_tokenizer_local
    view = getattr(local, "view", None)
    if view is None:
        with self._hcu_chat_tokenizer_lock:
            tokenizer = copy.deepcopy(self._hcu_chat_tokenizer_seed)
        view = copy.copy(self)
        view.tokenizer_manager = copy.copy(self.tokenizer_manager)
        view.tokenizer_manager.tokenizer = tokenizer
        view._chat_template_cache = OrderedDict()
        local.view = view
        logger.info(
            "GLM5 isolated tokenizer pid=%s thread=%s backend=%s",
            os.getpid(), threading.get_ident(), id(tokenizer._tokenizer),
        )
    view._prompt_text_round_trip_is_lossy = self._prompt_text_round_trip_is_lossy
    return original(view, *args, **kwargs)


def register_hooks():
    from hcu_sglang.utils.detection import is_hcu

    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.entrypoints.openai.serving_chat.OpenAIServingChat.__init__",
        around_initialize, HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.entrypoints.openai.serving_chat.OpenAIServingChat."
        "_render_and_encode_chat_template",
        around_render, HookType.AROUND,
    )
