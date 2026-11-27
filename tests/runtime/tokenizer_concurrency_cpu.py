import ast,concurrent.futures,hashlib,importlib.util,json,sys,threading,time,types
from collections import OrderedDict
from pathlib import Path
import orjson
from tokenizers import Tokenizer,models,pre_tokenizers
from transformers import PreTrainedTokenizerFast

import os
from unittest.mock import patch
root=Path(__file__).resolve().parents[2]
upstream=Path(os.environ['SGLANG_SOURCE_ROOT'])
src=upstream/'python/sglang/srt/entrypoints/openai/serving_chat.py'
source=src.read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='OpenAIServingChat')
method=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='_render_and_encode_chat_template')
ns={'orjson':orjson,'_CHAT_TEMPLATE_CACHE_MAX_SIZE':128,'parallel_prompt_encode':lambda tok,text,kw:tok.encode(text,**kw)}
exec(compile('from __future__ import annotations\n'+ast.unparse(method),str(src),'exec'),ns)
raw=ns['_render_and_encode_chat_template']
backend=Tokenizer(models.WordLevel({'[UNK]':0,'hello':1,'world':2,'user':3,'assistant':4},unk_token='[UNK]'))
backend.pre_tokenizer=pre_tokenizers.Whitespace()
tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',pad_token='[UNK]')
tokenizer.chat_template="{% for message in messages %}{{message['role']}} {{message['content']}} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"

def init(self,tok,model_type='glm5_next'):
    self.tokenizer_manager=types.SimpleNamespace(tokenizer=tok,model_config=types.SimpleNamespace(hf_config=types.SimpleNamespace(model_type=model_type)))
    self._chat_template_cache=OrderedDict()
    self._prompt_text_round_trip_is_lossy=False

def make():
    return type('OpenAIServingChat',(),{'__init__':types.FunctionType(init.__code__,init.__globals__,argdefs=init.__defaults__),'_render_and_encode_chat_template':types.FunctionType(raw.__code__,raw.__globals__)})
messages=[{'role':'user','content':'hello world '*8192}]
kwargs=dict(tools=None,template_kwargs={},encode_kwargs={'add_special_tokens':False},use_cache=False)
baseline=make()(tokenizer)
expected=baseline._render_and_encode_chat_template(messages,**kwargs)
def stress(service,n=64):
    gate=threading.Barrier(8)
    stop=threading.Event()
    background_errors=[]
    def processor():
        while not stop.is_set():
            try:
                service.tokenizer_manager.tokenizer(['hello world '*8192,'hello'],padding=True)
            except Exception as exc:
                background_errors.append(type(exc).__name__+': '+str(exc))
    writer=threading.Thread(target=processor)
    writer.start()
    def work(i):
        if i<8: gate.wait(timeout=30)
        try:
            result=service._render_and_encode_chat_template(messages,**kwargs)
            assert result==expected
            return None
        except Exception as exc:
            return type(exc).__name__+': '+str(exc)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            errors=[x for x in pool.map(work,range(n)) if x]
    finally:
        stop.set();writer.join(timeout=30)
    assert not writer.is_alive()
    return errors+background_errors
before=stress(baseline)
assert any('Already borrowed' in x for x in before),before
spec=importlib.util.spec_from_file_location('isolation',root/'hcu_sglang/tokenizer_concurrency.py')
fix=importlib.util.module_from_spec(spec); spec.loader.exec_module(fix)
spec=importlib.util.spec_from_file_location('native_hooks',upstream/'python/sglang/srt/plugins/hook_registry.py')
hooks=importlib.util.module_from_spec(spec); spec.loader.exec_module(hooks)
module=types.ModuleType('sglang.srt.entrypoints.openai.serving_chat')
module.OpenAIServingChat=make()
modules={'sglang.srt.entrypoints.openai.serving_chat':module,'sglang.srt.plugins.hook_registry':hooks}
for name in ('sglang','sglang.srt','sglang.srt.entrypoints','sglang.srt.entrypoints.openai','sglang.srt.plugins','hcu_sglang','hcu_sglang.utils'):
    p=types.ModuleType(name);p.__path__=[];modules[name]=p
d=types.ModuleType('hcu_sglang.utils.detection');d.is_hcu=lambda:True;modules[d.__name__]=d
with patch.dict(sys.modules,modules):
    fix.register_hooks()
    hooks.HookRegistry.apply_hooks()
    fixed=module.OpenAIServingChat(tokenizer)
assert hasattr(fixed,'_hcu_chat_tokenizer_local')
fallback=module.OpenAIServingChat(tokenizer,model_type='other_model')
fix.around_initialize(lambda self:None,fallback)
assert not hasattr(fallback,'_hcu_chat_tokenizer_local')
assert fix.around_render(lambda self:'original',fallback)=='original'
after=stress(fixed,128)
assert not after,after
for lossy in (False,True):
    fixed._prompt_text_round_trip_is_lossy=lossy
    reference=make()(tokenizer)
    reference._prompt_text_round_trip_is_lossy=lossy
    for cache in (False,True):
        opts=dict(kwargs,use_cache=cache)
        gold=reference._render_and_encode_chat_template(messages,**opts)
        got=fixed._render_and_encode_chat_template(messages,**opts)
        assert got==gold
        got[0].append(-99)
        assert fixed._render_and_encode_chat_template(messages,**opts)==gold
assert tokenizer._tokenizer is not fixed._hcu_chat_tokenizer_local.view.tokenizer_manager.tokenizer._tokenizer
assert not fixed._chat_template_cache
report={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'actual_upstream_render_method':True,'actual_hf_fast_tokenizer':True,'baseline_requests':64,'baseline_errors':len(before),'baseline_examples':before[:3],'fixed_requests':128,'fixed_errors':len(after),'threads':8,'token_ids_equal':True,'cache_and_lossy_branches':True,'actual_hook_registry_register_apply':True,'other_model_fallback':True,'gpu_service_test':'pending_for_final_hook_packaging','scope':'actual render vs one shared multimodal-style padding encoder; other call sites not patched'}

print(json.dumps(report,indent=2),flush=True)
