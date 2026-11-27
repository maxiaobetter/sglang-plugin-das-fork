import ast,json,sys,types,typing
from pathlib import Path
import torch
from torch import nn
from types import SimpleNamespace
src=(Path(__file__).resolve().parents[2]/'hcu_sglang/models/glm5_next/vision.py').read_text()
tree=ast.parse(src)
names={'Glm4vVisionMLP','Glm4vVisionBlock','Glm4vPatchMerger','_glm5_vision_parallel_group'}
nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
state=SimpleNamespace(tp_size=8,tp_rank=0,attn_tp_size=1,attn_tp_rank=0,dp=True)
class Linear(nn.Module):
 def __init__(self,*args,parallel_group='replicated',use_dp_attention_reduce=False,**kw):
  super().__init__();self.parallel_group=parallel_group;self.use_dp_attention_reduce=use_dp_attention_reduce
  self.tp_size=1 if parallel_group=='replicated' else getattr(state,parallel_group+'_size')
 def forward(self,x):return x,None
class Norm(nn.Module):
 def __init__(self,*a,**k):super().__init__()
class Attention(nn.Module):
 def __init__(self,*a,use_data_parallel=False,use_dp_attention_reduce=False,**k):
  super().__init__();self.tp_size=1 if use_data_parallel else state.attn_tp_size;self.use_dp_attention_reduce=use_dp_attention_reduce
scope=dict(torch=torch,nn=nn,F=torch.nn.functional,Optional=typing.Optional,QuantizationConfig=object,VisionAttentionMetadata=object,
 MergedColumnParallelLinear=Linear,RowParallelLinear=Linear,ReplicatedLinear=Linear,RMSNorm=Norm,LayerNorm=Norm,VisionAttention=Attention,
 get_parallel=lambda:state,add_prefix=lambda a,b:a,swiglu_clamped=lambda x,y:x,
 glm4v_vision_reduces_over_attn_tp=lambda flag:state.dp and not flag)
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<real-vision-definitions>','exec'),scope)
results=[]
for tp,attn,dp,use_encoder_dp in [(8,1,True,False),(8,2,True,False),(8,8,False,False),(8,1,True,True)]:
 state.tp_size=tp;state.attn_tp_size=attn;state.dp=dp
 block=scope['Glm4vVisionBlock'](16,32,8,use_data_parallel=use_encoder_dp)
 merger=scope['Glm4vPatchMerger'](16,32,use_data_parallel=use_encoder_dp)
 expected=1 if use_encoder_dp else attn
 owners=[block.attn.tp_size,block.mlp.gate_up_proj.tp_size,block.mlp.down_proj.tp_size,merger.gate_up_proj.tp_size,merger.down_proj.tp_size]
 reduce=dp and not use_encoder_dp
 flags=[block.attn.use_dp_attention_reduce,block.mlp.down_proj.use_dp_attention_reduce,merger.down_proj.use_dp_attention_reduce]
 ok=owners==[expected]*5 and flags==[reduce]*3
 results.append(dict(tp=tp,attn_tp=attn,dp=dp,encoder_dp=use_encoder_dp,owners=owners,reduce_flags=flags,passed=ok))
print(json.dumps(results,indent=2));sys.exit(0 if all(r['passed'] for r in results) else 1)
