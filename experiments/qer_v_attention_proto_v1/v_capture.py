import inspect
import torch
import torch.nn.functional as F
from v_common import LAYERS,MODULES,require,digest
from bridge import hidden_forward
from shared_teacher import shared_capture
from v_math import compare


def mapping_check(model):
    from transformers.models.llama.modeling_llama import repeat_kv,LlamaAttention
    c=model.config;require(c.hidden_size==4096 and c.num_attention_heads==32 and c.num_key_value_heads==8,'Unexpected GQA config')
    require(c.pretraining_tp==1 and c._attn_implementation=='eager','Wrong tensor parallel/attention backend')
    marker=torch.arange(c.num_key_value_heads).view(1,-1,1,1)
    mapping=repeat_kv(marker,c.num_attention_heads//c.num_key_value_heads).reshape(-1).tolist()
    require(mapping==[a//4 for a in range(32)],'Unexpected repeat_kv head order')
    for n in MODULES:require(tuple(model.get_submodule(n).weight.shape)==(1024,4096),'Wrong V shape')
    return dict(query_heads=32,kv_heads=8,head_dim=128,input_dim=4096,output_dim=1024,mapping=mapping,
                repeat_kv_source_hash=digest(inspect.getsource(repeat_kv)),attention_source_hash=digest(inspect.getsource(LlamaAttention.forward)))


def capture(model,ids,layers=LAYERS):
    states={i:{} for i in layers};handles=[]
    def force(module,args,kw):
        require(not args and kw.get('past_key_value') is None and not kw.get('use_cache',False),'Unexpected attention call convention/cache')
        return args,dict(kw,output_attentions=True)
    def attention(i):
        def hook(module,args,out):
            p=out[1];require(p is not None and p.dtype==torch.float32,'Missing native eager probability')
            states[i]['prob']=p[0].detach().cpu().clone()
        return hook
    def inp(i,key):
        def hook(module,args):states[i][key]=args[0].detach().cpu().clone()
        return hook
    def val(i):
        def hook(module,args,out):states[i]['v']=out.detach().cpu().clone()
        return hook
    def mlp(i):
        def hook(module,args,out):states[i]['mlp_output']=out.detach().cpu().clone()
        return hook
    try:
        for i in layers:
            block=model.model.layers[i];att=block.self_attn
            handles += [att.register_forward_pre_hook(force,with_kwargs=True),att.register_forward_hook(attention(i)),
                        att.v_proj.register_forward_pre_hook(inp(i,'x')),att.v_proj.register_forward_hook(val(i)),
                        att.o_proj.register_forward_pre_hook(inp(i,'z')),block.post_attention_layernorm.register_forward_pre_hook(inp(i,'r')),
                        block.mlp.register_forward_hook(mlp(i))]
        with torch.no_grad():ref=hidden_forward(model,ids).detach()
    finally:
        for h in handles:h.remove()
    return ref,states


def local_delta(model,layer,r,seed,expected=None):
    block=model.model.layers[layer];device=block.mlp.down_proj.weight.device
    r=r.to(device).detach().requires_grad_(True);seed=seed.to(device).reshape_as(r)
    out=block.mlp(block.post_attention_layernorm(r))
    if expected is not None:compare(out,expected,1e-5)
    extra=torch.autograd.grad(out,r,seed)[0];lam=seed+extra
    delta=F.linear(lam,block.self_attn.o_proj.weight.T)
    return lam.detach().cpu(),delta.detach().cpu()


def full_reference(teacher,ids,label,ref):
    names=[f'model.layers.{i}.{suffix}' for i in LAYERS for suffix in ('self_attn.v_proj','self_attn.o_proj','mlp.down_proj')]
    live_inputs={};handles=[];weights=[teacher.model.get_submodule(n).weight for n in MODULES]
    def hook(name):
        def save(module,args,out):
            if name not in live_inputs:live_inputs[name]=args[0]
        return save
    try:
        for w in weights:w.requires_grad_(True)
        for i in LAYERS:
            n=f'model.layers.{i}.self_attn.o_proj';handles.append(teacher.model.get_submodule(n).register_forward_hook(hook(n)))
        with shared_capture(teacher.model,names,gradient=True,recompute=True) as states:
            h=hidden_forward(teacher.model,ids);compare(h,ref,1e-5)
            seed=__import__('bridge').mo.sampled_seed(h.detach(),teacher.model.lm_head.weight,label,teacher.config['vocab_chunk'])
            inputs=[states[n]['h'] for n in names]+weights+[live_inputs[f'model.layers.{i}.self_attn.o_proj'] for i in LAYERS]
            grads=torch.autograd.grad(h,inputs,seed)
            result={n:g.detach().cpu() for n,g in zip(names,grads[:len(names)])}
            result.update({n+'.weight':g.detach().cpu() for n,g in zip(MODULES,grads[len(names):len(names)+4])})
            result.update({f'delta_{i}':g.detach().cpu() for i,g in zip(LAYERS,grads[-4:])})
            return result
    finally:
        for h in handles:h.remove()
        for w in weights:w.requires_grad_(False)


def check_aggregation(state,mapping,head_dim,device):
    prob=state['prob'].to(device);v=state['v'][0].to(device).reshape(-1,max(mapping)+1,head_dim)
    repeated=v.permute(1,0,2)[mapping]
    z=(prob@repeated).transpose(0,1).reshape_as(state['z'][0])
    return compare(z,state['z'][0].to(device),1e-5)
