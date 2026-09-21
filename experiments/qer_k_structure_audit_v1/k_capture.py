import inspect
import torch
from k_common import LAYERS,MODULES,require,digest,mo
from v_capture import capture as v_capture,mapping_check
from shared_teacher import shared_capture
from bridge import hidden_forward
from k_math import compare,rotate,transpose_rotate


def capture(model,ids):
    """Observe actual native RoPE arguments/results, never reconstruct frequencies."""
    from transformers.models.llama import modeling_llama as llama
    original=llama.apply_rotary_pos_emb;active=[None];extra={};handles=[]
    def pre(layer):
        def hook(module,args,kw):
            require(active[0] is None,'Nested target attention')
            active[0]=layer
            extra[layer]=dict(position_ids=kw['position_ids'].detach().cpu().clone(),
                              attention_mask=kw['attention_mask'].detach().cpu().clone())
        return hook
    def post(module,args,out):active[0]=None
    def wrapped(q,k,cos,sin,*args,**kw):
        out=original(q,k,cos,sin,*args,**kw)
        if active[0] is not None:
            require(not args and kw.get('unsqueeze_dim',1)==1,'Unexpected RoPE layout')
            extra[active[0]].update(q=q[0].detach().cpu().clone(),k=k[0].detach().cpu().clone(),
                qt=out[0][0].detach().cpu().clone(),kt=out[1][0].detach().cpu().clone(),
                cos=cos[0].detach().cpu().clone(),sin=sin[0].detach().cpu().clone())
        return out
    try:
        for layer in LAYERS:
            att=model.model.layers[layer].self_attn
            handles.extend([att.register_forward_pre_hook(pre(layer),with_kwargs=True),att.register_forward_hook(post)])
        llama.apply_rotary_pos_emb=wrapped
        ref,states=v_capture(model,ids)
    finally:
        llama.apply_rotary_pos_emb=original
        for h in handles:h.remove()
    require(set(extra)==set(LAYERS),'Missing native RoPE capture')
    for i in LAYERS:states[i].update(extra[i])
    return ref,states


def mapping(model,state):
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb,rotate_half
    row=mapping_check(model)
    row.update(rope_source_hash=digest(inspect.getsource(apply_rotary_pos_emb)),rotate_half_source_hash=digest(inspect.getsource(rotate_half)))
    for n in MODULES:require(tuple(model.get_submodule(n).weight.shape)==(1024,4096),'K shape differs')
    for layer,s in state.items():
        require(torch.equal(s['position_ids'],torch.arange(2048).view(1,-1)),'Unexpected position IDs')
        compare(rotate(s['q'],s['cos'],s['sin']),s['qt'],1e-5)
        compare(rotate(s['k'],s['cos'],s['sin']),s['kt'],1e-5)
        x=s['k'].double().requires_grad_();seed=torch.ones_like(x)
        actual=torch.autograd.grad(rotate(x,s['cos'].double(),s['sin'].double()),x,seed)[0]
        compare(transpose_rotate(seed,s['cos'].double(),s['sin'].double()),actual,1e-10)
        mask=s['attention_mask'][0,0]
        require(torch.all(mask.triu(1)[torch.ones_like(mask,dtype=torch.bool).triu(1)]<0),'Noncausal mask')
    return row


def full_reference(teacher,ids,label,reference):
    names=[f'model.layers.{i}.{suffix}' for i in LAYERS for suffix in ('self_attn.k_proj','self_attn.o_proj','mlp.down_proj')]
    weights=[teacher.model.get_submodule(n).weight for n in MODULES]
    try:
        for w in weights:w.requires_grad_(True)
        with shared_capture(teacher.model,names,gradient=True,recompute=True) as states:
            h=hidden_forward(teacher.model,ids);compare(h,reference,1e-5)
            seed=mo.sampled_seed(h.detach(),teacher.model.lm_head.weight,label,teacher.config['vocab_chunk'])
            grads=torch.autograd.grad(h,[states[n]['h'] for n in names]+weights,seed)
            result={n:g.detach().cpu() for n,g in zip(names,grads[:len(names)])}
            result.update({n+'.weight':g.detach().cpu() for n,g in zip(MODULES,grads[len(names):])})
            return result
    finally:
        for w in weights:w.requires_grad_(False)
