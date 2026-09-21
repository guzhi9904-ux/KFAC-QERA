"""Replay actual captured decoder arguments; preserve residual and MLP paths."""
import torch
import torch.nn.functional as F
from ko_common import mo,require,LAYERS
from bridge import hidden_forward
from shared_teacher import SharedTeacher


def tree(value,device=None):
    if isinstance(value,torch.Tensor):return value.detach().to('cpu' if device is None else device).clone()
    if isinstance(value,tuple):return tuple(tree(x,device) for x in value)
    if isinstance(value,list):return [tree(x,device) for x in value]
    if isinstance(value,dict):return {k:tree(v,device) for k,v in value.items()}
    require(value is None or isinstance(value,(bool,int,float,str)),'Unsupported cached decoder argument')
    return value


def first(output):return output[0] if isinstance(output,tuple) else output


def capture_blocks(model,ids,layers=LAYERS):
    states={};handles=[]
    def pre(i):
        def hook(module,args,kwargs):
            require(i not in states,'Repeated block capture')
            require(kwargs.get('past_key_value') is None and not kwargs.get('use_cache',False),'KV cache is forbidden')
            states[i]={'args':tree(args),'kwargs':tree(kwargs)}
        return hook
    def post(i):
        def hook(module,args,output):states[i]['output']=tree(first(output))
        return hook
    try:
        for i in layers:
            block=model.model.layers[i]
            handles += [block.register_forward_pre_hook(pre(i),with_kwargs=True),block.register_forward_hook(post(i))]
        with torch.no_grad():ref=hidden_forward(model,ids).detach()
    finally:
        for h in handles:h.remove()
    require(set(states)==set(layers),'Missing captured layer')
    return ref,states


def arguments(state,device,gradient=False):
    args=list(tree(state['args'],device));kw=tree(state['kwargs'],device)
    if args:args[0]=args[0].requires_grad_(gradient)
    else:kw['hidden_states']=kw['hidden_states'].requires_grad_(gradient)
    return args,kw


def local_gradient(model,layer,state,seed,names):
    block=model.model.layers[layer];device=next(block.parameters()).device
    args,kw=arguments(state,device,True);values={};handles=[]
    def hook(name):
        def record(module,args,out):values[name]=dict(x=args[0].detach(),h=out)
        return record
    try:
        for name in names:handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
        out=first(block(*args,**kw));error=mo.relative(out.detach().cpu(),state['output'])
        require(error<=1e-7,'Local block output replay differs')
        gs=torch.autograd.grad(out,tuple(values[n]['h'] for n in names),seed.to(device).reshape_as(out))
        result={n:dict(x=values[n]['x'].cpu(),g=g.detach().reshape(out.shape[1],-1).cpu()) for n,g in zip(names,gs)}
        require(all(torch.isfinite(t).all() for v in result.values() for t in v.values()),'Nonfinite local gradient')
        return result,error
    finally:
        for h in handles:h.remove()


def suffix_hidden(model,layer,state):
    block=model.model.layers[layer];args,kw=arguments(state,next(block.parameters()).device)
    hidden=first(block(*args,**kw))
    for block in model.model.layers[layer+1:]:
        device=next(block.parameters()).device
        kw=tree(state['kwargs'],device);kw.pop('hidden_states',None)
        hidden=first(block(hidden.to(device),**kw))
    norm=model.model.norm;hidden=norm(hidden.to(norm.weight.device))
    return hidden.to(model.lm_head.weight.device)


class ReplayTeacher(SharedTeacher):
    def scores_hidden(self,ids,reference_logits,actual):
        kl=[];nll=[];start=0
        with torch.no_grad():
            for reference in reference_logits:
                end=start+len(reference);logits=F.linear(actual[0,start:end],self.model.lm_head.weight)
                kl.append(mo.stable_kl(reference.to(logits.device),logits).cpu())
                labels=ids[0,start+1:end+1].to(logits.device)
                nll.append(-logits.double().log_softmax(-1).gather(1,labels[:,None]).flatten().cpu());start=end
        return dict(KL=float(torch.cat(kl).mean()),NLL_sum=float(torch.cat(nll).sum()),tokens=start)
    def suffix_scores(self,ids,logits,layer,state):
        with torch.no_grad():return self.scores_hidden(ids,logits,suffix_hidden(self.model,layer,state))
