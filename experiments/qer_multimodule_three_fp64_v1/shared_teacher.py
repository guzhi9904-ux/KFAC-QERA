"""One connected autograd graph provides all registered module output gradients."""
import contextlib
from functools import wraps
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from bridge import Teacher,hidden_forward,mo,require


@contextlib.contextmanager
def shared_capture(model,names,gradient=False,recompute=True):
    states={};handles=[];originals=[];phase={'recompute':0}
    @contextlib.contextmanager
    def replay_context():
        phase['recompute']+=1
        try:yield
        finally:phase['recompute']-=1
    def hook(name):
        def record(module,args,output):
            if phase['recompute']:return
            require(name not in states,'Repeated initial target forward: '+name)
            # Never detach an intermediate output: later targets must not sever
            # the path carrying gradients to earlier targets in the same graph.
            states[name]={'x':args[0].detach(),'h':output}
        return record
    try:
        for name in names:handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
        if gradient:
            def make_leaf(module,args,output):return output.detach().requires_grad_(True)
            handles.append(model.model.embed_tokens.register_forward_hook(make_leaf))
            if recompute:
                for layer in model.model.layers:
                    original=layer.forward
                    @wraps(original)
                    def wrapped(*args,_forward=original,**kwargs):
                        return checkpoint(_forward,*args,use_reentrant=False,preserve_rng_state=True,
                            context_fn=lambda:(contextlib.nullcontext(),replay_context()),**kwargs)
                    originals.append((layer,original));layer.forward=wrapped
        yield states
        require(set(states)==set(names),'Incomplete shared target capture')
    finally:
        for handle in handles:handle.remove()
        for layer,original in originals:layer.forward=original


def shared_gradient(model,names,ids,seed_fn,reference=None,reference_x=None,recompute=True):
    with shared_capture(model,names,gradient=True,recompute=recompute) as states:
        hidden=hidden_forward(model,ids)
        if reference is not None:require(mo.relative(hidden.detach(),reference)<=1e-7,'Shared checkpoint replay differs')
        if reference_x is not None:
            for name in names:require(torch.equal(states[name]['x'].detach().cpu(),reference_x[name]),'Shared input replay differs: '+name)
        initial=seed_fn(hidden.detach())
        gs=torch.autograd.grad(hidden,tuple(states[n]['h'] for n in names),grad_outputs=initial)
        result={name:dict(x=states[name]['x'].detach().cpu(),g=g.detach().reshape(ids.shape[1],-1).cpu()) for name,g in zip(names,gs)}
    require(all(torch.isfinite(t).all() for row in result.values() for t in row.values()),'Nonfinite shared gradient')
    return result


class SharedTeacher(Teacher):
    def reference_all(self,ids,names):
        self.load()
        with torch.no_grad(),shared_capture(self.model,names) as states:
            ref=hidden_forward(self.model,ids).detach()
            xs={name:states[name]['x'].cpu() for name in names}
        return ref,xs

    def gradient_all(self,ids,names,reference,xs,label):
        return shared_gradient(self.model,names,ids,
            lambda h:mo.sampled_seed(h,self.model.lm_head.weight,label,self.config['vocab_chunk']),reference,xs)

    def reference_logits(self,hidden):
        # FP32 logits preserve the exact stable_KL path while saving repeated
        # teacher LM-head GEMMs. About1GiB host memory for Llama128k,L2048.
        with torch.no_grad():
            return [F.linear(hidden[0,s:min(hidden.shape[1]-1,s+self.config['vocab_chunk'])],self.model.lm_head.weight).cpu()
                for s in range(0,hidden.shape[1]-1,self.config['vocab_chunk'])]

    def scores(self,ids,reference_logits):
        with torch.no_grad():
            actual=hidden_forward(self.model,ids);kl=[];nll=[];start=0
            for reference in reference_logits:
                end=start+len(reference)
                logits=F.linear(actual[0,start:end],self.model.lm_head.weight)
                kl.append(mo.stable_kl(reference.to(logits.device),logits).cpu())
                lp=logits.double().log_softmax(-1);labels=ids[0,start+1:end+1].to(lp.device)
                nll.append(-lp.gather(1,labels[:,None]).flatten().cpu());start=end
        return dict(KL=float(torch.cat(kl).mean()),NLL_sum=float(torch.cat(nll).sum()),tokens=start)

    @contextlib.contextmanager
    def deploy(self,weights):
        originals={};hashes={}
        try:
            for name,w in weights.items():
                p=self.model.get_submodule(name).weight;originals[name]=p.detach().cpu().clone();hashes[name]=mo.digest_tensor(originals[name])
                with torch.no_grad():p.copy_(w.to(p.device))
                require(mo.digest_tensor(p)==mo.digest_tensor(w),'Deployment hash differs')
            yield
        finally:
            for name,w in originals.items():
                p=self.model.get_submodule(name).weight
                with torch.no_grad():p.copy_(w.to(p.device))
                require(mo.digest_tensor(p)==hashes[name],'Teacher restoration failed')
