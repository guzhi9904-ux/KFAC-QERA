"""FP32 frozen-teacher operations, including explicit eval-mode recomputation."""
import contextlib
from functools import wraps

import torch
from torch.utils.checkpoint import checkpoint


@contextlib.contextmanager
def capture(module, gradient=False):
    state={}
    def hook(_module,args,output):
        if state:raise RuntimeError("Target hook executed twice in a forward")
        state["x"]=args[0].detach()
        if gradient:
            output=output.detach().requires_grad_(True)
        state["h"]=output
        return output
    handle=module.register_forward_hook(hook)
    try:
        yield state
        if not state:raise RuntimeError("Target hook did not execute")
    finally:handle.remove()


@contextlib.contextmanager
def recompute_suffix(model, target_layer, enabled):
    """Only layers AFTER the target: never recompute the detached target hook."""
    originals=[]
    if enabled:
        for layer in model.model.layers[target_layer+1:]:
            original=layer.forward
            @wraps(original)
            def wrapped(*args,_forward=original,**kwargs):
                return checkpoint(_forward,*args,use_reentrant=False,preserve_rng_state=True,**kwargs)
            originals.append((layer,original));layer.forward=wrapped
    try:yield
    finally:
        for layer,original in originals:layer.forward=original


def hidden_forward(model, ids):
    return model.model(input_ids=ids.to(model.model.embed_tokens.weight.device),
                       attention_mask=torch.ones_like(ids,device=model.model.embed_tokens.weight.device),
                       use_cache=False,return_dict=True).last_hidden_state
