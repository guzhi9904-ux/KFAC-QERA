"""Explicit placement for Llama weights, nonpersistent RoPE buffers and aliases."""
import torch


def model_device_map(layers=32, first=0, second=1):
    mapping = {'model.embed_tokens': first, 'model.rotary_emb': first,
               'model.norm': second, 'lm_head': second}
    mapping.update({f'model.layers.{i}': first if i < layers//2 else second for i in range(layers)})
    return mapping


def mapped_device(name, mapping):
    matches = [key for key in mapping if name == key or name.startswith(key+'.')]
    if not matches: raise RuntimeError('No device assignment for model tensor: '+name)
    value = mapping[max(matches, key=len)]
    return torch.device(f'cuda:{value}' if isinstance(value, int) else value)


def align_and_check_devices(model, mapping):
    rotary = {}
    for name, module in model.named_modules():
        if not name.endswith('rotary_emb'): continue
        device = mapped_device(name, mapping)
        before = {}
        for attr in ('inv_freq', 'original_inv_freq'):
            value = getattr(module, attr, None)
            if isinstance(value, torch.Tensor):
                if value.is_meta: raise RuntimeError('Unmaterialized RoPE tensor: '+name+'.'+attr)
                before[attr] = value.detach().cpu().clone()
        if 'inv_freq' not in before: raise RuntimeError('Missing RoPE inv_freq: '+name)
        module.to(device=device)  # No dtype conversion or frequency reconstruction.
        original = getattr(module, 'original_inv_freq', None)
        if isinstance(original, torch.Tensor):
            # This plain Tensor alias is not included in named_buffers/.to().
            module.original_inv_freq = original.to(device=device)
        for attr, old in before.items():
            value = getattr(module, attr)
            if value.device != device or value.dtype != old.dtype or not torch.equal(value.cpu(), old):
                raise RuntimeError('RoPE placement/value preservation failed: '+name+'.'+attr)
        rotary[name] = dict(device=str(device), dtype=str(module.inv_freq.dtype),
                            attributes_checked=list(before), exact_values_preserved=True)
    if 'model.rotary_emb' not in rotary: raise RuntimeError('Shared Llama RoPE module missing')
    count = 0
    for name, tensor in list(model.named_parameters())+list(model.named_buffers()):
        expected = mapped_device(name, mapping)
        if tensor.is_meta or tensor.device != expected:
            raise RuntimeError(f'Model tensor placement mismatch: {name}: {tensor.device}, expected {expected}')
        count += 1
    return dict(passed=True, tensors_checked=count, rotary=rotary,
                includes_nonpersistent_buffers=True, original_inv_freq_alias_checked=True)
