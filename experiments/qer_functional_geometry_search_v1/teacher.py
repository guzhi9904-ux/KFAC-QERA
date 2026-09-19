"""FP32 frozen teacher, one shared backprop per label for every candidate."""
import gc
import importlib.metadata
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from common import PLAN, HERE, read, save_json, mo, require, capture, recompute_suffix, hidden_forward

sys.path.insert(0, str(HERE.parent/'qer_functional_gradient_4090_v1'))
from config_identity import configuration_differences
from device_layout import model_device_map, align_and_check_devices
sys.path.insert(0, str(HERE))


class Teacher:
    def __init__(self, config, root, identity, timed):
        self.config = config; self.root = root; self.identity = identity; self.timed = timed
        self.model = None

    def load(self):
        if self.model is not None: return
        from transformers import AutoModelForCausalLM
        old = read(Path(self.config['assets'])/'exp03/environment.json')
        packages = {n: importlib.metadata.version(n) for n in old['packages']}
        require(packages == old['packages'], 'Parent package versions differ')
        require(torch.__version__.split('+')[0] == old['torch'].split('+')[0], 'Parent torch version differs')
        n = 1 if self.config['profile'] == 'a6000' else 2
        require(torch.cuda.device_count() == n, f'Exactly {n} visible GPUs required')
        names = [torch.cuda.get_device_name(i) for i in range(n)]
        require(all(('A6000' if n == 1 else '4090') in name for name in names), 'GPU profile mismatch')
        require(not torch.is_autocast_enabled(), 'Autocast must be disabled')
        mapping = model_device_map(32, 0, 0 if n == 1 else 1)
        torch.set_num_threads(self.config['cpu_threads'])
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        with self.timed('load_model'):
            self.model = AutoModelForCausalLM.from_pretrained(self.config['model'], torch_dtype=torch.float32,
                attn_implementation='eager', local_files_only=True, trust_remote_code=False,
                low_cpu_mem_usage=True, device_map=mapping)
            self.model.eval().requires_grad_(False); self.model.config.use_cache = False
            placement = align_and_check_devices(self.model, mapping)
        expected = read(Path(self.config['assets'])/'exp03/teacher_identity.json')
        require(not configuration_differences(expected['config'], self.model.config.to_dict()), 'Teacher config mismatch')
        with self.timed('teacher_parameter_identity'):
            names_seen = set()
            for name, p in self.model.named_parameters():
                require(p.dtype == torch.float32 and not p.requires_grad, 'Teacher precision/freeze mismatch')
                actual = dict(shape=list(p.shape), dtype=str(p.dtype), hash=mo.digest_tensor(p))
                require(actual == expected['tensor_hashes'].get(name), 'Teacher parameter differs: '+name)
                names_seen.add(name)
            require(names_seen == set(expected['tensor_hashes']), 'Teacher parameter set differs')
        require(not any(m.training for m in self.model.modules()), 'Teacher must be entirely in evaluation mode')
        save_json(self.root/'teacher_verification.json', dict(identity=self.identity, passed=True,
            exact_FP32_parameters=True, config_verified=True, packages=packages, torch=torch.__version__,
            cuda=torch.version.cuda, devices=names, device_map=mapping, placement=placement,
            autocast=False, tf32=False, dropout=False, attention='eager'))

    def unload(self):
        self.model = None; gc.collect()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i): torch.cuda.empty_cache()

    def reference(self, name, ids):
        self.load()
        with torch.no_grad(), capture(self.model.get_submodule(name)) as state:
            hidden = hidden_forward(self.model, ids).detach()
        return hidden, state['x'].detach(), state['h'].detach()

    def labels(self, reference, seeds):
        weight = self.model.lm_head.weight
        generators = [torch.Generator(device=weight.device).manual_seed(s) for s in seeds]
        labels = [[] for _ in seeds]; T = reference.shape[1]-1
        with torch.no_grad():
            for start in range(0, T, self.config['vocab_chunk']):
                prob = F.linear(reference[0, start:min(T, start+self.config['vocab_chunk'])], weight).double().softmax(-1)
                cdf = prob.cumsum(-1); cdf[:, -1] = 1
                for i, generator in enumerate(generators):
                    u = torch.rand((len(prob), 1), dtype=torch.float64, device=weight.device, generator=generator)
                    labels[i].append(torch.searchsorted(cdf, u).flatten().cpu())
        return [torch.cat(parts) for parts in labels]

    def gradient(self, name, ids, reference, x, labels, audit=False):
        target = self.model.get_submodule(name); target.weight.requires_grad_(audit)
        try:
            with recompute_suffix(self.model, int(name.split('.')[2]), True), capture(target, not audit) as state:
                hidden = hidden_forward(self.model, ids)
                require(torch.equal(state['x'], x), 'Gradient input replay differs')
                error = mo.relative(hidden.detach(), reference)
                require(error <= 1e-7, 'Checkpoint hidden replay differs')
                initial = mo.sampled_seed(hidden.detach(), self.model.lm_head.weight, labels, self.config['vocab_chunk'])
                inputs = (state['h'], target.weight) if audit else (state['h'],)
                gradients = torch.autograd.grad(hidden, inputs, grad_outputs=initial)
            g = gradients[0].detach().reshape(ids.shape[1], -1).double()
            checks = dict(hidden_relative_error=error, sum_NLL=True, positions=ids.shape[1]-1)
            if audit:
                s = g.T@x.reshape(ids.shape[1], -1).double()
                delta = mo.relative(s, gradients[1].double())
                require(delta <= PLAN['tolerances']['autograd'], 'S/shared-weight autograd disagreement')
                checks['S_autograd_relative_error'] = delta
            return g, checks
        finally:
            target.weight.requires_grad_(False)

    def kl(self, reference, actual):
        parts = []
        with torch.no_grad():
            for start in range(0, reference.shape[1]-1, self.config['vocab_chunk']):
                end = min(reference.shape[1]-1, start+self.config['vocab_chunk'])
                zr = F.linear(reference[0, start:end], self.model.lm_head.weight)
                za = F.linear(actual[0, start:end], self.model.lm_head.weight)
                parts.append(mo.stable_kl(zr, za).cpu())
        return float(torch.cat(parts).mean())

    def intervention_kl(self, name, ids, reference, x, h, weight, residual):
        target = self.model.get_submodule(name); original = target.weight.detach().clone()
        original_hash = mo.digest_tensor(original)
        try:
            with torch.no_grad(): target.weight.copy_(weight.to(target.weight.device))
            require(mo.digest_tensor(target.weight) == mo.digest_tensor(weight), 'Deployment hash differs')
            with torch.no_grad(), capture(target) as state:
                actual = hidden_forward(self.model, ids)
            require(torch.equal(state['x'], x), 'Intervention changed target input')
            expected = -x.double()@residual.to(x.device).T
            check = mo.comparison(state['h'].double()-h.double(), expected)
            require(check['relative_l2'] <= .02 and check['cosine'] >= .999, 'Intervention output path differs')
            value = self.kl(reference, actual)
            return value, check
        finally:
            with torch.no_grad(): target.weight.copy_(original)
            require(mo.digest_tensor(target.weight) == original_hash, 'Teacher restoration failed')
