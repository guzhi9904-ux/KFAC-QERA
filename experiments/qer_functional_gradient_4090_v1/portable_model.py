"""Two-device teacher and explicit cross-hardware replay gates."""
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time
from config_identity import configuration_differences, json_snapshot
from device_layout import model_device_map, align_and_check_devices
from bridge import (CONFIG, EXP01, EXP03, PLAN, OFFICIAL, read, torch, mo, sha_file,
                    save_json, capture, hidden_forward, read_tensors, slug, clean)


class PortableModel:
    def doctor(self):
        assert os.name == 'posix' and torch.cuda.device_count() == 2, 'Exactly two visible GPUs required'
        for i in range(2):
            p = torch.cuda.get_device_properties(i)
            assert '4090' in p.name and p.total_memory >= 23*2**30, p
        assert Path(self.config['model']).is_dir()
        for rel in OFFICIAL: self.official(rel)
        old = read(EXP03/'environment.json')
        packages = {n: importlib.metadata.version(n) for n in old['packages']}
        assert packages == old['packages'], ('Dependency versions differ', packages, old['packages'])
        assert torch.__version__.split('+')[0] == old['torch'].split('+')[0]
        save_json(self.root/'environment.json', dict(python=sys.version, platform=platform.platform(),
            torch=torch.__version__, cuda=torch.version.cuda, packages=packages,
            parent_torch=old['torch'], parent_python=old['python'], parent_cuda=old['cuda'],
            same_package_versions=True, dtype='float32', attention='eager', tf32=False,
            device_count=2, device_names=[torch.cuda.get_device_name(i) for i in range(2)],
            identity=self.identity, time=time.time(), execution='one process; layer model parallel'))

    def load_model(self):
        if self.model is not None: return
        from transformers import AutoModelForCausalLM
        mapping = model_device_map()
        self.device_map = mapping
        with self.timed('load_model', device_map=mapping):
            self.model = AutoModelForCausalLM.from_pretrained(self.config['model'], torch_dtype=torch.float32,
                attn_implementation='eager', local_files_only=True, trust_remote_code=False,
                low_cpu_mem_usage=True, device_map=mapping)
            self.model.eval(); self.model.requires_grad_(False); self.model.config.use_cache = False
            placement = align_and_check_devices(self.model, mapping)
            save_json(self.root/'device_placement.json', dict(identity=self.identity, **placement))
            print('DEVICE CHECK PASS: shared RoPE on cuda:0; all parameters and buffers checked', flush=True)
        expected = read(EXP03/'teacher_identity.json')
        # Compare all behavior-relevant config fields, allowing provenance/version/cache metadata only.
        actual_config = json_snapshot(self.model.config.to_dict())
        differences = configuration_differences(expected['config'], actual_config)
        assert not differences, ('Model configuration mismatch', differences)
        with self.timed('teacher_tensor_identity'):
            hashes = {}
            for name, p in self.model.named_parameters():
                assert p.dtype == torch.float32 and not p.requires_grad
                hashes[name] = {'shape': list(p.shape), 'dtype': str(p.dtype), 'hash': mo.digest_tensor(p)}
                assert hashes[name] == expected['tensor_hashes'].get(name), ('Teacher parameter differs', name)
            assert set(hashes) == set(expected['tensor_hashes'])
        manifest = Path(self.config['model'])/'DOWNLOAD_MANIFEST.json'
        record = dict(tensor_hashes=hashes, config=actual_config, device_map=mapping,
            checkpoint_manifest_hash=sha_file(manifest) if manifest.is_file() else None,
            parent_checkpoint_manifest_hash=expected['checkpoint_manifest_hash'],
            exact_FP32_parameter_identity=True, config_verified=True,
            provenance='Absent download manifest allowed only with exact verification of every FP32 parameter; not fabricated',
            parent_device_map=expected['device_map'], device_mapping_changed_explicitly=True)
        path = self.root/'teacher_identity.json'
        if path.exists(): assert read(path) == record
        else: save_json(path, record)
        assert self.model.get_submodule(self.name).weight.device == torch.device('cuda', self.compute_device)

    def unload(self):
        self.model = None
        # Release cached blocks on BOTH devices before full-channel linear algebra.
        import gc
        gc.collect()
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i): torch.cuda.empty_cache()
        torch.cuda.set_device(self.compute_device)

    def replay_parent(self, c, reference, x):
        """First old label in each old window: fixed tolerances, never new sampling."""
        from safetensors import safe_open
        path = self.root/'portability'/f'w{c:02d}.json'
        if path.exists():
            row = read(path)
            assert row['identity'] == self.identity and row['passed']
            assert row['target_x_hash'] == mo.digest_tensor(x)
            assert row['target_hidden_hash'] == mo.digest_tensor(reference)
            return
        source_x = EXP03/'cache'/slug(self.name)/f'x_w{c:02d}.safetensors'
        old, meta = read_tensors(source_x)
        assert mo.digest_tensor(old['x']) == meta['input_hash']
        assert meta['token_hash'] == mo.digest_tensor(self.fit['input_ids'][c])
        x_error = mo.relative(x.cpu(), old['x'])
        assert x_error <= PLAN['portability']['parent_input_relative_tolerance'], ('Parent input replay', x_error)
        labels, lm = self.old_label(c, 0)
        with self.timed('cross_hardware_parent_S_replay', module=self.name, window=c):
            g, audit = self.gradient(self.name, self.fit['input_ids'][c:c+1], reference, x, labels, weight_check=(c == 0))
            s = g.T@x.reshape(2048, -1).double()
            old_row = next(r for r in self.s_manifest() if r['window'] == c and r['replicate'] == 0)
            assert sha_file(old_row['path']) == old_row['file_sha256']
            with safe_open(old_row['path'], framework='pt') as f: old_s = f.get_tensor('S')
            assert mo.digest_tensor(old_s) == old_row['S_hash'] and old_row['label_hash'] == lm['label_hash']
            s_error = mo.relative(s.cpu(), old_s)
            assert s_error <= PLAN['portability']['parent_S_relative_tolerance'], ('Parent S replay', s_error)
        save_json(path, dict(passed=True, identity=self.identity, module=self.name, window=c,
            label_hash=lm['label_hash'], parent_x_file_sha256=sha_file(source_x), parent_S_hash=old_row['S_hash'],
            x_relative_error=x_error, S_relative_error=s_error, gradient_audit=audit,
            target_x_hash=mo.digest_tensor(x), target_hidden_hash=mo.digest_tensor(reference),
            parent_hidden_hash=meta['teacher_hidden_hash'],
            hidden_bitwise_equal=mo.digest_tensor(reference) == meta['teacher_hidden_hash'],
            acceptance='Exact teacher parameters, numerical parent x/S replay; source S and original labels unchanged'))
        del g, s, old_s, old
        clean()
