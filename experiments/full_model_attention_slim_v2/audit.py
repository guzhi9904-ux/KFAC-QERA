"""Read-only parent/resource inspection and exact complete-weight quantization."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import torch
from safetensors import safe_open
from fm_common import *

def audit(ctx):
    cfg = ctx.config
    require((cfg['n'], cfg['length'], cfg['rank'], cfg['eta_A'], cfg['eta_G']) == (256, 2048, 64, .001, .001), 'Scientific configuration changed')
    require(cfg['personal_free_GiB_minimum'] >= 200 and cfg['quota_evidence'], 'Personal quota is unverified')
    require(torch.cuda.device_count() == 2 and all('4090' in torch.cuda.get_device_name(i) for i in range(2)), 'Expected dual 4090')
    model_cfg = read(Path(cfg['model']) / 'config.json')
    expected = dict(hidden_size=4096, intermediate_size=14336, num_hidden_layers=32,
                    num_attention_heads=32, num_key_value_heads=8)
    require(all(model_cfg[k] == v for k, v in expected.items()), 'Model dimensions differ')
    from harness_word_ppl import verify_harness, HARNESS_COMMIT
    harness = verify_harness(cfg['harness'])
    rows = {}
    for role in ('calibration', 'wikitext2', 'validation'):
        p = Path(cfg[role])
        with safe_open(str(p), framework='pt') as f:
            rows[role] = dict(path=str(p), sha256=sha(p), shapes={k: f.get_slice(k).get_shape() for k in f.keys()})
        if role != 'validation':
            meta = read(p.with_suffix('.json'))
            require(meta['sha256'] == rows[role]['sha256'] and meta['model_path'] == cfg['model'], 'Historical tokens provenance differs')
            rows[role]['metadata'] = meta
    rows['old_statistics'] = dict(reused=False, reasons=['Recent K/V statistics are WikiText-2, not SlimPajama',
          'Historical original-A uses FP32 solve/accumulation and cannot meet the new FP64 contract'])
    rows['teacher'] = dict(path=cfg['model'], parent_identity_sha256=sha(Path(cfg['assets'])/'exp03/teacher_identity.json'),
                           dimensions=expected, FP32_tensor_verification='required during prepare-data')
    rows['storage'] = dict(personal_minimum_free_GiB=cfg['personal_free_GiB_minimum'], evidence=cfg['quota_evidence'],
                           df_is_not_quota=True, planned_peak_GiB=160)
    rows['resources'] = dict(gpus=[torch.cuda.get_device_name(i) for i in range(2)],
        cgroup_memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
        cgroup_memory_current=Path('/sys/fs/cgroup/memory.current').read_text().strip(),
        gpu_query=subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv'], text=True))
    rows['packages'] = {k: importlib.metadata.version(k) for k in ('torch','transformers','datasets','safetensors','numpy','accelerate','lm_eval')}
    rows['harness'] = dict(path=str(harness), commit=HARNESS_COMMIT)
    rows['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip()
    write(ctx.root/'audit/parent_assets_audit.json', rows)
    (ctx.root/'audit/reuse_plan.md').write_text(
        '# Reuse plan\n\nCalibration: frozen SlimPajama 256×2048. WT2 test: all 138 historical windows (verified actual shape). '
        'KL: 16 frozen development validation windows. Teacher and existing MXINT3 Wq are hash checked. '
        'All ordinary A, predictive labels, attention G, K one-step and V attention-aware statistics are new. '
        'For every layer 0–31, Q/K/V share ordinary A; gate/up share ordinary A; O/down are separate. '
        'No historical WT2 factors, CE gradients, BF16 model scores or previous candidate scores are reused.\n', encoding='utf-8')
    ctx.commit('audit/complete.json', [ctx.root/'audit/parent_assets_audit.json', ctx.root/'audit/reuse_plan.md'])

def quantizer(ctx):
    p = Path(ctx.config['assets'])/'vendor/quantize/quantizers/mxint.py'
    if not p.exists():
        p = Path(ctx.config['assets'])/'vendor/src/qera/quantize/quantizers/mxint.py'
    from assets import verify_quantizer_source
    from safetensors import safe_open
    freeze = read(Path(ctx.config['ko_run'])/'quantized/freeze.json')
    first = Path(next(iter(freeze['sources'])))
    with safe_open(str(first), framework='pt') as f:
        old = json.loads(f.metadata()['record'])
    require((old['width'], old['block_size'], old['block_axis']) == (3, 32, -1), 'Parent quantizer differs')
    verify_quantizer_source(p, old['quantizer_hash'])
    spec = importlib.util.spec_from_file_location('full_slim_frozen_mxint', p)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.mxint_quantizer, dict(path=str(p), sha256=sha(p), width=3, block_size=32, block_axis=-1)

def freeze_weights(ctx):
    if ctx.done('quantization_complete.json', modules=224):
        manifest = read(ctx.root/'quantization_manifest.json')
        for row in manifest['modules'].values():
            require(sha(row['path']) == row['file_sha256'], 'Frozen Wq file changed')
        return
    ctx.teacher.load(); model = ctx.teacher.model
    fn, info = quantizer(ctx)
    parent = Path(ctx.config['ko_run']); frozen = read(parent/'quantized/freeze.json')
    expected = read(Path(ctx.config['assets'])/'exp03/teacher_identity.json')
    rows = {}
    for i in range(32):
        for kind in KINDS:
            key = name(i, kind); w0 = model.get_submodule(key).weight.detach()
            w0hash = mo.digest_tensor(w0)
            require(w0hash == expected['tensor_hashes'][key+'.weight']['hash'], 'Teacher weight identity differs')
            rel = 'quantized/'+slug(key)+'.safetensors'; old = parent/rel
            if rel in frozen['files']:
                require(sha(old) == frozen['files'][rel], 'Parent Wq file changed')
                t = load_file(str(old)); require(mo.digest_tensor(t['W0']) == w0hash, 'Parent Wq teacher differs')
                wq = t['Wq']; path = old; reused = True
                # Verify quantizer equality for all reused targets, not merely its name.
                require(torch.equal(fn(w0, width=3, block_size=32, block_axis=-1).cpu(), wq), 'Parent Wq differs from frozen quantizer')
            else:
                path = ctx.root/'quantized'/(slug(key)+'.safetensors'); reused = False
                receipt = path.with_suffix('.json')
                if receipt.exists():
                    row = read(receipt)
                    require(row['identity'] == ctx.identity and row['W0_hash'] == w0hash and sha(path) == row['file_sha256'], 'Partial quantization changed')
                    wq = load_file(str(path))['Wq']
                else:
                    wq = fn(w0, width=3, block_size=32, block_axis=-1).cpu()
                    tensors(path, {'Wq': wq})
            row = dict(identity=ctx.identity, path=str(path), file_sha256=sha(path), W0_hash=w0hash,
                       Wq_hash=mo.digest_tensor(wq), shape=list(w0.shape), reused=reused, quantizer=info)
            if not reused:
                write(path.with_suffix('.json'), row)
            rows[key] = row
            del wq
        log('QUANTIZATION_LAYER', layer=i, modules=(i+1)*7)
    write(ctx.root/'quantization_manifest.json', dict(identity=ctx.identity, quantizer=info, modules=rows))
    write(ctx.root/'teacher_identity.json', expected)
    ctx.commit('quantization_complete.json', [ctx.root/'quantization_manifest.json', ctx.root/'teacher_identity.json'], modules=224)

def original_weight(ctx, key, device='cpu'):
    """Read a single BF16 checkpoint tensor and reproduce its exact FP32 teacher value."""
    root = Path(ctx.config['model'])
    index = read(root/'model.safetensors.index.json')
    with safe_open(str(root/index['weight_map'][key+'.weight']), framework='pt') as f:
        value = f.get_tensor(key+'.weight').float()
    expected = read(ctx.root/'teacher_identity.json')['tensor_hashes'][key+'.weight']['hash']
    require(mo.digest_tensor(value) == expected, 'Checkpoint-to-FP32 identity differs')
    return value.to(device)
