"""Carry only verified preparation into a new source identity after a failed pilot."""
import shutil
from fm_common import *
from data import checked_ids, freeze_tasks, label, tokenizer_identity


def checked_parent_file(root, relative, expected):
    path = (root / relative).resolve()
    require(root.resolve() in path.parents, 'Parent receipt escapes its run directory')
    require(sha(path) == expected, f'Preparation parent file changed: {relative}')
    return path


def parent_contract(ctx):
    parent = Path(ctx.config['preparation_parent'])
    require(parent.resolve() != ctx.root.resolve(), 'Recovery must use a new run directory')
    manifest = read(parent / 'manifest.json')
    identity = manifest['identity']
    require(identity == ctx.config['preparation_parent_identity'] and
            digest({k: v for k, v in manifest.items() if k != 'identity'}) == identity,
            'Preparation parent identity differs')
    operational = {'run', 'preparation_parent', 'preparation_parent_identity'}
    require({k: v for k, v in manifest['config'].items() if k not in operational} ==
            {k: v for k, v in ctx.config.items() if k not in operational},
            'Recovery changed scientific/data configuration')
    for path, expected in manifest['parents'].items():
        require(sha(path) == expected, f'Historical parent changed: {path}')
    current_source = read(ctx.root / 'manifest.json')['source']
    for path, expected in manifest['source'].items():
        if not path.startswith('experiments/full_model_attention_slim_v2/'):
            require(current_source.get(path) == expected, f'Borrowed scientific source changed: {path}')
    for relative in ('data/complete.json', 'quantization_complete.json'):
        record = read(parent / relative)
        require(record['identity'] == identity and record['passed'], 'Parent preparation incomplete')
        if relative == 'quantization_complete.json':
            require(record['modules'] == 224, 'Parent quantization incomplete')
        for filename, expected in record['files'].items():
            checked_parent_file(parent, filename, expected)
    require(not (parent / 'verification/pilot_complete.json').exists(),
            'This recovery path only accepts a failed, unaccepted pilot')
    require(not (parent / 'statistics').exists(), 'Preparation recovery cannot inherit production statistics')
    return parent, identity


def reuse_preparation(ctx):
    parent, identity = parent_contract(ctx)
    provenance = dict(parent=str(parent), parent_identity=identity, files={}, labels=[], Wq_modules=224,
                      statistics_reused=False, factors_reused=False, scores_reused=False)

    def copy(relative, expected=None):
        source = parent / relative
        source = checked_parent_file(parent, relative, expected or sha(source))
        target = ctx.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        require(sha(target) == sha(source), f'Preparation copy differs: {relative}')
        provenance['files'][relative] = sha(source)
        return target

    for relative, expected in read(parent / 'data/complete.json')['files'].items():
        copy(relative, expected)
    copy('teacher_identity.json')
    require(sha(ctx.root / 'teacher_identity.json') == sha(Path(ctx.config['assets']) / 'exp03/teacher_identity.json'),
            'Recovered teacher identity differs')
    dm = read(ctx.root / 'data_manifest.json')
    require(dm['identity'] == identity and dm['tokenizer'] == tokenizer_identity(ctx), 'Recovered tokenizer differs')
    for role, count in [('calibration', 256), ('wikitext2', 138), ('validation', 16), ('c4', 128)]:
        ids = checked_ids(ctx, role)
        require(len(ids) == count and sha(ctx.root / 'data' / f'{role}.safetensors') == dm['datasets'][role]['sha256'],
                f'Recovered {role} differs')

    # Only cache location/fingerprint may change. Ordered documents, prompts, metrics and harness stay exact.
    source = read(parent / 'data/arc_easy_transfer/source.json')
    copy('data/arc_easy_transfer/source.json')
    for row in source['files'].values():
        copy('data/arc_easy_transfer/' + row['local_file'], row['sha256'])
    old_eval = read(ctx.root / 'eval_manifest.json')
    require(old_eval['identity'] == identity, 'Recovered evaluation identity differs')
    fresh = freeze_tasks(ctx)
    def comparable(evidence):
        return {**evidence, 'tasks': {key: {k: v for k, v in row.items() if k not in ('fingerprint', 'cache_files')}
                                     for key, row in evidence['tasks'].items()}}
    require(comparable(fresh) == comparable(old_eval['downstream']), 'Recovered evaluation contract differs')
    provenance['downstream_content_and_config_exact'] = True
    dm['identity'] = ctx.identity
    write(ctx.root / 'data_manifest.json', dm)
    write(ctx.root / 'eval_manifest.json', dict(identity=ctx.identity, token_data=dm['datasets'], downstream=fresh))

    quant = read(ctx.root / 'quantization_manifest.json')
    require(quant['identity'] == identity and set(quant['modules']) == {name(i, k) for i in range(32) for k in KINDS},
            'Recovered quantization target set differs')
    teacher = read(ctx.root / 'teacher_identity.json')
    require(sha(quant['quantizer']['path']) == quant['quantizer']['sha256'], 'Recovered quantizer changed')
    for key, row in quant['modules'].items():
        require(row['identity'] == identity and sha(row['path']) == row['file_sha256'], 'Recovered Wq file differs')
        require(row['W0_hash'] == teacher['tensor_hashes'][key + '.weight']['hash'], 'Recovered Wq teacher differs')
        wq = load_file(row['path'])['Wq']
        require(wq.dtype == torch.float32 and list(wq.shape) == row['shape'] and mo.digest_tensor(wq) == row['Wq_hash'],
                'Recovered Wq tensor differs')
        del wq
        row.update(identity=ctx.identity, reused=True, preparation_parent_identity=identity)
    quant['identity'] = ctx.identity
    write(ctx.root / 'quantization_manifest.json', quant)
    ctx.commit('quantization_complete.json', [ctx.root / 'quantization_manifest.json', ctx.root / 'teacher_identity.json'],
               modules=224, preparation_parent_identity=identity)

    for receipt in sorted((parent / 'data/labels').glob('w*.json')):
        old = read(receipt)
        require(old['identity'] == identity and old['teacher_identity_sha256'] == sha(ctx.root / 'teacher_identity.json')
                and old['T'] == T and old['vocab_chunk'] == ctx.config['vocab_chunk'], 'Recovered label source differs')
        require(receipt.stem == f"w{old['window']:04d}" and 0 <= old['window'] < N, 'Invalid label receipt')
        relative = str(receipt.relative_to(parent))
        copy(str(receipt.with_suffix('.safetensors').relative_to(parent)), old['sha256'])
        provenance['files'][relative] = sha(receipt)
        write(ctx.root / relative, {**old, 'identity': ctx.identity, 'preparation_parent_identity': identity,
                                   'parent_receipt_sha256': sha(receipt)})
        label(ctx, old['window'])  # Checks seed, token binding, payload hash, dtype/range without resampling.
        provenance['labels'].append(old['window'])
    write(ctx.root / 'audit/preparation_reuse.json', provenance)
    files = [ctx.root / relative for relative in read(parent / 'data/complete.json')['files']]
    files += [ctx.root / 'quantization_complete.json', ctx.root / 'audit/preparation_reuse.json',
              ctx.root / 'data/arc_easy_transfer/source.json']
    ctx.commit('data/complete.json', files, preparation_parent_identity=identity)
    log('PREPARATION_REUSED', parent=str(parent), Wq_modules=224, labels=provenance['labels'],
        downstream_content_and_config_exact=True)
