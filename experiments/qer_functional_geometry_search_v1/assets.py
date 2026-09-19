"""Fail closed on teacher, quantizer, tokenizer, corpus and history identities."""
from pathlib import Path
from common import PLAN, read, read_tensors, sha_file, mo, require, slug, digest


def inventory(config):
    base = Path(config['assets']); old = base/'exp03'; parent = base/'exp01'
    identity = read(old/'identity.json'); pi = read(parent/'identity.json')
    require(identity['identity'] == PLAN['parent_identity'], 'Unexpected Exp-3 identity')
    for record in (identity, pi):
        require(digest({k:v for k,v in record.items() if k != 'identity'}) == record['identity'], 'Invalid parent identity digest')
    require(identity['plan']['parent_identity'] == pi['identity'], 'Parent lineage mismatch')
    meta = read(old/'data/fit_windows.json')
    files = {}
    def pin(path, expected=None):
        path = Path(path); require(path.is_file(), 'Missing required asset: '+str(path))
        value = sha_file(path)
        require(expected is None or value == expected, 'Asset hash differs: '+str(path))
        files[str(path)] = value
        return value
    for p in (old/'identity.json', parent/'identity.json', old/'teacher_identity.json', old/'environment.json', old/'data/fit_windows.json'):
        pin(p)
    # Authenticate transferred parent inputs when a bundle manifest exists.
    manifest_path = base/'migration_manifest.json'
    if manifest_path.exists():
        manifest = read(manifest_path)
        verified = read(base/'migration_verified.json')
        require(verified['passed'] and sha_file(manifest_path) == verified['manifest_sha256'], 'Migration manifest not verified')
        migration = {name: record['sha256'] for name, record in manifest['files'].items()}
    else:
        migration = None  # Native parent run: verify internal freeze below.
    frozen = read(old/'evaluation_freeze.json')
    pin(old/'evaluation_freeze.json'); pin(parent/'data_manifest.json')
    require(frozen['identity'] == PLAN['parent_identity'], 'Parent freeze identity mismatch')
    pin(old/'manifest.json', frozen['manifest_sha256'])
    for rel, value in frozen['files'].items():
        # Only the two selected module factors/deployments are needed, not every large intermediate.
        if any(slug(n) in rel for n in config['modules']) and rel.endswith('/marginal_solve.safetensors') and (old/rel).exists():
            pin(old/rel, value)
    tokenizer = {}
    for rel, value in meta['tokenizer_files'].items():
        tokenizer[rel] = pin(Path(config['model'])/rel, value)
    dataset = {}
    for rel, value in meta['source_sha256'].items():
        dataset[rel] = pin(Path(config['wikitext'])/rel, value)
    dm = read(parent/'data_manifest.json')
    pin(config['calibration'], dm['calibration_file_hash'])
    history = [str(Path(config['calibration'])), str(parent/'data/validation.safetensors'), str(old/'data/fit_windows.safetensors')]
    history += config.get('extra_history_windows', [])
    for p in history: pin(p)
    for p in config.get('extra_article_manifests', []): pin(p)
    factors = {}
    for name in config['modules']:
        # Two optional historical samples are used only for a numerical replay, never selection.
        for k in range(2): pin(old/'data/fit_samples'/f'w00_k{k:03d}.safetensors')
        for rel in (f'corrections/{slug(name)}/marginal.safetensors',
                    f'cache/{slug(name)}/x_w00.safetensors',
                    f'cache/{slug(name)}/w00_k000.safetensors'):
            if (old/rel).is_file():
                expected = frozen['files'].get(rel)
                pin(old/rel, expected)
        qp = parent/'quantized'/(slug(name)+'.safetensors'); pin(qp)
        qt, qm = read_tensors(qp)
        for key in ('W0','Wq'): require(mo.digest_tensor(qt[key]) == qm[key+'_hash'], 'Quantized tensor hash mismatch')
        require(qm['W0_hash'] == read(old/'teacher_identity.json')['tensor_hashes'][name+'.weight']['hash'], 'Wrong teacher E')
        fp = old/'factors'/slug(name)/'marginal_solve.safetensors'
        if fp.exists():
            pin(fp); ft, fm = read_tensors(fp)
            require(fm['identity'] == PLAN['parent_identity'], 'Wrong factor identity')
            require('A_solve' in ft and 'G_solve' in ft, 'Not Marginal solve factors')
            factors[name] = dict(path=str(fp), sha256=sha_file(fp), rebuilt=False)
        else:
            # Reconstruction uses original 8x4 saved labels; never new search/test data.
            for c in range(8):
                for k in range(4): pin(old/'data/fit_samples'/f'w{c:02d}_k{k:03d}.safetensors')
            factors[name] = dict(path=None, rebuilt=True, reason='Missing Marginal solve; reconstruct from original fit texts and labels')
    if migration is not None:
        for path, value in files.items():
            p = Path(path)
            if p.is_relative_to(base):
                rel = p.relative_to(base).as_posix()
                require(rel in migration and migration[rel] == value, 'Migrated parent file changed: '+rel)
    return dict(files=files, factors=factors, historical_windows=history, tokenizer_files=tokenizer,
        dataset_files=dataset, parent_identity=PLAN['parent_identity'],
        historical_lineage={
          'Exp1_Exp2_alpha_extension':'same eight frozen validation windows; 256 original SlimPajama calibration windows',
          'Exp3_diagnosis_functional_gradient':'same eight Exp3 fit windows; no new text in those protocols',
          'extra_history':'additional window files/article manifests from configuration; operator must include any other text runs'},
        factor_source='parent teacher sampled-label Marginal solve; eta_A=eta_G=.001 already applied')
