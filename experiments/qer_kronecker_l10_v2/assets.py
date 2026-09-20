"""Fail closed on teacher, quantizer, tokenizer, corpus and history identities."""
from pathlib import Path
import hashlib
from common import PLAN, read, read_tensors, sha_file, mo, require, slug, digest


def verify_quantizer_source(path,expected):
    # Match Exp1.official exactly: its recorded digest normalizes CRLF to LF.
    actual=hashlib.sha256(Path(path).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    require(actual==expected,'Official quantizer source differs')


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
        for c in range(8):
            for k in range(4): pin(old/'data/fit_samples'/f'w{c:02d}_k{k:03d}.safetensors')
        for rel in (f'corrections/{slug(name)}/marginal.safetensors',
                    f'cache/{slug(name)}/x_w00.safetensors',
                    f'cache/{slug(name)}/w00_k000.safetensors', f'cache/{slug(name)}/w00_k001.safetensors'):
            if (old/rel).is_file():
                expected = frozen['files'].get(rel)
                pin(old/rel, expected)
        qp = Path(config['quantized_source']); pin(qp)
        qt, qm = read_tensors(qp)
        require(qm['module']==name and (qm['width'],qm['block_size'],qm['block_axis'])==(3,32,-1),'Wrong L10 quantizer configuration')
        qi=read(qp.parents[2]/'identity.json');pin(qp.parents[2]/'identity.json')
        require(qm['identity']==qi['identity'] and digest({k:v for k,v in qi.items() if k!='identity'})==qi['identity'],'Invalid quantized source identity')
        quantizer=base/'vendor/quantize/quantizers/mxint.py'
        if not quantizer.exists():quantizer=base/'vendor/src/qera/quantize/quantizers/mxint.py'
        verify_quantizer_source(quantizer,qm['quantizer_hash']);pin(quantizer)
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
    from paired_data import source_inventory
    files.update(source_inventory(config))
    previous = Path(config['geometry_history'])
    geometry = read(previous/'manifest.json')
    require(geometry['identity'] == PLAN['geometry_history_identity'], 'Wrong immediately preceding geometry history')
    pin(previous/'manifest.json')
    gm = read(previous/'data/article_manifest.json'); pin(previous/'data/article_manifest.json')
    require(gm['identity']==geometry['identity'] and len(gm['windows'])==48,'Geometry history incomplete')
    for role in ('search','test'):
        p=previous/'data'/f'{role}_windows.safetensors';pin(p,gm['files'][role]);history.append(str(p))
    history_manifests=[str(old/'data/fit_windows.json'),str(previous/'data/article_manifest.json')]+config.get('extra_article_manifests',[])
    return dict(files=files, factors=factors, historical_windows=history, historical_manifests=history_manifests,tokenizer_files=tokenizer,
        dataset_files=dataset, parent_identity=PLAN['parent_identity'],
        historical_lineage={
          'Exp1_Exp2_alpha_extension':'same eight frozen validation windows; 256 original SlimPajama calibration windows',
          'Exp3_diagnosis_functional_gradient':'same eight Exp3 fit windows; no new text in those protocols',
          'geometry_search_20260920':'all 32 search and 16 test articles excluded, regardless of outcomes',
          'extra_history':'additional window files/article manifests from configuration; operator must include any other text runs'},
        factor_source='parent teacher sampled-label Marginal solve; eta_A=eta_G=.001 already applied')
