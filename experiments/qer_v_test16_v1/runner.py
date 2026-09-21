"""Evaluation only: frozen five-method V corrections, original test windows 0:16."""
import argparse
import math
import os
from pathlib import Path
import signal
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'qer_v_attention_proto_v1'))
from v_common import (MODULES, read, save_json, save_csv, require, digest, sha_file,
                      load_record, commit, checked_files, file_table, lock, ParentStore,
                      Resources, ReplayTeacher, mo, slug, identity as v_identity)
from ko_common import identity as ko_identity
from dataset import windows
from bridge import hidden_forward
import numpy as np
import torch

METHODS = ['A-only', 'Marginal', 'Token-joint', 'Sequence-one-step', 'Attention-aware', 'None']


def summarize(records):
    """Token-weighted absolute KL first, then recovery; no mean of window ratios."""
    rows = []
    expected = {(n, m, w) for n in MODULES for m in METHODS for w in range(16)}
    keys = [(r['module'], r['method'], r['window']) for r in records]
    require(len(keys) == len(set(keys)) and set(keys) == expected, 'Incomplete/duplicate test records')
    for n in MODULES:
        for m in METHODS:
            items = [r for r in records if r['module'] == n and r['method'] == m]
            tokens = sum(r['scores']['tokens'] for r in items)
            require(tokens == 16 * 2047, 'Unexpected test token count')
            rows.append(dict(module=n, method=m, windows=16, tokens=tokens,
                             KL=sum(r['scores']['KL'] * r['scores']['tokens'] for r in items) / tokens))
    baseline = {r['module']: r['KL'] for r in rows if r['method'] == 'None'}
    for r in rows:
        require(baseline[r['module']] > 1e-12, 'Degenerate recovery denominator')
        r['recovery_percent'] = 100 * (1 - r['KL'] / baseline[r['module']])
    return rows


def prepare(vroot, kroot):
    vm, km = read(vroot/'manifest.json'), read(kroot/'manifest.json')
    require(v_identity(vm['config']) == vm, 'Prototype frozen source/config changed')
    require(ko_identity(km['config']) == km, 'A-only frozen source/config changed')
    parent = Path(vm['config']['parent_run']).resolve()
    require(Path(km['config']['parent_run']).resolve() == parent, 'Different fit parents')
    pm = read(parent/'manifest.json')
    for root, manifest in ((vroot, vm), (kroot, km)):
        done = load_record(root/'complete.json', manifest['identity'])
        require(done['passed'], 'Source run incomplete')
        checked_files(root, done['files'])
    frozen = load_record(parent/'data/freeze.json', pm['identity'])
    checked_files(parent, frozen['files'])
    ids = windows(parent, 'test', pm['identity'])[:16]
    require(tuple(ids.shape) == (16, 2048), 'Expected original first16 test windows')
    require(read(parent/'data/test.json')['split'] == 'test', 'Not official test split')
    manifests = {parent: pm, vroot: vm, kroot: km}
    stores = {r: ParentStore(m['identity'], 0) for r, m in manifests.items()}
    freezes = {r: load_record(r/'candidate_freeze.json', m['identity']) for r, m in manifests.items()}
    weights, bindings = {}, []
    for n in MODULES:
        for method in METHODS:
            origin = vroot if method == 'Attention-aware' else kroot if method == 'A-only' else parent
            rel = (Path('corrections')/slug(n)/'attention_aware.safetensors' if method == 'Attention-aware'
                   else Path('quantized')/(slug(n)+'.safetensors') if method == 'None'
                   else Path('modules')/slug(n)/'corrections'/('N256__'+method+'.safetensors'))
            path = origin/rel
            checksum = sha_file(path)
            if method != 'None':
                require(freezes[origin]['files'].get(rel.as_posix()) == checksum, 'Candidate freeze mismatch')
            tensors, metadata = stores[origin].get(path)
            weight = tensors['Wq' if method == 'None' else 'W_deploy'].clone()
            require(tuple(weight.shape) == (1024,4096) and weight.dtype == torch.float32
                    and torch.isfinite(weight).all(), 'Invalid deployed V weight')
            weights[n,method] = weight
            bindings.append(dict(module=n, method=method, path=str(path), sha256=checksum,
                                 receipt_sha256=sha_file(path.with_suffix('.json')),
                                 identity=metadata['identity'], weight_hash=mo.digest_tensor(weight)))
    material = dict(version='qer_v_test16_v1', prototype=str(vroot), aonly=str(kroot), parent=str(parent),
                    source_manifests={str(r):sha_file(r/'manifest.json') for r in manifests},
                    source={p.name:sha_file(p) for p in HERE.iterdir() if p.suffix in ('.py','.md','.sh')},
                    candidates=bindings, test_indices=list(range(16)), token_hashes=[mo.digest_tensor(x) for x in ids],
                    selection='Original fixed prefix, previously evaluated with joint12 corrections; not pristine unseen test',
                    budget=256, rank=64, methods=METHODS, scope='Single-module actual teacher-KL; no fitting or test selection')
    return dict(identity=digest(material), **material), vm['config'], ids, weights


def check_score(row, ident, n, method, window, token_hash, frozen_hash):
    require(row['identity'] == ident and row['module'] == n and row['method'] == method
            and row['window'] == window and row['role'] == 'test'
            and row['token_hash'] == token_hash and row['freeze_hash'] == frozen_hash, 'Score binding mismatch')
    score = row['scores']
    require(score['tokens'] == 2047 and math.isfinite(score['KL']) and score['KL'] >= 0, 'Invalid KL')


def report(root, manifest):
    ident = manifest['identity']; frozen_hash = sha_file(root/'manifest.json'); records=[]; paths=[]
    for w in range(16):
        for n in MODULES:
            for method in METHODS:
                p=root/'scores/test'/f'w{w:04d}'/(slug(n)+'___'+method+'.json')
                r=load_record(p,ident)
                check_score(r,ident,n,method,w,manifest['token_hashes'][w],frozen_hash)
                records.append(r);paths.append(p)
    rows=summarize(records);save_csv(root/'summary/results.csv',rows)
    per=[dict(module=r['module'],method=r['method'],window=r['window'],**r['scores']) for r in records]
    save_csv(root/'summary/per_window.csv',per)
    # Paired exploratory intervals; identical resampled window indices across layers.
    indices=np.random.default_rng(20260921).integers(0,16,(2000,16));paired=[]
    lookup={(r['module'],r['method'],r['window']):r['scores']['KL'] for r in records}
    for n in MODULES:
        a=np.array([lookup[n,'Attention-aware',w] for w in range(16)])
        b=np.array([lookup[n,'None',w] for w in range(16)])
        for method in METHODS[:4]:
            delta=np.array([lookup[n,method,w] for w in range(16)])-a
            boot=100*delta[indices].sum(axis=1)/b[indices].sum(axis=1)
            lo,hi=np.quantile(boot,[.025,.975])
            paired.append(dict(module=n,reference=method,absolute_KL_difference=float(delta.mean()),
                               recovery_gain_pp=float(100*delta.sum()/b.sum()),wins=int((delta>0).sum()),
                               exploratory_low_pp=float(lo),exploratory_high_pp=float(hi)))
    save_csv(root/'summary/paired_comparisons.csv',paired)
    save_json(root/'summary/results.json',dict(identity=ident,rows=rows,paired=paired,
        caveat=manifest['selection']+'; window dependence and fit resampling not covered by bootstrap'))
    lines=['# V five-method test16', '', 'N256, rank64, original test windows0-15, single-module actual teacher-KL.',
           'Previously used for joint12 evaluation; not pristine unseen test. No test fitting or candidate selection.', '',
           '| Module | A-only | Marginal | Token-joint | Sequence-one-step | Attention-aware |', '|---|---:|---:|---:|---:|---:|']
    for n in MODULES:
        values={r['method']:r['recovery_percent'] for r in rows if r['module']==n}
        lines.append('| '+n+' | '+' | '.join(f'{values[m]:.4f}%' for m in METHODS[:-1])+' |')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    paths+=list((root/'summary').glob('*'))+[root/'RESULTS.md',root/'teacher_verification.json']
    commit(root/'complete.json',ident,passed=True,modules=4,methods=5,baselines=1,test_windows=16,
           candidate_KL=384,full_test=False,files=file_table(root,paths))
    print('V_TEST16_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prototype',type=Path,required=True)
    p.add_argument('--aonly',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--check-only',action='store_true');args=p.parse_args()
    for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[key]='1'
    require(not sys.flags.optimize,'Python -O forbidden')
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    vroot=args.prototype.resolve();kroot=args.aonly.resolve();root=args.output.resolve()
    parent=Path(read(vroot/'manifest.json')['config']['parent_run']).resolve()
    for src in (vroot,kroot,parent):
        require(not root.is_relative_to(src) and not src.is_relative_to(root),'Output overlaps frozen source')
    root.mkdir(parents=True,exist_ok=True)
    with lock(root):
        print('AUDITING_FROZEN_CANDIDATES_AND_TEST_WINDOWS',flush=True)
        manifest,config,ids,weights=prepare(vroot,kroot);ident=manifest['identity']
        if (root/'manifest.json').exists():require(read(root/'manifest.json')==manifest,'Test configuration/source changed')
        else:save_json(root/'manifest.json',manifest)
        print('TEST16_ASSETS_VERIFIED 24 weights; 16 windows; no fitting',flush=True)
        if args.check_only:return
        if (root/'complete.json').exists():
            checked_files(root,load_record(root/'complete.json',ident)['files']);print('V_TEST16_ALREADY_COMPLETE');return
        config=dict(config,budget_hours=2.,disk_limit_GiB=2.)
        resources=Resources(root,ident,config);teacher=ReplayTeacher(config,root,ident,resources.timed)
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:setattr(resources,'stop',True))
        frozen_hash=sha_file(root/'manifest.json')
        try:
            for window in range(16):
                resources.boundary();folder=root/'scores/test'/f'w{window:04d}';token_hash=manifest['token_hashes'][window]
                pending=[]
                for n in MODULES:
                    for method in METHODS:
                        path=folder/(slug(n)+'___'+method+'.json')
                        if path.exists():check_score(load_record(path,ident),ident,n,method,window,token_hash,frozen_hash)
                        else:pending.append((n,method,path))
                if not pending:
                    tr=load_record(folder/'teacher.json',ident)
                    require(tr['token_hash']==token_hash,'Teacher window mismatch');continue
                batch=ids[window:window+1]
                with resources.timed('test_teacher_reference',window=window):
                    teacher.load()
                    with torch.no_grad():hidden=hidden_forward(teacher.model,batch).detach()
                    logits=teacher.reference_logits(hidden);score=teacher.scores_hidden(batch,logits,hidden)
                    require(abs(score['KL'])<=1e-10 and score['tokens']==2047,'Teacher self-KL failed')
                    commit(folder/'teacher.json',ident,token_hash=token_hash,scores=score)
                for n,method,path in pending:
                    with resources.timed('test_actual_KL',window=window,module=n,method=method),teacher.deploy({n:weights[n,method]}):
                        score=teacher.scores(batch,logits)
                        if window==0:
                            repeat=teacher.scores(batch,logits)
                            require(abs(score['KL']-repeat['KL'])<=max(1e-12,abs(score['KL'])*1e-7),'Repeat evaluation failed')
                    row=dict(identity=ident,module=n,method=method,window=window,role='test',token_hash=token_hash,freeze_hash=frozen_hash,scores=score)
                    check_score(row,ident,n,method,window,token_hash,frozen_hash)
                    row.pop('identity');commit(path,ident,**row)
                del hidden,logits
                print('TEST_WINDOW_COMPLETE',window+1,'/16',flush=True)
            report(root,manifest)
        except BaseException as exc:
            save_json(root/'failure.json',dict(identity=ident,error=repr(exc)));raise
        finally:teacher.unload();resources.flush()


if __name__=='__main__':main()
