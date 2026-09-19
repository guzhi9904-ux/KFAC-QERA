"""Independent scalar/label/bootstrap audit, without repeating expensive backprop."""
import json
import math
from pathlib import Path
import numpy as np
from functional_analysis import CANDIDATES,RANKS


def close(a,b,scale=0.):
    assert abs(float(a)-float(b))<=2e-10*max(abs(float(a)),abs(float(b)))+64*np.finfo(float).eps*abs(scale),(a,b)


def verify_summary(records,summary,k):
    grid={(r['window'],r['replicate']):r for r in records}
    assert len(grid)==len(records)==8*k and set(grid)=={(c,j) for c in range(8) for j in range(k)}
    v=np.array([[[grid[c,j]['scores'][n]['b'] for n in CANDIDATES] for j in range(k)] for c in range(8)])
    means=np.array([math.fsum(v[:,:,i].flat)/(8*k) for i in range(17)])
    boot=None
    if k==16:
        indices=np.random.Generator(np.random.PCG64(2026091902)).integers(0,16,size=(2000,8,16),dtype=np.int64)
        counts=np.array([[np.bincount(indices[b,c],minlength=16) for c in range(8)] for b in range(2000)])
        boot=np.einsum('bck,cki->bi',counts,v)/(8*16)
    for i,name in enumerate(CANDIDATES):
        r=next(x for x in summary['curves'] if x['candidate']==name)
        close(r['q'],means[i],scale=means[i])
        if r['ratio_valid']:
            close(r['gamma'],1-means[i]/means[0],scale=1.)
            if k==16:
                lo,hi=np.quantile(1-boot[:,i]/boot[:,0],[.025,.975]);close(r['gamma_ci_low'],lo,1.);close(r['gamma_ci_high'],hi,1.)
        if k==16:
            lo,hi=np.quantile(boot[:,i],[.025,.975]);close(r['q_ci_low'],lo,means[i]);close(r['q_ci_high'],hi,means[i])
    for r in summary['paired']:
        i=CANDIDATES.index(r['reference']);j=CANDIDATES.index(r['candidate'])
        delta=means[i]-means[j];close(r['Delta'],delta,scale=means[0])
        if r['d'] is not None:close(r['d'],delta/means[0],1.)
        if k==16:
            variances=[]
            for c in range(8):
                u=[float(v[c,jj,i]-v[c,jj,j]) for jj in range(16)];mean=math.fsum(u)/16
                variances.append(math.fsum((x-mean)**2 for x in u)/15)
            close(r['SE'],math.sqrt(math.fsum(variances)/(64*16)),means[0])
            if r['d'] is not None:
                lo,hi=np.quantile((boot[:,i]-boot[:,j])/boot[:,0],[.025,.975])
                close(r['ci_low'],lo,1.);close(r['ci_high'],hi,1.)
                state='IMPROVED' if lo>.02 else 'DEGRADED' if hi<-.02 else 'SMALL_WITHIN_BUDGET' if lo>=-.02 and hi<=.02 else 'UNRESOLVED_AT_K16'
                assert r['status']==state
        else:assert 'SE' not in r and 'ci_low' not in r


def verify_run(root,require_complete=True):
    from bridge import read,PLAN,save_json,read_tensors,sha_file,mo,slug,source_identity
    root=Path(root);identity=read(root/'identity.json');assert identity==source_identity()
    completed=[m for m in PLAN['modules'] if (root/'modules'/slug(m)/'complete.json').exists()]
    if require_complete:assert completed==PLAN['modules']
    labels={}
    for p in sorted((root/'new_labels/samples').glob('*.safetensors')):
        t,meta=read_tensors(p);assert mo.digest_tensor(t['labels'])==meta['label_hash'] and meta['identity']==identity['identity']
        assert t['labels'].shape==(2047,)
        key=meta['window'],meta['replicate'];assert key not in labels
        assert meta['seed']==mo.stream_seed(*key,PLAN['sample_seed'])
        labels[key]=meta
    if require_complete:assert set(labels)=={(c,k) for c in range(8) for k in range(16)}
    for m in completed:
        folder=root/'modules'/slug(m);freeze=read(folder/'candidate_freeze.json')
        assert freeze['identity']==identity['identity'] and set(freeze['candidates'])==set(CANDIDATES)
        for n,h in freeze['files'].items():assert sha_file(folder/n)==h,n
        original=read(folder/'fit/records.json')['records']
        fresh=[read(p) for p in sorted((folder/'new_labels/records').glob('*.json'))]
        for phase,records,K in [('fit',original,4),('new_labels',fresh,16)]:
            for r in records:
                assert r['identity']==identity['identity'] and r['module']==m and set(r['scores'])==set(CANDIDATES)
                if K==16:
                    meta=labels[r['window'],r['replicate']]
                    assert r['label_hash']==meta['label_hash'] and r['input_hash']==meta['input_hash']
                for n,score in r['scores'].items():
                    close(score['b'],score['d']**2/(2*2047),score['b'])
                    assert score['R_hash']==freeze['candidates'][n]['R_hash']
            verify_summary(records,read(folder/phase/'summary.json'),K)
        for d in read(folder/'fit/direction_statistics.json'):
            if d['b']>0:close(d['beta'],d['a']/d['b'],abs(d['beta']))
            close(d['q_fit_raw_ideal'],d['q_fit_None']-d['a']+d['b']/2,d['q_fit_None']+abs(d['a'])+d['b'])
            close(d['q_fit_cal_ideal'],d['q_fit_None']-d['beta']*d['a']+d['beta']**2*d['b']/2,d['q_fit_None']+d['ideal_gain'])
            if d['method']=='Gradient':
                close(d['a'],d['proxy_curvature'],abs(d['a']))
                assert d['q_fit_cal_ideal']<=min(d['q_fit_None'],d['q_fit_raw_ideal'])+1e-10*max(d['q_fit_None'],d['q_fit_raw_ideal'])
        klrows=[read(p) for p in sorted((folder/'kl/records').glob('*.json'))]
        assert len(klrows)==40
        grid={(r['window'],r['candidate']):r for r in klrows};assert len(grid)==40
        quality=read(folder/'kl/summary.json')
        for r in klrows:
            assert r['R_hash']==freeze['candidates'][r['candidate']]['R_hash'] and r['W_hash']==freeze['candidates'][r['candidate']]['W_hash']
            if r['window']==0:assert r['repeated'] and r['repeat_difference']<=max(1e-12,1e-7*abs(r['KL']))
        means={r['candidate']:math.fsum(grid[c,r['candidate']]['KL'] for c in range(8))/8 for r in quality['quality']}
        for r in quality['quality']:
            close(r['KL'],means[r['candidate']],r['KL'])
            if r['gamma_KL'] is not None:close(r['gamma_KL'],1-means[r['candidate']]/means['None'],1.)
        for r in quality['paired']:
            suffix='' if r['comparison']=='raw' else '-Cal'
            delta=means[f'Residual-SVD{suffix}-r64']-means[f'Gradient-SVD{suffix}-r64']
            close(r['Delta'],delta,means['None'])
            if r['d'] is not None:close(r['d'],delta/means['None'],1.)
    result=dict(passed=True,identity=identity['identity'],modules_verified=completed,complete=len(completed)==28,
        fit_module_samples=len(completed)*32,new_module_samples=len(completed)*128,new_score_rows=len(completed)*2176,
        KL_main=len(completed)*40,KL_repeats=len(completed)*5,label_files=len(labels),
        scope='Independent scalar identities, real label tensor hashes, paired SE and multiplicity-weighted bootstrap; source/candidate files. Large contractions are not rerun by this verifier.')
    save_json(root/'verification.json',result);return result
