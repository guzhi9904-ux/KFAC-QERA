import math
from pathlib import Path
from bridge import (PLAN,read,read_tensors,slug,sha_file,torch,mo,save_tensors,save_json,save_csv,clean)
import functional_math as fm
from functional_analysis import CANDIDATES,RANKS,summarize


class Construction:
    def geometry(self):
        if self.is_parent:
            path=self.exp3/'factors'/slug(self.name)/'marginal_solve.safetensors'
            t,m=read_tensors(path);assert m['identity']==PLAN['exp3_identity']
            return t['A_solve'].cuda(),t['G_solve'].cuda(),dict(path=str(path),file_sha256=sha_file(path),inherited=True)
        path=self.root/'geometry.safetensors'
        if path.exists():
            t,m=read_tensors(path);assert m['identity']==self.identity
            return t['A_solve'].cuda(),t['G_solve'].cuda(),dict(path=str(path),file_sha256=sha_file(path),**m)
        t,m=read_tensors(self.root/'statistics.safetensors');assert m['identity']==self.identity
        a=t['A_sum'].cuda()/(8*2048);g=t['G_sum'].cuda()/(32*2048);del t
        # Exactly the parent's product-preserving canonical convention, once.
        norm=float(a.norm());assert norm>0
        a=a/norm;g=g*(2048/2047)*norm
        la=.001*float(a.trace())/len(a);lg=.001*float(g.trace())/len(g)
        aa=a+la*torch.eye(len(a),dtype=torch.float64,device='cuda')
        gg=g+lg*torch.eye(len(g),dtype=torch.float64,device='cuda')
        meta=dict(identity=self.identity,canonical='(A_marg, (L/T)*G_marg), product-preserving unit-Frobenius A gauge once',
                  eta_A=.001,eta_G=.001,lambda_A=la,lambda_G=lg,A_count=8*2048,G_count=32*2048,
                  A_raw_hash=mo.digest_tensor(a),G_raw_hash=mo.digest_tensor(g),A_solve_hash=mo.digest_tensor(aa),G_solve_hash=mo.digest_tensor(gg))
        save_tensors(path,{'A_raw':a,'G_raw':g,'A_solve':aa,'G_solve':gg},meta)
        return aa,gg,dict(path=str(path),file_sha256=sha_file(path),**meta)

    def construct(self):
        if (self.root/'candidate_freeze.json').exists():
            self.validate_freeze();return
        self.collect_fit();quant,qmeta=self.quantized();self.unload();self.boundary()
        w0,wq=quant['W0'],quant['Wq'];e=(w0.double()-wq.double()).cuda();del quant
        a,g,geometry=self.geometry();projected=[];m=torch.zeros_like(e)
        if self.is_parent:
            parent_scores=read(self.exp3/'eval'/f'{slug(self.name)}_metric.json')['scores']
            assert mo.digest_tensor(e)==next(r['R_hash'] for r in parent_scores if r['candidate']=='None')
        with self.timed('M_from_original_S',module=self.name):
            for row,s in self.stream_s():
                de=float((s*e).sum());m.add_(s,alpha=de/(32*2047))
                projected.append(dict(window=row['window'],replicate=row['replicate'],d_E=de,S_hash=row['S_hash'],label_hash=row['label_hash']))
                del s
        assert len(projected)==32
        q0=math.fsum(z['d_E']**2 for z in projected)/(2*32*2047)
        fm.check_scalar(float((m*e).sum()),2*q0,tolerance=1e-10,scale=float(m.norm())*float(e.norm()))
        save_tensors(self.root/'construction/M.safetensors',{'M':m},dict(identity=self.identity,module=self.name,M_hash=mo.digest_tensor(m),
            definition='mean(<S,E>*S)/T; original 8x4 samples only',E_hash=mo.digest_tensor(e),q_fit_None=q0))
        with self.timed('roots_and_two_dense_SVD',module=self.name):factors,audits=fm.decompose(e,m,a,g,128)
        if not self.is_parent:
            for side,lam in [('A',geometry['lambda_A']),('G',geometry['lambda_G'])]:
                lo=audits[side]['min_eigenvalue']-lam;hi=audits[side]['max_eigenvalue']-lam
                assert hi>0 and lo>=-1e-10*hi,('raw PSD',side,lo,hi)
                audits[side]['raw_min_eigenvalue']=lo
        # Both SVD prefixes come from their single full decomposition.
        for method,factor in factors.items():
            save_tensors(self.root/'construction'/(method+'_factors.safetensors'),factor,
                         dict(identity=self.identity,module=self.name,method=method,ranks=list(RANKS),**audits[method]))
        ds={(method,r):[] for method in factors for r in RANKS}
        with self.timed('fit_ray_projections',module=self.name):
            for row,s in self.stream_s():
                for method,factor in factors.items():
                    contributions=(factor['P']*(s@factor['Q'].T)).sum(0).cumsum(0)
                    for rank in RANKS:ds[method,rank].append(float(contributions[rank-1]))
                del s
        de=torch.tensor([r['d_E'] for r in projected],dtype=torch.float64)
        directions=[];candidate_meta={'None':dict(W_hash=mo.digest_tensor(wq),R_hash=mo.digest_tensor(e),origin=qmeta['path'])}
        deployed={'None':e.cpu()};self.checks['parent_replay']={}
        for method,factor in factors.items():
            for rank in RANKS:
                p,q=factor['P'][:,:rank],factor['Q'][:rank]
                c=p@q;norm=float(c.norm());d=torch.tensor(ds[method,rank],dtype=torch.float64)
                ray=fm.calibrated_ray(de,d,norm,2047);sigma2=float(factor['singular_values'][:rank].square().sum())
                linear=fm.inner_factors(m,p,q);proxy=fm.metric_factors(p,q,a,g)
                fm.check_scalar(ray['a'],linear,tolerance=1e-10,scale=float(m.norm())*norm)
                if method=='Gradient':
                    fm.check_scalar(linear,sigma2,scale=float(m.norm())*norm);fm.check_scalar(proxy,sigma2,scale=sigma2)
                    if norm>0:assert ray['a']>0 and ray['b']>0 and ray['beta']>0
                    complete_square=fm.check_scalar(q0-linear+proxy/2,
                        q0-audits[method]['input_norm_squared']/2+audits[method][str(rank)]['tail_squared']/2,
                        scale=q0+audits[method]['input_norm_squared'])
                ideal_raw=float((de-d).square().mean()/(2*2047))
                ideal_cal=float((de-ray['beta']*d).square().mean()/(2*2047))
                fm.check_scalar(q0-ideal_cal,ray['ideal_gain'],scale=q0)
                assert ideal_cal<=min(q0,ideal_raw)+1e-10*max(q0,ideal_raw)
                row=dict(module=self.name,method=method,rank=rank,norm_C=norm,**ray,proxy_a=sigma2,
                    proxy_curvature=proxy,true_to_proxy_curvature=ray['b']/proxy if proxy>0 else None,
                    a_minus_proxy_a=ray['a']-sigma2,b_minus_proxy_curvature=ray['b']-proxy,
                    proxy_predicted_gain=linear-proxy/2,proxy_predicted_calibrated_gain=ray['beta']*linear-ray['beta']**2*proxy/2,
                    q_fit_None=q0,q_fit_raw_ideal=ideal_raw,q_fit_cal_ideal=ideal_cal,
                    calibrated_norm=abs(ray['beta'])*norm)
                if method=='Gradient':row['surrogate_complete_square_error']=complete_square
                for calibrated in (False,True):
                    beta=ray['beta'] if calibrated else 1.
                    label=f'{method}-SVD'+('-Cal' if calibrated else '')+f'-r{rank}'
                    # The final compensated matrix is rounded first, then added in FP32 on CPU, exactly as parent.
                    c32=(c*beta).float().cpu();w=(wq+c32).float();residual=w0.double()-w.double()
                    assert bool(torch.isfinite(w).all()) and bool(torch.isfinite(c32).all())
                    meta=dict(identity=self.identity,module=self.name,candidate=label,rank=rank,beta=beta,
                        W_hash=mo.digest_tensor(w),R_hash=mo.digest_tensor(residual),C_FP32_hash=mo.digest_tensor(c32),
                        ideal_factor_file=f'construction/{method}_factors.safetensors',
                        deployment='FP32(Wq+FP32(beta*(P_r@Q_r))); R=FP64(W0)-FP64(W_deploy)')
                    path=self.root/'deployment'/(label+'.safetensors');save_tensors(path,{'W_deploy':w},meta)
                    candidate_meta[label]=dict(path=str(path.relative_to(self.root)),file_sha256=sha_file(path),**meta)
                    deployed[label]=residual
                    row['rounding_relative_'+('cal' if calibrated else 'raw')]=mo.relative(c32.double(),(c*beta).cpu())
                    if self.is_parent and method=='Residual' and rank==64 and not calibrated:
                        original,om=read_tensors(self.exp3/'corrections'/slug(self.name)/'marginal.safetensors')
                        c_error=mo.relative(c.cpu(),original['C64']);r_error=mo.relative(residual,original['R'])
                        j_old=fm.metric(original['R'].cuda(),a,g);j_new=fm.metric(residual.cuda(),a,g)
                        drift=abs(j_old-j_new)/max(abs(j_old),q0*1e-12)
                        assert c_error<=1e-8 and drift<=1e-4
                        self.checks['parent_replay']=dict(ideal_C_relative_error=c_error,R_relative_error=r_error,
                            R_bitwise_equal=torch.equal(residual,original['R']),W_bitwise_equal=torch.equal(w,original['W_deploy']),
                            metric_objective_relative_drift=drift,parent_R_hash=om['R_hash'],new_R_hash=meta['R_hash'],
                            note='Exact equality disclosed; numerical reconstruction accepted only within frozen tolerances')
                        del original
                    del c32,w,residual
                directions.append(row);del c
        assert set(candidate_meta)==set(CANDIDATES)
        fitrecords=[]
        with self.timed('fit_actual_deployment_scores',module=self.name):
            for row,s in self.stream_s():
                scores={}
                for label in CANDIDATES:
                    d=float((s*deployed[label].cuda()).sum());b=d*d/(2*2047)
                    assert math.isfinite(b)
                    scores[label]=dict(d=d,b=b,R_hash=candidate_meta[label]['R_hash'])
                fitrecords.append(dict(identity=self.identity,module=self.name,window=row['window'],replicate=row['replicate'],
                    S_hash=row['S_hash'],label_hash=row['label_hash'],scores=scores,T=2047))
                del s
        means={n:math.fsum(r['scores'][n]['b'] for r in fitrecords)/32 for n in CANDIDATES}
        for row in directions:
            for suffix,key in [('', 'raw'),('-Cal','cal')]:
                label=f"{row['method']}-SVD{suffix}-r{row['rank']}";value=means[label];ideal=row['q_fit_'+key+'_ideal']
                drift=abs(value-ideal)/max(abs(ideal),q0*1e-12)
                assert drift<=1e-4,('deployment drift',label,drift)
                row['q_fit_'+key+'_deploy']=value;row['deployment_q_relative_drift_'+key]=drift
        save_json(self.root/'fit/records.json',dict(identity=self.identity,records=fitrecords))
        save_json(self.root/'fit/direction_statistics.json',directions);save_csv(self.root/'fit/direction_statistics.csv',directions)
        summary=summarize(self.name,fitrecords,4,False);save_json(self.root/'fit/summary.json',summary)
        save_json(self.root/'construction/numerical_checks.json',dict(passed=True,identity=self.identity,roots_and_SVD=audits,**self.checks))
        del factors,a,g,m,e,deployed;clean()
        files={str(p.relative_to(self.root)):sha_file(p) for folder in ('construction','deployment','fit') for p in sorted((self.root/folder).rglob('*')) if p.is_file()}
        for filename in ('geometry.safetensors','quantized.safetensors','statistics.safetensors','fit_cache_manifest.json'):
            p=self.root/filename
            if p.exists():files[filename]=sha_file(p)
        save_json(self.root/'candidate_freeze.json',dict(identity=self.identity,module=self.name,files=files,candidates=candidate_meta,
            geometry=geometry,quantized=qmeta,frozen_at=__import__('time').time(),all_17_frozen_before_new_label_evaluation=True))
        self.status('CANDIDATES_FROZEN',module=self.name,candidates=17)

    def validate_freeze(self):
        record=read(self.root/'candidate_freeze.json');assert record['identity']==self.identity and record['module']==self.name
        for n,h in record['files'].items():assert sha_file(self.root/n)==h,n
        for ref in (record['geometry'],record['quantized']):
            assert sha_file(Path(ref['path']))==ref.get('file_sha256',ref.get('sha256'))
        return record

    def clean_temporary(self):
        if self.is_parent:return
        self.validate_freeze()
        assert read(self.root/'construction/numerical_checks.json')['passed']
        assert len(read(self.root/'fit/records.json')['records'])==32
        folder=(self.root/'temporary').resolve()
        assert folder.parent==self.root.resolve() and folder.is_relative_to(self.runroot)
        if not folder.exists():return
        files=list(folder.iterdir());assert all(p.is_file() and not p.is_symlink() for p in files)
        save_json(self.root/'temporary_cleanup.json',dict(identity=self.identity,
            removed=[dict(name=p.name,sha256=sha_file(p),bytes=p.stat().st_size) for p in files],
            rebuild_manifest='fit_cache_manifest.json',parent_cache_untouched=True))
        for path in files:path.unlink()
        folder.rmdir()
