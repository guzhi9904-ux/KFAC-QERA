import json
from pathlib import Path
import time
import torch
from safetensors import safe_open
import ag_math as am
import math_ops as mo
from storage import save_json,save_tensors,read_tensors,sha_file,save_csv
from model_ops import capture,hidden_forward
from run import slug,clean,sync

read=lambda p:json.loads(Path(p).read_text())


class FitStage:
    def sample_path(self,name,c,k):return self.root/'cache'/slug(name)/f'w{c:02d}_k{k:03d}.safetensors'

    def collect_fit(self,N,K,pilot=False):
        for name in self.plan['modules']:
            for c in range(N):
                missing=[k for k in range(K) if not self.sample_path(name,c,k).exists()]
                if not missing:continue
                ids=self.fit['input_ids'][c:c+1];target=self.model.get_submodule(name)
                with torch.no_grad(),capture(target) as state:reference=hidden_forward(self.model,ids).detach()
                x=state['x'].detach();del state
                xp=self.root/'cache'/slug(name)/f'x_w{c:02d}.safetensors'
                xm={'identity':self.identity,'module':name,'window':c,'input_hash':mo.digest_tensor(x),
                    'token_hash':mo.digest_tensor(ids[0]),'teacher_hidden_hash':mo.digest_tensor(reference),'positions':2048}
                if xp.exists():
                    old,om=read_tensors(xp);assert om==xm and torch.equal(old['x'],x.cpu());del old
                else:save_tensors(xp,{'x':x},xm)
                samples=self.fitting_samples(c,K,reference)
                for k in missing:
                    with self.timed('fit_gradient_and_statistics',module=name,window=c,replicate=k,pilot=pilot):
                        started=time.perf_counter()
                        g,audit=self.gradient(name,ids,reference,x,samples[k][0],weight_check=(pilot and k==0))
                        tick=time.perf_counter();s=g.T@x.reshape(2048,-1).double();gg=g.T@g;sync()
                        audit['S_and_G_seconds']=time.perf_counter()-tick
                        if pilot and k==0:
                            xd=x.reshape(2048,-1).double();a=xd.T@xd/2048
                            ug=s@a@s.T;alternative=g.T@(xd@a@xd.T)@g
                            ua=s.T@(gg/2048)@s;other=xd.T@(g@(gg/2048)@g.T)@xd
                            audit['SAS_vs_cross_position_relative_error']=am.rel(ug,alternative)
                            audit['StGS_vs_cross_position_relative_error']=am.rel(ua,other)
                            for key in ('SAS_vs_cross_position_relative_error','StGS_vs_cross_position_relative_error'):
                                if audit[key]>self.plan['contraction_tolerance']:raise RuntimeError(key)
                            r=self.directions[name]['R_none'].to(x.device).double()
                            projection1=float((s*r).sum());projection2=float((g*(xd@r.T)).sum())
                            audit['projection_absolute_difference']=abs(projection1-projection2)
                            audit['projection_scale']=float((s*r).abs().sum())
                            if abs(projection1-projection2)>1e-10*audit['projection_scale']:raise RuntimeError('S projection contraction failure')
                            del xd,a,ug,alternative,ua,other,r
                        meta={'identity':self.identity,'module':name,'window':c,'replicate':k,'L':2048,'T':2047,
                              'label_hash':samples[k][1],'input_hash':xm['input_hash'],'token_hash':xm['token_hash'],
                              'loss_reduction':'sum','S_definition':'sum_t g_t x_t.T; FP64; canonical H coefficient applied as 1/T in contractions',
                              'S_hash':mo.digest_tensor(s),'gradient_gram_hash':mo.digest_tensor(gg),'audit':audit,
                              'seconds_before_serialization':time.perf_counter()-started}
                        path=self.sample_path(name,c,k);save_tensors(path,{'S':s,'gradient_gram':gg},meta)
                        meta['file_sha256']=sha_file(path);meta['file_bytes']=path.stat().st_size
                        save_json(path.with_suffix('.json'),meta)
                        del s,gg,g
                    self.status('FIT_PILOT_COLLECTION' if pilot else 'COLLECTING_FIT',module=name,window=c,replicate=k,
                                committed_samples=len(list((self.root/'cache').glob('*/w*.json'))),target_samples=2*N*K)
                del x,reference,samples;clean()

    def raw_stats(self,name,N,K,tag):
        folder=self.root/tag/slug(name);path=folder/'marginal_statistics.safetensors'
        if path.exists():
            tensors,meta=read_tensors(path);assert meta['identity']==self.identity and (meta['N'],meta['K'])==(N,K)
            return {k:v.cuda() for k,v in tensors.items()},meta
        a=None;g=None;sample_manifest=[]
        for c in range(N):
            xx,xm=read_tensors(self.root/'cache'/slug(name)/f'x_w{c:02d}.safetensors')
            assert xm['identity']==self.identity and mo.digest_tensor(xx['x'])==xm['input_hash']
            x=xx['x'].reshape(2048,-1).cuda().double();term=x.T@x
            a=term if a is None else a+term
            for k in range(K):
                p=self.sample_path(name,c,k);meta=read(p.with_suffix('.json'))
                assert meta['identity']==self.identity and meta['input_hash']==xm['input_hash'] and sha_file(p)==meta['file_sha256']
                with safe_open(str(p),framework='pt') as f:gram=f.get_tensor('gradient_gram')
                assert mo.digest_tensor(gram)==meta['gradient_gram_hash']
                gram=gram.cuda();g=gram if g is None else g+gram
                sample_manifest.append({'module':name,'window':c,'replicate':k,'S_hash':meta['S_hash'],'label_hash':meta['label_hash'],
                                        'file_sha256':meta['file_sha256'],'relative_path':str(p.relative_to(self.root))})
                del gram
            del xx,x,term
        tensors={'A_sum':a,'G_sum':g,'A_marg':am.sym(a/(N*2048)),'G_marg':am.sym(g/(N*K*2048))}
        meta={'identity':self.identity,'N':N,'K':K,'A_count':N*2048,'G_count':N*K*2048,'module':name,
              'dtype':'float64','centered':False,'channel_diagonalization':False,'regularization':'none',
              'canonical_H_coefficient':1/2047,'canonical_marginal_G_multiplier':2048/2047,'samples':sample_manifest}
        save_tensors(path,tensors,meta);save_json(path.with_suffix('.json'),meta)
        return tensors,meta

    def stream_S(self,name,N,K):
        def samples():
            for c in range(N):
                for k in range(K):
                    with safe_open(str(self.sample_path(name,c,k)),framework='pt') as f:s=f.get_tensor('S')
                    yield s
        return samples

    def pilot_linear_algebra(self):
        audits=[]
        for name in self.plan['modules']:
            with self.timed('pilot_ALS_and_dense_SVD',module=name):
                stats,meta=self.raw_stats(name,1,2,'pilot_factors');a,g=am.gauge(stats['A_marg'],stats['G_marg']*(2048/2047))
                s=self.stream_S(name,1,2)
                af,gf,hist,stop=am.als(s,a,g,2047,1,self.plan['ALS_relative_J_tolerance'],self.plan['ALS_product_tolerance'])
                q,_=read_tensors(self.parent/'quantized'/(slug(name)+'.safetensors'));error=(q['W0'].double()-q['Wq'].double()).cuda()
                solved,svd=am.weighted_svd(error,af,gf,64,self.plan['eta_A'],self.plan['eta_G'])
                first=read(self.sample_path(name,0,0).with_suffix('.json'));second=read(self.sample_path(name,0,1).with_suffix('.json'))
                audits.append({'module':name,'gradient_audit':first['audit'],'normal_collection_seconds':second['seconds_before_serialization'],
                               'ALS_cycle_two_samples_seconds':hist[0]['seconds'],'SVD':svd,
                               'cache_bytes_per_sample':second['file_bytes'],'raw_counts':{'A':meta['A_count'],'G':meta['G_count']}})
                del stats,a,g,af,gf,q,error,solved;clean()
        estimated_collection=sum(r['normal_collection_seconds']*self.plan['N_fit']*self.plan['K_fit'] for r in audits)
        estimated_als=sum(r['ALS_cycle_two_samples_seconds']*(self.plan['N_fit']*self.plan['K_fit']/2)*self.plan['ALS_max_iterations'] for r in audits)
        estimated_svd=sum(r['SVD']['SVD_and_recovery_seconds']*2 for r in audits)
        estimated_cache=sum(r['cache_bytes_per_sample']*self.plan['N_fit']*self.plan['K_fit'] for r in audits)/2**30
        pilot={'identity':self.identity,'passed':True,'modules':audits,'fit_collection_estimated_seconds':estimated_collection,
               'ALS_maximum_estimated_seconds':estimated_als,'four_SVD_estimated_seconds':estimated_svd,'cache_estimated_GiB':estimated_cache,
               'evaluation_cost_note':'Evaluation uses the unchanged frozen FP32 backward path; estimate 512*sum(pilot normal unit times), plus 20% and 1200 seconds for KL/setup.',
               'conservative_total_estimated_seconds':estimated_collection+estimated_als+estimated_svd+512*sum(r['normal_collection_seconds'] for r in audits)*1.2+1200,
               'fit_only':True,'time':time.time()}
        save_json(self.root/'pilot.json',pilot);self.status('PILOT_COMPLETE',passed=True,pilot=pilot)

    def fit_and_solve(self):
        histories=[];all_samples=[]
        for name in self.plan['modules']:
            folder=self.root/'factors'/slug(name);N,K=self.plan['N_fit'],self.plan['K_fit']
            with self.timed('formal_marginal_statistics',module=name):stats,meta=self.raw_stats(name,N,K,'factors')
            all_samples.extend(meta['samples'])
            a,g=am.gauge(stats['A_marg'],stats['G_marg']*(2048/2047))
            save_tensors(folder/'marginal_raw.safetensors',{'A_raw':a,'G_raw':g},{'identity':self.identity,'canonical':'A_marg times (L/T)G_marg, then product-preserving gauge','N':N,'K':K})
            fp=folder/'full_fit_raw.safetensors'
            if fp.exists():
                factors,fmeta=read_tensors(fp);assert fmeta['identity']==self.identity
                af,gf=factors['A_raw'].cuda(),factors['G_raw'].cuda();history=read(folder/'history.json')['iterations'];stop=fmeta['stop_reason']
            else:
                history=[]
                def callback(row,aa,gg):
                    history.append(row);save_json(folder/'history.json',{'identity':self.identity,'iterations':history})
                    save_tensors(folder/'als_checkpoint.safetensors',{'A':aa,'G':gg},{'identity':self.identity,'iteration':row['iteration']})
                    self.status('FITTING_FULL_CURVATURE',module=name,iteration=row['iteration'],J=row['J_after_A'],product_change=row['product_relative_change'])
                with self.timed('formal_ALS',module=name):
                    af,gf,_,stop=am.als(self.stream_S(name,N,K),a,g,2047,self.plan['ALS_max_iterations'],
                                      self.plan['ALS_relative_J_tolerance'],self.plan['ALS_product_tolerance'],callback)
                save_tensors(fp,{'A_raw':af,'G_raw':gf},{'identity':self.identity,'stop_reason':stop,'iterations':len(history),
                    'canonical':'ALS target H=mean(vec(S)vec(S).T)/T; no extra L/T','J_marginal':history[0]['J_before'],'J_fitted':history[-1]['J_after_A'],
                    'claim':'Deterministic PSD block updates, not a global-optimality certificate'})
            histories.extend(dict(module=name,stop_reason=stop,**r) for r in history)
            q,_=read_tensors(self.parent/'quantized'/(slug(name)+'.safetensors'))
            error=(q['W0'].double()-q['Wq'].double()).cuda()
            for method,aa,gg in [('marginal',a,g),('full_fit',af,gf)]:
                destination=self.root/'corrections'/slug(name)/(method+'.safetensors')
                if destination.exists():continue
                with self.timed('weighted_SVD',module=name,method=method):
                    tensors,audit=am.weighted_svd(error,aa,gg,64,self.plan['eta_A'],self.plan['eta_G'])
                    # FP32 dense deployment; its actual residual is the common q_H/KL direction.
                    c32=tensors['C64'].float().cpu();deployed=(q['Wq']+c32).float()
                    residual=q['W0'].double()-deployed.double()
                    deployed_objective=am.qmetric(residual.cuda().double(),tensors['A_solve'],tensors['G_solve'])
                    drift=abs(deployed_objective/audit['tail_energy_half']-1)
                    if drift>self.plan['deployment_objective_relative_tolerance']:raise RuntimeError('Deployment solve objective drift')
                    audit.update(identity=self.identity,module=name,method=method,C_FP32_relative_error=mo.relative(c32.double(),tensors['C64'].cpu()),
                        deployment_objective=deployed_objective,deployment_objective_relative_drift=drift,rank_note='Ideal P64@Q64 has rank at most 64; dense FP32 rounding is disclosed',
                        deployment='FP32 W_deploy = Wq + C_FP32; exact difference of FP32 weights stored in FP64, same q_H/KL intervention',R_hash=mo.digest_tensor(residual),
                        W_deploy_hash=mo.digest_tensor(deployed),C_FP32_hash=mo.digest_tensor(c32))
                    save_tensors(folder/(method+'_solve.safetensors'),{k:v for k,v in tensors.items() if k in ('A_raw','G_raw','A_solve','G_solve')},audit)
                    save_tensors(destination,{'C64':tensors['C64'],'P64':tensors['P64'],'Q64':tensors['Q64'],'C_FP32':c32,
                                  'W_deploy':deployed,'R':residual,'singular_values':tensors['singular_values']},audit)
                    save_json(destination.with_suffix('.json'),audit);del tensors,c32,deployed,residual
            del stats,a,g,af,gf,q,error;clean()
        save_csv(self.root/'fit_history.csv',histories);save_json(self.root/'data/fit_sample_manifest.json',{'identity':self.identity,'samples':all_samples})

    def freeze_candidates(self):
        files={str(p.relative_to(self.root)):sha_file(p) for folder in ('factors','corrections','data')
               for p in sorted((self.root/folder).rglob('*')) if p.is_file()}
        path=self.root/'evaluation_freeze.json'
        record={'identity':self.identity,'manifest_sha256':sha_file(self.root/'manifest.json'),'files':files,'time':time.time()}
        if path.exists():assert read(path)['files']==files
        else:save_json(path,record)
        self.status('CANDIDATES_FROZEN',frozen_files=len(files))
