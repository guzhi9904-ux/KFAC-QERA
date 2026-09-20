"""One process: teacher collection, offline fitting, frozen evaluation, diagnostics."""
import gc
import math
from pathlib import Path
import torch
from common import PLAN,read,read_tensors,save_json,sha_file,mo,require,slug
from records import commit,load_record,checked_files,file_table
from runtime_io import TensorStore
from teacher import Teacher
import sketch_math as sm


class Experiment:
    def __init__(self,root,config,identity,resources,teacher=None,device='cuda:0'):
        self.root=root;self.config=config;self.identity=identity;self.resources=resources;self.device=device
        self.store=TensorStore(identity,config['cache_GiB']);self.name=PLAN['module']
        self.teacher=teacher or Teacher(config,root,identity,resources.timed)
        self.index=read(root/'data/sample_index.json');self.entries={r['id']:r for role in ('fit','eval') for r in self.index[role]}
        require(self.index['identity']==identity,'Sample identity differs')
        self.windows={role:read_tensors(root/'data'/f'{role}_windows.safetensors')[0]['input_ids'] for role in ('fit','eval')}
        self.documents={role:read(root/'data'/f'{role}_windows.json')['windows'] for role in ('fit','eval')}
        self.quantized=read_tensors(Path(config['quantized_source']) if config.get('quantized_source') else Path(config['assets'])/'exp01/quantized'/(slug(self.name)+'.safetensors'))[0]
        require(list(self.quantized['W0'].shape)==PLAN['shape'],'Wrong module shape')
        self.error=self.quantized['W0'].double()-self.quantized['Wq'].double()

    def path(self,kind,key):return self.root/'cache'/kind/(key+'.safetensors')
    def boundary(self):self.resources.boundary()
    def offline(self):self.teacher.unload();self.store.clear();gc.collect()
    def s(self,key,role='fit'):
        self.boundary();return self.store.get(self.path(role+'_S',key))[0]['S']
    def stream(self,keys,role='fit'):
        def iterator():
            for key in keys:yield self.s(key,role)
        from exact_ops import GroupedStream
        return GroupedStream(iterator,lambda:self.grouped_tokens(keys,role))
    def grouped_tokens(self,keys,role):
        groups={}
        for key in keys:groups.setdefault(self.entries[key]['window'],[]).append(key)
        for window,group in groups.items():
            self.boundary()
            x=self.store.get(self.path(role+'_x',f'w{window:02d}'))[0]['x'].reshape(PLAN['L'],-1)
            gradients=[self.store.get(self.path(role+'_g',key))[0]['g'] for key in group]
            yield x,gradients
    def token_stream(self,keys):
        def iterator():
            for key in keys:
                self.boundary();row=self.entries[key]
                x=self.store.get(self.path('fit_x',f'w{row["window"]:02d}'))[0]['x']
                g=self.store.get(self.path('fit_g',key))[0]['g']
                yield x.reshape(PLAN['L'],-1),g
        return iterator

    def labels(self,role,c,reference,entries):
        if role=='eval':require((self.root/'candidate_freeze.json').exists(),'Evaluation labels before candidate freeze')
        ids=self.windows[role][c:c+1];token_hash=mo.digest_tensor(ids[0]);missing=[]
        for row in entries:
            path=self.root/'data/labels'/role/(row['id']+'.safetensors')
            if path.exists():continue
            if self.config.get('paired_source'):
                from paired_data import copy_label
                copy_label(self,role,row,path)
            elif row['source']=='parent_saved':
                p=Path(self.config['assets'])/'exp03/data/fit_samples'/f'w{c:02d}_k{row["replicate"]:03d}.safetensors'
                t,m=read_tensors(p)
                require(m['identity']==PLAN['parent_identity'] and m['input_hash']==token_hash,'Original label input differs')
                require(m['label_hash']==mo.digest_tensor(t['labels']),'Original label changed')
                self.store.put(path,dict(labels=t['labels']),**row,original_file_sha256=sha_file(p),original_identity=m['identity'])
            else:missing.append(row)
        if missing:
            with self.resources.timed('sample_labels',role=role,window=c,count=len(missing)):
                labels=self.teacher.labels(reference,[r['seed'] for r in missing])
                for row,label in zip(missing,labels):self.store.put(self.root/'data/labels'/role/(row['id']+'.safetensors'),dict(labels=label),**row)
        result={}
        for row in entries:
            p=self.root/'data/labels'/role/(row['id']+'.safetensors');t,m=self.store.get(p)
            require(m['token_hash']==token_hash and m['id']==row['id'] and m['seed']==row['seed'],'Label binding changed')
            require(t['labels'].shape==(PLAN['T'],) and t['labels'].dtype==torch.int64,'Label dimensions differ')
            result[row['id']]=(t['labels'],dict(hash=m['tensor_hashes']['labels'],file_sha256=self.store.verified[str(p)][2]))
        return result

    def check_parent(self,c,k,x,s):
        result={};folder=Path(self.config['assets'])/'exp03/cache'/slug(self.name)
        xp=folder/f'x_w{c:02d}.safetensors';sp=folder/f'w{c:02d}_k{k:03d}.safetensors'
        if xp.exists():
            xt,xm=read_tensors(xp);require(mo.digest_tensor(xt['x'])==xm['input_hash'],'Parent x hash mismatch')
            result['parent_x_relative']=sm.relative(x.cpu().double(),xt['x'].double())
            require(result['parent_x_relative']<=PLAN['tolerances']['parent_x'],'Parent activation replay failed')
        if sp.exists():
            st,mm=read_tensors(sp);require(mo.digest_tensor(st['S'])==mm['S_hash'],'Parent S hash mismatch')
            result['parent_S_relative']=sm.relative(s.cpu(),st['S'])
            require(result['parent_S_relative']<=PLAN['tolerances']['parent_S'],'Parent S replay failed')
        result['old_cache_available']=sp.exists()
        return result

    def collect_fit(self,keys=None):
        chosen=set(keys or [r['id'] for r in self.index['fit']])
        for c in range(len(self.windows['fit'])):
            rows=[r for r in self.index['fit'] if r['window']==c and r['id'] in chosen]
            if not rows:continue
            missing=[r for r in rows if not self.path('fit_S',r['id']).exists() or not self.path('fit_g',r['id']).exists()]
            if not missing:
                for r in rows:self.store.get(self.path('fit_S',r['id']));self.store.get(self.path('fit_g',r['id']))
                continue
            self.boundary();ids=self.windows['fit'][c:c+1]
            with self.resources.timed('fit_reference',window=c):
                reference,x,h=self.teacher.reference(self.name,ids)
                selfkl=self.teacher.kl(reference,reference);require(abs(selfkl)<=PLAN['tolerances']['self_KL'],'Fit self-KL failed')
                xp=self.path('fit_x',f'w{c:02d}')
                self.store.put(xp,dict(x=x),role='fit',window=c,token_hash=mo.digest_tensor(ids[0]),self_KL=selfkl)
            samples=self.labels('fit',c,reference,rows)
            for row in missing:
                key=row['id'];self.boundary();label,lm=samples[key]
                with self.resources.timed('fit_gradient_cache',sample=key):
                    audit_sample=c==0 and row['replicate']<2
                    g,audit=self.teacher.gradient(self.name,ids,reference,x,label,audit=audit_sample)
                    xd=x.reshape(PLAN['L'],-1).double();s=g.T@xd
                    require(torch.equal(g,g.float().double()),'Gradient is not the exact FP32 teacher gradient')
                    if audit_sample:
                        r=self.error.to(g.device);d1=float((s*r).sum());d2=float((g*(xd@r.T)).sum())
                        audit['projection']=sm.scalar_check(d1,d2,float(s.norm()*r.norm()),PLAN['tolerances']['autograd'])
                        audit.update(self.check_parent(c,row['replicate'],x,s));del r
                    meta=dict(sample=row,label_hash=lm['hash'],label_file_sha256=lm['file_sha256'],x_hash=mo.digest_tensor(x),audit=audit,T=PLAN['T'],loss_reduction='sum')
                    self.store.put(self.path('fit_g',key),dict(g=g.float()),**meta)
                    self.store.put(self.path('fit_S',key),dict(S=s),**meta)
                    del g,s,xd
                print('FIT_COMMITTED',key,flush=True)
            del reference,x,h,samples

    def marginal_stats(self,keys,tag):
        p=self.root/'statistics'/(tag+'.safetensors')
        if p.exists():return self.store.get(p)[0]
        windows=sorted({self.entries[k]['window'] for k in keys})
        ap=self.root/'statistics'/('A8.safetensors' if windows==list(range(8)) else 'A32.safetensors' if windows==list(range(32)) else tag+'_A.safetensors')
        if ap.exists():a=self.store.get(ap)[0]['A_m'].to(self.device)
        else:
            a=None
            for c in windows:
                self.boundary();x=self.store.get(self.path('fit_x',f'w{c:02d}'))[0]['x'].reshape(PLAN['L'],-1).to(self.device).double()
                term=x.T@x;a=term if a is None else a+term
            a=sm.sym(a/(len(windows)*PLAN['L']))
            self.store.put(ap,dict(A_m=a),windows=windows,each_window_counted_once=True)
        g=None
        for key in keys:
            self.boundary();z=self.store.get(self.path('fit_g',key))[0]['g'].to(self.device).double()
            term=z.T@z;g=term if g is None else g+term
        g=sm.sym(g/(len(keys)*PLAN['T'])) # canonical (L/T)*G_m
        self.store.put(p,dict(A_m=a,G_canonical=g),samples=keys,A_asset=ap.relative_to(self.root).as_posix(),A_hash=mo.digest_tensor(a),normalization='A: D*L, G: N*T')
        return dict(A_m=a.cpu(),G_canonical=g.cpu())

    def moments(self,keys,tag,role='fit'):
        p=self.root/'statistics'/('moments_'+tag+'.safetensors')
        if p.exists():return self.store.get(p)[0]
        with self.resources.timed('sequence_moments',reference=tag,samples=len(keys)):
            a,g,n=sm.sequence_moments(self.stream(keys,role),PLAN['T'],self.device)
            self.store.put(p,dict(StS=a,SSt=g),samples=keys,N=n,T=PLAN['T'])
        return dict(StS=a.cpu(),SSt=g.cpu())

    def raw_factors(self,budget,method):
        folder=self.root/'factors'/budget/method;p=folder/'raw.safetensors'
        if p.exists():return self.store.get(p)[0]
        keys=self.index['budgets'][budget];stats=self.marginal_stats(keys,budget)
        a=stats['A_m'].to(self.device);g=stats['G_canonical'].to(self.device);history=[];stop='FIXED_CONSTRUCTION'
        if method=='Token-joint':
            g=torch.eye(len(g),dtype=torch.float64,device=self.device)
            for iteration in range(1,PLAN['token_rounds']+1):
                cp=folder/f'iteration_{iteration:02d}.safetensors'
                if cp.exists():
                    t,m=self.store.get(cp);a=t['A'].to(self.device);g=t['G'].to(self.device);history=m['history'];continue
                with self.resources.timed('token_joint_round',budget=budget,iteration=iteration,samples=len(keys)):
                    olda,oldg=a,g;a,g=sm.token_step(self.token_stream(keys),a,g,PLAN['T']);a,g=sm.gauge(a,g)
                    change,roundoff=sm.parent.product_change(a,g,olda,oldg)
                    row=dict(iteration=iteration,product_relative_change=change,distance_roundoff=roundoff,
                        A_spectrum=sm.parent.spectrum(a),G_spectrum=sm.parent.spectrum(g),synchronous=True)
                    history.append(row);self.store.put(cp,dict(A=a,G=g),history=history,samples=keys)
            stop='FIXED_THREE_SYNCHRONOUS_ROUNDS_NOT_CONVERGENCE_CLAIM'
        elif method=='Sequence-one-step':
            m=self.moments(keys,budget);a,g=sm.sequence_one_step(m['StS'].to(self.device),m['SSt'].to(self.device))
        elif method=='Full-fit':
            stable=0;a,g=sm.gauge(a,g);stop='MAX_ITERATIONS_REACHED'
            for iteration in range(1,PLAN['ALS_max_iterations']+1):
                cp=folder/f'iteration_{iteration:02d}.safetensors'
                if cp.exists():
                    t,m=self.store.get(cp);a=t['A'].to(self.device);g=t['G'].to(self.device);history=m['history'];stable=m['stable']
                else:
                    with self.resources.timed('full_fit_round',budget=budget,iteration=iteration,samples=len(keys)):
                        a,g,row=sm.full_step(self.stream(keys),a,g,PLAN['T']);row['iteration']=iteration;history.append(row)
                        stable=stable+1 if row['relative_J_improvement']<=PLAN['ALS_relative_J_tolerance'] and row['product_relative_change']<=PLAN['ALS_product_tolerance'] else 0
                        self.store.put(cp,dict(A=a,G=g),history=history,stable=stable,samples=keys)
                if stable>=PLAN['ALS_stable_rounds']:stop='TWO_CONSECUTIVE_CONVERGED_CYCLES';break
        else:require(method=='Marginal','Unexpected method')
        sm.parent.spectrum(a);sm.parent.spectrum(g);a,g=sm.gauge(sm.sym(a),sm.sym(g))
        self.store.put(p,dict(A_raw=a,G_raw=g),samples=keys,method=method,budget=budget,history=history,stop=stop)
        save_json(folder/'iterations.json',dict(identity=self.identity,method=method,budget=budget,history=history,stop=stop))
        return dict(A_raw=a.cpu(),G_raw=g.cpu())

    def candidate(self,key,a,g,budget,method):
        cp=self.root/'corrections'/(key+'.safetensors');fp=self.root/'factors'/key/'solve.safetensors'
        if cp.exists() and fp.exists():self.store.get(cp);self.store.get(fp);return
        with self.resources.timed('weighted_SVD',candidate=key):
            t,c,audit=sm.solve(self.error.to(self.device),a.to(self.device),g.to(self.device),PLAN['rank'],self.quantized['Wq'],self.quantized['W0'])
            self.store.put(fp,{k:t[k] for k in ('A_raw','G_raw','A_solve','G_solve')},candidate=key,budget=budget,method=method,audit=audit)
            self.store.put(cp,c,candidate=key,budget=budget,method=method,audit=audit,W_hash=mo.digest_tensor(c['W_deploy']))

    def fit(self):
        self.offline()
        # Shared A8 is committed serially before independent workers touch it.
        for budget in PLAN['budgets']:
            keys=self.index['budgets'][budget]
            with self.resources.timed('marginal_statistics',budget=budget):self.marginal_stats(keys,budget)
        def build(worker,budget):
            for method in PLAN['methods']:
                worker.boundary();t=worker.raw_factors(budget,method)
                worker.candidate(budget+'__'+method,t['A_raw'],t['G_raw'],budget,method)
        from parallel import offline_map
        offline_map(self,list(PLAN['budgets']),build)
        for key,budget in [('A8','S0'),('A32','S2')]:
            t=self.marginal_stats(self.index['budgets'][budget],budget);a=t['A_m'];g=torch.eye(PLAN['shape'][0],dtype=torch.float64)
            self.candidate(key,a,g,budget,'A-only')
        self.store.put(self.root/'corrections/None.safetensors',dict(R64=self.error,R_deploy=self.error.clone(),W_deploy=self.quantized['Wq']),
            candidate='None',budget=None,method='None',W_hash=mo.digest_tensor(self.quantized['Wq']))
        s0=self.store.get(self.root/'statistics/S0.safetensors')[1];s1=self.store.get(self.root/'statistics/S1.safetensors')[1]
        require(s0['A_asset']==s1['A_asset'] and s0['A_hash']==s1['A_hash'],'S0 and S1 must share exact A8')
        keys=[b+'__'+m for b in PLAN['budgets'] for m in PLAN['methods']]+['A8','A32','None']
        paths=[self.root/'corrections'/(k+'.safetensors') for k in keys]
        paths += [self.root/'factors'/k/'solve.safetensors' for k in keys if k!='None']
        paths += [self.root/'data/data_freeze.json',self.root/'data/sample_index.json',self.root/'manifest.json']
        commit(self.root/'candidate_freeze.json',self.identity,keys=keys,files=file_table(self.root,paths),
            A8_shared_exactly=True,evaluation_seen=False,selection='none: all 15 preregistered candidates evaluated')
        self.store.clear()

    def check_candidates(self):
        frozen=load_record(self.root/'candidate_freeze.json',self.identity);checked_files(self.root,frozen['files']);return frozen['keys']

    def project(self,s,key):
        tensors,_=self.store.get(self.root/'corrections'/(key+'.safetensors'))
        r=tensors['R_deploy'].to(s.device);d=float((s*r).sum());del r
        r=tensors['R64'].to(s.device);di=float((s*r).sum());del r
        return dict(d=d,squared=d*d,q=d*d/(2*PLAN['T']),ideal_d=di,ideal_q=di*di/(2*PLAN['T']),T=PLAN['T'])

    def evaluate(self):
        keys=self.check_candidates();self.store.clear()
        for c in range(PLAN['eval_windows']):
            rows=[r for r in self.index['eval'] if r['window']==c]
            qs=[self.root/'scores/atomic_q'/(r['id']+'.json') for r in rows]
            kls=[self.root/'scores/atomic_kl'/f'w{c:02d}_{k}.json' for k in keys]
            if all(p.exists() for p in qs+kls):
                for p in qs+kls:load_record(p,self.identity)
                continue
            self.boundary();ids=self.windows['eval'][c:c+1]
            with self.resources.timed('eval_reference',window=c):
                ref,x,h=self.teacher.reference(self.name,ids)
                selfkl=self.teacher.kl(ref,ref);require(abs(selfkl)<=PLAN['tolerances']['self_KL'],'Eval self-KL failed')
                commit(self.root/'scores/self_KL'/f'w{c:02d}.json',self.identity,window=c,value=selfkl,input_hash=mo.digest_tensor(ids[0]))
                self.store.put(self.path('eval_x',f'w{c:02d}'),dict(x=x),window=c,token_hash=mo.digest_tensor(ids[0]))
            labels=self.labels('eval',c,ref,rows)
            for row,qp in zip(rows,qs):
                self.boundary()
                if qp.exists():load_record(qp,self.identity);continue
                key=row['id'];sp=self.path('eval_S',key);gp=self.path('eval_g',key);label,lm=labels[key]
                if sp.exists() and gp.exists():
                    s=self.store.get(sp)[0]['S'].to(x.device);self.store.get(gp)
                else:
                    with self.resources.timed('eval_gradient_cache',sample=key):
                        g,audit=self.teacher.gradient(self.name,ids,ref,x,label,audit=False);s=g.T@x.reshape(PLAN['L'],-1).double()
                        self.store.put(sp,dict(S=s),sample=row,label_hash=lm['hash'],label_file_sha256=lm['file_sha256'],audit=audit,T=PLAN['T'])
                        require(torch.equal(g,g.float().double()),'Eval gradient not exactly FP32 representable')
                        self.store.put(gp,dict(g=g.float()),sample=row,label_hash=lm['hash'],label_file_sha256=lm['file_sha256'],audit=audit,T=PLAN['T'])
                        del g
                with self.resources.timed('all_candidate_projections',sample=key):
                    scores={candidate:self.project(s,candidate) for candidate in keys}
                    commit(qp,self.identity,sample=row,label_hash=lm['hash'],scores=scores,
                        candidate_freeze_sha256=sha_file(self.root/'candidate_freeze.json'),S_hash=mo.digest_tensor(s))
                del s
            for key,kp in zip(keys,kls):
                self.boundary()
                if kp.exists():load_record(kp,self.identity);continue
                t,m=self.store.get(self.root/'corrections'/(key+'.safetensors'))
                with self.resources.timed('actual_KL',window=c,candidate=key):
                    value,audit=self.teacher.intervention_kl(self.name,ids,ref,x,h,t['W_deploy'],t['R_deploy'])
                    require(math.isfinite(value) and value>=0,'Invalid KL')
                    repeat=path=None
                    if c==0:
                        second,_=self.teacher.intervention_kl(self.name,ids,ref,x,h,t['W_deploy'],t['R_deploy'])
                        allowed=max(PLAN['tolerances']['KL_repeat_absolute'],PLAN['tolerances']['KL_repeat_relative']*abs(value))
                        require(abs(second-value)<=allowed,'KL repeat failed')
                        alternate=self.teacher.output_perturbation_kl(self.name,ids,ref,x,t['R_deploy'])
                        path_allowed=max(PLAN['tolerances']['KL_repeat_absolute'],PLAN['tolerances']['KL_path_relative']*max(abs(value),abs(alternate)))
                        require(abs(alternate-value)<=path_allowed,'Direct/output-perturbation KL differs >2%')
                        repeat=dict(value=second,absolute_error=abs(second-value),allowed=allowed)
                        path=dict(value=alternate,absolute_error=abs(alternate-value),allowed=path_allowed)
                    commit(kp,self.identity,window=c,candidate=key,value=value,W_hash=m['W_hash'],self_KL=selfkl,
                        repeat=repeat,output_path=path,intervention=audit,input_hash=mo.digest_tensor(ids[0]))
                del t
            del ref,x,h,labels
            print('EVAL_WINDOW_COMPLETE',c+1,'/',PLAN['eval_windows'],flush=True)
        self.offline()

    def pilot(self):
        p=self.root/'pilot.json'
        if p.exists():
            result=load_record(p,self.identity);require(result['passed'],'Existing pilot failed')
            if not (self.root/'budget_freeze.json').exists():self.estimate()
            return result
        keys=[self.index['fit'][0]['id'],self.index['fit'][1]['id']]
        self.collect_fit(keys);self.offline()
        with self.resources.timed('pilot_marginal'):
            stats=self.marginal_stats(keys,'pilot');a=stats['A_m'].to(self.device);g=stats['G_canonical'].to(self.device)
        with self.resources.timed('pilot_token_contraction'):
            at,gt=sm.token_step(self.token_stream(keys),a,torch.eye(len(g),dtype=torch.float64,device=self.device),PLAN['T'])
        with self.resources.timed('pilot_token_fixed'):
            at,gt=sm.gauge(at,gt);sm.parent.product_change(at,gt,a,g)
            sm.parent.spectrum(at);sm.parent.spectrum(gt)
        with self.resources.timed('pilot_sequence_moments'):
            ma,mg,_=sm.sequence_moments(self.stream(keys),PLAN['T'],self.device)
            ao,go=sm.sequence_one_step(ma,mg);sm.parent.spectrum(ao);sm.parent.spectrum(go)
        with self.resources.timed('pilot_full_round'):af,gf,full_audit=sm.full_step(self.stream(keys),a,g,PLAN['T'])
        with self.resources.timed('pilot_SVD'):
            factors,correction,solve_audit=sm.solve(self.error.to(self.device),a,g,PLAN['rank'],self.quantized['Wq'],self.quantized['W0'])
            self.store.put(self.root/'pilot/correction.safetensors',correction,audit=solve_audit)
        with self.resources.timed('pilot_Gram'):
            gram,hnorm2=sm.gram_blocked(keys,lambda k:self.s(k),PLAN['T'],self.device,block=2,boundary=self.boundary)
            s0=self.s(keys[0]).to(self.device);s1=self.s(keys[1]).to(self.device)
            direct=float((s0*s1).sum());gram_check=sm.scalar_check(float(gram[0,1]),direct,float(s0.norm()*s1.norm()))
        # A throughput-only block benchmark reuses the SAME two fit samples. It
        # introduces no new samples and is not used to estimate curvature accuracy.
        self.store.clear()
        with self.resources.timed('pilot_Gram_two_sample_load'):
            packed=torch.stack([self.s(k).reshape(-1) for k in keys]).to(self.device)
        block=self.config['gram_block']
        with self.resources.timed('pilot_Gram_block_compute'):
            repeated=packed.repeat((math.ceil(block/2),1))[:block].contiguous()
            measured_gram=repeated@repeated.T
            require(bool(torch.isfinite(measured_gram).all()),'Gram throughput check failed')
        del packed,repeated,measured_gram
        with self.resources.timed('pilot_metric_contractions'):
            inner=sm.raw_metric_inner(self.stream(keys),factors['A_raw'],factors['G_raw'],PLAN['T'])
            expanded=sm.solve_metric_inner(inner,factors['A_raw'],factors['G_raw'],ma,mg,solve_audit['A_damping']['lambda'],solve_audit['G_damping']['lambda'])
            direct=sm.raw_metric_inner(self.stream(keys),factors['A_solve'],factors['G_solve'],PLAN['T'])
            expansion_check=sm.scalar_check(expanded,direct,abs(direct))
            # Independent contraction via x/g, preserving all cross-position terms.
            x,z=next(self.token_stream(keys)());x=x.to(self.device).double();z=z.to(self.device).double()
            lhs=s0@a@s0.T;rhs=z.T@(x@a@x.T)@z
            err=sm.relative(lhs,rhs);require(err<=PLAN['tolerances']['contraction'],'SAS versus x/g contraction failed')
        from exact_ops import validate_reassociation
        with self.resources.timed('pilot_optimized_equivalence'):
            equivalence=validate_reassociation(self.stream(keys),a,g,PLAN['T'])
        from parallel import offline_map
        def parallel_check(worker,tag):
            stats=worker.marginal_stats(keys,'pilot')
            with worker.resources.timed('pilot_parallel_equivalence',task=tag):
                return validate_reassociation(worker.stream(keys),stats['A_m'].to(worker.device),stats['G_canonical'].to(worker.device),PLAN['T'])
        parallel_checks=offline_map(self,[0,1],parallel_check)
        del at,gt,ao,go,af,gf,ma,mg,a,g,s0,s1,x,z,lhs,rhs,factors,correction
        self.store.clear();gc.collect();torch.cuda.empty_cache() if torch.cuda.is_available() else None
        ids=self.windows['fit'][:1];ref,x,h=self.teacher.reference(self.name,ids)
        t,_=self.store.get(self.root/'pilot/correction.safetensors')
        with self.resources.timed('pilot_KL'):
            value,check=self.teacher.intervention_kl(self.name,ids,ref,x,h,t['W_deploy'],t['R_deploy'])
            alternate=self.teacher.output_perturbation_kl(self.name,ids,ref,x,t['R_deploy'])
            allowed=max(PLAN['tolerances']['KL_repeat_absolute'],PLAN['tolerances']['KL_path_relative']*max(abs(value),abs(alternate)))
            require(abs(value-alternate)<=allowed,'Pilot KL path differs >2%')
        self.offline();self.resources.flush()
        peaks=self.resources.data.get('GPU_peaks',{})
        require(all(v['allocated_GiB']<=PLAN['pilot_allocated_GiB'] for v in peaks.values()),'Pilot GPU peak exceeds 21.5 GiB')
        result=commit(p,self.identity,passed=True,fit_only=True,fit_samples_reused=keys,full_fit_round=full_audit,
            optimized_equivalence=equivalence,parallel_equivalence=parallel_checks,SVD=solve_audit,Gram=gram_check,H_norm2=hnorm2,damping_expansion=expansion_check,
            SAS_independent_relative_error=err,KL=dict(direct=value,output_path=alternate,allowed=allowed,intervention=check),
            GPU_peaks=peaks,thresholds=PLAN['tolerances'],extra_evaluation_data_seen=False)
        self.estimate()
        return result

    def estimate(self):
        timings=self.resources.data['timings']
        def mean(stage):
            rows=[r['seconds'] for r in timings if r['stage']==stage and r['completed']]
            require(bool(rows),'Missing pilot timing '+stage);return sum(rows)/len(rows)
        full=read(self.root/'pilot.json')['full_fit_round'];block=self.config['gram_block']
        blocks=[math.ceil(n/block) for n in (224,256)]
        block_pairs=sum(n*(n+1)/2 for n in blocks)
        # One X load per block and Y loads below the diagonal: n(n+1)/2 loads.
        gram_seconds=block_pairs*(block/2*mean('pilot_Gram_two_sample_load')+mean('pilot_Gram_block_compute'))
        estimates=dict(fit_collection=224*mean('fit_gradient_cache'),
            token_joint_9_rounds=3*(32+128+128)/2*mean('pilot_token_contraction')+9*mean('pilot_token_fixed'),
            full_fit_max_60_rounds=20*(32+128+128)/2*full['contraction_seconds']+60*full['fixed_seconds'],
            sequence_moments_fit_and_eval=(32+128+128+256)/2*mean('pilot_sequence_moments'),
            all_14_SVD=14*mean('pilot_SVD'),eval_gradients=256*mean('fit_gradient_cache'),
            KL_240_plus_checks=(240+30)/2*mean('pilot_KL'),
            Gram_fit_eval=gram_seconds,
            curvature_metric_contractions=(4*(32+128+128)+12*256)/4*mean('pilot_metric_contractions'))
        # Fixed eigenspectrum costs scale by rounds, never by sample count.
        total=sum(estimates.values());reserved=total*1.35+600
        return commit(self.root/'budget_freeze.json',self.identity,stage_estimated_seconds=estimates,
            measured_upper_estimate_seconds=reserved,active_seconds_at_freeze=self.resources.data['active_seconds'],
            budget_hours=self.config['budget_hours'],fits_budget=reserved+self.resources.data['active_seconds']<3600*self.config['budget_hours'],
            safety_multiplier=1.35,extra_overhead_seconds=600,
            caveats='Conservative pilot extrapolation including worst-case 20 ALS rounds; filesystem cache and matrix-block throughput may change. No sample/method reduction.')
