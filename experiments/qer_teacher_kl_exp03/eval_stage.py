import contextlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import ag_math as am
import math_ops as mo
from storage import save_json,save_csv,read_tensors,sha_file
from model_ops import capture,hidden_forward
from run import slug,clean

read=lambda p:json.loads(Path(p).read_text())
CANDIDATES=('None','SVD64','A64','Marginal-AG64','Full-fit-AG64')
OLD={'None':'R_none','SVD64':'R_svd64','A64':'R_A64'}
NEW={'Marginal-AG64':'marginal','Full-fit-AG64':'full_fit'}


class EvalStage:
    def deployed_candidates(self,name):
        q,_=read_tensors(self.parent/'quantized'/(slug(name)+'.safetensors'));candidates={}
        for label,d in OLD.items():
            deployed=(q['W0'].double()-self.directions[name][d].double()).float()
            candidates[label]={'W':deployed,'R':q['W0'].double()-deployed.double(),'origin':'frozen parent residual, identical historical intervention'}
        assert torch.equal(candidates['None']['W'],q['Wq'])
        for label,method in NEW.items():
            t,m=read_tensors(self.root/'corrections'/slug(name)/(method+'.safetensors'))
            assert m['identity']==self.identity and mo.digest_tensor(t['W_deploy'])==m['W_deploy_hash']
            assert torch.equal(t['R'],q['W0'].double()-t['W_deploy'].double())
            candidates[label]={'W':t['W_deploy'],'R':t['R'],'origin':'new frozen correction'}
        for v in candidates.values():v.update(W_hash=mo.digest_tensor(v['W']),R_hash=mo.digest_tensor(v['R']))
        return candidates

    def metric_scores(self,name,candidates):
        path=self.root/'eval'/f'{slug(name)}_metric.json'
        if path.exists():
            r=read(path);assert r['identity']==self.identity;return r['scores']
        rows=[]
        with self.timed('metric_cross_scores',module=name):
            for method in ('marginal','full_fit'):
                t,m=read_tensors(self.root/'factors'/slug(name)/(method+'_solve.safetensors'));assert m['identity']==self.identity
                for variant in ('raw','solve'):
                    a,g=t['A_'+variant].cuda(),t['G_'+variant].cuda()
                    scores={c:am.qmetric(v['R'].cuda(),a,g) for c,v in candidates.items()}
                    assert all(math.isfinite(q) and q>=0 for q in scores.values())
                    for c,v in scores.items():
                        rows.append({'module':name,'metric':method,'variant':variant,'candidate':c,'q_K':v,
                                     'gamma_K':1-v/scores['None'] if scores['None']>self.plan['ratio_denominator_floor'] else None,
                                     'R_hash':candidates[c]['R_hash'],'factor_file_sha256':sha_file(self.root/'factors'/slug(name)/(method+'_solve.safetensors'))})
                    del a,g
                del t;clean()
        save_json(path,{'identity':self.identity,'scores':rows});return rows

    def full_unit(self,name,c,k,reference,x,rx,parent_rx,candidates):
        path=self.root/'eval/records'/f'{slug(name)}_w{c:02d}_k{k:03d}.json'
        if path.exists():
            row=read(path);assert row['identity']==self.identity and set(row['scores'])==set(CANDIDATES)
            for label in CANDIDATES:assert row['scores'][label]['R_hash']==candidates[label]['R_hash']
            return row
        t,meta=read_tensors(self.parent/'samples'/f'w{c:02d}_k{k:03d}.safetensors')
        assert meta['identity']==self.plan['parent_identity'] and mo.digest_tensor(t['labels'])==meta['label_hash']
        old=read(self.parent/'records/mc'/f'{slug(name)}_w{c:02d}_k{k:03d}.json')
        with self.timed('evaluate_full',module=name,window=c,replicate=k):
            g,audit=self.gradient(name,self.val['input_ids'][c:c+1],reference,x,t['labels'])
            scores={};replay={}
            for label in CANDIDATES:
                d=float((g*rx[label]).sum());b=d*d/(2*2047)
                assert math.isfinite(b)
                scores[label]={'d':d,'b':b,'R_hash':candidates[label]['R_hash']}
            for d in OLD.values():
                signed=float((g*parent_rx[d]).sum());previous=old['directions'][d]['d']
                replay[d]={'d':signed,'parent_d':previous,'absolute_difference':abs(signed-previous)}
            row={'identity':self.identity,'module':name,'window':c,'replicate':k,'L':2048,'T':2047,
                 'label_hash':meta['label_hash'],'loss_reduction':'sum','scores':scores,'parent_replay':replay,'audit':audit}
            save_json(path,row);del g
        self.status('EVALUATING_FULL',module=name,window=c,replicate=k,committed_records=len(list((self.root/'eval/records').glob('*.json'))))
        return row

    def replay_audit(self,records,pilot=False):
        rows=[]
        for name in self.plan['modules']:
            r=sorted((x for x in records if x['module']==name),key=lambda x:(x['window'],x['replicate']))
            expected=[(0,k) for k in range(4)] if pilot else [(c,k) for c in range(8) for k in range(64)]
            assert [(x['window'],x['replicate']) for x in r]==expected
            for d in OLD.values():
                new=np.array([x['parent_replay'][d]['d'] for x in r]);old=np.array([x['parent_replay'][d]['parent_d'] for x in r])
                relative=float(np.linalg.norm(new-old)/np.linalg.norm(old))
                if relative>self.plan['parent_projection_tolerance']:raise RuntimeError('Frozen baseline replay failed')
                rows.append({'module':name,'direction':d,'n':len(r),'relative_l2':relative,'max_absolute_difference':float(np.max(np.abs(new-old)))})
        return rows

    def actual_kl(self,name,c,label,candidate,reference,x,h):
        path=self.root/'eval/kl'/f'{slug(name)}_{label}_w{c:02d}.json'
        if path.exists():
            row=read(path);assert row['identity']==self.identity and row['W_hash']==candidate['W_hash'];return row
        target=self.model.get_submodule(name)
        if label in OLD and c>0:
            # Window zero is explicitly repeated first; remaining old KL is exact same deployed weight/path.
            oldpath=self.extension/'records/kl'/f'{slug(name)}_{OLD[label]}_w{c:02d}_a1.json'
            old=read(oldpath);assert old['valid'] and old['identity']==self.plan['extension_identity']
            row={'identity':self.identity,'module':name,'window':c,'candidate':label,'KL':old['KL_mean'],'T':2047,
                 'W_hash':candidate['W_hash'],'R_hash':candidate['R_hash'],'source':'historical alpha=1 KL after window-zero path replay',
                 'source_file':str(oldpath),'source_sha256':sha_file(oldpath),'repeat_difference':0.,'repeat_note':'not repeated in this window; inherited frozen result'}
            save_json(path,row);return row
        with self.timed('actual_KL_with_repeat',module=name,window=c,candidate=label):
            original=target.weight.detach().clone();original_hash=mo.digest_tensor(original);values=[];checks=[]
            try:
                for repeat in range(2):
                    with torch.no_grad():target.weight.copy_(candidate['W'].to(target.weight.device))
                    assert mo.digest_tensor(target.weight)==candidate['W_hash']
                    with torch.no_grad(),capture(target) as state:actual=hidden_forward(self.model,self.val['input_ids'][c:c+1])
                    assert torch.equal(state['x'],x)
                    expected=-(x.double()@candidate['R'].to(x.device).T)
                    audit=mo.comparison(state['h'].detach().double()-h.double(),expected)
                    if audit['relative_l2']>.02 or audit['cosine']<.999:raise RuntimeError('Actual intervention path failed')
                    result=self.kl_hidden(reference,actual);values.append(result['KL_mean']);checks.append(audit)
                    with torch.no_grad():target.weight.copy_(original)
                    assert mo.digest_tensor(target.weight)==original_hash
                    del state,actual,expected
            finally:
                with torch.no_grad():target.weight.copy_(original)
            if label in OLD:
                oldpath=self.extension/'records/kl'/f'{slug(name)}_{OLD[label]}_w{c:02d}_a1.json'
                old=read(oldpath)
                if abs(values[0]-old['KL_mean'])>max(1e-12,1e-7*old['KL_mean']):raise RuntimeError('Historical KL reproduction failed')
            row={'identity':self.identity,'module':name,'window':c,'candidate':label,'KL':values[0],'T':2047,
                 'W_hash':candidate['W_hash'],'R_hash':candidate['R_hash'],'source':'two FP32 full-vocabulary evaluations',
                 'repeat_KL':values[1],'repeat_difference':abs(values[1]-values[0]),'output_audits':checks,'restoration_verified':True}
            save_json(path,row)
        return row

    def evaluate(self):
        frozen=read(self.root/'evaluation_freeze.json');assert frozen['identity']==self.identity
        assert not (self.root/'eval/records').exists() or frozen['time']<min(p.stat().st_mtime for p in (self.root/'eval/records').glob('*.json'))
        all_candidates={n:self.deployed_candidates(n) for n in self.plan['modules']}
        pilot=[]
        # Numerical old-direction replay is a gate before any effect interpretation.
        for name in self.plan['modules']:
            reference,x,h=self.teacher_reference(name,0)
            rx={c:x.reshape(2048,-1).double()@v['R'].cuda().T for c,v in all_candidates[name].items()}
            oldrx={d:x.reshape(2048,-1).double()@r.cuda().double().T for d,r in self.directions[name].items()}
            for k in range(4):pilot.append(self.full_unit(name,0,k,reference,x,rx,oldrx,all_candidates[name]))
            del reference,x,h,rx,oldrx;clean()
        save_json(self.root/'eval/replay_pilot.json',{'identity':self.identity,'passed':True,'audits':self.replay_audit(pilot,True)})
        metric_rows=[];kl_rows=[]
        for name in self.plan['modules']:
            metric_rows.extend(self.metric_scores(name,all_candidates[name]))
            for c in range(8):
                reference,x,h=self.teacher_reference(name,c)
                self_kl=self.kl_hidden(reference,reference)['KL_mean']
                if self_kl>1e-10:raise RuntimeError('Self-KL floor')
                rx={label:x.reshape(2048,-1).double()@v['R'].cuda().T for label,v in all_candidates[name].items()}
                oldrx={d:x.reshape(2048,-1).double()@r.cuda().double().T for d,r in self.directions[name].items()}
                for k in range(64):self.full_unit(name,c,k,reference,x,rx,oldrx,all_candidates[name])
                del rx,oldrx
                for label in CANDIDATES:
                    kl_rows.append(self.actual_kl(name,c,label,all_candidates[name][label],reference,x,h))
                    self.status('EVALUATING_KL',module=name,window=c,candidate=label)
                del reference,x,h;clean()
        records=[read(p) for p in sorted((self.root/'eval/records').glob('*.json'))];assert len(records)==1024
        audits=self.replay_audit(records)
        for n,h in frozen['files'].items():assert sha_file(self.root/n)==h,'Frozen candidate changed: '+n
        save_json(self.root/'numerical_checks.json',{'identity':self.identity,'passed':True,'formal_replay':audits,
            'candidate_freeze_unchanged':True,'pilot_sha256':sha_file(self.root/'pilot.json'),
            'maximum_KL_repeat_difference':max(x['repeat_difference'] for x in kl_rows)})
        save_csv(self.root/'eval/metric_cross_scores.csv',metric_rows);save_csv(self.root/'eval/kl_by_window.csv',kl_rows)
        flat=[dict(module=r['module'],window=r['window'],replicate=r['replicate'],candidate=c,**v,label_hash=r['label_hash']) for r in records for c,v in r['scores'].items()]
        save_csv(self.root/'eval/full_scores.csv',flat)
