import math
import time
from bridge import read,save_json,torch,mo,clean,capture,hidden_forward,save_csv,sha_file
from functional_analysis import CANDIDATES,METHODS,summarize,summarize_kl


class Evaluation:
    def new_unit(self,c,k,reference,x,rx,candidates,sample,pilot=False):
        path=self.root/'new_labels/records'/f'w{c:02d}_k{k:03d}.json'
        if path.exists():
            record=read(path);assert record['identity']==self.identity and record['label_hash']==sample[1]['label_hash']
            assert set(record['scores'])==set(CANDIDATES)
            for n in CANDIDATES:assert record['scores'][n]['R_hash']==candidates[n]['R_hash']
            return record
        self.boundary()
        with self.timed('new_gradient_and_17_projections',module=self.name,window=c,replicate=k):
            gradient,audit=self.gradient(self.name,self.fit['input_ids'][c:c+1],reference,x,sample[0])
            scores={};projection_audits={}
            if c==0 and k==0:s=gradient.T@x.reshape(2048,-1).double()
            for n in CANDIDATES:
                d=float((gradient*rx[n]).sum());b=d*d/(2*2047);assert math.isfinite(b)
                scores[n]=dict(d=d,b=b,R_hash=candidates[n]['R_hash'])
                if c==0 and k==0:
                    product=s*candidates[n]['R'].cuda();alternate=float(product.sum());scale=float(product.abs().sum())
                    difference=abs(d-alternate);assert difference<=1e-10*scale,(n,difference,scale)
                    projection_audits[n]=dict(d_direct=d,d_S=alternate,absolute_difference=difference,absolute_sum_scale=scale)
                    del product
            record=dict(identity=self.identity,module=self.name,window=c,replicate=k,T=2047,L=2048,
                label_hash=sample[1]['label_hash'],input_hash=sample[1]['input_hash'],scores=scores,
                gradient_audit=audit,projection_audits=projection_audits,pilot_sample_included_once=(c==0 and k==0))
            save_json(path,record);del gradient
            if c==0 and k==0:del s
        self.status('NEW_LABELS',module=self.name,records=len(list(path.parent.glob('*.json'))),total=128)
        return record

    def kl_unit(self,c,label,candidate,reference,x,h):
        path=self.root/'kl/records'/f'w{c:02d}_{label}.json'
        if path.exists():
            row=read(path);assert row['identity']==self.identity and row['R_hash']==candidate['R_hash'] and row['W_hash']==candidate['W_hash']
            return row
        self.boundary();target=self.model.get_submodule(self.name)
        with self.timed('actual_KL',module=self.name,window=c,candidate=label,repeats=2 if c==0 else 1):
            original=target.weight.detach().clone();original_hash=mo.digest_tensor(original);values=[];audits=[]
            try:
                for repeat in range(2 if c==0 else 1):
                    with torch.no_grad():target.weight.copy_(candidate['W'].to(target.weight.device))
                    assert mo.digest_tensor(target.weight)==candidate['W_hash']
                    with torch.no_grad(),capture(target) as state:
                        actual=hidden_forward(self.model,self.fit['input_ids'][c:c+1])
                    assert torch.equal(state['x'],x)
                    expected=-x.double()@candidate['R'].cuda().T
                    audit=mo.comparison(state['h'].detach().double()-h.double(),expected)
                    assert audit['relative_l2']<=.02 and audit['cosine']>=.999,('Intervention output path',audit)
                    value=self.kl_hidden(reference,actual)['KL_mean'];assert math.isfinite(value) and value>=0
                    values.append(value);audits.append(audit)
                    with torch.no_grad():target.weight.copy_(original)
                    assert mo.digest_tensor(target.weight)==original_hash
                    del state,actual,expected
            finally:
                with torch.no_grad():target.weight.copy_(original)
                assert mo.digest_tensor(target.weight)==original_hash
            difference=abs(values[0]-values[1]) if c==0 else 0.
            if c==0:assert difference<=max(1e-12,1e-7*abs(values[0]))
            row=dict(identity=self.identity,module=self.name,window=c,candidate=label,T=2047,KL=values[0],
                repeat_KL=values[1] if c==0 else None,repeat_difference=difference,repeated=c==0,
                W_hash=candidate['W_hash'],R_hash=candidate['R_hash'],output_audits=audits,
                restoration_verified=True,distribution='full-vocabulary teacher-to-fixed-deployment KL; FP64 stable KL, FP32 teacher/eager')
            save_json(path,row);del original
        return row

    def evaluate(self,pilot=False):
        frozen=self.validate_freeze();self.checked_model();candidates=self.candidates()
        kl_names=('None',)+tuple(f'{m}-r64' for m in METHODS)
        for n,v in candidates.items():
            if n not in kl_names:v['W']=None # Keep only five dense weights; all R on host, projected Rx on device.
        for c in ([0] if pilot else range(8)):
            self.boundary();reference,x,h=self.reference(c)
            assert self.kl_hidden(reference,reference)['KL_mean']<=1e-10
            samples=self.new_samples(c,reference)
            with self.timed('precompute_17_Rx',module=self.name,window=c):
                xd=x.reshape(2048,-1).double()
                rx={n:xd@candidate['R'].cuda().T for n,candidate in candidates.items()}
                del xd
            for k in ([0] if pilot else range(16)):self.new_unit(c,k,reference,x,rx,candidates,samples[k],pilot)
            del rx;clean()
            for n in kl_names:self.kl_unit(c,n,candidates[n],reference,x,h)
            del reference,x,h,samples;clean()
        self.unload();self.validate_freeze()
        del candidates;clean()
        if pilot:
            row=read(self.root/'new_labels/records/w00_k000.json')
            assert len(row['projection_audits'])==17
            save_json(self.root/'pilot_complete.json',dict(identity=self.identity,module=self.name,passed=True,
                shape=read(self.exp3/'teacher_identity.json')['tensor_hashes'][self.name+'.weight']['shape'],
                sample_included_once=True,new_sample_checks=row['projection_audits'],
                complete_shapes=True,roots_and_both_SVD=True,all_17_projections=True,all_5_KL_and_repeats=True))
            self.status('MODULE_PILOT_COMPLETE',module=self.name)
        else:
            records=[read(p) for p in sorted((self.root/'new_labels/records').glob('*.json'))]
            klrows=[read(p) for p in sorted((self.root/'kl/records').glob('*.json'))]
            assert len(records)==128 and len(klrows)==40
            analysis=summarize(self.name,records,16,True);klanalysis=summarize_kl(self.name,klrows)
            save_json(self.root/'new_labels/summary.json',analysis);save_json(self.root/'kl/summary.json',klanalysis)
            rows=[dict(module=self.name,window=r['window'],replicate=r['replicate'],candidate=n,
                label_hash=r['label_hash'],input_hash=r['input_hash'],T=2047,**score) for r in records for n,score in r['scores'].items()]
            assert len(rows)==2176
            save_csv(self.root/'new_labels/scores.csv',rows);save_csv(self.root/'kl/by_window.csv',klrows)
            save_json(self.root/'complete.json',dict(identity=self.identity,module=self.name,passed=True,
                new_module_samples=128,new_score_rows=2176,fit_module_samples=32,KL_rows=40,KL_repeats=5,
                candidate_freeze_sha256=sha_file(self.root/'candidate_freeze.json'),completed=time.time()))
            self.status('MODULE_COMPLETE',module=self.name)
