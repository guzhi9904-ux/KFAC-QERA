"""Pre-score engineering acceptance. Pilot statistics are discarded, labels retained."""
import gc
import tempfile
import time
import torch
from fm_common import *
from data import checked_ids,label
from collect import ordinary_forward,functional_capture,functional_terms
from solve import solve_one
from audit import original_weight
from evaluate import score_hidden,make_harness_model

def mathematical_checks(device='cpu'):
    generator=torch.Generator().manual_seed(7919)
    x=torch.randn(17,11,generator=generator,dtype=torch.float64).to(device)
    error=torch.randn(9,11,generator=generator,dtype=torch.float64).to(device)*.02
    a=x.T@x/17;w0=torch.randn(9,11,generator=generator).to(device);wq=w0-error.float()
    actual,audit=solve_one(error,a,None,w0,wq,rank=3)
    reference,ref_audit=sm.parent.weighted_svd(error,a,torch.eye(9,dtype=torch.float64,device=device),3,.001,.001)
    check=compare(actual['P64']@actual['Q64'],reference['C64'],1e-10)
    return dict(A_only_scalar_parent=check,audit=audit,parent_audit=ref_audit)

def resume_check(ctx):
    binding={'pilot':'two commits plus an uncommitted increment'}
    key='pilot_resume';folder=ctx.root/'temporary'/key
    if folder.exists():ctx.cleanup_temporary(folder)
    acc=Accumulator(ctx,key,{'A':(3,3)},binding)
    first=torch.eye(3,dtype=torch.float64);second=torch.ones((3,3),dtype=torch.float64)
    acc.add({'A':first},0);acc.save();acc.add({'A':second},1)
    resumed=Accumulator(ctx,key,{'A':(3,3)},binding)
    require(resumed.count==1 and torch.equal(resumed.values['A'],first),'Uncommitted state leaked')
    resumed.add({'A':second},1);resumed.save()
    verified=Accumulator(ctx,key,{'A':(3,3)},binding)
    require(verified.count==2 and torch.equal(verified.values['A'],first+second),'Resume omitted/duplicated a window')
    ctx.cleanup_temporary(folder)
    return {'passed':True,'exact_tensor_equality':True,'count':2}

def pilot(ctx):
    require(ctx.done('data/complete.json'),'Data/teacher/Wq must freeze before pilot')
    if ctx.done('verification/pilot_complete.json'):return
    ctx.teacher.load();model=ctx.teacher.model;ids=checked_ids(ctx,'calibration');checks={};timing={}
    require(model.config._attn_implementation=='eager' and model.config.pretraining_tp==1,'Teacher attention backend differs')
    from transformers.models.llama.modeling_llama import repeat_kv
    mapping=repeat_kv(torch.arange(8).reshape(1,8,1,1),4).flatten().tolist()
    require(mapping==[a//4 for a in range(32)],'GQA head mapping differs')
    checks['mapping']=mapping;checks['checkpoint_resume']=resume_check(ctx)
    down_a=None
    for layers in (list(range(4)),list(range(16,20))):
        ordinary=[];references=[]
        for w in range(2):
            h,terms=ordinary_forward(model,ids[w:w+1],layers)
            references.append(h.cpu());ordinary.append({i:terms[f'{i}.qkv'] for i in layers})
            if layers[0]==0:
                down_a=terms['0.down'].clone() if down_a is None else down_a+terms['0.down']
            y=label(ctx,w,h)
            require(torch.equal(y,ctx.teacher.labels(h,[mo.stream_seed(w,0,ctx.config['base_seed'])])[0]),'Frozen labels are not exactly reproducible')
            del terms,h
        am={i:(ordinary[0][i]+ordinary[1][i])/(2*L) for i in layers};del ordinary
        k_samples={i:[] for i in layers};k_terms={i:[] for i in layers}
        for w in range(2):
            start=time.monotonic()
            hidden,rows=functional_capture(model,ids[w:w+1],label(ctx,w),layers,ctx.config['vocab_chunk'])
            for d in range(2):torch.cuda.synchronize(d)
            timing[f'group_{layers[0]}_capture_{w}']=time.monotonic()-start
            compare(hidden,references[w],1e-7)
            start=time.monotonic();terms,checks_v=functional_terms(rows,layers,am,explicit=True)
            timing[f'group_{layers[0]}_contraction_with_explicit_{w}']=time.monotonic()-start
            checks[f'V_{layers[0]}_{w}']=checks_v
            # A separate parent shared-backward pass, without the new delta/P collector.
            names=[name(i,k) for i in layers for k in ('q','k','v','o')]
            reference=shared_gradient(model,names,ids[w:w+1],
                lambda h:mo.sampled_seed(h,model.lm_head.weight,label(ctx,w),ctx.config['vocab_chunk']))
            checks[f'parent_gradient_{layers[0]}_{w}']={key:compare(rows[key]['g'],reference[key]['g'],1e-5) for key in names}
            for i in layers:
                k=rows[name(i,'k')];k_samples[i].append((k['x'].reshape(L,-1).double(),k['g'].double()))
                k_terms[i].append((terms[f'{i}.K_A'],terms[f'{i}.K_G']))
            del rows,terms,reference
        for i in layers:
            device=f'cuda:{int(i>=16)}';a=am[i].to(device)
            expected_a,expected_g=sm.token_step(lambda:iter(k_samples[i]),a,torch.eye(1024,device=device,dtype=torch.float64),T)
            got_a=sum(pair[0] for pair in k_terms[i])/(2*T*1024)
            got_g=sum(pair[1] for pair in k_terms[i])/(2*T*am[i].square().sum())
            checks[f'K_first_step_{i}']=dict(A=compare(got_a,expected_a.cpu(),1e-10),G=compare(got_g,expected_g.cpu(),1e-10),
                                             initialization='two-window pilot A_M,I; production uses complete N256 A_M')
        del am,k_samples,k_terms,references
        gc.collect();torch.cuda.empty_cache()
    with torch.no_grad():
        start=time.monotonic();h,terms=ordinary_forward(model,ids[:1],list(range(8)))
        for d in range(2):torch.cuda.synchronize(d)
        timing['pass_a_group8_window']=time.monotonic()-start;del terms
        self_score=score_hidden(h,ids[:1],model.lm_head.weight,ctx.config['vocab_chunk'],h)
        require(self_score['KL']<=1e-12,'Teacher self-KL failure');checks['self_score']=self_score
        z=h[:,:128];targets=ids[:1,:128]
        score=score_hidden(z,targets,model.lm_head.weight,17)
        logits=torch.nn.functional.linear(z[0,:-1],model.lm_head.weight)
        direct=-logits.double().log_softmax(-1).gather(1,targets[0,1:].to(logits.device)[:,None]).sum()
        require(abs(score['NLL_sum']-float(direct))/max(abs(float(direct)),1e-30)<=1e-5,'Chunked CE/reference differs')
        checks['chunked_CE_relative_error']=abs(score['NLL_sum']-float(direct))/max(abs(float(direct)),1e-30)
        del logits,h,z
        long_ids=torch.cat([ids[:1],ids[1:2]],dim=1)
        h=hidden_forward(model,long_ids);require(torch.isfinite(h).all(),'4096-token downstream memory pilot failed');del h
    lm=make_harness_model(ctx)
    requests=[(None,ids[0,:64].tolist(),ids[0,64:72].tolist())]
    new=lm._loglikelihood_tokens(requests,disable_tqdm=True)
    from lm_eval.models.huggingface import HFLM
    old=HFLM._loglikelihood_tokens(lm,requests,disable_tqdm=True)
    require(new[0][1]==old[0][1] and abs(new[0][0]-old[0][0])/max(abs(old[0][0]),1e-30)<=1e-5,'Harness head-chunking reference mismatch')
    checks['harness_reference']=dict(chunked=new,parent=old,tolerance=1e-5)
    del lm;ctx.teacher.unload();gc.collect();warm_and_probe(['cuda:0','cuda:1'])
    checks['A_only_equivalence']=mathematical_checks('cuda:0')
    # Largest matrix pilot: exact 14336-dimensional A eigensolve and full down SVD.
    key=name(0,'down');row=read(ctx.root/'quantization_manifest.json')['modules'][key]
    wq=load_file(row['path'])['Wq'].to('cuda:0');w0=original_weight(ctx,key,'cuda:0')
    start=time.monotonic()
    f,audit=solve_one(w0.double()-wq.double(),(down_a/(2*L)).to('cuda:0'),None,w0,wq)
    timing['largest_down_solve']=time.monotonic()-start;checks['largest_down_solver']=audit
    del down_a,w0,wq,f;gc.collect();torch.cuda.empty_cache()
    checks['resources']=ctx.resource_snapshot();checks['timing']=timing
    write(ctx.root/'verification/pilot.json',checks)
    # Conservative: explicit V-Gram verification is included in this upper estimate.
    capture=sum(v for k,v in timing.items() if '_capture_' in k)/4
    contraction=sum(v for k,v in timing.items() if '_contraction_' in k)/4
    eta_a=4*N*timing['pass_a_group8_window'];eta_b=8*N*(capture+contraction)
    text=('# Pilot cost update\n\nPass A estimate %.2f h; Pass B conservative estimate %.2f h. '
          'Pass B pilot includes expensive explicit V algebra checks that production omits. '
          'Largest down root/SVD pilot %.1f s. Solve/evaluation ETA remains uncertain; downstream 4–12 h planning range. '
          'These are wall-time estimates; checkpoint/hash/candidate switch overhead is recorded separately. '
          'No unconfirmed currency rate is used. Personal free space >=200 GiB was user-confirmed.\n')%(eta_a/3600,eta_b/3600,timing['largest_down_solve'])
    (ctx.root/'audit/estimated_cost.md').write_text(text,encoding='utf-8')
    ctx.commit('verification/pilot_complete.json',[ctx.root/'verification/pilot.json',ctx.root/'audit/estimated_cost.md'],
               windows=[0,1],pilot_statistics_in_production=False,pilot_labels_reused=True)
    log('PILOT_ACCEPTED',pass_a_hours=eta_a/3600,pass_b_upper_hours=eta_b/3600)
