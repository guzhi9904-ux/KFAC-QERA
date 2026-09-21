import torch
import torch.nn.functional as F
from d_common import (LAYERS,MODULES,require,save_json,commit,load_record,TensorStore,
                      slug,checked_files,file_table)
from v_statistics import checkpoint,restore
from k_capture import capture,mapping,full_reference
from v_capture import local_delta,check_aggregation
from k_math import compare
from d_math import effective,normalize


def accept_pilot(root,ident,resources):
    complete=root/'pilot/complete.json'
    if complete.exists():
        checked_files(root,load_record(complete,ident)['files']);return
    paths=[root/'pilot'/f'w{w:04d}.json' for w in (0,1)]
    for p in paths:require(load_record(p,ident)['passed'],'Incomplete pilot')
    commit(root/'pilot/complete.json',ident,passed=True,windows=[0,1],layers=LAYERS,files=file_table(root,paths),
           no_surrogate_true_score_gate=True,pilot_samples_counted_once=True)
    records=resources.data['timings']
    stages=['shared_capture','local_MLP_VJP','direct_query_statistics','statistics_checkpoint']
    seconds=sum(r['seconds'] for r in records if r['stage'] in stages and r['completed']
                and (r.get('window',256)<2 or r.get('windows',256)<=2))
    save_json(root/'pilot/resource_estimate.json',dict(remaining_statistics_seconds=seconds/2*254,
         note='Pilot uses extra true/proxy score and explicit-Gram checks; estimate is conservative. Excludes solve/eval.',
         loading_seconds=sum(r['seconds'] for r in records if r['stage'] in ('load_model','teacher_parameter_identity')),
         GPU_peak=resources.data.get('GPU_peaks')))
    print('DIRECT_QUERY_PILOT_PASSED',flush=True)


def collect(root,config,ident,resources,teacher,source):
    final=root/'statistics/complete.json'
    if final.exists():checked_files(root,load_record(final,ident)['files']);return
    teacher.load();devices={i:teacher.model.model.layers[i].self_attn.k_proj.weight.device for i in LAYERS}
    count,sums=restore(root,ident,devices)
    if count>=2:accept_pilot(root,ident,resources)
    for window in range(count,256):
        resources.boundary();row=source.index['fit'][window];ids=source.ids[window:window+1];label=source.label(row)
        with resources.timed('shared_capture',window=window):ref,states=capture(teacher.model,ids)
        pilot=window<2;full=None;checks=[]
        if pilot:
            mp=mapping(teacher.model,states);commit(root/'head_mapping.json',ident,passed=True,**mp)
            with resources.timed('pilot_full_backward',window=window):full=full_reference(teacher,ids,label,ref)
        else:mp=load_record(root/'head_mapping.json',ident)
        for layer in (20,0,10,31):
            name=f'model.layers.{layer}.self_attn.k_proj';st=states[layer];device=devices[layer]
            with resources.timed('required_asset_read',window=window,layer=layer):
                x=source.x(name,window);require(torch.equal(x,st['x']),'Exact frozen X mismatch')
                down=f'model.layers.{layer}.mlp.down_proj';seed=source.down(layer,row,label)
            with resources.timed('local_MLP_VJP',window=window,layer=layer):
                lam,delta=local_delta(teacher.model,layer,st['r'],seed,st['mlp_output'])
            require(torch.count_nonzero(delta[:,-1])==0,'Final loss-free delta nonzero')
            with resources.timed('direct_query_statistics',window=window,layer=layer):values=effective(st,delta,device,pilot)
            entry=dict(window=window,layer=layer,head_checks=values['checks'])
            if pilot:
                old=source.kgrad(name,row,label);trueg=values['true_g'];xx=x.reshape(2048,4096).to(device).double()
                actual_delta=F.linear(full[f'model.layers.{layer}.self_attn.o_proj'].to(device),teacher.model.model.layers[layer].self_attn.o_proj.weight.T)
                entry.update(down_cache=compare(seed,full[down].reshape_as(seed),1e-5),
                    K_cache=compare(old,full[name].reshape_as(old),1e-5),
                    delta_connected_backward=compare(delta,actual_delta,1e-5),
                    lambda_connected_backward=compare(lam,full[f'model.layers.{layer}.self_attn.o_proj'],1e-5),
                    native_attention_aggregation=check_aggregation(st,mp['mapping'],128,device),
                    real_RoPE_chain_vs_cache=compare(trueg,old,1e-5),
                    real_RoPE_score_vs_autograd=compare(trueg.T@xx,full[name+'.weight'],1e-5),
                    explicit_gram=values['gram_identity'],surrogate_internal_identity=values['proxy_score_identity'])
                del actual_delta,xx,trueg
            sums[layer]['A_sum'].add_(values['A_sum']);sums[layer]['G_blocks_sum'].add_(values['G_blocks_sum'])
            checks.append(entry);del values,states[layer]
        if pilot:commit(root/'pilot'/f'w{window:04d}.json',ident,passed=True,window=window,checks=checks,token_hash=row['token_hash'])
        save_json(root/'statistics/window_checks'/f'w{window:04d}.json',dict(identity=ident,window=window,checks=checks))
        count=window+1
        with resources.timed('statistics_checkpoint',windows=count):checkpoint(root,ident,sums,count)
        print('DIRECT_QUERY_STATISTICS_COMMITTED',count,'/256',flush=True)
        del ref,states,full
        if count==2:accept_pilot(root,ident,resources)
    teacher.unload();store=TensorStore(ident,0);paths=[]
    for layer,name in zip(LAYERS,MODULES):
        a,g,gbar=normalize(sums[layer]['A_sum'],sums[layer]['G_blocks_sum'],256,2048,32)
        p=root/'statistics'/slug(name)/'raw.safetensors'
        store.put(p,dict(A_raw=a,G_raw=g,G_bar=gbar),N=256,L=2048,T=2047,Hq=32,Hkv=8,d=128,
                  A_denominator=256*2048*32,G_sum_includes_inverse_d=True,G_denominator=256*2047,
                  Gbar_denominator=256*2048*32,G_over_Gbar=2048*32/2047,
                  query='pre-RoPE',include_all_L_queries_even_if_e_zero=True,block_diagonal=True)
        paths.append(p)
    commit(final,ident,windows=256,files=file_table(root,paths),pilot_included_exactly_once=True)
    print('DIRECT_QUERY_STATISTICS_COMPLETE',flush=True)
