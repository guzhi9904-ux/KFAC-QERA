import torch
from ko_common import LAYERS,NEW,OLD,require,commit,load_record,mo,slug,TensorStore
from ko_replay import capture_blocks,local_gradient,suffix_hidden
from bridge import hidden_forward


def accept(root,config,ident,resources,teacher,source):
    p=root/'acceptance.json'
    if p.exists():return load_record(p,ident)
    teacher.load();checks=[];local_pass=True;prefix_pass=True;store=TensorStore(ident,0)
    names=NEW+[n for n in OLD if not n.endswith('down_proj')]
    for index in (0,1):
        row=source.index['fit'][index];ids=source.ids[index:index+1];label=source.label(row)
        with resources.timed('pilot_reference',window=index):ref,states=capture_blocks(teacher.model,ids)
        for device in range(2):torch.cuda.reset_peak_memory_stats(device)
        with resources.timed('pilot_full_shared_backward',window=index):full=teacher.gradient_all(ids,names,ref,None,label)
        for layer in LAYERS:
            selected=[n for n in names if int(n.split('.')[2])==layer]
            seed=source.gradient(f'model.layers.{layer}.mlp.down_proj',row,label)
            # Parent q/v must agree with a fresh full graph even if local replay falls back.
            for n in selected:
                if n.endswith(('.q_proj','.v_proj')):
                    old=source.gradient(n,row,label);error=mo.relative(full[n]['g'].double(),old.double())
                    require(error<=config['gradient_tolerance'],'Full teacher versus frozen q/v gradient differs')
                    checks.append(dict(window=index,module=n,check='full_vs_parent',relative=error))
            try:
                with resources.timed('pilot_local_backward',window=index,layer=layer):got,out_error=local_gradient(teacher.model,layer,states[layer],seed,selected)
                for n in selected:
                    error=mo.relative(got[n]['g'].double(),full[n]['g'].double())
                    ok=error<=config['gradient_tolerance'] and torch.equal(got[n]['x'],full[n]['x'])
                    local_pass=local_pass and ok
                    checks.append(dict(window=index,module=n,check='local_vs_full',relative=error,output_relative=out_error,passed=ok))
                del got
            except RuntimeError as exc:
                local_pass=False;checks.append(dict(window=index,layer=layer,local_replay_error=str(exc)))
        # Independent weight-gradient identity for each new module, once per pilot.
        if index==0:
            for n in NEW:
                x=full[n]['x'].to(teacher.model.get_submodule(n).weight.device)
                with resources.timed('pilot_weight_autograd',module=n):g,audit=teacher.gradient(n,ids,ref,x,label,audit=True)
                error=mo.relative(g.float().cpu().double(),full[n]['g'].double())
                require(error<=config['gradient_tolerance'],'Independent versus shared gradient differs')
                checks.append(dict(module=n,check='independent_weight_autograd',gradient_relative=error,audit=audit))
                del g,x
        del full
        logits=teacher.reference_logits(ref)
        for n in config['modules']:
            w=store.get(root/'quantized'/(slug(n)+'.safetensors'))[0]['Wq'];layer=int(n.split('.')[2])
            with resources.timed('pilot_prefix_equivalence',window=index,module=n),teacher.deploy({n:w}),torch.no_grad():
                full_h=hidden_forward(teacher.model,ids);full_score=teacher.scores_hidden(ids,logits,full_h)
                try:
                    fast_h=suffix_hidden(teacher.model,layer,states[layer]);fast_score=teacher.scores_hidden(ids,logits,fast_h)
                    error=mo.relative(fast_h,full_h)
                    ok=(error<=config['forward_tolerance'] and abs(fast_score['KL']-full_score['KL'])<=max(1e-12,abs(full_score['KL'])*1e-7)
                        and abs(fast_score['NLL_sum']-full_score['NLL_sum'])<=1e-7)
                    prefix_pass=prefix_pass and ok
                    checks.append(dict(window=index,module=n,check='prefix_quantized_vs_full',hidden_relative=error,passed=ok))
                    del fast_h
                except RuntimeError as exc:
                    prefix_pass=False;checks.append(dict(window=index,module=n,prefix_error=str(exc)))
                del full_h
        del logits,ref,states
    peaks={str(i):torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)}
    require(all(x<=21.5 for x in peaks.values()),'Pilot peak exceeds21.5GiB')
    result=commit(p,ident,passed=True,fit_windows=[0,1],fit_only=True,checks=checks,GPU_peak_allocated_GiB=peaks,
                  gradient_mode='local_down_seed' if local_pass else 'shared_full',
                  evaluation_mode='cached_prefix' if prefix_pass else 'full_forward',
                  loss_reduction='sum',parent_identity=source.identity,
                  fallback='Numerical failure disables shortcut; no relaxed thresholds or approximate gradients')
    teacher.unload();print('INCREMENT_ACCEPTANCE_PASS',result['gradient_mode'],result['evaluation_mode'],flush=True)
    return result
