import time
import torch
from v_common import LAYERS,require,commit,load_record,save_json
from v_capture import capture,mapping_check,local_delta,full_reference,check_aggregation
from v_math import effective,compare


def pilot(root,config,ident,resources,teacher,source):
    p=root/'pilot/complete.json'
    if p.exists():return load_record(p,ident)
    teacher.load();mapping=mapping_check(teacher.model);save_json(root/'head_mapping.json',mapping)
    gradient=[];replay=[];reassociation=[];started=time.monotonic();counts_before=len(resources.data['timings'])
    for window in (0,1):
        row=source.index['fit'][window];ids=source.ids[window:window+1];label=source.label(row)
        with resources.timed('pilot_capture',window=window):ref,states=capture(teacher.model,ids)
        with resources.timed('pilot_complete_teacher_backward',window=window):full=full_reference(teacher,ids,label,ref)
        for layer in (20,0,10,31):
            name=f'model.layers.{layer}.self_attn.v_proj';state=states[layer];device=teacher.model.get_submodule(name).weight.device
            old_v=source.gradient(name,row,label);down_name=f'model.layers.{layer}.mlp.down_proj';seed=source.gradient(down_name,row,label)
            require(torch.equal(state['x'],source.x(name,window)),'Exact cached QKV input replay differs')
            checks=dict(window=window,layer=layer,down_cache=compare(seed,full[down_name].reshape_as(seed),1e-5),
                        V_cache=compare(old_v,full[name].reshape_as(old_v),1e-5),attention_aggregation=check_aggregation(state,mapping['mapping'],128,device))
            with resources.timed('pilot_local_MLP_VJP',window=window,layer=layer):lam,delta=local_delta(teacher.model,layer,state['r'],seed,state['mlp_output'])
            checks.update(lambda_replay=compare(lam,full[f'model.layers.{layer}.self_attn.o_proj'],1e-5),delta_replay=compare(delta,full[f'delta_{layer}'],1e-5))
            replay.append(checks)
            with resources.timed('pilot_explicit_and_fast_statistics',window=window,layer=layer):
                values=effective(state['prob'],state['x'],delta,mapping['mapping'],8,device,explicit=True)
                reassociation.append(dict(window=window,layer=layer,check=compare(values['A_sum'],values['explicit_A_sum'],1e-10)))
                x=state['x'].reshape(2048,-1).to(device).double();cached_s=old_v.to(device).double().T@x
                gradient.append(dict(window=window,layer=layer,
                    FP64_reordering=compare(values['S_effective'],values['S_source'],1e-10),
                    FP32_weight_autograd=compare(values['S_effective'],full[name+'.weight'],1e-5),
                    cached_weight_gradient=compare(values['S_effective'],cached_s,1e-5),
                    native_V_gradient=compare(values['g_v'],old_v,1e-5)))
                del values,x,cached_s
            # Time the exact formal path separately; explicit A is pilot-only.
            with resources.timed('pilot_fast_statistics',window=window,layer=layer):
                fast=effective(state['prob'],state['x'],delta,mapping['mapping'],8,device)
            del fast
        del ref,states,full
    peaks={str(i):torch.cuda.max_memory_allocated(i)/2**30 for i in range(2)}
    require(all(v<=21.5 for v in peaks.values()),'Pilot exceeds21.5GiB allocated')
    commit(root/'pilot/gradient_identity.json',ident,passed=True,checks=gradient)
    commit(root/'pilot/local_replay.json',ident,passed=True,checks=replay)
    commit(root/'pilot/effective_A_reassociation.json',ident,passed=True,checks=reassociation)
    timings=resources.data['timings'][counts_before:];fast=sum(t['seconds'] for t in timings if t['stage']=='pilot_fast_statistics')/2
    capture_seconds=sum(t['seconds'] for t in timings if t['stage']=='pilot_capture')/2
    local=sum(t['seconds'] for t in timings if t['stage']=='pilot_local_MLP_VJP')/2
    estimate=dict(pilot_elapsed_seconds=time.monotonic()-started,formal_serial_core_estimate_seconds=256*(fast+capture_seconds+local),
                  note='Core-only estimate excludes required parent reads/checkpoints; no claimed twofold GPU scaling',timings=timings,GPU_peak_allocated_GiB=peaks)
    save_json(root/'pilot/resource_estimate.json',estimate)
    result=commit(p,ident,passed=True,windows=[0,1],layers=LAYERS,method_selection=False,FP64_tolerance=1e-10,FP32_tolerance=1e-5)
    teacher.unload();print('CHECK1_PASSED_ALL_FOUR_LAYERS',estimate['formal_serial_core_estimate_seconds'],flush=True)
    return result
