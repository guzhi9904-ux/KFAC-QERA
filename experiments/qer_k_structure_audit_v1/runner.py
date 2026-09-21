import argparse
import os
import signal
import time
import torch
from k_common import (Path,LAYERS,WINDOWS,PLAN,read,save_json,require,identity,assets,
                      commit,load_record,checked_files,file_table,lock,Resources,ReplayTeacher,slug,mo)
from k_capture import capture,mapping,full_reference
from v_capture import local_delta
from k_math import compare,probe_layer
from k_report import report
from v_capture import check_aggregation
from storage import save_tensors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--ko-run',type=Path,required=True)
    parser.add_argument('--v-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--assets-only',action='store_true');args=parser.parse_args()
    for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[key]='1'
    require(not __import__('sys').flags.optimize,'Python -O forbidden')
    torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=args.output.resolve();kroot=args.ko_run.resolve();vroot=args.v_run.resolve()
    manifest=identity(kroot,vroot);ident=manifest['identity'];config=manifest['ko_manifest']['config']
    for src in (kroot,vroot,Path(config['parent_run'])):
        require(not root.is_relative_to(src) and not src.is_relative_to(root),'Output overlaps source')
    root.mkdir(parents=True,exist_ok=True)
    with lock(root):
        if (root/'manifest.json').exists():require(read(root/'manifest.json')==manifest,'Frozen audit identity changed')
        else:save_json(root/'manifest.json',manifest)
        commit(root/'frozen_probe_plan.json',ident,**PLAN)
        if (root/'complete.json').exists():checked_files(root,load_record(root/'complete.json',ident)['files']);print('AUDIT_ALREADY_COMPLETE');return
        config=dict(config,budget_hours=2.,disk_limit_GiB=8.)
        resources=Resources(root,ident,config);teacher=ReplayTeacher(config,root,ident,resources.timed)
        for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:setattr(resources,'stop',True))
        try:
            source,ks,residuals,baselines=assets(root,config,ident,kroot,resources)
            print('K_REQUIRED_ASSETS_VERIFIED',flush=True)
            if args.assets_only:return
            for j,window in enumerate(WINDOWS):
                pending=[layer for layer in (20,0,10,31) if not (root/'probes'/f'w{window:04d}_L{layer}.json').exists()]
                for layer in LAYERS:
                    p=root/'probes'/f'w{window:04d}_L{layer}.json'
                    if p.exists():checked_files(root,load_record(p,ident)['files'])
                if not pending:continue
                if j>=2:
                    acceptance_path=root/'acceptance/gradient_identity.json'
                    if not acceptance_path.exists():
                        paths=[root/'probes'/f'w{w:04d}_L{layer}.json' for w in WINDOWS[:2] for layer in LAYERS]
                        for p in paths:require(load_record(p,ident)['passed'],'Pilot not accepted')
                        commit(acceptance_path,ident,passed=True,windows=WINDOWS[:2],layers=LAYERS,
                               files=file_table(root,paths),FP32=1e-5,FP64=1e-10)
                    load_record(acceptance_path,ident)
                pilot=j<2;row=source.index['fit'][window];label=source.label(row);ids=source.ids[window:window+1]
                teacher.load();started=time.monotonic()
                with resources.timed('shared_capture',window=window):ref,states=capture(teacher.model,ids)
                mp=mapping(teacher.model,states)
                commit(root/'acceptance/rope_gqa_mapping.json',ident,passed=True,**mp)
                full=None
                if pilot:
                    with resources.timed('full_teacher_backward',window=window):full=full_reference(teacher,ids,label,ref)
                for layer in pending:
                    resources.boundary();name=f'model.layers.{layer}.self_attn.k_proj';state=states[layer]
                    require(torch.equal(state['x'],source.x(name,window)),'Cached X differs')
                    down=f'model.layers.{layer}.mlp.down_proj';seed=source.gradient(down,row,label)
                    tensors,meta=ks.get(kroot/'cache/g'/slug(name)/(row['id']+'.safetensors'));g=tensors['g']
                    with resources.timed('local_MLP_VJP',window=window,layer=layer):
                        lam,delta=local_delta(teacher.model,layer,state['r'],seed,state['mlp_output'])
                    device=teacher.model.get_submodule(name).weight.device
                    acceptance=dict(attention_aggregation=check_aggregation(state,mp['mapping'],128,device))
                    if pilot:
                        acceptance.update(down_cache=compare(seed,full[down].reshape_as(seed),1e-5),
                                          K_cache=compare(g,full[name].reshape_as(g),1e-5),
                                          lambda_replay=compare(lam,full[f'model.layers.{layer}.self_attn.o_proj'],1e-5))
                    with resources.timed('layer_mechanism_audit',window=window,layer=layer):
                        result,vectors=probe_layer(state,delta,g,residuals[name],device,window,layer,j,resources,
                                                   pilot=pilot,full=full[name+'.weight'] if pilot else None)
                    tp=root/'contractions'/f'w{window:04d}_L{layer}.safetensors'
                    with resources.timed('probe_IO',window=window,layer=layer):
                        save_tensors(tp,vectors,dict(identity=ident,window=window,layer=layer,methods=PLAN['methods'],
                                                    note='FP64 native-edge source/query; cached_source also retained'))
                        commit(root/'probes'/f'w{window:04d}_L{layer}.json',ident,passed=True,window=window,layer=layer,
                               pilot=pilot,acceptance=acceptance,token_hash=row['token_hash'],label_hash=mo.digest_tensor(label),
                               files=file_table(root,[tp]),**result)
                    del states[layer],result,vectors
                    print('K_MODULE_AUDIT_COMMITTED',window,layer,flush=True)
                if j==1:
                    paths=[root/'probes'/f'w{w:04d}_L{layer}.json' for w in WINDOWS[:2] for layer in LAYERS]
                    for p in paths:require(load_record(p,ident)['passed'],'Pilot not accepted')
                    commit(root/'acceptance/gradient_identity.json',ident,passed=True,windows=WINDOWS[:2],layers=LAYERS,
                           files=file_table(root,paths),FP32=1e-5,FP64=1e-10)
                    timings=resources.data['timings'];layer_cost=sum(t['seconds'] for t in timings if t['stage']=='layer_mechanism_audit')/8
                    forward=sum(t['seconds'] for t in timings if t['stage']=='shared_capture')/2
                    save_json(root/'resources/pilot_estimate.json',dict(remaining_core_seconds=6*(4*layer_cost+forward),
                              note='Measured pilot extrapolation, excludes IO and local VJP; no N256 candidate',GPU_peak=resources.data.get('GPU_peak_allocated_GiB')))
                    print('STAGE_A_PASSED_ALL_FOUR_K_LAYERS',flush=True)
                del ref,states,full
                print('K_AUDIT_WINDOW_COMPLETE',j+1,'/8',round(time.monotonic()-started,2),'seconds',flush=True)
            report(root,ident,resources,baselines)
        except BaseException as exc:
            done=[str(p.relative_to(root)) for p in (root/'probes').glob('*.json')]
            save_json(root/'incomplete.json',dict(identity=ident,error=repr(exc),committed_modules=done,required_modules=32,
                                                resume='Same command; no silent budget extension or reduced scope'))
            print('K_AUDIT_STOPPED',repr(exc),flush=True);raise
        finally:
            teacher.unload();resources.flush()
            if not (root/'complete.json').exists():
                save_json(root/'resources/timings.json',resources.data)
                save_json(root/'resources/peak_memory.json',dict(GPU_peak_allocated_GiB=resources.data.get('GPU_peak_allocated_GiB'),
                          output_GiB=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())/2**30))


if __name__=='__main__':main()
