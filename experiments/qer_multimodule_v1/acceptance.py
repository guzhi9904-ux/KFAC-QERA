"""Fit-only real-model check: shared gradients versus12 independent backwards."""
import argparse
import gc
from pathlib import Path
import time
import torch
from bridge import MODULES,identity,read,read_tensors,save_json,mo,require,commit,load_record
from shared_teacher import SharedTeacher
from resources import Resources
from planning import defaults


def acceptance(root,config,manifest,resources):
    p=root/'acceptance.json'
    if p.exists():return load_record(p,manifest['identity'])
    teacher=SharedTeacher(config,root,manifest['identity'],resources.timed)
    # Existing fit data only: no validation/test data or results used by pilot.
    parent=Path(config['assets'])/'exp03'
    ids,m=read_tensors(parent/'data/fit_windows.safetensors');ids=ids['input_ids'][:1]
    lt,lm=read_tensors(parent/'data/fit_samples/w00_k000.safetensors');label=lt['labels']
    from common import PLAN
    require(m['identity']==lm['identity']==PLAN['parent_identity'],'Pilot parent identity differs')
    require(lm['input_hash']==mo.digest_tensor(ids[0]) and lm['label_hash']==mo.digest_tensor(label),'Pilot parent data changed')
    names=config['modules'];checks={};timings={}
    try:
        ref,xs=teacher.reference_all(ids,names)
        for i in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(i)
        start=time.monotonic()
        with resources.timed('pilot_shared_backward',modules=len(names)):shared=teacher.gradient_all(ids,names,ref,xs,label)
        timings['shared_backward_seconds']=time.monotonic()-start
        timings['independent_backward_seconds']=0.
        for name in names:
            target=teacher.model.get_submodule(name);x=xs[name].to(target.weight.device);start=time.monotonic()
            with resources.timed('pilot_independent_backward',module=name):
                g,audit=teacher.gradient(name,ids,ref,x,label,audit=True)
                relative=mo.relative(shared[name]['g'].double(),g.cpu())
                require(relative<=1e-5,'Shared versus independent gradient differs: '+name)
                checks[name]=dict(gradient_relative=relative,independent_weight_autograd=audit)
            timings['independent_backward_seconds']+=time.monotonic()-start
            del g,x
        # A separate uninstrumented-gradient baseline excludes weight-autograd
        # audit work. Both paths materialize native FP32 gradients on the host.
        start=time.monotonic()
        with resources.timed('pilot_independent_backward_without_weight_audit',modules=len(names)):
            for name in names:
                x=xs[name].to(teacher.model.get_submodule(name).weight.device)
                g,_=teacher.gradient(name,ids,ref,x,label,audit=False)
                host_g=g.float().cpu();del g,host_g,x
        timings['independent_without_weight_audit_seconds']=time.monotonic()-start
        logits=teacher.reference_logits(ref);score=teacher.scores(ids,logits)
        require(abs(score['KL'])<=1e-10,'Cached reference self-KL failed')
        peaks={str(i):torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())}
        require(all(v<=21.5 for v in peaks.values()),'Pilot GPU peak exceeds21.5GiB')
        return commit(p,manifest['identity'],passed=True,fit_only=True,modules=names,checks=checks,timings=timings,
            GPU_peak_allocated_GiB=peaks,self_score=score,scope='Shared-gradient and evaluation arithmetic acceptance; not a complete12-module timing benchmark')
    finally:teacher.unload();gc.collect()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--from-config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=read(args.from_config);config=defaults()
    config.update({k:base[k] for k in ('assets','model','cpu_threads','vocab_chunk')});config.update(profile='dual4090',budget_hours=1.,modules=MODULES)
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True);manifest=identity(config)
    if (root/'manifest.json').exists():require(read(root/'manifest.json')==manifest,'Pilot source/config changed; use a fresh output')
    else:save_json(root/'manifest.json',manifest)
    from bridge import lock
    with lock(root):
        resources=Resources(root,manifest['identity'],config)
        try:acceptance(root,config,manifest,resources)
        finally:resources.flush()
    print('SHARED_GRADIENT_ACCEPTANCE_PASS',flush=True)


if __name__=='__main__':main()
