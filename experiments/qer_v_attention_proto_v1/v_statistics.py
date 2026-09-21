import torch
from v_common import (LAYERS,MODULES,require,read,save_json,save_tensors,read_tensors,sha_file,mo,digest,
                      commit,load_record,TensorStore,slug)
from v_capture import capture,local_delta
from v_math import effective,compare,normalize


def checkpoint(root,ident,sums,count):
    folder=root/'statistics/progress';folder.mkdir(parents=True,exist_ok=True)
    tensors={f'{k}_{layer}':v for layer,values in sums.items() for k,v in values.items()}
    path=folder/f'generation_{count:06d}.safetensors'
    meta=dict(identity=ident,count=count,tensor_hashes={k:mo.digest_tensor(v) for k,v in tensors.items()})
    save_tensors(path,tensors,meta)
    pointer=dict(identity=ident,count=count,file=path.name,file_sha256=sha_file(path),tensor_hashes=meta['tensor_hashes'])
    save_json(folder/'latest.json',dict(pointer,record_sha256=digest(pointer)))
    # Current and previous complete generations only; no parent files are touched.
    generations=sorted(folder.glob('generation_*.safetensors'))
    keep={path}
    older=[p for p in generations if p.name<path.name]
    if older:keep.add(older[-1])
    for old in generations:
        if old not in keep:
            require(old.resolve().parent==folder.resolve(),'Checkpoint deletion escaped own output')
            old.unlink()


def restore(root,ident,devices):
    pointer=root/'statistics/progress/latest.json'
    if not pointer.exists():return 0,{i:dict(A_sum=torch.zeros(4096,4096,dtype=torch.float64,device=devices[i]),
                                                G_blocks_sum=torch.zeros(8,128,128,dtype=torch.float64,device=devices[i])) for i in LAYERS}
    row=load_record(pointer,ident);path=pointer.parent/row['file']
    require(path.resolve().parent==pointer.parent.resolve(),'Unsafe checkpoint path')
    require(sha_file(path)==row['file_sha256'],'Checkpoint file checksum failed')
    tensors,meta=read_tensors(path);require(meta['identity']==ident and meta['count']==row['count'] and meta['tensor_hashes']==row['tensor_hashes'],'Checkpoint metadata differs')
    require(0<row['count']<=256 and all(mo.digest_tensor(t)==row['tensor_hashes'][k] for k,t in tensors.items()),'Checkpoint content differs')
    return row['count'],{i:{k:tensors[f'{k}_{i}'].to(devices[i],copy=True) for k in ('A_sum','G_blocks_sum')} for i in LAYERS}


def collect_statistics(root,config,ident,resources,teacher,source):
    if (root/'statistics/complete.json').exists():
        from v_common import checked_files
        checked_files(root,load_record(root/'statistics/complete.json',ident)['files']);return
    teacher.load();head=read(root/'head_mapping.json');devices={i:teacher.model.model.layers[i].self_attn.v_proj.weight.device for i in LAYERS}
    count,sums=restore(root,ident,devices);start=count;checks=[]
    for window in range(start,256):
        resources.boundary();row=source.index['fit'][window];ids=source.ids[window:window+1];label=source.label(row)
        with resources.timed('shared_attention_capture',window=window):ref,states=capture(teacher.model,ids)
        for layer,name in zip(LAYERS,MODULES):
            state=states[layer];device=devices[layer]
            with resources.timed('required_asset_read',window=window,layer=layer,device=str(device)):
                x=source.x(name,window);require(torch.equal(x,state['x']),'Exact QKV input changed')
                seed=source.gradient(f'model.layers.{layer}.mlp.down_proj',row,label);old_v=source.gradient(name,row,label)
            with resources.timed('local_MLP_VJP',window=window,layer=layer,device=str(device)):
                _,delta=local_delta(teacher.model,layer,state['r'],seed,state['mlp_output'])
            with resources.timed('attention_effective_statistics',window=window,layer=layer,device=str(device)):
                values=effective(state['prob'],x,delta,head['mapping'],8,device)
                check=compare(values['g_v'],old_v,1e-5)
                require(torch.count_nonzero(delta[:,-1])==0 and torch.count_nonzero(old_v[-1])==0,'Nonzero final loss-free position gradient')
                sums[layer]['A_sum'].add_(values['A_sum']);sums[layer]['G_blocks_sum'].add_(values['G_blocks_sum'])
                checks.append(dict(window=window,layer=layer,**check));del values
            del states[layer]
        count=window+1
        # Store per-window checks independently; incomplete generations are recomputed.
        save_json(root/'statistics/window_checks'/f'w{window:04d}.json',dict(identity=ident,checks=checks[-4:]))
        if count%config['checkpoint_every']==0 or count==256:
            with resources.timed('statistics_checkpoint',windows=count):checkpoint(root,ident,sums,count)
            print('STATISTICS_COMMITTED',count,'/256',flush=True)
        del ref,states
    teacher.unload();store=TensorStore(ident,0);paths=[]
    for layer,name in zip(LAYERS,MODULES):
        a,g=normalize(sums[layer]['A_sum'],sums[layer]['G_blocks_sum'],count,2048,32)
        p=root/'statistics'/slug(name)/'raw.safetensors'
        store.put(p,dict(A_raw=a,G_raw=g),N=count,L=2048,T=2047,Hq=32,Hkv=8,d=128,
                  A_denominator=count*2048*32,G_block_denominator=count*2047,
                  canonical_G_multiplier=2048*32/2047,block_diagonal=True,head_mapping=head['mapping'])
        paths.append(p)
    from v_common import file_table
    commit(root/'statistics/complete.json',ident,windows=count,files=file_table(root,paths),all_1024_gradient_checks=True)
    print('STATISTICS_COMPLETE',flush=True)
