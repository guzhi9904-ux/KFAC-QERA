"""Compact shared X/G cache: one receipt only after all modules are committed."""
from bridge import TensorStore,read,mo,slug,require,commit,load_record,checked_files,file_table
from planning import input_group
from dataset import windows


def x_path(root,name,window):return root/'cache/x'/slug(input_group(name))/f'w{window:04d}.safetensors'
def g_path(root,name,key):return root/'cache/g'/slug(name)/(key+'.safetensors')


def collect(root,config,identity,resources,teacher,limit=None):
    store=TensorStore(identity,config['cache_GiB']);index=read(root/'data/index.json');ids_all=windows(root,'fit',identity)
    names=config['modules'];chosen=index['fit'] if limit is None else index['fit'][:limit]
    for c in sorted({r['window'] for r in chosen}):
        rows=[r for r in chosen if r['window']==c];missing=[]
        for row in rows:
            p=root/'cache/commits'/(row['id']+'.json')
            if p.exists():checked_files(root,load_record(p,identity)['files'])
            else:missing.append(row)
        if not missing:continue
        ids=ids_all[c:c+1]
        with resources.timed('shared_reference',window=c):ref,xs=teacher.reference_all(ids,names)
        for name in names:store.put(x_path(root,name,c),dict(x=xs[name]),window=c,input_group=input_group(name),token_hash=mo.digest_tensor(ids[0]))
        for row in missing:
            lp=root/'cache/labels'/(row['id']+'.safetensors')
            if not lp.exists():
                with resources.timed('sample_labels',window=c):label=teacher.labels(ref,[row['seed']])[0]
                store.put(lp,dict(labels=label),sample=row)
            t,lm=store.get(lp);require(lm['sample']==row,'Label sample binding mismatch');label=t['labels']
            with resources.timed('shared_gradient_cache',sample=row['id'],modules=len(names)):
                gradients=teacher.gradient_all(ids,names,ref,xs,label)
                for name in names:
                    item=gradients[name];require(mo.digest_tensor(item['x'])==mo.digest_tensor(xs[name]),'Shared X changed')
                    store.put(g_path(root,name,row['id']),dict(g=item['g']),module=name,sample=row,label_hash=mo.digest_tensor(label),x_hash=mo.digest_tensor(xs[name]))
                paths=[lp]+[x_path(root,n,c) for n in names]+[g_path(root,n,row['id']) for n in names]
                commit(root/'cache/commits'/(row['id']+'.json'),identity,sample=row,files=file_table(root,set(paths)))
            del gradients
            print('SHARED_FIT_COMMITTED',row['id'],len(names),'modules',flush=True)
        del ref,xs
    store.clear();teacher.unload()
