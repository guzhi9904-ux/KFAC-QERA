from pathlib import Path
from ko_common import (read,save_json,require,load_record,checked_files,commit,file_table,
                       ParentStore,TensorStore,mo,slug,source_x)
from dataset import windows
from assets_local import prepare_quantized


class Source:
    def __init__(self,config):
        self.root=Path(config['parent_run']);self.manifest=read(self.root/'manifest.json');self.identity=self.manifest['identity']
        self.index=read(self.root/'data/index.json');require(self.index['identity']==self.identity,'Parent index identity changed')
        require(self.index['windows']['fit']==256 and self.index['windows']['validation']==16,'Wrong parent windows')
        frozen=load_record(self.root/'data/freeze.json',self.identity);checked_files(self.root,frozen['files'])
        self.ids=windows(self.root,'fit',self.identity);self.validation=windows(self.root,'validation',self.identity)
        self.store=ParentStore(self.identity,0)
    def label(self,row):
        t,m=self.store.get(self.root/'cache/labels'/(row['id']+'.safetensors'))
        require(m['sample']==row and row['token_hash']==mo.digest_tensor(self.ids[row['window']]),'Parent label binding changed')
        return t['labels']
    def gradient(self,name,row,label):
        t,m=self.store.get(self.root/'cache/g'/slug(name)/(row['id']+'.safetensors'))
        require(m['sample']==row and m['module']==name and m['label_hash']==mo.digest_tensor(label),'Parent gradient binding changed')
        return t['g']
    def x(self,name,window):
        t,m=self.store.get(source_x(self.root,name,window))
        require(m['token_hash']==mo.digest_tensor(self.ids[window]),'Parent input binding changed')
        return t['x']


def prepare(root,config,ident,source):
    freeze=root/'data/freeze.json'
    if freeze.exists():checked_files(root,load_record(freeze,ident)['files']);return
    index=dict(source.index,identity=ident,budgets={'N256':source.index['budgets']['N256']})
    require(len(index['fit'])==256 and all(r['replicate']==0 for r in index['fit']),'Require one sample per256 windows')
    save_json(root/'data/index.json',index)
    prepare_quantized(root,config,ident)
    commit(freeze,ident,files=file_table(root,[root/'data/index.json',root/'quantized/freeze.json']),
           parent_identity=source.identity,source_run=str(source.root),fit_windows=256,validation_windows=16,
           labels='Parent frozen labels; no resampling',test_windows=0)


def new_x(root,name,window):return root/'cache/x'/slug(name+'.input')/f'w{window:04d}.safetensors'
def new_g(root,name,key):return root/'cache/g'/slug(name)/(key+'.safetensors')


def collect(root,config,ident,resources,teacher,source,acceptance):
    from ko_common import LAYERS,NEW
    from ko_replay import capture_blocks,local_gradient
    store=TensorStore(ident,0)
    for row in source.index['fit']:
        resources.boundary();cp=root/'cache/commits'/(row['id']+'.json')
        if cp.exists():checked_files(root,load_record(cp,ident)['files']);continue
        window=row['window'];ids=source.ids[window:window+1];label=source.label(row)
        with resources.timed('collect_reference',window=window):
            teacher.load();ref,states=capture_blocks(teacher.model,ids)
        paths=[]
        if acceptance['gradient_mode']=='shared_full':
            with resources.timed('shared_full_fallback',window=window):
                got=teacher.gradient_all(ids,NEW,ref,None,label)
        for layer in LAYERS:
            names=[n for n in NEW if int(n.split('.')[2])==layer]
            with resources.timed('local_block_gradient_cache',window=window,layer=layer):
                if acceptance['gradient_mode']=='local_down_seed':
                    seed=source.gradient(f'model.layers.{layer}.mlp.down_proj',row,label)
                    got,_=local_gradient(teacher.model,layer,states[layer],seed,names)
                for name in names:
                    x=got[name]['x'];g=got[name]['g'];require(x.dtype==g.dtype==__import__('torch').float32,'Native cache must be FP32')
                    if name.endswith('k_proj'):
                        require(__import__('torch').equal(x,source.x(name,window)),'Shared k input changed')
                    else:
                        xp=new_x(root,name,window);store.put(xp,dict(x=x),window=window,token_hash=row['token_hash']);paths.append(xp)
                    gp=new_g(root,name,row['id']);store.put(gp,dict(g=g),module=name,sample=row,
                        label_hash=mo.digest_tensor(label),x_hash=mo.digest_tensor(x),loss_reduction='sum',gradient_mode=acceptance['gradient_mode'])
                    paths.append(gp)
            if acceptance['gradient_mode']=='local_down_seed':del got
        commit(cp,ident,sample=row,files=file_table(root,paths),parent_identity=source.identity)
        print('INCREMENT_FIT_COMMITTED',window+1,'/256',flush=True)
        del ref,states
    teacher.unload()
