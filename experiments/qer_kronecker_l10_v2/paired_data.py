"""Reuse the completed L31 texts/labels, with independent L10 identities and caches."""
from pathlib import Path
from common import PLAN,read,read_tensors,save_json,save_tensors,sha_file,mo,require


def source_inventory(config):
    source=Path(config['paired_source']);expected=PLAN['paired_L31_identity']
    files={}
    def pin(p,expected_hash=None):
        value=sha_file(p);require(expected_hash is None or value==expected_hash,'Paired source changed: '+str(p))
        files[str(p)]=value
    for name in ('manifest.json','verification.json','status.json','candidate_freeze.json','data/data_freeze.json'):
        record=read(source/name);require(record['identity']==expected,'Unexpected paired L31 run')
        pin(source/name)
    require(read(source/'status.json')['status']=='COMPLETE' and read(source/'verification.json')['passed'],'L31 source not completed/verified')
    frozen=read(source/'data/data_freeze.json')
    for rel,value in frozen['files'].items():
        p=(source/rel).resolve();require(p.is_relative_to(source.resolve()),'Unsafe source freeze path');pin(p,value)
    index=read(source/'data/sample_index.json')
    require(len(index['fit'])==224 and len(index['eval'])==256,'Paired sample count differs')
    for role in ('fit','eval'):
        for row in index[role]:
            lp=source/'data/labels'/role/(row['id']+'.safetensors')
            receipt=read(lp.with_suffix('.json'))
            require(receipt['identity']==expected and receipt['id']==row['id'],'Wrong paired label binding')
            pin(lp,receipt['file_sha256']);pin(lp.with_suffix('.json'))
    return files


def prepare_paired(root,config,identity):
    source=Path(config['paired_source'])
    from data import sample_views
    rows={}
    for role in ('fit','eval'):
        path=source/'data'/f'{role}_windows.safetensors';t,m=read_tensors(path)
        require(m['identity']==PLAN['paired_L31_identity'],'Paired text identity differs')
        meta=read(source/'data'/f'{role}_windows.json');rows[role]=meta['windows']
        require(all(mo.digest_tensor(t['input_ids'][i])==r['token_hash'] for i,r in enumerate(rows[role])),'Paired token hash differs')
        save_tensors(root/'data'/path.name,t,{**m,'identity':identity,'paired_source_sha256':sha_file(path)})
        save_json(root/'data'/f'{role}_windows.json',{**meta,'identity':identity})
    index=dict(identity=identity,**sample_views(rows['fit'],rows['eval']))
    old=read(source/'data/sample_index.json')
    require({k:v for k,v in old.items() if k!='identity'}=={k:v for k,v in index.items() if k!='identity'},'Cross-layer sample index changed')
    save_json(root/'data/sample_index.json',index)
    files={p.relative_to(root).as_posix():sha_file(p) for p in (root/'data').glob('*') if p.is_file()}
    save_json(root/'data/data_freeze.json',dict(identity=identity,files=files,paired_source_identity=PLAN['paired_L31_identity'],
        selection='Exact same frozen L31 texts and labels; exploratory layer extension after observing L31 results',
        source_data_freeze_sha256=sha_file(source/'data/data_freeze.json')))
    return index


def copy_label(e,role,row,path):
    source=Path(e.config['paired_source'])/'data/labels'/role/(row['id']+'.safetensors')
    pinned=read(e.root/'asset_identity.json')['files'];require(sha_file(source)==pinned[str(source)],'Paired label changed after inventory')
    t,m=read_tensors(source)
    require(m['identity']==PLAN['paired_L31_identity'],'Paired label identity differs')
    require(all(m[k]==v for k,v in row.items()),'Paired label sample/seed binding differs')
    require(mo.digest_tensor(t['labels'])==m['tensor_hashes']['labels'],'Paired label hash differs')
    e.store.put(path,dict(labels=t['labels']),**row,paired_source_sha256=sha_file(source),paired_source_identity=m['identity'])
