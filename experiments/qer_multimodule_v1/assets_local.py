"""Import exact existing quantizations; never change the old run."""
from pathlib import Path
from bridge import TensorStore,read,read_tensors,sha_file,mo,slug,require,commit,load_record,checked_files,file_table


def prepare_quantized(root,config,identity):
    p=root/'quantized/freeze.json'
    if p.exists():
        row=load_record(p,identity);checked_files(root,row['files'])
        for source,value in row['teacher_files'].items():require(sha_file(source)==value,'Frozen teacher asset changed')
        return row['shapes']
    from assets import verify_quantizer_source
    expected=read(Path(config['assets'])/'exp03/teacher_identity.json')['tensor_hashes']
    quantizer=Path(config['assets'])/'vendor/quantize/quantizers/mxint.py'
    if not quantizer.exists():quantizer=Path(config['assets'])/'vendor/src/qera/quantize/quantizers/mxint.py'
    store=TensorStore(identity,0);paths=[];sources={};shapes={}
    for name in config['modules']:
        source=Path(config['quantized_run'])/'modules'/slug(name)/'quantized.safetensors'
        if not source.exists():source=Path(config['assets'])/'exp01/quantized'/(slug(name)+'.safetensors')
        t,m=read_tensors(source)
        require(m.get('module')==name and (m['width'],m['block_size'],m['block_axis'])==(3,32,-1),'Wrong existing quantization')
        verify_quantizer_source(quantizer,m['quantizer_hash'])
        for k in ('W0','Wq'):require(mo.digest_tensor(t[k])==m[k+'_hash'],'Existing quantized tensor changed')
        require(m['W0_hash']==expected[name+'.weight']['hash'],'Quantized teacher identity differs')
        dest=root/'quantized'/(slug(name)+'.safetensors')
        store.put(dest,t,module=name,source=str(source),source_sha256=sha_file(source),quantizer_sha256=sha_file(quantizer))
        paths.append(dest);sources[str(source)]=sha_file(source);shapes[name]=list(t['W0'].shape)
    teacher_files={str(Path(config['assets'])/'exp03'/f):sha_file(Path(config['assets'])/'exp03'/f) for f in ('teacher_identity.json','environment.json','identity.json')}
    commit(p,identity,files=file_table(root,paths),sources=sources,shapes=shapes,teacher_files=teacher_files)
    return shapes
