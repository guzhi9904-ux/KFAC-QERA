#!/usr/bin/env python3
"""Small private bundle for raw train/validation text and historical calibration only.

No model, S caches, network, SSH, GPU or credentials. Extraction never overwrites a directory.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''): h.update(block)
    return h.hexdigest()


def allowed(name):
    valid={'calibration.safetensors'} | {'wikitext2/'+role+'/'+file
        for role in ('train','validation') for file in ('dataset_info.json','state.json','data-00000-of-00001.arrow')}
    return name in valid and str(PurePosixPath(name))==name


def pack(exp01,exp03,output,wikitext=None,calibration=None):
    fit=json.loads((exp03/'data/fit_windows.json').read_text())
    dm=json.loads((exp01/'data_manifest.json').read_text())
    identity=json.loads((exp01/'identity.json').read_text())
    wikitext=wikitext or Path(fit['dataset_path'])
    calibration=calibration or Path(identity['config']['calibration'])
    sources={('wikitext2/'+rel):wikitext/rel for rel in fit['source_sha256']}
    expected={('wikitext2/'+rel):value for rel,value in fit['source_sha256'].items()}
    sources['calibration.safetensors']=calibration; expected['calibration.safetensors']=dm['calibration_file_hash']
    records={}
    for name,path in sources.items():
        if not allowed(name) or path.is_symlink() or not path.is_file(): raise ValueError('Missing/unsupported source: '+str(path))
        actual=sha(path)
        if actual!=expected[name]: raise ValueError('Parent corpus/calibration hash differs: '+str(path))
        records[name]=dict(sha256=actual,bytes=path.stat().st_size)
    manifest=dict(schema=1,purpose='geometry-new-article-inputs',files=records,parent_identity=fit['identity'])
    data=(json.dumps(manifest,sort_keys=True,indent=2)+'\n').encode()
    output.mkdir(parents=True,exist_ok=False)
    (output/'manifest.json').write_bytes(data)
    with tarfile.open(output/'inputs.tar.partial','w') as tar:
        for name,path in sources.items():
            info=tar.gettarinfo(str(path),arcname=name); info.mode=0o600; info.uid=info.gid=0; info.uname=info.gname=''
            with path.open('rb') as stream: tar.addfile(info,stream)
            if sha(path)!=records[name]['sha256']: raise ValueError('Source changed during packing')
        info=tarfile.TarInfo('manifest.json'); info.size=len(data); info.mode=0o600
        tar.addfile(info,io.BytesIO(data))
    (output/'inputs.tar.partial').rename(output/'inputs.tar')
    print(json.dumps(dict(manifest_sha256=hashlib.sha256(data).hexdigest(),bytes=sum(r['bytes'] for r in records.values()),
        output=str(output.resolve()),compute_started=False)))


def extract(archive,manifest_path,manifest_hash,destination):
    data=manifest_path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=manifest_hash: raise ValueError('Manifest SHA256 mismatch')
    manifest=json.loads(data); expected=manifest['files']
    if manifest.get('schema')!=1 or manifest.get('purpose')!='geometry-new-article-inputs': raise ValueError('Wrong manifest schema')
    if len(expected)!=7 or not all(allowed(n) for n in expected): raise ValueError('Unexpected file set')
    destination=destination.resolve(); destination.mkdir(parents=True,exist_ok=False); seen=set()
    with tarfile.open(archive,'r|') as tar:
        for member in tar:
            if member.name in seen or not member.isfile(): raise ValueError('Duplicate or nonregular tar member')
            seen.add(member.name)
            if member.name=='manifest.json':
                if member.size!=len(data) or tar.extractfile(member).read()!=data: raise ValueError('Internal manifest differs')
                continue
            if not allowed(member.name) or member.name not in expected: raise ValueError('Unexpected archive path')
            record=expected[member.name]
            if member.size!=record['bytes'] or member.size>2**31: raise ValueError('Unexpected archive size')
            target=destination/member.name
            if not target.resolve().is_relative_to(destination): raise ValueError('Path escaped destination')
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as out, tar.extractfile(member) as inp:
                for chunk in iter(lambda:inp.read(2**20),b''): out.write(chunk)
            if sha(target)!=record['sha256']: raise ValueError('Payload SHA256 mismatch')
    if seen!=set(expected)|{'manifest.json'}: raise ValueError('Missing payload')
    (destination/'input_manifest.json').write_bytes(data)
    print(json.dumps(dict(passed=True,destination=str(destination),files=len(expected))))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('pack')
    for k in ('exp01','exp03','output'): a.add_argument('--'+k,type=Path,required=True)
    for k in ('wikitext','calibration'): a.add_argument('--'+k,type=Path)
    a=sub.add_parser('extract')
    for k in ('archive','manifest','destination'): a.add_argument('--'+k,type=Path,required=True)
    a.add_argument('--manifest-sha256',required=True)
    args=p.parse_args()
    if args.command=='pack': pack(args.exp01,args.exp03,args.output,args.wikitext,args.calibration)
    else: extract(args.archive,args.manifest,args.manifest_sha256,args.destination)


if __name__=='__main__': main()
