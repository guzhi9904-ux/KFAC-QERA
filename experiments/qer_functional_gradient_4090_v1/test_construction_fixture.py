"""Small CPU orchestration fixture. CUDA transfer is mocked; not a GPU pilot."""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
from unittest.mock import patch
import torch


def run_fixture():
    import functional_math
    from functional_analysis import CANDIDATES
    from verify import verify_summary
    from safetensors.torch import save_file,load_file
    torch.set_num_threads(2)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    digest=lambda t:hashlib.sha256(str((str(t.dtype),tuple(t.shape))).encode()+t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    def save_json(path,value):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,allow_nan=False),encoding='utf8')
    def save_tensors(path,tensors,metadata):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        save_file({k:v.contiguous() for k,v in tensors.items()},str(path),metadata={'record':json.dumps(metadata)})
    def read_tensors(path):
        from safetensors import safe_open
        with safe_open(str(path),framework='pt') as f:meta=json.loads(f.metadata()['record'])
        return load_file(str(path)),meta
    bridge=types.ModuleType('bridge')
    bridge.PLAN={};bridge.read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
    bridge.read_tensors=read_tensors;bridge.slug=lambda n:n.replace('.','__');bridge.sha_file=sha
    bridge.torch=torch;bridge.mo=types.SimpleNamespace(digest_tensor=digest,relative=functional_math.relative)
    bridge.save_tensors=save_tensors;bridge.save_json=save_json;bridge.save_csv=lambda path,rows:save_json(path,rows);bridge.clean=lambda:None
    with patch.dict(sys.modules,{'bridge':bridge}):
        spec=importlib.util.spec_from_file_location('fixture_construction',Path(__file__).with_name('construction.py'))
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='qer-fg-unit-') as tmp:
        root=Path(tmp).resolve()
        assert root.name.startswith('qer-fg-unit-') and root.parent==Path(tempfile.gettempdir()).resolve()
        class Fixture(module.Construction):
            def __init__(self):
                self.root=root;self.runroot=root;self.name='fixture';self.identity='fixture';self.is_parent=False;self.checks={}
                rng=torch.Generator().manual_seed(613019)
                self.w0=torch.randn(140,132,dtype=torch.float64,generator=rng).float()
                self.wq=(self.w0.double()+.2*torch.randn(140,132,dtype=torch.float64,generator=rng)).float()
                self.s=torch.randn(32,140,132,dtype=torch.float64,generator=rng)
                self.a=torch.diag(torch.linspace(1.,2.,132,dtype=torch.float64));self.g=torch.diag(torch.linspace(.8,1.5,140,dtype=torch.float64))
                self.qpath=root/'quantized.safetensors';save_tensors(self.qpath,{'W0':self.w0,'Wq':self.wq},{'identity':self.identity})
                self.gpath=root/'geometry.safetensors';save_tensors(self.gpath,{'A_solve':self.a,'G_solve':self.g},{'identity':self.identity})
            def collect_fit(self):pass
            def quantized(self):return {'W0':self.w0,'Wq':self.wq},dict(path=str(self.qpath),sha256=sha(self.qpath))
            def geometry(self):return self.a,self.g,dict(path=str(self.gpath),file_sha256=sha(self.gpath),lambda_A=0.,lambda_G=0.)
            def unload(self):pass
            def boundary(self):pass
            def stream_s(self):
                for j,s in enumerate(self.s):yield dict(window=j//4,replicate=j%4,S_hash=digest(s),label_hash=f'label{j}'),s
            @contextlib.contextmanager
            def timed(self,*args,**kwargs):yield
            def status(self,*args,**kwargs):pass
        fixture=Fixture()
        with patch.object(torch.Tensor,'cuda',lambda self,*args,**kwargs:self):fixture.construct()
        freeze=fixture.validate_freeze();assert set(freeze['candidates'])==set(CANDIDATES)
        records=bridge.read(root/'fit/records.json')['records'];summary=bridge.read(root/'fit/summary.json')
        verify_summary(records,summary,4)
        assert len(records)==32 and len(bridge.read(root/'fit/direction_statistics.json'))==8
        # Resuming an already frozen constructor must not write or recompute.
        before={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
        with patch.object(torch.Tensor,'cuda',side_effect=AssertionError('Unexpected recomputation')):fixture.construct()
        assert before=={str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}
        result=dict(passed=True,shape=[140,132],original_samples=32,candidates=17,
                    independent_fit_summary=True,frozen_resume_read_only=True,CUDA_transfer_mocked=True)
    return result


if __name__=='__main__':print(json.dumps(run_fixture(),indent=2))
