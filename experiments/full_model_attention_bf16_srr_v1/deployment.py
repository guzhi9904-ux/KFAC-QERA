"""BF16 base plus two BF16 low-rank GEMMs; never merge the dense correction."""
from common import *

def factor_cast(values, shape):
    p,q=values['P64'],values['Q64'];out_dim,in_dim=shape
    require(p.dtype==q.dtype==torch.float64 and p.shape==(out_dim,64) and q.shape==(64,in_dim),'Invalid FP64 factors')
    require(torch.isfinite(p).all() and torch.isfinite(q).all(),'Nonfinite FP64 factors')
    a=q.T.to(torch.bfloat16).contiguous();b=p.T.to(torch.bfloat16).contiguous()
    require(torch.isfinite(a).all() and torch.isfinite(b).all(),'BF16 factor overflow')
    return a,b

def correction_hook(a,b,hits,key):
    def hook(module,args,output):
        x=args[0]
        require(x.dtype==output.dtype==a.dtype==b.dtype==torch.bfloat16,'Correction is not BF16')
        first=x@a;second=first@b
        require(first.dtype==second.dtype==torch.bfloat16,'GEMM output dtype differs')
        hits[key]+=1
        return output+second
    return hook

class Deployment:
    def __init__(self,ctx):
        from transformers import AutoModelForCausalLM
        self.ctx=ctx;self.handles=[];self.hits={};self.bits={}
        with timed(ctx,'load_BF16_teacher'):
            mapping=layout.model_device_map()
            self.model=AutoModelForCausalLM.from_pretrained(str(ctx.model_path),torch_dtype=torch.bfloat16,
                attn_implementation='eager',local_files_only=True,trust_remote_code=False,
                device_map=mapping,low_cpu_mem_usage=True,max_position_embeddings=4096)
            self.model.eval().requires_grad_(False);self.model.config.use_cache=False
            placement=layout.align_and_check_devices(self.model,mapping)
            expected=read(ctx.parent/'teacher_identity.json')['tensor_hashes']
            for name,param in self.model.named_parameters():
                require(param.dtype==torch.bfloat16 and tensor_hash(param.float())==expected[name]['hash'],
                        'BF16 checkpoint/parent teacher mismatch: '+name)
            targets=set(read(ctx.parent/'quantization_manifest.json')['modules'])
            self.non_target={n:tensor_hash(p) for n,p in self.model.named_parameters()
                             if n.removesuffix('.weight') not in targets}
            self.originals={key:self.model.get_submodule(key).weight.detach().cpu().clone()
                            for key in targets}
            freeze(ctx.root/'verification/teacher.json',dict(identity=ctx.identity,passed=True,all_parameters_BF16=True,
                parent_FP32_values_exact=True,placement=placement,autocast=False,tf32=False,model_max_length=4096))

    def activate(self,state):
        ctx=self.ctx
        for h in self.handles:h.remove()
        self.handles=[];self.hits={};self.bits={}
        candidate=read(ctx.root/'candidates'/f'{state}.json')
        for key,row in candidate['modules'].items():
            module=self.model.get_submodule(key);device=module.weight.device;self.hits[key]=0
            if state=='Teacher':w=self.originals[key]
            else:w=load_file(str(ctx.verify(row['path'],row['file_sha256'])))['Wq'].bfloat16()
            require(tensor_hash(w)==row['base_BF16_hash'],'BF16 base bits differ')
            with torch.no_grad():module.weight.copy_(w.to(device))
            if row['family']:
                values=load_file(str(ctx.verify(row['factor'],row['factor_sha256'])))
                a,b=factor_cast(values,row['shape'])
                require(tensor_hash(a)==row['A_BF16_hash'] and tensor_hash(b)==row['B_BF16_hash'],'Factor cast bits differ')
                a,b=a.to(device),b.to(device);self.bits[key]=(a,b)
                hook=correction_hook(a,b,self.hits,key)
            else:
                def hook(_m,args,output,key=key):
                    require(args[0].dtype==output.dtype==torch.bfloat16,'Base activation dtype differs')
                    self.hits[key]+=1;return output
            self.handles.append(module.register_forward_hook(hook))
        require(all(tensor_hash(p)==self.non_target[n] for n,p in self.model.named_parameters() if n in self.non_target),
                'Non-target parameters changed')
        self.state=state;self.candidate=candidate['identity']
        freeze(ctx.root/'verification/deployment'/f'{state}.json',dict(identity=ctx.identity,candidate=self.candidate,
            base_dtype='bfloat16',factor_dtype='bfloat16',intermediates_dtype='bfloat16',addition_dtype='bfloat16',
            modules=224,compensated_modules=len(self.bits),two_GEMMs=True,dense_merge=False))
        log('CANDIDATE_ACTIVE',state=state,candidate=self.candidate,BF16_factors=len(self.bits))

    def close(self):
        for h in self.handles:h.remove()
        self.handles=[];self.bits={};self.originals={};self.model=None;gc.collect()
        for d in range(torch.cuda.device_count()):
            with torch.cuda.device(d):torch.cuda.empty_cache()
