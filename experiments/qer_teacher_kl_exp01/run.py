#!/usr/bin/env python3
"""Protocol v2: prepare -> audited A -> directions -> pilot -> paired MC/KL.

Execute inside a cck labgpu lease. No scheduler actions or admin access here.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback

os.environ.setdefault("HF_HUB_OFFLINE","1")
os.environ.setdefault("HF_DATASETS_OFFLINE","1")
os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
sys.dont_write_bytecode=True
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

import math_ops as mo
from model_ops import capture, recompute_suffix, hidden_forward
from storage import (atomic_bytes,read_json,read_tensors,save_csv,save_json,save_tensors,sha_file)

MODULES=("model.layers.31.self_attn.q_proj","model.layers.10.self_attn.v_proj")
DIRECTIONS=("R_none","R_svd64","R_A64")
OFFICIAL={"quantize/quantizers/mxint.py":"74643c6306fd7109a57d76c559ee8dfe82d702e3285af6cddc8296722d68c9c4",
          "datasets/wikitext2.py":"1991a661b1f2a2f336a22622f6693135caf96d92ad944788bd701e0c94fc0555"}


def slug(name):return name.replace(".","__")
def sync():
    for i in range(torch.cuda.device_count()):torch.cuda.synchronize(i)
def log(*args):print(time.strftime("%Y-%m-%d %H:%M:%S"),*args,flush=True)
def clean():gc.collect();torch.cuda.empty_cache()


class Paused(Exception):pass
def request_pause(signum,frame):raise Paused(f"signal {signum}")


class Experiment:
    def __init__(self,args):
        self.args=args;self.root=Path(args.output).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.config=read_json(args.config)
        self.model=None;self.resources=[]
        if (self.root/"resource_records.json").exists():self.resources=read_json(self.root/"resource_records.json")
        code={p.name:sha_file(p) for p in sorted(Path(__file__).parent.iterdir())
              if p.suffix in {".py",".sh",".json",".md"}}
        identity_data={"config":self.config,"source":code,"protocol":sha_file(args.protocol)}
        self.identity=hashlib.sha256(json.dumps(identity_data,sort_keys=True).encode()).hexdigest()
        if (self.root/"identity.json").exists():
            if read_json(self.root/"identity.json")["identity"]!=self.identity:
                raise RuntimeError("RUN_IDENTITY_MISMATCH: use a new run directory")
        else:
            save_json(self.root/"identity.json",{"identity":self.identity,**identity_data})
            save_json(self.root/"source_manifest.json",code)
            atomic_bytes(self.root/"protocol.md",Path(args.protocol).read_bytes())
            import yaml
            atomic_bytes(self.root/"config.resolved.yaml",yaml.safe_dump(self.config,sort_keys=False).encode())
        torch.set_num_threads(self.config["cpu_threads"])
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.manual_seed(20260918)
        self.cal=None;self.val=None

    def status(self,state,**extra):
        save_json(self.root/"status.json",{"status":state,"identity":self.identity,"time":time.time(),**extra})
        log(state,extra)

    @contextlib.contextmanager
    def timed(self,stage,**extra):
        sync()
        for i in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(i)
        start=time.perf_counter()
        try:yield
        finally:
            sync()
            row={"stage":stage,"seconds":time.perf_counter()-start,
                 "rss_lifetime_peak_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
                 "allocated_peak_GiB":{str(i):torch.cuda.max_memory_allocated(i)/2**30 for i in range(torch.cuda.device_count())},
                 "reserved_peak_GiB":{str(i):torch.cuda.max_memory_reserved(i)/2**30 for i in range(torch.cuda.device_count())},
                 "CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),"checkpoint_suffix":self.config["checkpoint_suffix"],**extra}
            self.resources.append(row);save_json(self.root/"resource_records.json",self.resources)
            save_csv(self.root/"resource_usage.csv",self.resources)
            log("COST",stage,round(row["seconds"],3),row["allocated_peak_GiB"])

    def official(self,relative):
        path=Path(self.config["qera"])/"src/qera"/relative
        digest=hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
        if digest!=OFFICIAL[relative]:raise RuntimeError("Official source digest mismatch: "+relative)
        spec=importlib.util.spec_from_file_location("frozen_"+path.stem,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def doctor(self):
        if os.name!="posix" or os.getuid()!=1001 or os.environ.get("USER")!="cck":
            raise RuntimeError("Only the cck account is permitted")
        cgroup=Path("/proc/self/cgroup").read_text()
        if "/labgpu.slice/" not in cgroup:raise RuntimeError("Must run inside a labgpu lease")
        if torch.cuda.device_count()<self.config["gpus"]:raise RuntimeError("Not enough leased GPUs")
        for key in ("model","calibration","wikitext","qera"):
            if not Path(self.config[key]).exists():raise RuntimeError("MISSING_ASSET: "+key)
        for rel in OFFICIAL:self.official(rel)
        parent=Path("/sys/fs/cgroup")/cgroup.split("0::",1)[1].strip().lstrip("/")
        limits={}
        while str(parent).startswith("/sys/fs/cgroup"):
            for name in ("memory.max","memory.current","memory.swap.max","cpu.max"):
                p=parent/name
                if p.exists():limits[str(p)]=p.read_text().strip()
            if parent==Path("/sys/fs/cgroup"):break
            parent=parent.parent
        env={"python":sys.version,"platform":platform.platform(),"torch":torch.__version__,"cuda":torch.version.cuda,
             "packages":{n:importlib.metadata.version(n) for n in ("transformers","datasets","numpy","safetensors","accelerate")},
             "cgroup":cgroup,"limits":limits,"CUDA_VISIBLE_DEVICES":os.environ.get("CUDA_VISIBLE_DEVICES"),
             "gpu_query":subprocess.run(["nvidia-smi","--query-gpu=uuid,name,memory.total,memory.used,memory.free","--format=csv"],capture_output=True,text=True).stdout,
             "compute_processes":subprocess.run(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,used_memory","--format=csv"],capture_output=True,text=True).stdout,
             "disk":subprocess.run(["df","-h",str(self.root)],capture_output=True,text=True).stdout,
             "identity":self.identity,"time":time.time(),"tf32":False,"autocast":False,"dtype":"float32","attention":"eager"}
        save_json(self.root/"environment.json",env)
        log("DOCTOR_OK",env["gpu_query"])

    def prepare(self):
        path=self.root/"data/validation.safetensors"
        if path.exists():
            self.val,meta=read_tensors(path)
            if meta["identity"]!=self.identity:raise RuntimeError("Data identity mismatch")
            self.cal=load_file(self.config["calibration"])
            return
        self.status("PREPARING_DATA")
        from transformers import AutoTokenizer
        import datasets
        with self.timed("prepare_validation"):
            tokenizer=AutoTokenizer.from_pretrained(self.config["model"],local_files_only=True,trust_remote_code=False)
            raw=datasets.load_from_disk(self.config["wikitext"],keep_in_memory=True)
            # Run the frozen function on validation as its sole named split. Explicit
            # metadata records the original split; no historical test data is used.
            original=raw["validation"]
            wrapper=datasets.DatasetDict(train=original)
            prepared=self.official("datasets/wikitext2.py").preprocess_data_module_wikitext2(
                wrapper,tokenizer,2048,num_proc=8)["train"]
            ids=[];masks=[];indices=[]
            for i,row in enumerate(prepared):
                if len(row["input_ids"])==2048 and all(row["attention_mask"]):
                    ids.append(row["input_ids"]);masks.append(row["attention_mask"]);indices.append(i)
                if len(ids)==8:break
            if len(ids)!=8:raise RuntimeError("MISSING_VALIDATION_WINDOWS")
            self.val={"input_ids":torch.tensor(ids,dtype=torch.int64),"attention_mask":torch.tensor(masks,dtype=torch.int64)}
            self.cal=load_file(self.config["calibration"])
            if self.cal["input_ids"].shape!=(256,2048) or not self.cal["attention_mask"].all():
                raise RuntimeError("Calibration must be the fixed 256 complete windows")
            cal_hashes=[mo.digest_tensor(t) for t in self.cal["input_ids"]]
            val_hashes=[mo.digest_tensor(t) for t in self.val["input_ids"]]
            overlap=sorted(set(cal_hashes)&set(val_hashes))
            if overlap:raise RuntimeError("Exact calibration/validation window overlap")
            metadata={"identity":self.identity,"source_split":"validation","original_rows":len(original),
                "processed_indices":indices,"window_ids":list(range(8)),"window_tensor_hashes":val_hashes,
                "calibration_window_hashes":cal_hashes,"calibration_file_hash":sha_file(self.config["calibration"]),
                "preprocessor_hash":OFFICIAL["datasets/wikitext2.py"],"workers":8,"map_batch_size":1000,
                "complete_window_overlap":overlap,"overlap_scope":"Exact token windows only; no contamination claim"}
            save_tensors(path,self.val,metadata);save_json(self.root/"data_manifest.json",metadata)

    def load_model(self):
        if self.model is not None:return
        from transformers import AutoModelForCausalLM
        n=self.config["gpus"]
        mapping={"model.embed_tokens":0,"model.norm":n-1,"lm_head":n-1}
        mapping.update({f"model.layers.{i}":min(i*n//32,n-1) for i in range(32)})
        self.device_map=mapping
        with self.timed("load_model",device_map=mapping):
            self.model=AutoModelForCausalLM.from_pretrained(self.config["model"],torch_dtype=torch.float32,
                attn_implementation="eager",local_files_only=True,trust_remote_code=False,
                low_cpu_mem_usage=True,device_map=mapping)
            self.model.eval();self.model.requires_grad_(False);self.model.config.use_cache=False
            if any(p.dtype!=torch.float32 or p.requires_grad for p in self.model.parameters()):
                raise RuntimeError("Teacher must be entirely frozen FP32")
        identity_path=self.root/"teacher_identity.json"
        with self.timed("teacher_tensor_identity"):
            tensors={name:{"shape":list(p.shape),"dtype":str(p.dtype),"hash":mo.digest_tensor(p)}
                     for name,p in self.model.named_parameters()}
            record={"tensor_hashes":tensors,"config":self.model.config.to_dict(),"device_map":mapping,
                    "checkpoint_manifest_hash":sha_file(Path(self.config["model"])/"DOWNLOAD_MANIFEST.json")}
            if identity_path.exists():
                previous=read_json(identity_path)
                if previous["tensor_hashes"]!=tensors or previous["device_map"]!=mapping:
                    raise RuntimeError("Teacher identity changed")
            else:save_json(identity_path,record)

    def unload(self):
        self.model=None;clean()

    def quantize(self):
        quantizer=self.official("quantize/quantizers/mxint.py").mxint_quantizer
        for name in MODULES:
            path=self.root/"quantized"/(slug(name)+".safetensors")
            weight=self.model.get_submodule(name).weight.detach()
            if path.exists():
                tensors,meta=read_tensors(path)
                if meta["identity"]!=self.identity or mo.digest_tensor(weight)!=meta["W0_hash"]:raise RuntimeError("W0 mismatch")
                continue
            with self.timed("quantize",module=name):
                wq=quantizer(weight.float(),width=3,block_size=32,block_axis=-1)
                save_tensors(path,{"W0":weight,"Wq":wq},{"identity":self.identity,"module":name,
                    "W0_hash":mo.digest_tensor(weight),"Wq_hash":mo.digest_tensor(wq),
                    "quantizer_hash":OFFICIAL["quantize/quantizers/mxint.py"],"width":3,"block_size":32,"block_axis":-1})

    def collect(self):
        if (self.root/"input_stats/complete.json").exists():return
        self.status("A_COLLECTOR_PILOT")
        self.load_model();self.quantize()
        pilot_path=self.root/"input_stats/pilot.json"
        if not pilot_path.exists():
            with self.timed("A_pilot"):
                states={}
                with contextlib.ExitStack() as stack:
                    for name in MODULES:states[name]=stack.enter_context(capture(self.model.get_submodule(name)))
                    with torch.no_grad():hidden_forward(self.model,self.cal["input_ids"][:1])
                audits={}
                for name,state in states.items():
                    x=state["x"].reshape(-1,state["x"].shape[-1]);a=mo.gram(x)/len(x);ref=mo.gram(x,double_product=True)/len(x)
                    w,_=read_tensors(self.root/"quantized"/(slug(name)+".safetensors"),str(x.device))
                    error=w["W0"].double()-w["Wq"].double()
                    generator=torch.Generator(device=x.device).manual_seed(613)
                    probe=torch.randn((3,x.shape[-1]),dtype=torch.float64,device=x.device,generator=generator)
                    checks={}
                    for key,r in (("E",error),("probe",probe)):
                        direct=float((x.double()@r.T).square().sum()/len(x));trace=mo.objective(r,a)
                        checks[key]={"direct_SSE":direct,"trace":trace,"relative_error":abs(trace-direct)/abs(direct) if direct else 0.}
                    audits[name]={"gram_relative_error":mo.relative(a,ref),"N":len(x),"objectives":checks}
                    del w,error,probe,a,ref
                production_double=any(v["gram_relative_error"]>1e-5 or any(q["relative_error"]>1e-5 for q in v["objectives"].values()) for v in audits.values())
                if production_double:
                    raise RuntimeError("GRAM_PILOT_FAILED: new FP64-statistics run required")
                save_json(pilot_path,{"identity":self.identity,"production":"fp32_gram_fp64_accumulate","audits":audits})
                del states;clean()
        checkpoint=self.root/"input_stats/progress.safetensors"
        if checkpoint.exists():
            sums,meta=read_tensors(checkpoint)
            if meta["identity"]!=self.identity or meta["completed"]!=list(range(len(meta["completed"]))):raise RuntimeError("A progress mismatch")
            start=len(meta["completed"]);count=meta["N"]
        else:
            sums={slug(name):torch.zeros((4096,4096),dtype=torch.float64) for name in MODULES};start=0;count=0
        if count!=start*2048:raise RuntimeError("A denominator mismatch")
        self.status("COLLECTING_A",completed=start,total=256)
        for base in range(start,256,8):
            with self.timed("A_batch",window_start=base):
                for window in range(base,min(base+8,256)):
                    seen=set();handles=[]
                    for name in MODULES:
                        def hook(_module,args,output,_name=name):
                            if _name in seen:raise RuntimeError("Repeated A hook")
                            seen.add(_name);x=args[0].detach().reshape(-1,4096)
                            sums[slug(_name)].add_(mo.gram(x,256).cpu())
                        handles.append(self.model.get_submodule(name).register_forward_hook(hook))
                    try:
                        with torch.no_grad():hidden_forward(self.model,self.cal["input_ids"][window:window+1])
                        if seen!=set(MODULES):raise RuntimeError("Missing A hook")
                    finally:
                        for handle in handles:handle.remove()
                    count+=2048
                save_tensors(checkpoint,sums,{"identity":self.identity,"N":count,"completed":list(range(window+1)),
                    "input_mask":"all 2048 positions","gram_dtype":"float32","accumulator_dtype":"float64"})
            self.status("COLLECTING_A",completed=window+1,total=256)
        for name in MODULES:
            save_tensors(self.root/"input_stats"/(slug(name)+".safetensors"),
                         {"S_A":sums[slug(name)],"A_raw":sums[slug(name)]/count},
                         {"identity":self.identity,"module":name,"N_A":count,"completed_windows":list(range(256)),
                          "calibration_file_hash":sha_file(self.config["calibration"]),"progress_hash":sha_file(checkpoint)})
        save_json(self.root/"input_stats/complete.json",{"identity":self.identity,"N_A":count,"windows":256})

    def solve(self):
        self.unload();self.status("SOLVING_DIRECTIONS")
        audits=[]
        for name in MODULES:
            path=self.root/"directions"/(slug(name)+".safetensors")
            audit_path=self.root/"solve_metrics"/(slug(name)+".json")
            if path.exists() and audit_path.exists():audits.append(read_json(audit_path));continue
            with self.timed("matrix_solve",module=name):
                raw,raw_meta=read_tensors(self.root/"input_stats"/(slug(name)+".safetensors"))
                quant,_=read_tensors(self.root/"quantized"/(slug(name)+".safetensors"))
                error=quant["W0"].double()-quant["Wq"].double();a=raw["A_raw"]
                started=time.perf_counter();solve,l,s,audit=mo.choose_metric(a);audit["metric_total_seconds"]=time.perf_counter()-started
                p,q,sv,chol=mo.low_rank(error,l,64,True)
                ps,qs,svs,symmetric=mo.low_rank(error,s,64)
                pu,qu,svu,plain=mo.low_rank(error,None,64)
                ra=error-p@q;rs=error-ps@qs;ru=error-pu@qu
                j=mo.objective(ra,solve);js=mo.objective(rs,solve)
                checks={"chol_vs_sym":abs(j-js)/abs(js),"chol_vs_tail":abs(j-chol["tail_objective"])/abs(chol["tail_objective"]),
                        "sym_vs_tail":abs(js-symmetric["tail_objective"])/abs(symmetric["tail_objective"])}
                audit.update({"module":name,"identity":self.identity,"N_A":raw_meta["N_A"],"J_solve_chol":j,"J_solve_sym":js,
                    "J_raw_chol":mo.objective(ra,a),"J_raw_sym":mo.objective(rs,a),"J_raw_none":mo.objective(error,a),
                    "C_relative_difference":mo.relative(p@q,ps@qs),"checks":checks,"cholesky":chol,"symmetric":symmetric,"plain_svd":plain,
                    "A_raw_hash":mo.digest_tensor(a)})
                if any(v>1e-6 for v in checks.values()):
                    save_json(audit_path,audit);raise RuntimeError("SOLVE_AUDIT_FAILED")
                directions={"R_none":error.float(),"R_svd64":ru.float(),"R_A64":ra.float()}
                audit["FP32_direction_conversion"]={d:mo.relative(directions[d].double(),r) for d,r in zip(DIRECTIONS,(error,ru,ra))}
                save_tensors(self.root/"solve_metrics"/(slug(name)+".safetensors"),{"A_solve":solve,"L":l,"S_symmetric":s},audit)
                save_tensors(self.root/"factors"/(slug(name)+".safetensors"),
                    {"A_P":p,"A_Q":q,"svd_P":pu,"svd_Q":qu,"symmetric_P":ps,"symmetric_Q":qs,
                     "A_singular":sv,"symmetric_singular":svs,"plain_singular":svu},{"identity":self.identity,"module":name})
                save_tensors(path,directions,{"identity":self.identity,"module":name,"rank":64,
                    "direction_hashes":{d:mo.digest_tensor(t) for d,t in directions.items()},
                    "quantized_file_hash":sha_file(self.root/"quantized"/(slug(name)+".safetensors")),"A_raw_hash":audit["A_raw_hash"]})
                save_json(audit_path,audit);audits.append(audit)
            save_csv(self.root/"a_audit.csv",audits)
        save_json(self.root/"directions/frozen.json",{"identity":self.identity,
            "files":{name:sha_file(self.root/"directions"/(slug(name)+".safetensors")) for name in MODULES}})

    def read_record(self,path):
        row=read_json(path)
        if row["identity"]!=self.identity:raise RuntimeError("Record identity mismatch")
        return row

    def samples(self,window,k,hidden):
        paths=[self.root/"samples"/f"w{window:02d}_k{j:03d}.safetensors" for j in range(k)]
        missing=[j for j,p in enumerate(paths) if not p.exists()]
        if missing:
            weight=self.model.lm_head.weight;source=hidden[0,:-1]
            generators={j:torch.Generator(device=weight.device).manual_seed(mo.stream_seed(window,j)) for j in missing}
            labels={j:[] for j in missing}
            with torch.no_grad():
                for start in range(0,len(source),self.config["vocab_chunk"]):
                    p=F.linear(source[start:start+self.config["vocab_chunk"]],weight).double().softmax(-1)
                    cdf=p.cumsum(-1);cdf[:,-1]=1.0
                    for j in missing:
                        u=torch.rand((len(p),1),dtype=torch.float64,device=weight.device,generator=generators[j])
                        labels[j].append(torch.searchsorted(cdf,u,right=False).flatten().cpu())
            for j in missing:
                label=torch.cat(labels[j])
                save_tensors(paths[j],{"labels":label},{"identity":self.identity,"window":window,"replicate":j,
                    "seed":mo.stream_seed(window,j),"label_hash":mo.digest_tensor(label),"distribution":"full_vocab_FP64_CDF",
                    "device":str(weight.device),"prediction_tokens":2047})
        result=[]
        for p in paths:
            tensors,meta=read_tensors(p)
            if meta["identity"]!=self.identity or mo.digest_tensor(tensors["labels"])!=meta["label_hash"]:raise RuntimeError("Label identity mismatch")
            result.append((tensors["labels"],meta["label_hash"]))
        return result

    def teacher_reference(self,name,window):
        mod=self.model.get_submodule(name)
        with torch.no_grad(),capture(mod) as state:
            hidden=hidden_forward(self.model,self.val["input_ids"][window:window+1])
        return hidden.detach(),state["x"].detach(),state["h"].detach()

    def kl_hidden(self,reference,actual):
        values=[];max_diff=0.
        with torch.no_grad():
            for start in range(0,2047,self.config["vocab_chunk"]):
                end=min(start+self.config["vocab_chunk"],2047)
                zr=F.linear(reference[0,start:end],self.model.lm_head.weight)
                za=F.linear(actual[0,start:end],self.model.lm_head.weight)
                max_diff=max(max_diff,float((zr-za).abs().max()))
                values.append(mo.stable_kl(zr,za).cpu())
        values=torch.cat(values)
        return {"KL_sum":float(values.sum()),"KL_mean":float(values.mean()),"KL_max_token":float(values.max()),"logits_max_difference":max_diff}

    def kl_point(self,name,direction,window,alpha,reference,x,h,residual,self_floor,pilot=False):
        folder="pilot" if pilot else "records/kl"
        path=self.root/folder/(f"{slug(name)}_{direction}_w{window:02d}_a{alpha:g}.json")
        if path.exists():return self.read_record(path)
        module=self.model.get_submodule(name)
        with self.timed("pilot_KL" if pilot else "KL",module=name,direction=direction,window=window,alpha=alpha):
            with mo.intervention(module,residual,alpha) as weight_audit:
                with torch.no_grad(),capture(module) as actual:
                    candidate=hidden_forward(self.model,self.val["input_ids"][window:window+1])
                same_input=torch.equal(x,actual["x"])
                expected=-alpha*(x.double()@residual.to(x.device).double().T)
                output_audit=mo.comparison(actual["h"].detach().double()-h.double(),expected)
                kl=self.kl_hidden(reference,candidate)
            reasons=[]
            if not same_input:reasons.append("module_input_changed")
            if alpha==0:
                if kl["KL_mean"]>1e-10:reasons.append("self_KL_floor")
            else:
                if weight_audit["cosine"]<.999999 or weight_audit["relative_l2"]>1e-3:reasons.append("weight_path")
                if output_audit["cosine"]<.999 or output_audit["relative_l2"]>.02:reasons.append("output_path")
                if kl["KL_mean"]<=100*max(self_floor,1e-12):reasons.append("below_numerical_floor")
            row={"identity":self.identity,"module":name,"direction":direction,"window":window,"alpha":alpha,"T":2047,
                 **kl,"self_KL":self_floor,"weight_audit":weight_audit,"output_audit":output_audit,
                 "input_identical":same_input,"R_hash":mo.digest_tensor(residual),"restoration_verified":True,
                 "valid":not reasons,"invalid_reasons":reasons}
            save_json(path,row)
        return row

    def projection(self,name,window,replicate,labels,label_hash,directions,reference,reference_x,pilot=False):
        path=self.root/("pilot" if pilot else "records/mc")/f"{slug(name)}_w{window:02d}_k{replicate:03d}.json"
        if path.exists():return self.read_record(path)
        target=self.model.get_submodule(name);layer=int(name.split(".")[2])
        with self.timed("pilot_backward" if pilot else "backward",module=name,window=window,replicate=replicate):
            with recompute_suffix(self.model,layer,self.config["checkpoint_suffix"]),capture(target,True) as state:
                hidden=hidden_forward(self.model,self.val["input_ids"][window:window+1])
                if not torch.equal(state["x"],reference_x):raise RuntimeError("MC input changed")
                hidden_difference=mo.relative(hidden.detach(),reference)
                if hidden_difference>1e-7:raise RuntimeError("Checkpoint forward changed teacher")
                seed=mo.sampled_seed(hidden.detach(),self.model.lm_head.weight,labels,self.config["vocab_chunk"])
                gradient=torch.autograd.grad(hidden,state["h"],grad_outputs=seed)[0]
            results={}
            with torch.no_grad():
                for direction,r in directions.items():
                    rx=state["x"].double()@r.to(state["x"].device).double().T
                    d=float((gradient.double()*rx).sum());b=d*d/(2*2047)
                    if not math.isfinite(b):raise RuntimeError("Nonfinite direction projection")
                    results[direction]={"d":d,"b":b,"R_hash":mo.digest_tensor(r)}
            row={"identity":self.identity,"module":name,"window":window,"replicate":replicate,"T":2047,
                 "label_hash":label_hash,"directions":results,"teacher_hidden_relative_difference":hidden_difference,
                 "loss_reduction":"sum","projection_reduction":"square_after_sum_all_module_positions"}
            save_json(path,row)
            del hidden,seed,gradient,state
        return row

    def pilot(self):
        if (self.root/"pilot/complete.json").exists():return
        self.status("CURVATURE_PILOT");self.load_model()
        records=[]
        for name in MODULES:
            directions,_=read_tensors(self.root/"directions"/(slug(name)+".safetensors"))
            with self.timed("pilot_teacher",module=name):reference,x,h=self.teacher_reference(name,0)
            sample=self.samples(0,1,reference)[0]
            # Compare the analytic seed with direct autograd on full vocabulary
            # for a bounded token chunk on the actual final teacher hidden.
            hs=reference[:,:5].detach().requires_grad_();labels=sample[0][:4].to(hs.device)
            logits=F.linear(hs[0,:-1],self.model.lm_head.weight)
            loss=-logits.double().log_softmax(-1)[torch.arange(4,device=hs.device),labels].sum()
            direct=torch.autograd.grad(loss,hs)[0]
            analytic=mo.sampled_seed(hs.detach(),self.model.lm_head.weight,labels,4)
            seed_error=mo.relative(analytic,direct)
            if seed_error>1e-5:raise RuntimeError("GPU_ANALYTIC_SEED_AUDIT_FAILED")
            del hs,logits,loss,direct,analytic
            zero=self.kl_point(name,"R_A64",0,0.,reference,x,h,directions["R_A64"],0.,True)
            one=self.kl_point(name,"R_A64",0,.1,reference,x,h,directions["R_A64"],zero["KL_mean"],True)
            self.projection(name,0,0,*sample,directions,reference,x,True)
            if not zero["valid"] or not one["valid"]:raise RuntimeError("GPU_KL_PILOT_INVALID")
            records.append({"module":name,"analytic_seed_relative_error":seed_error,"zero":zero,"alpha_0.1":one})
            del reference,x,h,directions;clean()
        save_json(self.root/"pilot/complete.json",{"identity":self.identity,"modules":records})

    def mc_records(self,name,k=None):
        rows=[self.read_record(p) for p in sorted((self.root/"records/mc").glob(slug(name)+"_*.json"))]
        return [r for r in rows if k is None or r["replicate"]<k]

    def measure_mc(self,name,k):
        directions,_=read_tensors(self.root/"directions"/(slug(name)+".safetensors"))
        for window in range(8):
            missing=[j for j in range(k) if not (self.root/"records/mc"/f"{slug(name)}_w{window:02d}_k{j:03d}.json").exists()]
            if not missing:continue
            with self.timed("MC_teacher",module=name,window=window):reference,x,h=self.teacher_reference(name,window)
            samples=self.samples(window,k,reference)
            for j in missing:
                self.projection(name,window,j,*samples[j],directions,reference,x)
                self.status("SAMPLING_MC",module=name,window=window,replicate=j,K_target=k)
            del reference,x,h;clean()
        stats={d:mo.monte_carlo(self.mc_records(name,k),d) for d in DIRECTIONS}
        save_json(self.root/"records"/f"{slug(name)}_K{k}.json",stats)
        return stats

    def pooled_points(self,name,direction):
        folder=self.root/"records/kl"
        rows=[self.read_record(p) for p in folder.glob(f"{slug(name)}_{direction}_*.json")]
        points=[]
        for alpha in sorted(set(r["alpha"] for r in rows if r["alpha"]>0)):
            batch=[r for r in rows if r["alpha"]==alpha]
            if sorted(r["window"] for r in batch)!=list(range(8)):continue
            points.append({"alpha":alpha,"valid":all(r["valid"] for r in batch),
                           "KL_mean":sum(r["KL_sum"] for r in batch)/sum(r["T"] for r in batch),
                           "invalid_reasons":sorted(set(x for r in batch for x in r["invalid_reasons"]))})
        return points

    def measure_kl(self,name,alphas,direction_subset=DIRECTIONS):
        directions,_=read_tensors(self.root/"directions"/(slug(name)+".safetensors"))
        for window in range(8):
            missing=[(d,a) for d in direction_subset for a in alphas if not (self.root/"records/kl"/
                     f"{slug(name)}_{d}_w{window:02d}_a{a:g}.json").exists()]
            if not missing:continue
            with self.timed("KL_teacher",module=name,window=window):reference,x,h=self.teacher_reference(name,window)
            control=self.kl_point(name,"control",window,0.,reference,x,h,directions["R_none"],0.)
            if not control["valid"]:raise RuntimeError("NUMERICAL_INVALID: self control")
            for direction,alpha in missing:
                self.kl_point(name,direction,window,alpha,reference,x,h,directions[direction],control["KL_mean"])
            del reference,x,h;clean()

    def measure(self):
        self.load_model();self.pilot()
        for name in MODULES:
            self.measure_mc(name,4)
        for name in MODULES:
            stats=self.measure_mc(name,16)
            if any(v["relative_halfwidth"]>.20 for v in stats.values()):self.measure_mc(name,64)
        for name in MODULES:
            self.status("MEASURING_KL",module=name)
            self.measure_kl(name,[.05,.1,.2])
            for alpha in (.025,.0125,.00625):
                need=[]
                for d in DIRECTIONS:
                    points=self.pooled_points(name,d)
                    if mo.plateau(points) is None:
                        smallest=min(points,key=lambda p:p["alpha"])
                        if "below_numerical_floor" not in smallest["invalid_reasons"]:need.append(d)
                if not need:break
                self.measure_kl(name,[alpha],need)
        self.unload();self.report()

    def report(self):
        from analyze import write_report
        write_report(self)

    def run(self,stage):
        self.doctor();self.prepare()
        if stage in ("all","collect"):self.collect()
        if stage in ("all","solve"):self.solve()
        if stage=="pilot":self.pilot()
        if stage in ("all","measure"):self.measure()
        if stage=="report":self.report()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--protocol",required=True)
    parser.add_argument("--stage",choices=["all","collect","solve","pilot","measure","report"],default="all")
    args=parser.parse_args()
    import fcntl
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    with (output/".run.lock").open("w") as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM,request_pause);signal.signal(signal.SIGINT,request_pause)
        experiment=Experiment(args)
        try:experiment.run(args.stage)
        except Paused as error:
            experiment.status("PAUSED",reason=str(error));return 75
        except BaseException as error:
            experiment.status("FAILED",error=repr(error),traceback=traceback.format_exc());raise
    return 0


if __name__=="__main__":sys.exit(main())
