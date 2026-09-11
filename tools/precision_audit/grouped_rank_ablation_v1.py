#!/usr/bin/env python3
"""Frozen Llama MXINT3 residual SVD baseline and independent rank-group deletion.

Four methods, rank64, eight groups. Existing FA factors are never re-solved.
Token NLL keeps the frozen batch8 protocol. KL is separately labelled batch1,
teacher on GPU0 and student on GPU1, with the same full evaluation windows.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import signal
import shutil
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
import numpy as np
import torch
from safetensors import safe_open
import full_a_all_ranks_dual_ppl_v1 as dual

parent, single, core, prev = dual.parent, dual.single, dual.core, dual.prev
VERSION = "grouped_rank_ablation_v1"
METHODS = ("plain_svd", "full_gi", "full_gd", "full_gf")
GROUPS = tuple(tuple(range(8*g, 8*g+8)) for g in range(8))
WINDOWS, LENGTH, TOKENS = 138, 2048, 282486
CONFIGS = [("teacher", 0), ("wq", 0)] + [(m, g) for m in METHODS for g in range(9)]
HELPERS = ("full_a_all_ranks_dual_ppl_v1.py", "full_a_all_precision_r8_v1.py",
           "full_g_precision_r8_v1.py", "full_g_target_probe_v1.py", "full_g_rank_audit_v1.py")


def label(method, group):
    if method in ("teacher", "wq"):
        if group != 0:
            raise ValueError("Controls have no removed group")
        return "BF16" if method == "teacher" else "W3_MXINT"
    if method not in METHODS or group not in range(9):
        raise ValueError("Unknown method/group")
    return method.upper()+("_R64" if group == 0 else f"_DROP_G{group}")


def masked_factors(left, right, group):
    """Zero right-factor rows; preserve rank64 GEMM shapes and component order."""
    if (group not in range(9) or left.ndim != 2 or right.ndim != 2
            or left.shape[1] != 64 or right.shape[0] != 64
            or left.dtype != torch.bfloat16 or right.dtype != torch.bfloat16
            or not torch.isfinite(left).all() or not torch.isfinite(right).all()):
        raise ValueError("Require finite BF16 rank64 factors and group 0..8")
    a, b = left.contiguous(), right.clone().contiguous()
    if group:
        b[8*(group-1):8*group] = 0
    return a, b


def plain_solve(error_t):
    """Same frozen residual values, FP64 full SVD; A=I, G=I, no re-quantization."""
    e = error_t.double()
    if e.ndim != 2 or min(e.shape) < 64 or not torch.isfinite(e).all():
        raise ValueError("Invalid residual for rank64 SVD")
    u, s, vh = torch.linalg.svd(e, full_matrices=True)
    left, right = u[:, :64].clone(), (s[:64, None]*vh[:64]).clone()
    residual = e-left @ right
    before, after, tail = core.norm2(e), core.norm2(residual), core.norm2(s[64:])
    excess = abs(after-tail)/max(before, 1e-30)
    if excess > 1e-9 or after > before+max(before, 1e-30)*1e-9:
        raise RuntimeError("Plain SVD rank64 tail certificate failed")
    a, b = left.bfloat16(), right.bfloat16()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise RuntimeError("Plain SVD BF16 overflow")
    return {"A_fp64":left, "B_fp64":right, "singular_values":s,
            "A":a, "B":b}, {"before":before, "after":after, "tail64":tail,
                             "relative_tail_error":excess, "full_matrices":True}


def group_geometry(error_t, a_root, left, right):
    """Common weight / frozen-A disturbance, computed with rank64 Gram matrices.

    Uses actual BF16-rounded factors in FP64, not activation-execution evidence.
    Reports both removed-component energy and residual-error change (cross terms).
    """
    e, a, l, b = (x.double() for x in (error_t, a_root, left, right))
    al, ae = a @ l, a @ e
    gl, ga, gb = l.T @ l, al.T @ al, b @ b.T
    pe, pa = l.T @ e, al.T @ ae
    rows = []
    for group in range(1, 9):
        ix = slice(8*(group-1), 8*group)
        w = float((gl[ix, ix]*gb[ix, ix]).sum())
        out = float((ga[ix, ix]*gb[ix, ix]).sum())
        cross_w = float((pe[ix]*b[ix]).sum()-(gl[ix]*gb[ix]).sum())
        cross_a = float((pa[ix]*b[ix]).sum()-(ga[ix]*gb[ix]).sum())
        row = {"group":group, "first_component":8*(group-1)+1, "last_component":8*group,
               "removed_weight_fro2":w, "removed_output_sse_calibration_A":out,
               "delta_total_weight_sse":w+2*cross_w,
               "delta_total_output_sse_calibration_A":out+2*cross_a}
        if not all(math.isfinite(v) for v in row.values()) or w < -1e-8 or out < -1e-8:
            raise RuntimeError("Nonfinite/negative disturbance energy")
        rows.append(row)
    return rows


def factor_path(ctx, name):
    return ctx.output/"plain_factors"/(core.safe_name(name)+".json")


def read_plain(ctx, name):
    path = factor_path(ctx, name)
    if not path.exists():
        return None
    rec = core.read_json(path)
    if (rec.get("experiment_identity") != ctx.identity or rec.get("module") != name
            or rec.get("binding") != {k:ctx.tasks[name][k] for k in ("model","quant","shape")}
            or rec.get("status") != "PASS"
            or Path(rec["file"]["path"]).resolve() != path.with_suffix(".safetensors").resolve()):
        raise RuntimeError("Plain factor identity/path mismatch")
    values = ctx.inputs.tensors(rec["file"])
    do, di = ctx.tasks[name]["shape"]
    if set(values) != {"A_fp64","B_fp64","singular_values","A","B"}:
        raise RuntimeError("Plain factor payload keys mismatch")
    for key, shape, dtype in (("A_fp64",(di,64),torch.float64),("B_fp64",(64,do),torch.float64),
                              ("A",(di,64),torch.bfloat16),("B",(64,do),torch.bfloat16),
                              ("singular_values",(min(di,do),),torch.float64)):
        if values[key].shape != shape or values[key].dtype != dtype or not torch.isfinite(values[key]).all():
            raise RuntimeError("Plain factor shape/dtype/finiteness mismatch")
    if (not torch.equal(values["A"],values["A_fp64"].bfloat16())
            or not torch.equal(values["B"],values["B_fp64"].bfloat16())
            or rec["bits"] != {k:single.tensor_record(values[k]) for k in ("A","B")}):
        raise RuntimeError("Plain deployment does not match direct FP64 rounding")
    return rec


def residual(ctx, name):
    binding = ctx.tasks[name]
    with safe_open(str(ctx.inputs.verify(binding["model"])), framework="pt", device="cpu") as f:
        w = f.get_tensor(name+".weight").clone()
    q = ctx.inputs.tensors(binding["quant"])["weight_q"]
    if w.dtype != torch.bfloat16 or q.dtype != torch.bfloat16 or list(w.shape) != binding["shape"] or w.shape != q.shape:
        raise RuntimeError("Frozen W/Wq mismatch")
    return (w.float()-q.float()).T.to(ctx.device)


def prepare_plain(ctx, name):
    existing = read_plain(ctx, name)
    if existing:
        return existing
    path = factor_path(ctx, name); path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad(), core.heartbeat("NEW plain residual SVD "+name):
        values, metrics = plain_solve(residual(ctx, name))
        rec = {"experiment_identity":ctx.identity,"module":name,"status":"PASS",
               "binding":{k:ctx.tasks[name][k] for k in ("model","quant","shape")},
               "metrics":metrics,"bits":{k:single.tensor_record(values[k]) for k in ("A","B")},
               "file":single.atomic_tensors(path.with_suffix(".safetensors"),values)}
        core.atomic_json(path, rec)
    dual.clear_gpu()
    return rec


def factor_record(ctx, method, name):
    return read_plain(ctx,name) if method == "plain_svd" else dual.read_checked(ctx,method,name)


def factor_values(ctx, method, name):
    rec = factor_record(ctx, method, name)
    if rec is None:
        raise RuntimeError("Incomplete factors: "+method+" "+name)
    values = ctx.inputs.tensors(rec["file"])
    return values["A"], values["B"]


def import_checked(ctx, method, name):
    """Copy the already certified rank64 endpoint, verifying direct rounding.

    Source certificates and evaluated deployment bits are bound during setup;
    avoid repeating the frozen evaluator's large FP64 matrix products.
    """
    existing=dual.read_checked(ctx,method,name)
    if existing is not None: return existing
    source=ctx.rank64_checked[method][name]
    values=ctx.inputs.tensors(source["file"])
    raw=ctx.inputs.tensors(ctx.factors[method][name]["file"])
    for k in ("A","B"):
        if not torch.equal(values[k],raw[k+"_fp64"].bfloat16()):
            raise RuntimeError("Saved rank64 endpoint differs from direct FP64 rounding")
    path=dual.checked_path(ctx,method,name);path.parent.mkdir(parents=True,exist_ok=True)
    rec=copy.deepcopy(source)
    rec.update(experiment_identity=ctx.identity,
               imported_source=ctx.rank64_checked_metadata[method][name],
               source_check_elapsed_seconds=rec.pop("elapsed_seconds",None),
               note="Imported existing rank64 certificate; direct BF16 rounding rechecked; no SVD or root rebuild",
               file=single.atomic_tensors(path.with_suffix(".safetensors"),values))
    core.atomic_json(path,rec)
    return rec


def prepare_geometry(ctx, name, method):
    rec = factor_record(ctx,method,name)
    path = ctx.output/"geometry"/method/(core.safe_name(name)+".json")
    identity = {"experiment_identity":ctx.identity,"module":name,"method":method,
                "factor":rec["file"],"a_root":ctx.tasks[name]["root"]}
    if path.exists():
        old = core.read_json(path)
        if old.get("binding") != identity or len(old.get("rows",[])) != 8:
            raise RuntimeError("Geometry identity mismatch")
        return old
    path.parent.mkdir(parents=True,exist_ok=True)
    with torch.no_grad(), core.heartbeat("GROUP geometry "+method+" "+name):
        l,b = factor_values(ctx,method,name)
        a = ctx.inputs.tensors(ctx.tasks[name]["root"])["full"].to(ctx.device)
        rows = group_geometry(residual(ctx,name),a,l.to(ctx.device),b.to(ctx.device))
        record = {"binding":identity,"rows":rows,
                  "metric":"FP64 product of BF16 factors; calibration effective FA, not heldout activation execution"}
        core.atomic_json(path,record)
    dual.clear_gpu()
    return record


def prepare_all(ctx, stop, names=None):
    names = list(ctx.tasks) if names is None else names
    for name in names:
        for m in METHODS:
            if factor_record(ctx,m,name) is None:
                stop.check()
                prepare_plain(ctx,name) if m == "plain_svd" else import_checked(ctx,m,name)
                stop.done += 1
            stop.check()
            prepare_geometry(ctx,name,m)


def require_factors(ctx):
    for m in METHODS:
        for n in ctx.tasks:
            if factor_record(ctx,m,n) is None:
                raise RuntimeError("Run prepare/run first: missing "+m+" "+n)


def routes(ctx, method):
    if method == "teacher":
        return {}
    result = {}
    for n,b in ctx.tasks.items():
        result[n] = {"quant":b["quant"]}
        if method in METHODS:
            rec = factor_record(ctx,method,n)
            if rec is None:
                raise RuntimeError("Incomplete factor set")
            result[n]["correction"] = rec["file"]
    return result


def install(model, artifacts, group, inputs):
    handles, bits, modules = [], {}, dict(model.named_modules())
    try:
        with torch.no_grad():
            for n, files in artifacts.items():
                mod = modules[n]
                q = inputs.tensors(files["quant"])["weight_q"]
                if q.dtype != torch.bfloat16 or mod.weight.dtype != torch.bfloat16 or mod.weight.shape != q.shape:
                    raise RuntimeError("Wq installation shape/dtype mismatch")
                mod.weight.copy_(q.to(mod.weight.device))
                if single.tensor_record(mod.weight) != single.tensor_record(q):
                    raise RuntimeError("Wq bits changed")
                if "correction" in files:
                    v = inputs.tensors(files["correction"])
                    if v["A"].shape != (mod.in_features,64) or v["B"].shape != (64,mod.out_features):
                        raise RuntimeError("Factor orientation mismatch")
                    a,b = masked_factors(v["A"],v["B"],group)
                    a,b = a.to(mod.weight.device),b.to(mod.weight.device)
                    bits[n] = {"A":single.tensor_record(a),"B":single.tensor_record(b)}
                    handles.append(mod.register_forward_hook(prev.correction_hook(a,b)))
            deployment = {"parameter_bits":{n:single.tensor_record(t) for n,t in model.named_parameters()},
                          "buffer_bits":{n:single.tensor_record(t) for n,t in model.named_buffers()},
                          "device_map":model.hf_device_map,"correction_bits":bits}
        return handles, deployment
    except BaseException:
        for h in handles: h.remove()
        raise


def bind_deployment(ctx, folder, name, deployed, artifacts, method, group):
    expected=ctx.rank64_deployments["teacher" if method=="teacher" else "wq"]["deployment"]
    if any(deployed[k]!=expected[k] for k in ("parameter_bits","buffer_bits")):
        raise RuntimeError("Deployment parameters/buffers differ from frozen BF16/Wq source")
    rec = {"experiment_identity":ctx.identity,"arm":name,"group":group,
           "route_sha256":core.fingerprint(artifacts),"deployment":deployed}
    dual.freeze_json(folder/("deployment_"+name+".json"),rec)
    if method != "teacher":
        dual.freeze_json(folder/"common_wq.json",{k:deployed[k] for k in ("parameter_bits","buffer_bits","device_map")})
    if method in METHODS:
        # Exact correction bits must agree across token and KL placements.
        dual.freeze_json(ctx.output/"deployment_factors"/(name+".json"),
                         {"experiment_identity":ctx.identity,"bits":deployed["correction_bits"]})
    return rec


def windows(ctx):
    w = ctx.inputs.tensors(ctx.manifest["payload"]["data"]["wikitext2"])
    if (w["input_ids"].shape != (WINDOWS,LENGTH) or w["attention_mask"].shape != (WINDOWS,LENGTH)
            or not bool((w["attention_mask"] == 1).all())):
        raise RuntimeError("Require original 138x2048 unpadded evaluation windows")
    return w


def token_evaluate(ctx, method, group, stop):
    name = label(method,group); folder=ctx.output/"token_ppl"; folder.mkdir(exist_ok=True)
    state = single.read_state(folder,name,ctx.identity,ctx.inputs)
    artifacts = routes(ctx,method)
    if state["records"] and state.get("route_sha256") != core.fingerprint(artifacts):
        raise RuntimeError("Token resume routes changed")
    if state["complete"]:
        return state
    stop.check(); data=windows(ctx)
    loading=dict(ctx.manifest["payload"]["source_config"])
    loading["max_memory"]=ctx.manifest["payload"]["config"]["eval_max_memory"]
    model, handles=None,[]
    try:
        torch.manual_seed(1234)
        with core.heartbeat("TOKEN load/install "+name):
            model=ctx.legacy.load_model(loading,"bfloat16","balanced"); dual.resident(model)
            handles,deployed=install(model,artifacts,group,ctx.inputs)
            bind_deployment(ctx,folder,name,deployed,artifacts,method,group)
        state.update(route_sha256=core.fingerprint(artifacts),deployment_sha256=core.fingerprint(deployed))
        device=ctx.legacy._input_device(model)
        directory=folder/"tokens"/name; directory.mkdir(parents=True,exist_ok=True)
        with torch.inference_mode():
            for start in range(len(state["records"]),WINDOWS,8):
                stop.check(); end=min(start+8,WINDOWS)
                with core.heartbeat(f"TOKEN {name} {start+1}-{end}/{WINDOWS}"):
                    ids=data["input_ids"][start:end].to(device); mask=data["attention_mask"][start:end].to(device)
                    logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
                    values,tokens=single.nll_with_tokens(logits,ids,mask,256)
                    if start==0 and values != ctx.legacy._chunked_window_nll(logits,ids,mask,256):
                        raise RuntimeError("NLL reduction instrumentation mismatch")
                del logits,ids,mask
                state["records"].extend({"window":start+i,"tokens":n,"nll_sum":loss,
                    "token_nll_sum_fp64":float(tokens[i].double().sum())} for i,(loss,n) in enumerate(values))
                f=single.atomic_tensors(directory/f"batch_{start:04d}.safetensors",{"nll":tokens})
                state["batches"].append({"start":start,"end":end,"file":f}); state["complete"]=end==WINDOWS
                core.atomic_json(folder/(name+".json"),state); stop.done+=1
    finally:
        for h in handles: h.remove()
        handles.clear(); model=None; dual.clear_gpu()
    return state


def token_gate(ctx, method, current):
    """Existing rank64 and frozen BF16/Wq controls gate all deletion endpoints."""
    source=ctx.rank64_states[method]
    check=prev.control_check(current["records"],source["records"],1e-5)
    check["max_window_difference"]=max(abs(x["nll_sum"]-y["nll_sum"])
                                      for x,y in zip(current["records"],source["records"]))
    current_dep=core.read_json(ctx.output/"token_ppl"/("deployment_"+label(method,0)+".json"))["deployment"]
    if current_dep != ctx.rank64_deployments[method]["deployment"]:
        raise RuntimeError("All-retained rank64 deployment differs from frozen source")
    if check["max_window_difference"] > 1e-3: check["status"]="FAIL"
    dual.freeze_json(ctx.output/"token_ppl"/(method+"_control.json"),{"experiment_identity":ctx.identity,**check})
    if check["status"] != "PASS": raise RuntimeError("Rank64/token control failed: "+method)


def run_token(ctx, stop):
    for method in ("teacher","wq",*parent.METHODS):
        token_gate(ctx,method,token_evaluate(ctx,method,0,stop)); summarize(ctx)
    for method,group in CONFIGS:
        token_evaluate(ctx,method,group,stop); summarize(ctx)


def kl_tokens(teacher_logits, student_logits, chunk=128):
    """Full-vocabulary KL(Pteacher||Pstudent), FP64; no truncation/top-k.

    Center the logit difference under Pteacher before logsumexp. This avoids
    subtracting two large normalizers. Only roundoff-negative KL is clamped.
    """
    if (teacher_logits.shape != student_logits.shape or teacher_logits.ndim != 3
            or min(teacher_logits.shape) < 1 or teacher_logits.shape[1] < 2 or chunk < 1):
        raise ValueError("KL logits shape mismatch")
    out=[]; minimum=0.
    for row in range(teacher_logits.shape[0]):
        parts=[]
        for start in range(0,teacher_logits.shape[1]-1,chunk):
            end=min(start+chunk,teacher_logits.shape[1]-1)
            pz=teacher_logits[row,start:end].to(student_logits.device,dtype=torch.float64)
            qz=student_logits[row,start:end].double()
            if not torch.isfinite(pz).all() or not torch.isfinite(qz).all():
                raise RuntimeError("Nonfinite KL logits")
            logp=pz.log_softmax(-1); prob=logp.exp(); difference=qz-pz
            centered=difference-(prob*difference).sum(-1,keepdim=True)
            value=torch.logsumexp(logp+centered,-1)
            raw_min=float(value.min()); minimum=min(minimum,raw_min)
            if not torch.isfinite(value).all() or raw_min < -1e-10:
                raise RuntimeError("Invalid KL beyond FP64 roundoff")
            parts.append(value.clamp_min(0).cpu())
        out.append(torch.cat(parts))
    return torch.stack(out),minimum


def kl_state(ctx, name):
    folder=ctx.output/"teacher_kl"; path=folder/(name+".json")
    if not path.exists():
        return {"experiment_identity":ctx.identity,"arm":name,"records":[],"complete":False}
    state=core.read_json(path)
    if state.get("experiment_identity") != ctx.identity or state.get("arm") != name:
        raise RuntimeError("KL identity mismatch")
    rr=state["records"]
    if len(rr)>WINDOWS or state["complete"] != (len(rr)==WINDOWS):
        raise RuntimeError("KL completion mismatch")
    deployed=core.read_json(folder/("deployment_"+name+".json"))
    teacher=core.read_json(folder/"teacher.json")
    if (core.fingerprint(deployed)!=state["deployment_sha256"]
            or core.fingerprint(teacher)!=state["teacher_sha256"]):
        raise RuntimeError("KL deployment/teacher binding changed")
    for i,row in enumerate(rr):
        expected=folder/"tokens"/name/f"window_{i:04d}.safetensors"
        if (row["window"]!=i or row["tokens"]!=LENGTH-1
                or Path(row["file"]["path"]).resolve()!=expected.resolve()):
            raise RuntimeError("KL window/path mismatch")
        values=ctx.inputs.tensors(row["file"])
        if (set(values)!={"kl"} or values["kl"].dtype!=torch.float64
                or values["kl"].shape!=(LENGTH-1,) or not torch.isfinite(values["kl"]).all()
                or (values["kl"]<0).any() or float(values["kl"].sum())!=row["kl_sum"]):
            raise RuntimeError("KL token payload mismatch")
    return state


def load_kl_model(ctx, device):
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(ctx.manifest["payload"]["source_config"]["model_path"],
        torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation="eager",device_map={"":device})
    model.eval(); model.config.use_cache=False
    if any(p.device != torch.device("cuda",device) for p in model.parameters()):
        raise RuntimeError("KL model must fit fully on its dedicated GPU")
    return model


def kl_evaluate(ctx, teacher, student, method, group, stop):
    name=label(method,group); folder=ctx.output/"teacher_kl"; folder.mkdir(exist_ok=True)
    state=kl_state(ctx,name)
    if state["complete"]: return state
    stop.check(); data=windows(ctx); artifacts=routes(ctx,method); handles=[]
    try:
        with core.heartbeat("KL install "+name):
            handles,deployed=install(student,artifacts,group,ctx.inputs)
            rec=bind_deployment(ctx,folder,name,deployed,artifacts,method,group)
            tref={"experiment_identity":ctx.identity,"parameter_bits":{n:single.tensor_record(t) for n,t in teacher.named_parameters()},
                  "buffer_bits":{n:single.tensor_record(t) for n,t in teacher.named_buffers()},"device_map":teacher.hf_device_map}
            expected=ctx.rank64_deployments["teacher"]["deployment"]
            if any(tref[k]!=expected[k] for k in ("parameter_bits","buffer_bits")):
                raise RuntimeError("KL teacher differs from frozen BF16 source")
            dual.freeze_json(folder/"teacher.json",tref)
            state.update(deployment_sha256=core.fingerprint(rec),teacher_sha256=core.fingerprint(tref))
        tdev=next(teacher.parameters()).device; sdev=next(student.parameters()).device
        directory=folder/"tokens"/name; directory.mkdir(parents=True,exist_ok=True)
        with torch.inference_mode():
            for i in range(len(state["records"]),WINDOWS):
                stop.check()
                with core.heartbeat(f"KL {name} window={i+1}/{WINDOWS}"):
                    ids=data["input_ids"][i:i+1]; mask=data["attention_mask"][i:i+1]
                    tl=teacher(input_ids=ids.to(tdev),attention_mask=mask.to(tdev),use_cache=False).logits
                    sl=student(input_ids=ids.to(sdev),attention_mask=mask.to(sdev),use_cache=False).logits
                    values,minimum=kl_tokens(tl,sl)
                    if method=="teacher" and float(values.max())>1e-9:
                        raise RuntimeError("Two-device BF16 teacher self-KL gate failed")
                del tl,sl
                f=single.atomic_tensors(directory/f"window_{i:04d}.safetensors",{"kl":values[0]})
                state["records"].append({"window":i,"tokens":LENGTH-1,"kl_sum":float(values.sum()),
                                          "raw_min_before_roundoff_clamp":minimum,"file":f})
                state["complete"]=i+1==WINDOWS
                core.atomic_json(folder/(name+".json"),state); stop.done+=1
    finally:
        for h in handles: h.remove()
    return state


def run_kl(ctx, stop):
    folder=ctx.output/"teacher_kl"; folder.mkdir(exist_ok=True)
    pending=[(m,g) for m,g in CONFIGS if not kl_state(ctx,label(m,g))["complete"]]
    if not pending: return
    stop.check(); teacher=student=None
    try:
        with core.heartbeat("KL load two resident BF16 models, one per GPU"):
            teacher=load_kl_model(ctx,0); student=load_kl_model(ctx,1)
        # Always validate/reuse a complete clean self-KL before modifying student.
        state=kl_evaluate(ctx,teacher,student,"teacher",0,stop)
        if not state["complete"]: raise RuntimeError("Incomplete self-KL control")
        for method,group in CONFIGS:
            if method=="teacher": continue
            kl_evaluate(ctx,teacher,student,method,group,stop); summarize(ctx)
    finally:
        teacher=student=None; dual.clear_gpu()


def damage_stats(base, removed, metric, seed=20260911, draws=2000, block=8):
    if len(base)!=WINDOWS or len(removed)!=WINDOWS:
        raise ValueError("Damage requires two complete paired evaluations")
    if any(x["window"]!=i or y["window"]!=i or x["tokens"]!=y["tokens"]
           or x["tokens"]<=0 or not math.isfinite(x[metric]) or not math.isfinite(y[metric])
           for i,(x,y) in enumerate(zip(base,removed))):
        raise RuntimeError("Damage pairing mismatch")
    delta=np.asarray([y[metric]-x[metric] for x,y in zip(base,removed)],dtype=np.float64)
    counts=np.asarray([x["tokens"] for x in base],dtype=np.float64)
    rng=np.random.default_rng(seed)
    starts=rng.integers(0,WINDOWS,size=(draws,math.ceil(WINDOWS/block)))
    indices=((starts[:,:,None]+np.arange(block))%WINDOWS).reshape(draws,-1)[:,:WINDOWS]
    boot=delta[indices].sum(1)/counts[indices].sum(1)
    low,high=np.quantile(boot,[.025,.975])
    return {"damage_per_token":float(delta.sum()/counts.sum()),"ci95_low":float(low),"ci95_high":float(high),
            "positive_damage_windows":int((delta>0).sum()),"negative_damage_windows":int((delta<0).sum()),
            "paired_windows":WINDOWS,"bootstrap":"circular moving blocks of 8 windows; exploratory, unadjusted",
            "bootstrap_draws":draws,"bootstrap_seed":seed}


def summarize(ctx):
    for protocol,key in (("token_ppl","nll_sum"),("teacher_kl","kl_sum")):
        folder=ctx.output/protocol
        if not folder.exists(): continue
        summaries=[]; detail=[]; records={}; paired=[]; damage=[]
        for m,g in CONFIGS:
            name=label(m,g)
            state=single.read_state(folder,name,ctx.identity,ctx.inputs) if protocol=="token_ppl" else kl_state(ctx,name)
            if not state["complete"]: continue
            rr=state["records"]; records[name]=rr
            total=sum(x[key] for x in rr); count=sum(x["tokens"] for x in rr)
            if count!=TOKENS: raise RuntimeError("Complete evaluation has wrong scored token count")
            row={"configuration":name,"method":m,"removed_group":g,"tokens":count,
                 "sum":total,"mean_per_token":total/count,"protocol":protocol}
            if protocol=="token_ppl": row["ppl"]=math.exp(total/count)
            summaries.append(row)
            detail.extend({"configuration":name,"window":x["window"],"tokens":x["tokens"],key:x[key]} for x in rr)
        for m in METHODS:
            baseline=label(m,0)
            if baseline not in records: continue
            for g in range(1,9):
                name=label(m,g)
                if name not in records: continue
                stats=damage_stats(records[baseline],records[name],key)
                damage.append({"method":m,"removed_group":g,"baseline":baseline,"configuration":name,**stats})
                paired.extend({"method":m,"group":g,"window":x["window"],"tokens":x["tokens"],
                               "delta":y[key]-x[key]} for x,y in zip(records[baseline],records[name]))
        for file,rows in (("summary.csv",summaries),("per_window.csv",detail),("damage.csv",damage),("paired_windows.csv",paired)):
            prev.write_csv(folder/file,rows)
        core.atomic_json(folder/"status.json",{"experiment_identity":ctx.identity,"complete":len(summaries)==len(CONFIGS),
            "completed":len(summaries),"expected":len(CONFIGS),"damage_rows":len(damage),
            "note":"Different method baselines; non-additive global interventions; no IID-token significance claim."})
    rows=[]
    for m in METHODS:
        for name in ctx.tasks:
            p=ctx.output/"geometry"/m/(core.safe_name(name)+".json")
            if p.exists(): rows.extend({"method":m,"module":name,**x} for x in core.read_json(p)["rows"])
    prev.write_csv(ctx.output/"group_geometry.csv",rows)
    totals=[]
    for m in METHODS:
        for g in range(1,9):
            selected=[r for r in rows if r["method"]==m and r["group"]==g]
            if len(selected)==len(ctx.tasks) and selected:
                totals.append({"method":m,"group":g,"modules":len(selected),
                    **{k:sum(r[k] for r in selected) for k in (
                        "removed_weight_fro2","removed_output_sse_calibration_A",
                        "delta_total_weight_sse","delta_total_output_sse_calibration_A")}})
    prev.write_csv(ctx.output/"group_geometry_totals.csv",totals)


def setup(args):
    # Delegate frozen source/environment audit without changing its implementation.
    ctx=dual.setup(SimpleNamespace(**{**vars(args),"protocol":"token"}))
    source=args.source_rank64_dir.resolve(); core.disjoint(ctx.output,[source])
    source_exp=core.read_json(source/"experiment.json"); source_id=core.fingerprint(source_exp)
    if (source_exp.get("version")!=dual.VERSION or source_exp.get("source_identity")!=ctx.source_identity
            or source_exp.get("source_factor_metadata")!=ctx.factor_metadata
            or source_exp.get("token_protocol")!={"windows":138,"length":2048,"batch":8,"ce_chunk":256}
            or source_exp.get("ranks")!=[8,16,32,64] or source_exp.get("methods")!=list(parent.METHODS)
            or source_exp.get("script_sha256")!=core.sha(dual.__file__)):
        raise RuntimeError("Rank64 source is not the matching frozen full-A dual evaluator")
    ctx.rank64_states={}; ctx.rank64_deployments={}; refs={}
    for m in ("teacher","wq",*parent.METHODS):
        name=dual.label(m,64 if m in parent.METHODS else None)
        folder=source/"token_ppl"
        state=single.read_state(folder,name,source_id,ctx.inputs)
        if not state["complete"]: raise RuntimeError("Missing completed source token control: "+name)
        dep=core.read_json(folder/("deployment_"+name+".json"))
        ctx.rank64_states[m]=state; ctx.rank64_deployments[m]=dep
        refs[m]={"state":single.file_record(folder/(name+".json")),"deployment":single.file_record(folder/("deployment_"+name+".json"))}
    src_ctx=copy.copy(ctx);src_ctx.output=source;src_ctx.identity=source_id
    ctx.rank64_checked={m:{} for m in parent.METHODS}
    ctx.rank64_checked_metadata={m:{} for m in parent.METHODS}
    for m in parent.METHODS:
        for name in ctx.tasks:
            rec=dual.read_checked(src_ctx,m,name)
            if (rec is None or rec.get("source_certificate")!=ctx.factor_metadata[m][name]
                    or rec.get("root_bits")!=ctx.factors[m][name]["root_bits"]
                    or rec["bits"]!=ctx.rank64_deployments[m]["deployment"]["correction_bits"][name]):
                raise RuntimeError("Source checked rank64 is not the evaluated frozen factor: "+m+" "+name)
            ctx.rank64_checked[m][name]=rec
            ctx.rank64_checked_metadata[m][name]=single.file_record(dual.checked_path(src_ctx,m,name))
    ctx.experiment={"version":VERSION,"script_sha256":core.sha(__file__),
        "helpers":{n:core.sha(Path(__file__).with_name(n)) for n in HELPERS},"source_audit":ctx.experiment,
        "rank64_source_experiment":single.file_record(source/"experiment.json"),"references":refs,
        "rank64_checked_metadata":ctx.rank64_checked_metadata,
        "methods":list(METHODS),"groups_1based":[[i+1 for i in x] for x in GROUPS],
        "intervention":"independent removal across all 224 modules; right rows zeroed; rank64 GEMM shapes",
        "plain_svd":"FP64 full SVD of identical frozen FP32 residual values; A=I,G=I",
        "token":{"windows":WINDOWS,"length":LENGTH,"batch":8,"ce_chunk":256,"scored_tokens":TOKENS},
        "kl":{"windows":WINDOWS,"length":LENGTH,"batch":1,"teacher_device":0,"student_device":1,
              "vocabulary":"full","reduction":"FP64 centered-logit logsumexp","chunk":128,
              "checkpoint":"one whole window","same_teacher":True,"self_kl_max":1e-9},
        "bootstrap":{"seed":20260911,"draws":2000,"circular_block_windows":8},
        "no_word_ppl":True,"no_collection":True,"no_model_merge":True}
    ctx.identity=core.fingerprint(ctx.experiment)
    return ctx


def pilot(ctx,stop):
    prepare_all(ctx,stop,list(parent.PILOT))
    data=windows(ctx); model=None; teacher=student=None
    try:
        stop.check()
        loading=dict(ctx.manifest["payload"]["source_config"])
        loading["max_memory"]=ctx.manifest["payload"]["config"]["eval_max_memory"]
        model=ctx.legacy.load_model(loading,"bfloat16","balanced"); dual.resident(model)
        device=ctx.legacy._input_device(model)
        with torch.inference_mode(),core.heartbeat("PILOT frozen batch8 teacher smoke"):
            ids=data["input_ids"][:8].to(device); mask=data["attention_mask"][:8].to(device)
            logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
            rr,_=single.nll_with_tokens(logits,ids,mask,256)
            difference=max(abs(x[0]-y["nll_sum"]) for x,y in zip(rr,ctx.rank64_states["teacher"]["records"][:8]))
            if difference>1e-3: raise RuntimeError("Pilot teacher batch8 control failed")
            del logits,ids,mask
        model=None; dual.clear_gpu(); stop.check()
        teacher=load_kl_model(ctx,0); student=load_kl_model(ctx,1)
        with torch.inference_mode(),core.heartbeat("PILOT batch1 KL memory and self control"):
            ids=data["input_ids"][:1]; mask=data["attention_mask"][:1]
            tl=teacher(input_ids=ids.to("cuda:0"),attention_mask=mask.to("cuda:0"),use_cache=False).logits
            sl=student(input_ids=ids.to("cuda:1"),attention_mask=mask.to("cuda:1"),use_cache=False).logits
            value,_=kl_tokens(tl,sl)
            if float(value.max())>1e-9: raise RuntimeError("Pilot teacher self-KL failed")
            self_max=float(value.max()); del sl
            # Exercise the exact BF16 masked-hook path on the two large pilot
            # modules. All other targets use Wq only; this is not an endpoint.
            artifacts=routes(ctx,"wq")
            for n in parent.PILOT:
                artifacts[n]["correction"]=factor_record(ctx,"full_gf",n)["file"]
            handles=[]
            try:
                handles,_=install(student,artifacts,8,ctx.inputs)
                sl=student(input_ids=ids.to("cuda:1"),attention_mask=mask.to("cuda:1"),use_cache=False).logits
                smoke,_=kl_tokens(tl,sl)
                smoke_mean=float(smoke.mean()); del sl
            finally:
                for h in handles: h.remove()
                handles.clear()
            del tl
        dual.freeze_json(ctx.output/"pilot.json",{"experiment_identity":ctx.identity,"status":"PASS",
            "modules":list(parent.PILOT),"teacher_window_nll_max_difference":difference,
            "self_kl_max":self_max,"two_module_gf_drop8_smoke_kl":smoke_mean,
            "scope":"two modules, teacher batch8 and KL batch1 masked-hook smoke; not full mechanism PASS"})
    finally:
        model=teacher=student=None; dual.clear_gpu()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("doctor","pilot","prepare","run","token","kl","summarize"))
    for key in ("run-dir","repo-dir","output-dir","source-fp64-dir","source-rank64-dir",
                "official-qera-root","harness-source","word-reference-dir"):
        p.add_argument("--"+key,required=True,type=Path)
    p.add_argument("--max-hours",type=float,default=10)
    p.add_argument("--max-new-units",type=int)
    args=p.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours<=0 or (args.max_new_units is not None and args.max_new_units<1):
        p.error("Budgets must be positive")
    stop=prev.Budget(args.max_hours,args.max_new_units)
    signal.signal(signal.SIGINT,stop.signal); signal.signal(signal.SIGTERM,stop.signal)
    ctx=setup(args)
    if args.command=="doctor":
        core.log("DOCTOR PASS: read-only source audit; 4 methods x 9 conditions + BF16/Wq; no experiment executed")
        return 0
    if ctx.output.exists() and not (ctx.output/"experiment.json").exists() and any(ctx.output.iterdir()):
        raise RuntimeError("Refusing unowned nonempty output")
    ctx.output.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(ctx.output).free<8*2**30: raise RuntimeError("Need at least 8 GiB free")
    with core.audit_lock(ctx.output):
        dual.freeze_json(ctx.output/"experiment.json",ctx.experiment)
        try:
            if args.command=="summarize": summarize(ctx); return 0
            if args.command=="pilot":
                pilot(ctx,stop); core.log("PILOT COMPLETE; full factor preparation and endpoint gates still required"); return 0
            if args.command in ("prepare","run"): prepare_all(ctx,stop)
            if args.command=="prepare": summarize(ctx); return 0
            require_factors(ctx)
            if args.command in ("token","run"): run_token(ctx,stop)
            if args.command in ("kl","run"):
                # KL cannot bypass the exact all-retained endpoint gates.
                for m in ("teacher","wq",*parent.METHODS):
                    state=single.read_state(ctx.output/"token_ppl",label(m,0),ctx.identity,ctx.inputs)
                    if not state["complete"]: raise RuntimeError("Run token/run before KL")
                    token_gate(ctx,m,state)
                run_kl(ctx,stop)
            summarize(ctx); core.log("REQUESTED STAGES COMPLETE; interpret damage with method-specific baselines")
        except prev.Paused:
            summarize(ctx); core.log("PAUSED (75): rerun same command/output; no source modifications"); return 75
        except Exception as exc:
            core.atomic_json(ctx.output/"last_failure.json",{"experiment_identity":ctx.identity,"error":str(exc),"type":type(exc).__name__})
            raise
    return 0


if __name__=="__main__":
    raise SystemExit(main())
