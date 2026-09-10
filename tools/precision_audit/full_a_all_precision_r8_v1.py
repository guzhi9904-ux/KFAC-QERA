#!/usr/bin/env python3
"""All-module FA+GI/GD/GF FP64 solve diagnostic; fixed r8 BF16 deployment.

Frozen sources read-only. No new statistics, A-root construction, damping,
pseudoinverse, reduced SVD or mixed per-module precision fallback.
"""
from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
import signal
import shutil
import sys
import time
from types import SimpleNamespace

sys.dont_write_bytecode = True
import torch
from safetensors import safe_open
import full_g_precision_r8_v1 as single

core, previous = single.core, single.previous
VERSION = "full_a_all_precision_r8_v1"
SINGLE_SHA = "793f1bc5ef0c27362af9acae4989cb089aedd6ddc9597213ab64ca15b47f4e7e"
METHODS = ("full_gi", "full_gd", "full_gf")
PILOT = ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.down_proj")
RANK, WINDOWS, LENGTH = 8, 138, 2048


class Budget(previous.Budget):
    def __init__(self, hours=None, batches=None, solves=None):
        super().__init__(hours, batches)
        self.solves, self.solved = solves, 0

    def check(self):
        super().check()
        if self.solves is not None and self.solved >= self.solves:
            raise previous.Paused()


def memory():
    import resource
    return {"cpu_process_peak_GiB":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            "gpu_peak_GiB":[round(torch.cuda.max_memory_allocated(i)/2**30,3)
                            for i in range(torch.cuda.device_count())],
            "gpu_reserved_GiB":[round(torch.cuda.memory_reserved(i)/2**30,3)
                                for i in range(torch.cuda.device_count())]}


def factor_paths(output, method, name):
    path = output/"factors"/method/(core.safe_name(name)+".safetensors")
    return path,path.with_suffix(".json")


def solved_record(output, method, name, binding, identity, inputs):
    path,meta = factor_paths(output,method,name)
    if not meta.exists():
        return None
    record = core.read_json(meta)
    if (record.get("experiment_identity") != identity or record.get("status") != "PASS"
            or record.get("method") != method or record.get("module") != name
            or record.get("source_inputs") != binding
            or Path(record["file"]["path"]).resolve() != path.resolve()):
        raise RuntimeError("Generated factor identity/binding mismatch: "+method+" "+name)
    values = inputs.tensors(record["file"])
    single.check_factors(values,binding["shape"][1],binding["shape"][0])
    return record


def make_g_root(method, binding, dg, inputs, device):
    d_out = binding["shape"][0]
    if method == "full_gi":
        return torch.eye(d_out,dtype=torch.float32,device=device),{"representation":"identity"}
    if method == "full_gd":
        return torch.diag(previous.diagonal_root(dg).to(device)),{"representation":"original_diagonal"}
    if method != "full_gf":
        raise ValueError(method)
    raw = inputs.tensors(binding["gram"])
    if raw["gram"].dtype != torch.float64 or tuple(raw["gram"].shape) != (d_out,d_out):
        raise RuntimeError("Expected original FP64 Full-G statistics")
    diagonal_drift = core.rel(raw["gram"].diagonal(),raw["diagonal"])
    dg_drift = core.rel(raw["diagonal"],dg)
    if diagonal_drift > 1e-10 or dg_drift > 1e-6:
        raise RuntimeError("Full-G diagonal no longer agrees with frozen DG")
    g,info = core.full_root(raw["gram"].to(device),256*2047,1e-6)
    return g,{**info,"direct_diagonal_drift":diagonal_drift,"original_dg_drift":dg_drift}


def prepare_one(name, method, binding, dg, inputs, output, identity, device):
    """One transaction per module/method. Exceptions must abort; no fallback."""
    record = solved_record(output,method,name,binding,identity,inputs)
    if record is not None:
        return record
    path,meta = factor_paths(output,method,name)
    path.parent.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(),core.heartbeat("SOLVE "+method+" "+name+" FP64 full-SVD/rank64 -> BF16 r8"):
        a = inputs.tensors(binding["root"])["full"].to(device,copy=True)
        d_out,d_in = binding["shape"]
        if a.dtype != torch.float32 or tuple(a.shape) != (d_in,d_in) or not torch.isfinite(a).all():
            raise RuntimeError("Require unchanged stored FP32 Full-A root")
        with safe_open(str(inputs.verify(binding["model"])),framework="pt",device="cpu") as handle:
            w = handle.get_tensor(name+".weight").clone()
        q = inputs.tensors(binding["quant"])["weight_q"]
        if w.dtype != torch.bfloat16 or q.dtype != torch.bfloat16 or tuple(w.shape) != (d_out,d_in) or w.shape != q.shape:
            raise RuntimeError("Frozen W/Wq dtype/shape mismatch")
        e = (w.float()-q.float()).T.to(device)
        del w,q
        if not torch.isfinite(e).all():
            raise RuntimeError("Nonfinite quantization error")
        g,g_info = make_g_root(method,binding,dg,inputs,device)
        if g.dtype != torch.float32 or tuple(g.shape) != (d_out,d_out):
            raise RuntimeError("Require frozen-protocol FP32 G root")
        root_bits = {"a":single.tensor_record(a),"g":single.tensor_record(g)}
        left,right,s,inverse = single.solve_fp64(e,a,g)
        metrics = previous.metrics(e,a,g,left,right,RANK)
        tail = core.norm2(s[RANK:])
        excess = (metrics["sse_after"]-tail)/max(metrics["sse_before"],1e-30)
        if not math.isfinite(excess) or abs(excess) > 1e-9:
            raise RuntimeError("FP64 r8 objective disagrees with SVD tail; no correction committed")
        old = inputs.tensors(binding["baseline_factors"][method])
        if (old["A"].dtype != torch.float32 or old["B"].dtype != torch.float32
                or old["A"].shape != (d_in,64) or old["B"].shape != (64,d_out)
                or not torch.isfinite(old["A"]).all() or not torch.isfinite(old["B"]).all()):
            raise RuntimeError("Invalid frozen baseline factors")
        old_metrics = previous.metrics(e,a,g,old["A"].to(device),old["B"].to(device),RANK)
        values = {"A_fp64":left.cpu(),"B_fp64":right.cpu()}
        values.update(A_bf16_r8=values["A_fp64"][:,:RANK].bfloat16(),
                      B_bf16_r8=values["B_fp64"][:RANK].bfloat16())
        single.check_factors(values,d_in,d_out)
        # FP64 product of rounded factors is a rounding proxy, not BF16 activation execution.
        rounded_residual = e.double()-values["A_bf16_r8"].to(device).double() @ values["B_bf16_r8"].to(device).double()
        rounded_sse = core.norm2(a.double() @ rounded_residual @ g.double())
        if not math.isfinite(rounded_sse):
            raise RuntimeError("Nonfinite rounded-factor objective")
        record = {"experiment_identity":identity,"status":"PASS","module":name,"method":method,
                  "rank":8,"solve_rank":64,"source_inputs":binding,"root_bits":root_bits,
                  "g_diagnostics":g_info,"inverse":inverse,"new_fp64_metrics":metrics,
                  "saved_old_metrics":old_metrics,"svd_tail_sse_fp64":tail,
                  "relative_tail_difference":excess,"rounded_factor_sse_fp64_proxy":rounded_sse,
                  "rounded_objective_increased_flag":rounded_sse > metrics["sse_before"]*(1+1e-5)+1e-20,
                  "deployment":{k:single.tensor_record(values[k]) for k in ("A_bf16_r8","B_bf16_r8")},
                  "file":single.atomic_tensors(path,values),"elapsed_seconds":time.monotonic()-started,
                  "memory":memory(),
                  "note":"Stored FP32 A root; frozen-protocol FP32 G root. No root-construction precision upgrade. PASS is numerical checks, not scientific success."}
        core.atomic_json(meta,record)
    del a,g,e,left,right,s,values,old,rounded_residual
    gc.collect(); torch.cuda.empty_cache()
    core.log(f"COMMITTED {method} {name} elapsed={record['elapsed_seconds']:.1f}s memory={record['memory']}")
    return record


def solve_all(ctx, stop, names=None):
    names = list(ctx.tasks) if names is None else names
    for index,name in enumerate(names,1):
        for method in METHODS:
            binding = ctx.tasks[name]
            if solved_record(ctx.output,method,name,binding,ctx.identity,ctx.inputs) is not None:
                continue
            stop.check()
            core.log(f"module={index}/{len(names)} method={method} name={name}")
            prepare_one(name,method,binding,ctx.dg[name],ctx.inputs,ctx.output,ctx.identity,ctx.device)
            stop.solved += 1
    write_solve_summary(ctx)


def write_solve_summary(ctx):
    rows=[]; counts={m:0 for m in METHODS}
    for name,binding in ctx.tasks.items():
        for method in METHODS:
            record=solved_record(ctx.output,method,name,binding,ctx.identity,ctx.inputs)
            if record is None: continue
            counts[method]+=1
            row={"module":name,"method":method,"rank":8,"elapsed_seconds":record["elapsed_seconds"],
                 "relative_tail_difference":record["relative_tail_difference"],
                 "rounded_factor_sse_fp64_proxy":record["rounded_factor_sse_fp64_proxy"],
                 "rounded_objective_increased_flag":record["rounded_objective_increased_flag"],**record["inverse"]}
            for key in ("sse_before","sse_after","correction_over_error_norm","bf16_rounding_product_relative"):
                row["old_"+key]=record["saved_old_metrics"][key]
                row["new_"+key]=record["new_fp64_metrics"][key]
            rows.append(row)
    previous.write_csv(ctx.output/"solve_metrics.csv",rows)
    status={"experiment_identity":ctx.identity,"status":"COMPLETE" if all(c==len(ctx.tasks) for c in counts.values()) else "INCOMPLETE",
            "completed":counts,"expected_per_method":len(ctx.tasks),
            "rounded_objective_flags":sum(r["rounded_objective_increased_flag"] for r in rows)}
    core.atomic_json(ctx.output/"solve_status.json",status)
    return status


def routes(ctx,method,arm):
    result={}
    for name,b in ctx.tasks.items():
        if arm=="OLD":
            factor=b["baseline_factors"][method]
        elif arm=="FP64":
            record=solved_record(ctx.output,method,name,b,ctx.identity,ctx.inputs)
            if record is None: raise RuntimeError("Cannot evaluate incomplete FP64 factors: "+method+" "+name)
            factor=record["file"]
        else: raise ValueError(arm)
        result[name]={"quant":b["quant"],"correction":factor}
    return result


def install(model,artifacts,arm,inputs):
    modules=dict(model.named_modules()); handles=[]; bits={}
    try:
        with torch.no_grad():
            for name,files in artifacts.items():
                module=modules[name]
                q=inputs.tensors(files["quant"])["weight_q"]
                if q.dtype != torch.bfloat16 or module.weight.dtype != torch.bfloat16 or q.shape != module.weight.shape:
                    raise RuntimeError("Deployment Wq dtype/shape mismatch")
                module.weight.copy_(q.to(module.weight.device))
                if single.tensor_record(q)!=single.tensor_record(module.weight):
                    raise RuntimeError("Installed Wq differs from frozen bits")
                factors=inputs.tensors(files["correction"])
                if arm=="FP64":
                    single.check_factors(factors,module.in_features,module.out_features)
                    left,right=factors["A_bf16_r8"],factors["B_bf16_r8"]
                elif arm=="OLD":
                    if (factors["A"].dtype!=torch.float32 or factors["B"].dtype!=torch.float32
                            or factors["A"].shape!=(module.in_features,64) or factors["B"].shape!=(64,module.out_features)):
                        raise RuntimeError("Frozen baseline factor dtype/shape mismatch")
                    left,right=factors["A"][:,:8].bfloat16(),factors["B"][:8].bfloat16()
                else: raise ValueError(arm)
                left,right=left.to(module.weight.device),right.to(module.weight.device)
                if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                    raise RuntimeError("Nonfinite deployed correction")
                bits[name]={"A":single.tensor_record(left),"B":single.tensor_record(right)}
                handles.append(module.register_forward_hook(previous.correction_hook(left,right)))
            deployment={"parameter_bits":{n:single.tensor_record(t) for n,t in model.named_parameters()},
                        "buffer_bits":{n:single.tensor_record(t) for n,t in model.named_buffers()},
                        "device_map":model.hf_device_map,"correction_bits":bits}
        return handles,deployment
    except BaseException:
        for h in handles: h.remove()
        raise


def assert_common_deployment(reference,current):
    for key in ("parameter_bits","buffer_bits","device_map"):
        if reference[key]!=current[key]: raise RuntimeError("All-module arms differ in "+key)
    if set(reference["correction_bits"])!=set(current["correction_bits"]):
        raise RuntimeError("Correction module coverage differs")


def eval_folder(ctx,method):
    return ctx.output/"evaluation"/method


def evaluate(ctx,legacy,method,arm,stop):
    folder=eval_folder(ctx,method); folder.mkdir(parents=True,exist_ok=True)
    state=single.read_state(folder,arm,ctx.identity,ctx.inputs)
    artifacts=routes(ctx,method,arm)
    route_sha=core.fingerprint(artifacts)
    if state["records"] and state.get("route_sha256")!=route_sha:
        raise RuntimeError("Evaluation checkpoint routed to different artifacts")
    if state["complete"]:
        core.log(method+" "+arm+" already complete")
        return state
    stop.check()
    windows=ctx.inputs.tensors(ctx.manifest["payload"]["data"]["wikitext2"])
    if (tuple(windows["input_ids"].shape)!=(WINDOWS,LENGTH)
            or windows["attention_mask"].shape!=windows["input_ids"].shape or not (windows["attention_mask"]==1).all()):
        raise RuntimeError("Require original all-valid 138 x 2048 evaluation windows")
    loading=dict(ctx.manifest["payload"]["source_config"])
    loading["max_memory"]=ctx.manifest["payload"]["config"]["eval_max_memory"]
    model,handles=None,[]
    try:
        torch.manual_seed(1234)
        with core.heartbeat(method+" "+arm+" load BF16 / install / hash deployment"):
            model=legacy.load_model(loading,"bfloat16","balanced")
            if not {str(v) for v in model.hf_device_map.values()}<={"0","1","cuda:0","cuda:1"}:
                raise RuntimeError("Model must be fully resident on the two GPUs")
            handles,deployment=install(model,artifacts,arm,ctx.inputs)
        common_path=ctx.output/"evaluation/common_deployment.json"
        if common_path.exists():
            common=core.read_json(common_path)
            if common["experiment_identity"]!=ctx.identity: raise RuntimeError("Common deployment identity mismatch")
            assert_common_deployment(common["deployment"],deployment)
        else:
            if arm!="OLD": raise RuntimeError("OLD deployment required first")
            core.atomic_json(common_path,{"experiment_identity":ctx.identity,"deployment":deployment})
        deploy_record={"experiment_identity":ctx.identity,"arm":arm,"method":method,
                       "route_sha256":route_sha,"deployment":deployment}
        deploy_path=folder/("deployment_"+arm+".json")
        if deploy_path.exists() and core.read_json(deploy_path)!=deploy_record:
            raise RuntimeError("Actual deployment changed on resume")
        if not deploy_path.exists(): core.atomic_json(deploy_path,deploy_record)
        state.update(deployment_sha256=core.fingerprint(deployment),route_sha256=route_sha)
        token_dir=folder/"tokens"/arm; token_dir.mkdir(parents=True,exist_ok=True)
        device=legacy._input_device(model); checked=False
        with torch.inference_mode():
            for start in range(len(state["records"]),WINDOWS,8):
                stop.check(); end=min(start+8,WINDOWS)
                ids=windows["input_ids"][start:end].to(device); mask=windows["attention_mask"][start:end].to(device)
                with core.heartbeat(f"EVAL {method} {arm} rank=8 windows={start+1}-{end}/138"):
                    logits=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
                    values,tokens=single.nll_with_tokens(logits,ids,mask,256)
                    if not checked:
                        if values!=legacy._chunked_window_nll(logits,ids,mask,256):
                            raise RuntimeError("Instrumentation changed frozen NLL reduction")
                        checked=True
                del logits,ids,mask
                state["records"].extend({"window":start+i,"tokens":n,"nll_sum":loss,
                                         "token_nll_sum_fp64":float(tokens[i].double().sum())}
                                        for i,(loss,n) in enumerate(values))
                previous.valid_records(state["records"],WINDOWS)
                record=single.atomic_tensors(token_dir/f"batch_{start:04d}.safetensors",{"nll":tokens})
                state["batches"].append({"start":start,"end":end,"file":record})
                state["complete"]=end==WINDOWS
                core.atomic_json(folder/(arm+".json"),state)
                stop.done+=1
                core.log(f"COMMITTED {method} {arm} window={end}/138")
    finally:
        for h in handles: h.remove()
        handles.clear(); model=None
        gc.collect(); torch.cuda.empty_cache()
    return state


def verify_evaluations(ctx):
    common_path=ctx.output/"evaluation/common_deployment.json"
    common=core.read_json(common_path) if common_path.exists() else None
    if common is not None and common["experiment_identity"]!=ctx.identity:
        raise RuntimeError("Common deployment identity mismatch")
    for method in METHODS:
        for arm in ("OLD","FP64"):
            folder=eval_folder(ctx,method)
            state=single.read_state(folder,arm,ctx.identity,ctx.inputs)
            path=folder/("deployment_"+arm+".json")
            if path.exists():
                record=core.read_json(path)
                if (common is None or record["experiment_identity"]!=ctx.identity
                        or record["method"]!=method or record["arm"]!=arm):
                    raise RuntimeError("Deployment report mismatch")
                assert_common_deployment(common["deployment"],record["deployment"])
                route_sha=core.fingerprint(routes(ctx,method,arm))
                if record["route_sha256"]!=route_sha or (state["records"] and state.get("route_sha256")!=route_sha):
                    raise RuntimeError("Deployment/checkpoint artifact route mismatch")


def controls(ctx,legacy,stop):
    for method in METHODS:
        old=evaluate(ctx,legacy,method,"OLD",stop)
        check=previous.control_check(old["records"],ctx.references[method]["records"],
                                     ctx.manifest["payload"]["config"]["control_ppl_tolerance"])
        core.atomic_json(eval_folder(ctx,method)/"control_check.json",{"experiment_identity":ctx.identity,**check})
        core.log(method+" control "+str(check))
        if check["status"]!="PASS":
            raise RuntimeError("OLD replay FAILED: "+method+"; FP64 evaluation blocked")


def summarize(ctx):
    folder=ctx.output/"evaluation"; folder.mkdir(exist_ok=True)
    complete={}; summary=[]; records=[]; counts={}; control_pass={}
    for method in METHODS:
        gate=eval_folder(ctx,method)/"control_check.json"
        value=core.read_json(gate) if gate.exists() else {}
        control_pass[method]=value.get("experiment_identity")==ctx.identity and value.get("status")=="PASS"
        for arm in ("OLD","FP64"):
            state=single.read_state(eval_folder(ctx,method),arm,ctx.identity,ctx.inputs)
            label=method+"_"+arm; counts[label]=len(state["records"])
            if not state["complete"]: continue
            complete[label]=state
            nll=sum(r["nll_sum"] for r in state["records"])
            summary.append({"configuration":label,"method":method,"arm":arm,"rank":8,"windows":138,
                            "prediction_tokens":138*2047,"nll_sum":nll,"mean_nll":nll/(138*2047),"ppl":math.exp(nll/(138*2047))})
            records.extend({"configuration":label,**r} for r in state["records"])
    pairs=[(m+"_OLD",m+"_FP64") for m in METHODS]
    pairs += [("full_gi_FP64","full_gd_FP64"),("full_gi_FP64","full_gf_FP64"),("full_gd_FP64","full_gf_FP64")]
    comparisons=[]; paired=[]
    for first,second in pairs:
        if first not in complete or second not in complete: continue
        a,b=complete[first],complete[second]; label=second+"_minus_"+first
        dn=sum(y["nll_sum"]-x["nll_sum"] for x,y in zip(a["records"],b["records"]))
        wins=losses=ties=0; absolute=token_delta=0.
        for x,y in zip(a["batches"],b["batches"]):
            delta=ctx.inputs.tensors(y["file"])["nll"].double()-ctx.inputs.tensors(x["file"])["nll"].double()
            wins+=int((delta<0).sum()); losses+=int((delta>0).sum()); ties+=int((delta==0).sum())
            absolute+=float(delta.abs().sum()); token_delta+=float(delta.sum())
        comparisons.append({"comparison":label,"delta_nll":dn,"delta_mean_nll":dn/(138*2047),
                            "delta_ppl":previous.ppl(b["records"])-previous.ppl(a["records"]),
                            "improved_windows":sum(y["nll_sum"]<x["nll_sum"] for x,y in zip(a["records"],b["records"])),
                            "improved_tokens":wins,"worsened_tokens":losses,"tied_tokens":ties,
                            "token_delta_mean_fp64":token_delta/(138*2047),"token_mean_absolute_delta":absolute/(138*2047)})
        paired.extend({"comparison":label,"window":x["window"],"tokens":x["tokens"],
                       "first_nll":x["nll_sum"],"second_nll":y["nll_sum"],"delta_nll":y["nll_sum"]-x["nll_sum"]}
                      for x,y in zip(a["records"],b["records"]))
    for name,rows in (("ppl_summary",summary),("per_window",records),("comparisons",comparisons),("paired_windows",paired)):
        previous.write_csv(folder/(name+".csv"),rows)
    status={"experiment_identity":ctx.identity,"status":"COMPLETE" if len(complete)==6 and all(control_pass.values()) else "INCOMPLETE",
            "completed_configurations":list(complete),"committed_windows":counts,"control_passed":control_pass,
            "note":"All-module stored-root FP64 solve diagnostic at r8, not root-construction precision. No IID-token significance claims."}
    core.atomic_json(folder/"status.json",status)
    return status


def setup(args):
    if core.sha(single.__file__)!=SINGLE_SHA: raise RuntimeError("Frozen single-module helper changed")
    # Reuse the tested source audit, strict environment check, output isolation and numerical policy.
    proxy=SimpleNamespace(**vars(args)); proxy.command="prepare" if args.command=="solve" else "run"
    manifest,_,_,_,inputs,output,parent=single.setup(proxy)
    payload=manifest["payload"]; baseline=payload["full_g_protocol"]["baseline_inputs"]
    states=core.committed_states(args.run_dir.resolve(),manifest)
    tasks={layer["name"]:core.metadata(args.run_dir.resolve(),manifest,group,layer,states)
           for group in payload["groups"] for layer in group["layers"]}
    if len(tasks)!=224 or any(name not in tasks for name in PILOT): raise RuntimeError("Expected all Llama 224 modules")
    artifacts={m:{} for m in METHODS}
    for name,b in tasks.items():
        b["baseline_factors"]={m:b["corrections"]["full_gf"]["file"] if m=="full_gf" else baseline[m][name]["correction"] for m in METHODS}
        for method in METHODS:
            if method!="full_gf" and baseline[method][name]["quant"]!=b["quant"]:
                raise RuntimeError("GI/DG/GF use different Wq")
            artifacts[method][name]={"quant":b["quant"],"correction":b["baseline_factors"][method]}
    references={}; ref_files={}
    for method in METHODS:
        name=method.upper()+"_R8"
        path=args.run_dir.resolve()/"evaluation/configurations"/(name+".json")
        record=core.read_json(path)
        expected=core.fingerprint({"manifest":manifest["sha256"],"name":name,"artifacts":artifacts[method]})
        if record.get("protocol_sha256")!=expected or record.get("complete") is not True or len(record["records"])!=138:
            raise RuntimeError("Missing/mismatched frozen r8 reference: "+method)
        previous.valid_records(record["records"],138)
        references[method]=record; ref_files[method]=single.file_record(path)
    protocol=payload["full_g_protocol"]
    inputs.verify(protocol["diagonal_g_state_file"])
    dg_state=protocol["diagonal_g_state"]
    if (core.read_json(protocol["diagonal_g_state_file"]["path"])!=dg_state
            or dg_state["windows_completed"]!=256 or dg_state["prediction_tokens"]!=256*2047):
        raise RuntimeError("Original DG checkpoint mismatch")
    dg=inputs.tensors(dg_state["file"])
    if set(dg)!=set(tasks): raise RuntimeError("Original DG coverage mismatch")
    for name,value in dg.items():
        if value.dtype!=torch.float64 or value.shape!=(tasks[name]["shape"][0],):
            raise RuntimeError("DG dtype/shape mismatch")
        previous.diagonal_root(value)
    experiment={"version":VERSION,"script_sha256":core.sha(__file__),"single_helper_sha256":SINGLE_SHA,
                "frozen_parent_audit":parent,"methods":list(METHODS),"modules":list(tasks),"rank":8,"solve_rank":64,
                "all_bindings_sha256":core.fingerprint(tasks),"references":ref_files,
                "dg_state_file":protocol["diagonal_g_state_file"],"pilot_modules":list(PILOT),
                "inverse_policy":"FP64 torch.linalg.solve, no fallback/damping/pinv; dense GI/DG/GF G route",
                "g_root_policy":"FP32 roots from unchanged original normalization/floor; G root recomputed, original root bits not archived",
                "new_a_root":False,"deploy":"BF16 two-GEMM rank8 hooks","expected_evaluations":6}
    return SimpleNamespace(manifest=manifest,tasks=tasks,references=references,inputs=inputs,output=output,
                           experiment=experiment,identity=core.fingerprint(experiment),dg=dg,device="cuda:0")


def load_legacy(ctx,args):
    sys.path.insert(0,str(args.repo_dir.resolve()/"experiments"))
    from qera_diag_g_isolation import pipeline as legacy
    from qera_original_a_isolation import pipeline as original
    verified={Path(r["path"]).resolve() for r in ctx.manifest["payload"]["code"].values()}
    if any(Path(m.__file__).resolve() not in verified for m in (legacy,original)):
        raise RuntimeError("Imported model/evaluation helpers not manifest-bound")
    for record in ctx.manifest["payload"]["model_files"].values(): ctx.inputs.verify(record)
    return legacy


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("pilot","run","solve","evaluate"))
    for key in ("run-dir","repo-dir","output-dir"): parser.add_argument("--"+key,required=True,type=Path)
    parser.add_argument("--max-hours",type=float,default=10)
    parser.add_argument("--max-new-solves",type=int)
    parser.add_argument("--max-new-batches",type=int)
    args=parser.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours<=0: parser.error("max-hours must be positive finite")
    if any(v is not None and v<1 for v in (args.max_new_solves,args.max_new_batches)): parser.error("Budgets must be positive")
    if args.command=="pilot" and (args.max_new_solves is not None or args.max_new_batches is not None):
        parser.error("pilot has a fixed two-module + one-eval-batch budget")
    if args.command=="solve" and args.max_new_batches is not None: parser.error("solve has no evaluation batches")
    if args.command=="evaluate" and args.max_new_solves is not None: parser.error("evaluate has no new solves")
    stop=Budget(args.max_hours,args.max_new_batches,args.max_new_solves)
    signal.signal(signal.SIGINT,stop.signal); signal.signal(signal.SIGTERM,stop.signal)
    ctx=setup(args)
    if ctx.output.exists() and not (ctx.output/"experiment.json").exists() and any(ctx.output.iterdir()):
        raise RuntimeError("Refusing unowned nonempty output directory")
    ctx.output.mkdir(parents=True,exist_ok=True)
    needed_bytes=sum(sum(b["shape"]) for b in ctx.tasks.values())*3*(64*8+8*2)
    free_bytes=shutil.disk_usage(ctx.output).free
    core.log(f"storage: generated_factor_payload_GiB={needed_bytes/2**30:.2f}; free_GiB={free_bytes/2**30:.2f}")
    if free_bytes < max(2*2**30,needed_bytes):
        raise RuntimeError("Insufficient disk headroom for generated factors and transactions")
    with core.audit_lock(ctx.output):
        path=ctx.output/"experiment.json"
        if path.exists() and core.read_json(path)!=ctx.experiment:
            raise RuntimeError("Experiment changed; do not mix configurations on resume")
        if not path.exists(): core.atomic_json(path,ctx.experiment)
        try:
            stop.check(); verify_evaluations(ctx)
            if args.command=="pilot":
                solve_all(ctx,stop,list(PILOT))
                legacy=load_legacy(ctx,args)
                stop.batches=1
                pilot_state=single.read_state(eval_folder(ctx,"full_gf"),"OLD",ctx.identity,ctx.inputs)
                if len(pilot_state["records"]) < 8:
                    evaluate(ctx,legacy,"full_gf","OLD",stop)
                core.log("PILOT COMPLETE; use run for all modules and methods")
            elif args.command=="solve":
                solve_all(ctx,stop)
            else:
                legacy=load_legacy(ctx,args)
                controls(ctx,legacy,stop)
                if args.command=="run": solve_all(ctx,stop)
                if write_solve_summary(ctx)["status"]!="COMPLETE":
                    raise RuntimeError("FP64 factor set incomplete; run solve/run before evaluate")
                for method in METHODS:
                    evaluate(ctx,legacy,method,"FP64",stop)
                    summarize(ctx)
            verify_evaluations(ctx)
            core.log(str(summarize(ctx)))
        except previous.Paused:
            write_solve_summary(ctx); summarize(ctx)
            core.log("PAUSED: committed methods/batches retained. Resume with run (omit pilot/unit budgets).")
            return 75
        except Exception as exc:
            core.atomic_json(ctx.output/"last_failure.json",{"experiment_identity":ctx.identity,
                             "time":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"type":type(exc).__name__,"message":str(exc),
                             "note":"No automatic numerical fallback; previous committed methods/batches retained."})
            raise
    return 0


if __name__=="__main__": raise SystemExit(main())
