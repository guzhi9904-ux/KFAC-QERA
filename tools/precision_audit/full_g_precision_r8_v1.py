#!/usr/bin/env python3
"""Three-arm, one-module r8 precision intervention. Frozen sources are read-only.

OLD: saved FA+GF throughout. FP64: only L0 o_proj uses newly solved FP64 factors,
directly converted to BF16. ZERO: only L0 o_proj correction omitted, Wq retained.
No shrinkage, new calibration, KL, parameter training, or other rank evaluation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import signal
import sys
import uuid

sys.dont_write_bytecode = True
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
import full_g_rank_audit_v1 as core
import full_g_target_probe_v1 as previous

VERSION = "full_g_precision_r8_v1"
PREVIOUS_SHA = "eb2db6b5d544132fa16fa9f463edf3ee7039eefea261212e13b7f4b4983bf827"
TARGET = previous.TARGET
ARMS = ("OLD", "FP64", "ZERO")
RANK, WINDOWS, LENGTH = 8, 138, 2048


def tensor_record(tensor):
    value = tensor.detach().cpu().contiguous()
    raw = memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B")
    return {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": hashlib.sha256(raw).hexdigest()}


def file_record(path):
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": core.sha(path)}


def atomic_tensors(path, tensors):
    temporary = path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp")
    save_file({k:v.detach().cpu().contiguous() for k,v in tensors.items()}, str(temporary))
    with temporary.open("r+b") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    core.sync_dir(path.parent)
    return file_record(path)


def check_factors(values, d_in, d_out):
    expected = {"A_fp64": ((d_in,64),torch.float64), "B_fp64": ((64,d_out),torch.float64),
                "A_bf16_r8": ((d_in,8),torch.bfloat16), "B_bf16_r8": ((8,d_out),torch.bfloat16)}
    if set(values) != set(expected):
        raise RuntimeError("Unexpected generated factor keys")
    for key,(shape,dtype) in expected.items():
        if tuple(values[key].shape) != shape or values[key].dtype != dtype or not torch.isfinite(values[key]).all():
            raise RuntimeError("Generated factor dtype/shape/finiteness mismatch")
    if (tensor_record(values["A_bf16_r8"]) != tensor_record(values["A_fp64"][:,:8].bfloat16())
            or tensor_record(values["B_bf16_r8"]) != tensor_record(values["B_fp64"][:8].bfloat16())):
        raise RuntimeError("Deployment factors are not direct FP64-to-BF16 r8 slices")


def solve_fp64(error, a, g, solve_rank=64):
    # Match the preceding diagnostic: full SVD and rank64 solves, then r8 slices.
    e, sa, sg = error.double(), a.double(), g.double()
    u,s,vh = torch.linalg.svd(sa @ e @ sg, full_matrices=True)
    u, vh = u[:,:solve_rank].clone(), vh[:solve_rank].clone()
    target = s[:solve_rank,None]*vh
    left = torch.linalg.solve(sa,u)
    right = torch.linalg.solve(sg.T,target.T).T
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("Nonfinite FP64 solve")
    residual_a, residual_g = core.rel(sa @ left,u), core.rel(right @ sg,target)
    if max(residual_a,residual_g) > 1e-9:
        raise RuntimeError("FP64 inverse residual exceeds 1e-9; no fallback or damping allowed")
    return left, right, s, {"a_inverse_residual_fp64":residual_a,"g_inverse_residual_fp64":residual_g}


def prepare_factors(binding, inputs, output, identity, device):
    folder = output/"factors"
    folder.mkdir(exist_ok=True)
    path = folder/"l0_o_fp64.safetensors"
    metadata = folder/"l0_o_fp64.json"
    d_out,d_in = binding["shape"]
    if metadata.exists():
        record = core.read_json(metadata)
        if (record.get("experiment_identity") != identity or record.get("source_inputs") != binding
                or Path(record["file"]["path"]).resolve() != path.resolve()):
            raise RuntimeError("Generated factor binding mismatch")
        values = inputs.tensors(record["file"])
        check_factors(values,d_in,d_out)
        core.log("Frozen FP64 factors already complete and verified")
        return record
    with torch.no_grad(), core.heartbeat("PREPARE: rebuild unchanged G root and solve ONLY L0 o_proj in FP64"):
        a = inputs.tensors(binding["root"])["full"].to(device)
        raw = inputs.tensors(binding["gram"])
        if core.rel(raw["gram"].diagonal(),raw["diagonal"]) > 1e-10:
            raise RuntimeError("Raw G diagonal inconsistency")
        g,g_info = core.full_root(raw["gram"].to(device),256*2047,1e-6)
        del raw
        with safe_open(str(inputs.verify(binding["model"])),framework="pt",device="cpu") as stream:
            w = stream.get_tensor(TARGET+".weight")
        q = inputs.tensors(binding["quant"])["weight_q"]
        if w.dtype != torch.bfloat16 or q.dtype != torch.bfloat16 or w.shape != q.shape:
            raise RuntimeError("Frozen W/Wq mismatch")
        e = (w.float()-q.float()).T.to(device)
        if a.dtype != torch.float32 or a.shape != (d_in,d_in) or not torch.isfinite(a).all():
            raise RuntimeError("Expected unchanged stored FP32 Full-A root")
        left,right,s,inverse = solve_fp64(e,a,g)
        metrics = previous.metrics(e,a,g,left,right,RANK)
        tail = core.norm2(s[RANK:])
        excess = (metrics["sse_after"]-tail)/metrics["sse_before"]
        if not math.isfinite(excess) or abs(excess) > 1e-9:
            raise RuntimeError("FP64 r8 residual disagrees with SVD tail; stop before deployment")
        values = {"A_fp64":left.cpu(),"B_fp64":right.cpu()}
        values.update(A_bf16_r8=values["A_fp64"][:,:RANK].bfloat16(),
                      B_bf16_r8=values["B_fp64"][:RANK].bfloat16())
        check_factors(values,d_in,d_out)
        old = inputs.tensors(binding["corrections"]["full_gf"]["file"])
        old_metrics = previous.metrics(e,a,g,old["A"].to(device),old["B"].to(device),RANK)
        record = {"experiment_identity":identity,"module":TARGET,"rank":RANK,
            "file":atomic_tensors(path,values),"source_inputs":binding,
            "stored_a_root":tensor_record(a),"rebuilt_g_root":tensor_record(g),
            "g_diagnostics":g_info,"inverse":inverse,"saved_old_metrics":old_metrics,
            "new_fp64_metrics":metrics,"svd_tail_sse_fp64":tail,"relative_tail_difference":excess,
            "deployment":{k:tensor_record(values[k]) for k in ("A_bf16_r8","B_bf16_r8")},
            "note":"No new A/G statistics or regularization. Stored FP32 roots promoted to FP64; factors cast directly to BF16."}
        core.atomic_json(metadata,record)
    del a,g,w,q,e,left,right,s,values,old
    gc.collect()
    torch.cuda.empty_cache()
    core.log("COMMITTED generated factors; no source factor was replaced")
    return record


def nll_with_tokens(logits, ids, mask, chunk_tokens):
    """Exact legacy CE/reduction order plus saved per-token losses, no extra forward."""
    labels = ids[:,1:].to(logits.device)
    valid = mask[:,1:].to(logits.device).bool()
    if not valid.all():
        raise RuntimeError("This frozen WikiText2 protocol has no padding")
    losses = torch.empty(labels.shape,dtype=logits.dtype,device=logits.device)
    for row in range(labels.shape[0]):
        for start in range(0,labels.shape[1],chunk_tokens):
            end = min(start+chunk_tokens,labels.shape[1])
            losses[row,start:end] = F.cross_entropy(logits[row,start:end,:],labels[row,start:end],reduction="none")
    metrics = [(float(losses[row][valid[row]].sum().item()),int(valid[row].sum().item())) for row in range(labels.shape[0])]
    return metrics,losses.detach().cpu().contiguous()


def assert_deployment(old, current, arm):
    if current["parameter_bits"] != old["parameter_bits"] or current["buffer_bits"] != old["buffer_bits"]:
        raise RuntimeError("Model parameters/buffers differ between arms")
    if current["device_map"] != old["device_map"]:
        raise RuntimeError("Device placement differs between arms")
    if set(current["correction_bits"]) != set(old["correction_bits"]):
        raise RuntimeError("Correction module coverage differs")
    changed = [name for name in old["correction_bits"] if old["correction_bits"][name] != current["correction_bits"][name]]
    if changed != ([] if arm == "OLD" else [TARGET]):
        raise RuntimeError(f"Unexpected changed corrections: {changed}")
    if arm == "ZERO" and current["correction_bits"][TARGET] != {"mode":"zero_no_hook"}:
        raise RuntimeError("ZERO must omit only target correction")


def install(model, base, generated, arm, inputs):
    handles, bits = [], {}
    modules = dict(model.named_modules())
    try:
        with torch.no_grad():
            for name,files in base.items():
                module = modules[name]
                weight_q = inputs.tensors(files["quant"])["weight_q"]
                if weight_q.dtype != torch.bfloat16 or module.weight.dtype != torch.bfloat16 or weight_q.shape != module.weight.shape:
                    raise RuntimeError("Frozen quantized weight mismatch")
                module.weight.copy_(weight_q.to(module.weight.device))
                if tensor_record(module.weight) != tensor_record(weight_q):
                    raise RuntimeError("Installed Wq is not bit-identical to source")
                del weight_q
                if name == TARGET and arm == "ZERO":
                    bits[name] = {"mode":"zero_no_hook"}
                    continue
                if name == TARGET and arm == "FP64":
                    left,right = generated["A_bf16_r8"],generated["B_bf16_r8"]
                else:
                    values = inputs.tensors(files["correction"])
                    if (values["A"].dtype != torch.float32 or values["B"].dtype != torch.float32
                            or values["A"].shape != (module.in_features,64) or values["B"].shape != (64,module.out_features)):
                        raise RuntimeError("Frozen correction shape/dtype mismatch")
                    left,right = values["A"][:,:RANK].bfloat16(),values["B"][:RANK].bfloat16()
                left,right = left.to(module.weight.device),right.to(module.weight.device)
                if not torch.isfinite(left).all() or not torch.isfinite(right).all():
                    raise RuntimeError("Nonfinite BF16 deployment factors")
                bits[name] = {"A":tensor_record(left),"B":tensor_record(right)}
                handles.append(module.register_forward_hook(previous.correction_hook(left,right)))
        deployment = {"parameter_bits":{name:tensor_record(t) for name,t in model.named_parameters()},
                      "buffer_bits":{name:tensor_record(t) for name,t in model.named_buffers()},
                      "correction_bits":bits,"device_map":model.hf_device_map}
        return handles,deployment
    except BaseException:
        for h in handles:
            h.remove()
        raise


def read_state(folder, arm, identity, inputs):
    path = folder/(arm+".json")
    if not path.exists():
        return {"experiment_identity":identity,"arm":arm,"records":[],"batches":[],"complete":False}
    state = core.read_json(path)
    if state.get("experiment_identity") != identity or state.get("arm") != arm:
        raise RuntimeError("Evaluation checkpoint identity mismatch")
    records = previous.valid_records(state["records"],WINDOWS)
    deployment = core.read_json(folder/("deployment_"+arm+".json"))
    if (deployment.get("experiment_identity") != identity or deployment.get("arm") != arm
            or core.fingerprint(deployment["deployment"]) != state.get("deployment_sha256")):
        raise RuntimeError("Checkpoint does not match its deployment tensor hashes")
    if state["complete"] != (len(records) == WINDOWS):
        raise RuntimeError("Completion marker mismatch")
    cursor = 0
    for batch in state["batches"]:
        start,end = batch["start"],batch["end"]
        if start != cursor or end != min(start+8,WINDOWS):
            raise RuntimeError("Evaluation batch boundary mismatch")
        expected_path = folder/"tokens"/arm/f"batch_{start:04d}.safetensors"
        if Path(batch["file"]["path"]).resolve() != expected_path.resolve():
            raise RuntimeError("Token checkpoint path mismatch")
        tokens = inputs.tensors(batch["file"])["nll"]
        if tokens.shape != (end-start,LENGTH-1) or not torch.isfinite(tokens).all() or (tokens < 0).any():
            raise RuntimeError("Token NLL checkpoint mismatch")
        if len(records[start:end]) != end-start:
            raise RuntimeError("Token checkpoint has more windows than committed records")
        for offset,row in enumerate(records[start:end]):
            # Native GPU and CPU reduction kernels may round differently. Stored
            # headline NLL remains authoritative; FP64 sum is independently tied.
            if float(tokens[offset].double().sum()) != row["token_nll_sum_fp64"]:
                raise RuntimeError("Token losses do not match committed FP64 diagnostic sum")
        cursor = end
    if cursor != len(records):
        raise RuntimeError("Token batch coverage does not match window records")
    return state


def evaluate_arm(legacy, manifest, base, generated, arm, inputs, output, identity, stop):
    folder = output/"evaluation"
    folder.mkdir(exist_ok=True)
    state = read_state(folder,arm,identity,inputs)
    if state["complete"]:
        core.log(arm+" already complete")
        return state
    stop.check()
    windows = inputs.tensors(manifest["payload"]["data"]["wikitext2"])
    if tuple(windows["input_ids"].shape) != (WINDOWS,LENGTH) or windows["attention_mask"].shape != windows["input_ids"].shape or not (windows["attention_mask"] == 1).all():
        raise RuntimeError("Require original 138 x 2048 all-valid evaluation windows")
    config = manifest["payload"]["config"]
    loading = dict(manifest["payload"]["source_config"])
    loading["max_memory"] = config["eval_max_memory"]
    model,handles = None,[]
    try:
        torch.manual_seed(1234)
        with core.heartbeat(arm+": load BF16 model, install factors and hash actual deployment tensors"):
            model = legacy.load_model(loading,"bfloat16","balanced")
            if not {str(x) for x in model.hf_device_map.values()} <= {"0","1","cuda:0","cuda:1"}:
                raise RuntimeError("Model must reside on the two GPUs")
            handles,deployment = install(model,base,generated,arm,inputs)
        deploy_path = folder/("deployment_"+arm+".json")
        old_path = folder/"deployment_OLD.json"
        old = core.read_json(old_path)["deployment"] if old_path.exists() else deployment
        if arm != "OLD" and not old_path.exists():
            raise RuntimeError("OLD deployment record required before interventions")
        assert_deployment(old,deployment,arm)
        deploy_record = {"experiment_identity":identity,"arm":arm,"deployment":deployment}
        if deploy_path.exists() and core.read_json(deploy_path) != deploy_record:
            raise RuntimeError("Deployment bits changed on resume")
        if not deploy_path.exists():
            core.atomic_json(deploy_path,deploy_record)
        deploy_hash = core.fingerprint(deployment)
        if state["records"] and state.get("deployment_sha256") != deploy_hash:
            raise RuntimeError("Evaluation records bound to a different deployment")
        state["deployment_sha256"] = deploy_hash
        device = legacy._input_device(model)
        token_dir = folder/"tokens"/arm
        token_dir.mkdir(parents=True,exist_ok=True)
        nll_kernel_checked = False
        with torch.inference_mode():
            for start in range(len(state["records"]),WINDOWS,8):
                stop.check()
                end = min(start+8,WINDOWS)
                ids = windows["input_ids"][start:end].to(device)
                mask = windows["attention_mask"][start:end].to(device)
                with core.heartbeat(f"{arm} rank=8 windows={start+1}-{end}/{WINDOWS}"):
                    logits = model(input_ids=ids,attention_mask=mask,use_cache=False).logits
                    values,token_nll = nll_with_tokens(logits,ids,mask,256)
                    # Verify instrumentation against the frozen evaluator once
                    # per (re)loaded model before any batch is committed.
                    if not nll_kernel_checked:
                        original = legacy._chunked_window_nll(logits,ids,mask,256)
                        if original != values:
                            raise RuntimeError("Token-NLL instrumentation changed the frozen headline NLL")
                        nll_kernel_checked = True
                del logits,ids,mask
                for offset,(nll,count) in enumerate(values):
                    state["records"].append({"window":start+offset,"tokens":count,"nll_sum":nll,
                                             "token_nll_sum_fp64":float(token_nll[offset].double().sum())})
                previous.valid_records(state["records"],WINDOWS)
                record = atomic_tensors(token_dir/f"batch_{start:04d}.safetensors",{"nll":token_nll})
                state["batches"].append({"start":start,"end":end,"file":record})
                state["complete"] = end == WINDOWS
                core.atomic_json(folder/(arm+".json"),state)
                stop.done += 1
                core.log(f"COMMITTED {arm} window={end}/{WINDOWS}")
    finally:
        for h in handles:
            h.remove()
        handles.clear()
        model = None
        gc.collect()
        torch.cuda.empty_cache()
    return state


def summarize(output, identity, inputs):
    folder = output/"evaluation"
    folder.mkdir(exist_ok=True)
    states = {arm:read_state(folder,arm,identity,inputs) for arm in ARMS}
    complete = {arm:state for arm,state in states.items() if state["complete"]}
    summary,window_rows,pair_rows = [],[],[]
    for arm,state in complete.items():
        records = state["records"]
        nll = sum(r["nll_sum"] for r in records)
        summary.append({"arm":arm,"rank":8,"windows":138,"prediction_tokens":138*2047,
                        "nll_sum":nll,"mean_nll":nll/(138*2047),"ppl":math.exp(nll/(138*2047)),
                        "token_nll_sum_fp64":sum(r["token_nll_sum_fp64"] for r in records)})
        window_rows.extend({"arm":arm,**row} for row in records)
    comparisons = []
    for first,second in (("OLD","FP64"),("OLD","ZERO"),("ZERO","FP64")):
        if first not in complete or second not in complete:
            continue
        aa,bb = complete[first],complete[second]
        improved=tied=worsened=0
        absolute=0.
        token_delta_sum=0.
        for ab,bbatch in zip(aa["batches"],bb["batches"]):
            x = inputs.tensors(ab["file"])["nll"].double()
            y = inputs.tensors(bbatch["file"])["nll"].double()
            delta=y-x
            improved += int((delta < 0).sum()); worsened += int((delta > 0).sum()); tied += int((delta == 0).sum())
            absolute += float(delta.abs().sum()); token_delta_sum += float(delta.sum())
        pair_name = second+"_minus_"+first
        for a,b in zip(aa["records"],bb["records"]):
            pair_rows.append({"comparison":pair_name,"window":a["window"],"tokens":a["tokens"],
                "first_nll":a["nll_sum"],"second_nll":b["nll_sum"],"delta_nll":b["nll_sum"]-a["nll_sum"],
                "delta_token_sum_fp64":b["token_nll_sum_fp64"]-a["token_nll_sum_fp64"]})
        delta_nll = sum(b["nll_sum"]-a["nll_sum"] for a,b in zip(aa["records"],bb["records"]))
        comparisons.append({"comparison":pair_name,"delta_nll":delta_nll,"delta_mean_nll":delta_nll/(138*2047),
            "delta_ppl":previous.ppl(bb["records"])-previous.ppl(aa["records"]),
            "improved_windows":sum(b["nll_sum"]<a["nll_sum"] for a,b in zip(aa["records"],bb["records"])),
            "improved_tokens":improved,"worsened_tokens":worsened,"tied_tokens":tied,
            "token_mean_absolute_delta":absolute/(138*2047),"token_delta_mean_fp64":token_delta_sum/(138*2047)})
    previous.write_csv(folder/"ppl_summary.csv",summary)
    previous.write_csv(folder/"per_window.csv",window_rows)
    previous.write_csv(folder/"paired_windows.csv",pair_rows)
    previous.write_csv(folder/"comparisons.csv",comparisons)
    control_path = folder/"control_check.json"
    control = core.read_json(control_path) if control_path.exists() else {}
    passed = control.get("experiment_identity") == identity and control.get("status") == "PASS"
    status = {"experiment_identity":identity,"status":"COMPLETE" if len(complete)==3 and passed else "INCOMPLETE",
        "completed_arms":list(complete),"control_passed":passed,
        "committed_windows":{arm:len(s["records"]) for arm,s in states.items()},
        "note":"Post-hoc r8 diagnostic, not proof of all-model correctness. No IID-token significance claims."}
    core.atomic_json(folder/"status.json",status)
    return status


def verify_deployment_reports(output, identity):
    folder=output/"evaluation"
    old_path=folder/"deployment_OLD.json"
    if not old_path.exists():
        return
    old_record=core.read_json(old_path)
    if old_record.get("experiment_identity") != identity or old_record.get("arm") != "OLD":
        raise RuntimeError("OLD deployment identity mismatch")
    for arm in ARMS:
        path=folder/("deployment_"+arm+".json")
        if path.exists():
            record=core.read_json(path)
            if record.get("experiment_identity") != identity or record.get("arm") != arm:
                raise RuntimeError("Deployment report identity mismatch")
            assert_deployment(old_record["deployment"],record["deployment"],arm)


def run_arms(legacy,manifest,base,generated,reference,inputs,output,identity,stop):
    old=evaluate_arm(legacy,manifest,base,generated,"OLD",inputs,output,identity,stop)
    check=previous.control_check(old["records"],reference["records"],manifest["payload"]["config"]["control_ppl_tolerance"])
    core.atomic_json(output/"evaluation/control_check.json",{"experiment_identity":identity,**check})
    summarize(output,identity,inputs)
    if check["status"] != "PASS":
        raise RuntimeError("OLD control PPL replay FAILED; FP64 and ZERO arms are blocked")
    for arm in ("FP64","ZERO"):
        evaluate_arm(legacy,manifest,base,generated,arm,inputs,output,identity,stop)
        summarize(output,identity,inputs)


def setup(args):
    if core.sha(core.__file__) != previous.HELPER_SHA or core.sha(previous.__file__) != PREVIOUS_SHA:
        raise RuntimeError("Required helper scripts have changed")
    run,repo,output = args.run_dir.resolve(),args.repo_dir.resolve(),args.output_dir.resolve()
    manifest=core.read_json(run/"manifest.json")
    payload=manifest["payload"]; config=payload["config"]
    if manifest["sha256"] != core.fingerprint(payload) or core.read_json(run/"config.json") != config:
        raise RuntimeError("Source manifest/config mismatch")
    if (config["experiment_variant"] != "mxint3_full_g_v1" or config["ranks"] != [8,16,32,64]
            or config["g_relative_floor"] != 1e-6 or config["eval_batch_size"] != 8
            or config["eval_ce_chunk_tokens"] != 256 or Path(config["run_dir"]).resolve() != run):
        raise RuntimeError("Require frozen original Llama MXINT3 Full-G protocol")
    protected=[run.parent,repo,Path(__file__).resolve().parent]
    for dictionary in (config,payload["source_config"]):
        protected += [dictionary[k] for k in ("run_dir","source_run_dir","model_path","qera_source_dir") if k in dictionary]
    for rec in payload["code"].values():
        parts=Path(rec["path"]).parts
        if "experiments" in parts:
            protected.append(Path(*parts[:parts.index("experiments")]))
    core.disjoint(output,protected)
    if args.output_dir.is_symlink() or (output.exists() and any(p.is_symlink() for p in output.rglob("*"))):
        raise RuntimeError("No symlinks allowed in new output")
    inputs=core.Inputs()
    for record in payload["code"].values(): inputs.verify(record)
    states=core.committed_states(run,manifest)
    tasks={layer["name"]:core.metadata(run,manifest,group,layer,states) for group in payload["groups"] for layer in group["layers"]}
    if len(tasks)!=224 or TARGET not in tasks: raise RuntimeError("Expected all 224 modules")
    baseline=payload["full_g_protocol"]["baseline_inputs"]
    base={name:{"quant":b["quant"],"correction":b["corrections"]["full_gf"]["file"]} for name,b in tasks.items()}
    for name,b in tasks.items():
        rec=b["corrections"]["full_gf"]
        if rec.get("gi_reference")!=baseline["full_gi"][name]["correction"] or rec.get("gd_reference")!=baseline["full_gd"][name]["correction"]:
            raise RuntimeError("Frozen GF baseline binding mismatch")
        for key in ("identity_product_drift","diagonal_product_drift"):
            drift=rec[key]
            if set(drift)!={"8","16","32","64"} or any(not math.isfinite(v) or not 0<=v<=config["identity_product_tolerance"] for v in drift.values()):
                raise RuntimeError("Frozen GF regression gate not passed")
    reference_path=run/"evaluation/configurations/FULL_GF_R8.json"
    reference=core.read_json(reference_path)
    expected=core.fingerprint({"manifest":manifest["sha256"],"name":"FULL_GF_R8","artifacts":base})
    if reference.get("protocol_sha256")!=expected or reference.get("complete") is not True or len(reference["records"])!=138:
        raise RuntimeError("Original r8 GF evaluation is missing/incomplete or has different inputs")
    previous.valid_records(reference["records"],138)
    if not torch.cuda.is_available() or str(torch.__version__)!="2.3.0+cu121":
        raise RuntimeError("Use original qera-original-a torch 2.3.0+cu121 environment")
    needed=1 if args.command=="prepare" else 2
    if torch.cuda.device_count()<needed: raise RuntimeError(f"Need {needed} visible idle 4090 GPU(s)")
    for i in range(needed):
        if "4090" not in torch.cuda.get_device_name(i) or torch.cuda.mem_get_info(i)[0]<18*2**30:
            raise RuntimeError("Required 4090 GPU(s) must be idle with at least 18 GiB free")
    torch.set_num_threads(14)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    experiment={"version":VERSION,"script_sha256":core.sha(__file__),"core_sha256":previous.HELPER_SHA,
        "previous_helper_sha256":PREVIOUS_SHA,"source_manifest":manifest["sha256"],"run_dir":str(run),"repo_dir":str(repo),
        "target":TARGET,"rank":8,"arms":list(ARMS),"base_artifacts":core.fingerprint(base),
        "target_binding":core.fingerprint(tasks[TARGET]),"reference":file_record(reference_path),
        "wikitext2":payload["data"]["wikitext2"],"torch":str(torch.__version__),"cuda":torch.version.cuda,
        "transformers":importlib.metadata.version("transformers"),"accelerate":importlib.metadata.version("accelerate"),
        "tf32":False,"eval_batch_size":8,"eval_ce_chunk_tokens":256,"seed":1234,"cpu_threads":14,
        "precision_policy":"stored FP32 roots -> FP64 weighted product/full SVD/rank64 solves -> first8 direct BF16 cast",
        "nll_policy":"original native reduction for PPL; additional saved per-token losses and FP64 diagnostic sums",
        "control_ppl_tolerance":config["control_ppl_tolerance"],"new_regularization":False,"teacher_kl":False}
    return manifest,base,tasks[TARGET],reference,inputs,output,experiment


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","run"))
    parser.add_argument("--run-dir",required=True,type=Path)
    parser.add_argument("--repo-dir",required=True,type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--max-hours",type=float,default=10)
    parser.add_argument("--max-new-batches",type=int)
    args=parser.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours<=0 or (args.max_new_batches is not None and args.max_new_batches<1):
        parser.error("Budgets must be positive finite")
    if args.command=="prepare" and args.max_new_batches is not None: parser.error("Batch budget is only for run")
    stop=previous.Budget(args.max_hours,args.max_new_batches)
    signal.signal(signal.SIGTERM,stop.signal); signal.signal(signal.SIGINT,stop.signal)
    manifest,base,binding,reference,inputs,output,experiment=setup(args)
    identity=core.fingerprint(experiment)
    if output.exists() and not (output/"experiment.json").exists() and any(output.iterdir()):
        raise RuntimeError("Refusing unowned nonempty output directory")
    output.mkdir(parents=True,exist_ok=True)
    with core.audit_lock(output):
        path=output/"experiment.json"
        if path.exists() and core.read_json(path)!=experiment:
            raise RuntimeError("Experiment inputs/settings changed; use a separate new output directory")
        if not path.exists(): core.atomic_json(path,experiment)
        try:
            stop.check()
            factor_record=prepare_factors(binding,inputs,output,identity,"cuda:0")
            if args.command=="prepare":
                core.log("PREPARE COMPLETE; no model evaluation performed")
                return 0
            stop.check()
            sys.path.insert(0,str(args.repo_dir.resolve()/"experiments"))
            from qera_diag_g_isolation import pipeline as legacy
            from qera_original_a_isolation import pipeline as original
            verified={Path(rec["path"]).resolve() for rec in manifest["payload"]["code"].values()}
            if any(Path(module.__file__).resolve() not in verified for module in (legacy,original)):
                raise RuntimeError("Imported evaluator helpers are not manifest-bound")
            for record in manifest["payload"]["model_files"].values(): inputs.verify(record)
            for files in base.values():
                for record in files.values(): inputs.verify(record)
            generated=inputs.tensors(factor_record["file"])
            verify_deployment_reports(output,identity)
            run_arms(legacy,manifest,base,generated,reference,inputs,output,identity,stop)
            verify_deployment_reports(output,identity)
            core.log(str(summarize(output,identity,inputs)))
        except previous.Paused:
            summarize(output,identity,inputs)
            core.log("PAUSED: rerun SAME command; remove pilot batch limit to finish all 3 arms")
            return 75
    return 0


if __name__=="__main__":
    raise SystemExit(main())
