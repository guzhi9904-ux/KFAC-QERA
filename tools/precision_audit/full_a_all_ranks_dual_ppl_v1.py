#!/usr/bin/env python3
"""Read-only A5 factors -> checked BF16 prefixes -> two isolated PPL protocols.

No collection, SVD, inverse solve, A-root reconstruction, or quantization.
Token checkpoints: 8 windows. Word checkpoints: one FULL harness configuration.
"""
from __future__ import annotations
import argparse
import copy
import gc
import importlib.util
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
import full_a_all_precision_r8_v1 as parent

single, core, prev = parent.single, parent.core, parent.previous
VERSION = "full_a_all_ranks_dual_ppl_v1"
PARENT_SHA = "23ec871264e6d6312d3bd4c7fb08ddc89dbc745cd2b1f0ae9fa1289307833f4f"
WORD_HELPER_SHA = "b83f214b15c75b39784dc86eade3d4b657d973960c041da985812c8e3589dbe6"
RANKS = (8, 16, 32, 64)
METHODS = parent.METHODS
CONFIGS = [("teacher", None), ("wq", None)] + [(m, r) for m in METHODS for r in RANKS]


def label(method, rank):
    return "BF16" if method == "teacher" else "W3_MXINT" if method == "wq" else f"{method.upper()}_R{rank}"


def freeze_json(path, value):
    if path.exists():
        if core.read_json(path) != value:
            raise RuntimeError("Immutable new-run record changed: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        core.atomic_json(path, value)


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def prefix(values, rank):
    if rank not in RANKS:
        raise ValueError("Only ranks 8/16/32/64")
    a, b = values["A_fp64"], values["B_fp64"]
    single.check_factors(values, a.shape[0], b.shape[1])
    left, right = a[:, :rank].bfloat16().contiguous(), b[:rank].bfloat16().contiguous()
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise RuntimeError("BF16 prefix overflow")
    return left, right


def check_rank_metrics(error, a, g, left, right, parent_tail8):
    """Certify saved prefixes without running another SVD or changing factors.

    Anchor tail energies to the source r8 SVD certificate, then subtract the
    saved orthogonal components 9..r. This is not an independent new SVD proof.
    """
    e, sa, sg, l, b = [x.double() for x in (error, a, g, left, right)]
    y = sa @ e @ sg
    u, d = sa @ l, b @ sg
    before = core.norm2(y)
    scale = max(before, 1e-30)
    energies = d.square().sum(dim=1)
    orth = core.rel(u.T @ u, torch.eye(64, dtype=torch.float64, device=u.device))
    row_orth = core.rel(d @ d.T, torch.diag(energies))
    stationarity = core.rel(u.T @ y, d)
    if max(orth, row_orth, stationarity) > 1e-8:
        raise RuntimeError(f"Saved weighted singular factors fail checks: {orth}, {row_orth}, {stationarity}")
    if bool((energies[1:] > energies[:-1] + scale * 1e-10).any()):
        raise RuntimeError("Saved singular components are not in descending order")
    result = []
    last = before
    for rank in RANKS:
        correction = l[:, :rank] @ b[:rank]
        after = core.norm2(sa @ (e - correction) @ sg)
        expected = parent_tail8 - float(energies[8:rank].sum())
        excess = (after - expected) / scale
        if (not math.isfinite(after) or expected < -scale*1e-8 or abs(excess) > 1e-8
                or after > last + scale*1e-8):
            raise RuntimeError(f"Saved rank {rank} objective / anchored tail mismatch: {excess}")
        lr, br = l[:, :rank].bfloat16().double(), b[:rank].bfloat16().double()
        if not torch.isfinite(lr).all() or not torch.isfinite(br).all():
            raise RuntimeError("Nonfinite BF16 deployment factors")
        rounded = core.norm2(sa @ (e - lr @ br) @ sg)
        if not math.isfinite(rounded):
            raise RuntimeError("Nonfinite BF16-rounded objective")
        result.append({"rank": rank, "sse_before": before, "sse_after_fp64": after,
                       "anchored_tail_sse": expected, "relative_tail_difference": excess,
                       "left_orthogonality": orth, "right_orthogonality": row_orth,
                       "projected_residual": stationarity, "bf16_rounded_sse_proxy": rounded,
                       "bf16_objective_increased_flag": rounded > before*(1+1e-5)+1e-20,
                       "correction_over_error_norm": math.sqrt(core.norm2(correction)/max(core.norm2(e), 1e-30))})
        last = after
    return result


def checked_path(ctx, method, name):
    return ctx.output/"checked_factors"/method/(core.safe_name(name)+".json")


def read_checked(ctx, method, name):
    path = checked_path(ctx, method, name)
    if not path.exists():
        return None
    r = core.read_json(path)
    source = ctx.factors[method][name]
    if (r.get("experiment_identity") != ctx.identity or r.get("source_factor") != source["file"]
            or r.get("module") != name or r.get("method") != method or r.get("status") != "PASS"
            or Path(r["file"]["path"]).resolve() != path.with_suffix(".safetensors").resolve()
            or [x["rank"] for x in r["metrics"]] != list(RANKS)):
        raise RuntimeError("Checked factor identity mismatch: " + str(path))
    v = ctx.inputs.tensors(r["file"])
    d_out, d_in = ctx.tasks[name]["shape"]
    if set(v) != {"A", "B"} or v["A"].shape != (d_in, 64) or v["B"].shape != (64, d_out):
        raise RuntimeError("Checked factor shape mismatch")
    for key in v:
        if v[key].dtype != torch.bfloat16 or not torch.isfinite(v[key]).all() or single.tensor_record(v[key]) != r["bits"][key]:
            raise RuntimeError("Checked BF16 factor mismatch")
    return r


def check_one(ctx, name, method):
    existing = read_checked(ctx, method, name)
    if existing is not None:
        return existing
    binding, source = ctx.tasks[name], ctx.factors[method][name]
    path = checked_path(ctx, method, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with torch.no_grad(), core.heartbeat(f"CHECK ranks={RANKS} {method} {name} (no SVD/solve)"):
        values = ctx.inputs.tensors(source["file"])
        a = ctx.inputs.tensors(binding["root"])["full"].to(ctx.device, copy=True)
        g, _ = parent.make_g_root(method, binding, ctx.dg[name], ctx.inputs, ctx.device)
        if source["root_bits"] != {"a": single.tensor_record(a), "g": single.tensor_record(g)}:
            raise RuntimeError("Root bits differ from source A5. Do not silently change root/protocol.")
        with safe_open(str(ctx.inputs.verify(binding["model"])), framework="pt", device="cpu") as f:
            w = f.get_tensor(name+".weight").clone()
        q = ctx.inputs.tensors(binding["quant"])["weight_q"]
        error = (w.float()-q.float()).T.to(ctx.device)
        metrics = check_rank_metrics(error, a, g, values["A_fp64"].to(ctx.device),
                                     values["B_fp64"].to(ctx.device), source["svd_tail_sse_fp64"])
        left, right = prefix(values, 64)
        record = {"experiment_identity": ctx.identity, "status": "PASS", "method": method, "module": name,
                  "source_factor": source["file"], "source_certificate": ctx.factor_metadata[method][name],
                  "metrics": metrics, "root_bits": source["root_bits"],
                  "bits": {"A": single.tensor_record(left), "B": single.tensor_record(right)},
                  "file": single.atomic_tensors(path.with_suffix(".safetensors"), {"A": left, "B": right}),
                  "elapsed_seconds": time.monotonic()-started,
                  "note": "BF16 rounded objective is a proxy, flags retained; not actual activation execution."}
        core.atomic_json(path, record)
    del values, a, g, w, q, error, left, right
    clear_gpu()
    core.log(f"COMMITTED CHECK {method} {name} elapsed={record['elapsed_seconds']:.1f}s")
    return record


def check_all(ctx, stop, names=None):
    names = list(ctx.tasks) if names is None else names
    for i, name in enumerate(names, 1):
        for method in METHODS:
            if read_checked(ctx, method, name) is not None:
                continue
            stop.check()
            core.log(f"check module={i}/{len(names)} method={method}")
            check_one(ctx, name, method)
            stop.done += 1
    write_check_summary(ctx)


def write_check_summary(ctx):
    rows = []
    for m in METHODS:
        for name in ctx.tasks:
            rec = read_checked(ctx, m, name)
            if rec:
                rows.extend({"method": m, "module": name, **x} for x in rec["metrics"])
    prev.write_csv(ctx.output/"rank_checks.csv", rows)
    status = {"experiment_identity": ctx.identity, "checked_module_methods": len(rows)//4,
              "expected": len(ctx.tasks)*3, "status": "COMPLETE" if len(rows)==len(ctx.tasks)*12 else "INCOMPLETE",
              "bf16_objective_flags": sum(r["bf16_objective_increased_flag"] for r in rows)}
    core.atomic_json(ctx.output/"rank_check_status.json", status)
    return status


def artifact_routes(ctx, method, rank, old=False):
    routes = {}
    if method == "teacher":
        return routes
    for name, b in ctx.tasks.items():
        route = {"quant": b["quant"]}
        if method != "wq":
            if old:
                route["correction"] = b["baseline_factors"][method]
            else:
                checked = read_checked(ctx, method, name)
                if checked is None:
                    raise RuntimeError("Run check before evaluating: " + method + " " + name)
                route["correction"] = checked["file"]
        routes[name] = route
    return routes


def install(model, routes, rank, inputs, old=False):
    modules, handles, bits = dict(model.named_modules()), [], {}
    try:
        with torch.no_grad():
            for name, files in routes.items():
                mod = modules[name]
                q = inputs.tensors(files["quant"])["weight_q"]
                if q.dtype != torch.bfloat16 or mod.weight.dtype != torch.bfloat16 or q.shape != mod.weight.shape:
                    raise RuntimeError("Wq shape/dtype mismatch")
                mod.weight.copy_(q.to(mod.weight.device))
                if single.tensor_record(q) != single.tensor_record(mod.weight):
                    raise RuntimeError("Wq installation changed bits")
                if "correction" not in files:
                    continue
                v = inputs.tensors(files["correction"])
                dtype = torch.float32 if old else torch.bfloat16
                if (set(v) != {"A", "B"} or v["A"].dtype != dtype or v["B"].dtype != dtype
                        or v["A"].shape != (mod.in_features,64) or v["B"].shape != (64,mod.out_features)):
                    raise RuntimeError("Deployment factor shape/dtype mismatch")
                a = v["A"][:, :rank].to(device=mod.weight.device, dtype=torch.bfloat16).contiguous()
                b = v["B"][:rank].to(device=mod.weight.device, dtype=torch.bfloat16).contiguous()
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise RuntimeError("Nonfinite deployment")
                bits[name] = {"A": single.tensor_record(a), "B": single.tensor_record(b)}
                handles.append(mod.register_forward_hook(prev.correction_hook(a, b)))
            deployment = {"parameter_bits": {n: single.tensor_record(t) for n,t in model.named_parameters()},
                          "buffer_bits": {n: single.tensor_record(t) for n,t in model.named_buffers()},
                          "device_map": model.hf_device_map, "correction_bits": bits}
        return handles, deployment
    except BaseException:
        for handle in handles:
            handle.remove()
        raise


def bind_deployment(ctx, folder, name, deployment, routes, method):
    record = {"experiment_identity": ctx.identity, "arm": name,
              "route_sha256": core.fingerprint(routes), "deployment": deployment}
    freeze_json(folder/("deployment_"+name+".json"), record)
    if method != "teacher":
        common_path = folder/"common_wq.json"
        common = {k: deployment[k] for k in ("parameter_bits", "buffer_bits", "device_map")}
        freeze_json(common_path, {"experiment_identity": ctx.identity, "deployment": common})
    if method in METHODS:
        # The two protocols may dispatch differently; correction bits must not.
        freeze_json(ctx.output/"deployment_factors"/(name+".json"),
                    {"experiment_identity": ctx.identity, "bits": deployment["correction_bits"]})
    return record


def token_evaluate(ctx, legacy, method, rank, stop):
    name = label(method, rank)
    folder = ctx.output/"token_ppl"; folder.mkdir(exist_ok=True)
    state = single.read_state(folder, name, ctx.identity, ctx.inputs)
    routes = artifact_routes(ctx, method, rank)
    route_sha = core.fingerprint(routes)
    if state["records"] and state.get("route_sha256") != route_sha:
        raise RuntimeError("Token checkpoint routes changed")
    if state["complete"]:
        return state
    stop.check()
    windows = ctx.inputs.tensors(ctx.manifest["payload"]["data"]["wikitext2"])
    if (windows["input_ids"].shape != (138,2048) or windows["attention_mask"].shape != (138,2048)
            or not bool((windows["attention_mask"] == 1).all())):
        raise RuntimeError("Token protocol requires frozen 138 x 2048 unpadded windows")
    loading = dict(ctx.manifest["payload"]["source_config"])
    loading["max_memory"] = ctx.manifest["payload"]["config"]["eval_max_memory"]
    model, handles = None, []
    try:
        torch.manual_seed(1234)
        with core.heartbeat("TOKEN load/install/hash " + name):
            model = legacy.load_model(loading, "bfloat16", "balanced")
            resident(model)
            handles, deployment = install(model, routes, rank, ctx.inputs)
            bind_deployment(ctx, folder, name, deployment, routes, method)
        state.update(route_sha256=route_sha, deployment_sha256=core.fingerprint(deployment))
        device = legacy._input_device(model)
        directory = folder/"tokens"/name; directory.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            for start in range(len(state["records"]), 138, 8):
                stop.check(); end = min(start+8, 138)
                with core.heartbeat(f"TOKEN {name} windows={start+1}-{end}/138"):
                    ids = windows["input_ids"][start:end].to(device)
                    mask = windows["attention_mask"][start:end].to(device)
                    logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                    values, tokens = single.nll_with_tokens(logits, ids, mask, 256)
                    if start == 0 and values != legacy._chunked_window_nll(logits, ids, mask, 256):
                        raise RuntimeError("Token NLL instrumentation changed reduction")
                del logits, ids, mask
                state["records"].extend({"window": start+i, "tokens": n, "nll_sum": loss,
                                          "token_nll_sum_fp64": float(tokens[i].double().sum())}
                                         for i,(loss,n) in enumerate(values))
                f = single.atomic_tensors(directory/f"batch_{start:04d}.safetensors", {"nll": tokens})
                state["batches"].append({"start": start, "end": end, "file": f})
                state["complete"] = end == 138
                core.atomic_json(folder/(name+".json"), state)
                stop.done += 1
                core.log(f"COMMITTED TOKEN {name} window={end}/138")
    finally:
        for h in handles:
            h.remove()
        handles.clear(); model = None; clear_gpu()
    return state


def resident(model):
    if not {str(v) for v in model.hf_device_map.values()} <= {"0", "1", "cuda:0", "cuda:1"}:
        raise RuntimeError("Require model fully resident on two GPUs, no CPU/disk offload")


def token_gate(ctx, method, current):
    source = ctx.source_states[method]
    check = prev.control_check(current["records"], source["records"], 1e-5)
    # A new r8 path must preserve full deployed bits, not just a similar PPL.
    old = ctx.source_deployments[method]["deployment"]
    new = core.read_json(ctx.output/"token_ppl"/("deployment_"+label(method,8)+".json"))["deployment"]
    if old != new:
        raise RuntimeError("r8 deployment is not identical to completed source A5")
    check["max_window_difference"] = max(abs(a["nll_sum"]-b["nll_sum"]) for a,b in zip(current["records"],source["records"]))
    if check["max_window_difference"] > 1e-3:
        check["status"] = "FAIL"
    freeze_json(ctx.output/"token_ppl"/(method+"_r8_control.json"), {"experiment_identity": ctx.identity, **check})
    if check["status"] != "PASS":
        raise RuntimeError("New token evaluation path failed source r8 replay")


def load_word_helper():
    path = Path(__file__).with_name("frozen_harness_word_ppl.py")
    if not path.is_file() or core.sha(path) != WORD_HELPER_SHA:
        raise RuntimeError("Bundled frozen harness helper missing/changed")
    spec = importlib.util.spec_from_file_location("frozen_harness_word_ppl", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prepare_word(ctx, args):
    """Read-only preflight before numerical work; no downloads or installations."""
    helper = load_word_helper()
    helper.verify_harness(args.harness_source, check_import=True)
    from qera_original_a_isolation.common import import_official_qera
    official = import_official_qera({"qera_source_dir": str(args.official_qera_root),
                                     "official_qera_commit": "bd7fc86a2e44d41f95b9b0421f27f5624dd37064"})
    from qera.utils import QERA_SRC_DIR
    from lm_eval.tasks import TaskManager, get_task_dict
    if Path(QERA_SRC_DIR).resolve() != (args.official_qera_root/"src").resolve():
        # Some QERA versions define QERA_SRC_DIR as src/qera. Match the actual wrapper's parent rule below.
        if Path(QERA_SRC_DIR).resolve() != (args.official_qera_root/"src/qera").resolve():
            raise RuntimeError("Wrong imported QERA source")
    manager = TaskManager(verbosity="INFO", include_path=str(Path(QERA_SRC_DIR).parent/"qera_harness_tasks"), include_defaults=True)
    task = get_task_dict(["wikitext"], manager)["wikitext"]
    data = helper.dataset_identity(task)
    ref_path = args.word_reference_dir/"protocol.json"
    if not ref_path.exists():
        raise RuntimeError(f"Need original 4096 word-PPL protocol.json: {ref_path}. Use --word-reference-dir; do not substitute token results.")
    ref = core.read_json(ref_path)
    fixed = ref["fixed_protocol"]
    for k,v in {"task":"wikitext", "context_length":4096, "max_position_embeddings":4096,
                "dtype":"bfloat16", "attn_implementation":"eager", "device_map":"auto-balanced",
                "lm_eval_batch_size":"auto", "num_fewshot":None}.items():
        if fixed.get(k) != v:
            raise RuntimeError("Old word protocol is not the 4096 reference: " + k)
    if ref["harness_commit"] != helper.HARNESS_COMMIT or ref["harness_hashes"] != helper.HARNESS_HASHES or ref["official_qera_commit"] != official["commit"]:
        raise RuntimeError("Old/new harness or QERA revision mismatch")
    if ref["dataset"] != data or ref["packages"] != helper.core_versions():
        raise RuntimeError("Old/new word dataset or package versions differ")
    files = ctx.manifest["payload"]["model_files"]
    model_files = {n:{"sha256":r["sha256"],"bytes":r["bytes"]} for n,r in files.items()}
    if ref["model"]["files"] != model_files:
        raise RuntimeError("Old word model/tokenizer files differ from frozen A5")
    source_refs = {}
    for name,relative in (("BF16", "BF16"), ("OLD_FULL_GD_R8", "mxint3/FULL_GD_R8")):
        directory = args.word_reference_dir/relative
        state = core.read_json(directory/"complete.json")
        result = core.read_json(directory/"results.json")
        if state.get("status") != "PASS" or state["results_sha256"] != core.sha(directory/"results.json"):
            raise RuntimeError("Old word reference incomplete/tampered: " + relative)
        if helper.extract_word_ppl(result) != state["word_ppl"]:
            raise RuntimeError("Old word reference metric mismatch")
        if name == "BF16":
            if state["protocol_sha256"] != core.fingerprint(ref):
                raise RuntimeError("Old BF16 word protocol binding mismatch")
        else:
            identity = state["artifact_identity"]
            expected = {"configuration":"FULL_GD_R8", "method":"full_gd", "rank":8,
                        "artifacts": {n:{"quant":{k:b["quant"][k] for k in ("path","sha256","bytes")},
                                         "correction":{k:b["baseline_factors"]["full_gd"][k] for k in ("path","sha256","bytes")}}
                                      for n,b in sorted(ctx.tasks.items())}}
            if identity != expected:
                raise RuntimeError("Old word correction/Wq inputs differ")
            expected_hash = core.fingerprint({"base_protocol_sha256":core.fingerprint(ref),"artifact_run":"mxint3",
                                             "width":3,"manifest_sha256":state["artifact_manifest_sha256"],**expected})
            if state["protocol_sha256"] != expected_hash:
                raise RuntimeError("Old word artifact protocol binding mismatch")
        source_refs[name] = {"state":state, "state_file":single.file_record(directory/"complete.json"),
                             "result_file":single.file_record(directory/"results.json")}
    return SimpleNamespace(helper=helper, task=task, manager=manager, data=data, references=source_refs,
                           protocol={"context":4096,"harness":helper.HARNESS_COMMIT,"official":official["commit"],
                                     "data":data,"packages":helper.core_versions(),"old_protocol":single.file_record(ref_path),
                                     "references":source_refs,"hflm":"HFLM(model) as old QERA wrapper",
                                     "bootstrap_iters":0,"limit":None,"chat_template":False})


def word_documents(result, task, helper):
    ppl = helper.extract_word_ppl(result)
    count = result["n-samples"]["wikitext"]
    if int(count["original"]) <= 0 or count["original"] != count["effective"]:
        raise RuntimeError("Incomplete harness test set")
    samples = result.get("samples", {}).get("wikitext")
    if not samples or len(samples) != result["n-samples"]["wikitext"]["effective"]:
        raise RuntimeError("Missing full harness per-document samples; no partial word result may commit")
    rows = []
    for s in samples:
        metrics = task.process_results(s["doc"], s["filtered_resps"])
        loglike, words = metrics["word_perplexity"]
        if not math.isfinite(loglike) or loglike > 1e-7 or words < 0 or int(words) != words:
            raise RuntimeError("Invalid harness document loglikelihood/word count")
        rows.append({"document":int(s["doc_id"]),"document_sha256":core.fingerprint(s["doc"]),
                     "nll_sum":-float(loglike),"words":int(words)})
    rows.sort(key=lambda r:r["document"])
    if len({r["document"] for r in rows}) != len(rows):
        raise RuntimeError("Duplicate harness document IDs")
    words = sum(r["words"] for r in rows); nll = sum(r["nll_sum"] for r in rows)
    if words <= 0 or abs(math.exp(nll/words)-ppl) > 1e-9:
        raise RuntimeError("Word-PPL cannot be reconciled with official task sample metrics")
    return {"ppl":ppl,"nll_sum":nll,"scored_words":words,"documents":len(rows),"mean_nll_per_word":nll/words}, rows


def word_existing(ctx, name):
    folder = ctx.output/"word_ppl"/name; path = folder/"complete.json"
    if not path.exists():
        return None
    state = core.read_json(path)
    if state.get("experiment_identity") != ctx.identity or state.get("status") != "PASS" or state.get("configuration") != name:
        raise RuntimeError("Word checkpoint identity mismatch")
    for key, filename in (("results_file","results.json"),("documents_file","documents.json")):
        if Path(state[key]["path"]).resolve() != (folder/filename).resolve():
            raise RuntimeError("Word checkpoint path mismatch")
        ctx.inputs.verify(state[key])
    deployed = core.read_json(ctx.output/"word_ppl"/("deployment_"+name+".json"))
    if core.fingerprint(deployed) != state["deployment_record_sha256"]:
        raise RuntimeError("Word deployment checkpoint mismatch")
    summary, rows = word_documents(core.read_json(state["results_file"]["path"]), ctx.word.task, ctx.word.helper)
    if summary != state["summary"] or rows != core.read_json(state["documents_file"]["path"]):
        raise RuntimeError("Word saved summary/documents differ from raw harness result")
    return state


def word_evaluate(ctx, args, method, rank, stop, old=False):
    name = "OLD_FULL_GD_R8" if old else label(method, rank)
    existing = word_existing(ctx, name)
    if existing:
        return existing
    stop.check()
    routes = artifact_routes(ctx, method, rank, old)
    folder = ctx.output/"word_ppl"; folder.mkdir(exist_ok=True)
    directory = folder/name; directory.mkdir(exist_ok=True)
    from transformers import AutoModelForCausalLM
    from accelerate import dispatch_model
    from qera.utils import create_device_map
    from lm_eval.models.huggingface import HFLM
    from lm_eval.evaluator import simple_evaluate
    model, lm, handles = None, None, []
    started = time.monotonic()
    try:
        torch.manual_seed(1234)
        with core.heartbeat("WORD load/install/hash " + name):
            model = AutoModelForCausalLM.from_pretrained(ctx.manifest["payload"]["source_config"]["model_path"],
                        torch_dtype=torch.bfloat16, local_files_only=True, _attn_implementation="eager", max_position_embeddings=4096)
            model.eval(); model.config.use_cache = False
            model = dispatch_model(model, device_map=create_device_map(model, "auto-balanced"))
            resident(model)
            handles, deployed = install(model, routes, rank, ctx.inputs, old)
            deploy_record = bind_deployment(ctx, folder, name, deployed, routes, method)
            lm = HFLM(model)
            if lm.max_length != 4096:
                raise RuntimeError(f"Actual harness context {lm.max_length} != 4096")
            freeze_json(directory/"runtime.json", {"experiment_identity":ctx.identity,"context":lm.max_length,
                                                   "hflm_batch_size":lm.batch_size,"device_map":model.hf_device_map})
        # Read-only budget hook; no logits/labels/cache modifications. If interrupted,
        # restart this configuration in full, never aggregate partial documents.
        handles.append(model.register_forward_pre_hook(lambda _m,_a: stop.check()))
        with core.heartbeat("WORD " + name + " full WikiText2 documents; configuration-level checkpoint"), torch.no_grad():
            result = simple_evaluate(model=lm, tasks=[ctx.word.task], task_manager=ctx.word.manager,
                    num_fewshot=None, batch_size="auto", limit=None, use_cache=None, bootstrap_iters=0,
                    log_samples=True, apply_chat_template=False, random_seed=0, numpy_random_seed=1234,
                    torch_random_seed=1234, fewshot_random_seed=1234)
        summary, documents = word_documents(result, ctx.word.task, ctx.word.helper)
        if summary["documents"] != ctx.word.data["documents"]:
            raise RuntimeError("Harness evaluated document count differs from frozen task")
        # Match old JSON serialization for numpy and torch scalar metadata.
        import json
        def convert(x):
            item = getattr(x,"item",None)
            return item() if callable(item) else str(x)
        payload = json.loads(json.dumps(result, default=convert))
        core.atomic_json(directory/"results.json", payload)
        core.atomic_json(directory/"documents.json", documents)
        record = {"experiment_identity":ctx.identity,"status":"PASS","configuration":name,"method":method,"rank":rank,
                  "summary":summary,"results_file":single.file_record(directory/"results.json"),
                  "documents_file":single.file_record(directory/"documents.json"),
                  "deployment_record_sha256":core.fingerprint(deploy_record),"elapsed_seconds":time.monotonic()-started}
        core.atomic_json(directory/"complete.json",record)
        stop.done += 1
        core.log(f"COMMITTED WORD {name} word_ppl={summary['ppl']:.9f} documents={summary['documents']}")
        return record
    finally:
        for h in handles:
            h.remove()
        handles.clear(); lm = None; model = None; clear_gpu()


def word_gate(ctx, name, state):
    reference = ctx.word.references[name]["state"]["word_ppl"]
    delta = state["summary"]["ppl"]-reference
    # 1e-4 is engineering replay tolerance, not a significance threshold.
    record = {"experiment_identity":ctx.identity,"configuration":name,"reference_ppl":reference,
              "observed_ppl":state["summary"]["ppl"],"delta_ppl":delta,"tolerance":1e-4,
              "status":"PASS" if abs(delta)<=1e-4 else "FAIL"}
    freeze_json(ctx.output/"word_ppl"/(name+"_control.json"), record)
    if record["status"] != "PASS":
        raise RuntimeError("Word protocol replay failed; new corrections blocked: " + name)


def summarize(ctx):
    for protocol in ("token_ppl", "word_ppl"):
        folder = ctx.output/protocol
        if not folder.exists():
            continue
        summary, detail = [], []; records = {}
        for method,rank in CONFIGS:
            name = label(method,rank)
            if protocol == "token_ppl":
                state = single.read_state(folder,name,ctx.identity,ctx.inputs)
                if not state["complete"]:
                    continue
                rr = state["records"]; nll = sum(r["nll_sum"] for r in rr)
                values = {"ppl":prev.ppl(rr),"nll_sum":nll,"prediction_tokens":282486,"windows":138,"context":2048}
            else:
                if ctx.word is None:
                    continue
                state = word_existing(ctx,name)
                if state is None:
                    continue
                rr = core.read_json(state["documents_file"]["path"])
                values = {**state["summary"],"context":4096}
            summary.append({"configuration":name,"method":method,"rank":rank,"metric":protocol,**values})
            records[name] = rr
            detail.extend({"configuration":name,**r} for r in rr)
        comparisons, paired = [], []
        byname = {r["configuration"]:r for r in summary}
        for rank in RANKS:
            for first, second in (("full_gi","full_gd"),("full_gi","full_gf"),("full_gd","full_gf")):
                a,b=label(first,rank),label(second,rank)
                if a not in records or b not in records:
                    continue
                key = "window" if protocol=="token_ppl" else "document"
                for x,y in zip(records[a],records[b]):
                    if x[key]!=y[key] or (protocol=="word_ppl" and (x["document_sha256"]!=y["document_sha256"] or x["words"]!=y["words"])):
                        raise RuntimeError("Paired evaluation inputs differ")
                    paired.append({"comparison":b+"_minus_"+a,key:x[key],"delta_nll":y["nll_sum"]-x["nll_sum"]})
                comparisons.append({"comparison":b+"_minus_"+a,"rank":rank,
                                    "delta_ppl":byname[b]["ppl"]-byname[a]["ppl"],
                                    "delta_nll":byname[b]["nll_sum"]-byname[a]["nll_sum"],
                                    "improved_units":sum(y["nll_sum"]<x["nll_sum"] for x,y in zip(records[a],records[b])),
                                    "units":len(records[a]),"unit":key})
        prev.write_csv(folder/"ppl_summary.csv",summary)
        prev.write_csv(folder/"per_unit.csv",detail)
        prev.write_csv(folder/"comparisons.csv",comparisons)
        prev.write_csv(folder/"paired_units.csv",paired)
        core.atomic_json(folder/"status.json",{"experiment_identity":ctx.identity,"status":"COMPLETE" if len(summary)==14 else "INCOMPLETE",
                                               "completed":len(summary),"expected":14,"metric":protocol,
                                               "note":"No IID-token/document significance claims; word and token protocols are separate."})


def setup(args):
    if core.sha(parent.__file__) != PARENT_SHA:
        raise RuntimeError("Frozen A5 helper changed")
    # This existing setup is read-only: it audits frozen inputs/environment.
    ctx = parent.setup(SimpleNamespace(**{**vars(args),"command":"run"}))
    source = args.source_fp64_dir.resolve()
    core.disjoint(ctx.output, [source,args.word_reference_dir,args.harness_source,args.official_qera_root])
    exp_path = source/"experiment.json"
    if core.read_json(exp_path) != ctx.experiment:
        raise RuntimeError("Source A5 experiment does not match frozen helpers/inputs/environment")
    ctx.source_identity = ctx.identity
    src_ctx = copy.copy(ctx); src_ctx.output = source
    ctx.factors = {m:{} for m in METHODS}; ctx.factor_metadata = {m:{} for m in METHODS}
    for m in METHODS:
        for name,b in ctx.tasks.items():
            rec = parent.solved_record(source,m,name,b,ctx.source_identity,ctx.inputs)
            if rec is None:
                raise RuntimeError("A5 FP64 factor set incomplete")
            ctx.factors[m][name] = rec
            ctx.factor_metadata[m][name] = single.file_record(parent.factor_paths(source,m,name)[1])
    ctx.source_states = {}; ctx.source_deployments = {}; source_controls = {}
    for m in METHODS:
        folder = source/"evaluation"/m
        gate = core.read_json(folder/"control_check.json")
        if gate.get("experiment_identity") != ctx.source_identity or gate.get("status") != "PASS":
            raise RuntimeError("Source A5 OLD control not passed: " + m)
        source_controls[m] = single.file_record(folder/"control_check.json")
        ctx.source_states[m] = single.read_state(folder,"FP64",ctx.source_identity,ctx.inputs)
        if not ctx.source_states[m]["complete"]:
            raise RuntimeError("Source A5 r8 evaluation incomplete")
        ctx.source_deployments[m] = core.read_json(folder/"deployment_FP64.json")
        if ctx.source_deployments[m]["route_sha256"] != core.fingerprint(parent.routes(src_ctx,m,"FP64")):
            raise RuntimeError("Source r8 evaluation routed to different factor files")
    ctx.legacy = parent.load_legacy(ctx,args)
    ctx.word = prepare_word(ctx,args) if args.protocol in ("both","word") else None
    ctx.experiment = {"version":VERSION,"script_sha256":core.sha(__file__),"parent_sha256":PARENT_SHA,
                      "source_experiment":single.file_record(exp_path),"source_identity":ctx.source_identity,
                      "source_factor_metadata":ctx.factor_metadata,"source_controls":source_controls,
                      "source_states_sha256":core.fingerprint(ctx.source_states),"ranks":list(RANKS),"methods":list(METHODS),
                      "protocol_selection":args.protocol,"token_protocol":{"windows":138,"length":2048,"batch":8,"ce_chunk":256},
                      "word_protocol":None if ctx.word is None else ctx.word.protocol,
                      "rank_check":"orthogonality + stationarity + source-r8 anchored tails; no new SVD",
                      "deployment":"source FP64 rank64 -> direct BF16 prefixes; two-GEMM hooks",
                      "word_resume":"configuration boundary, not partial-doc accumulation","new_root_precision":False}
    ctx.identity = core.fingerprint(ctx.experiment)
    return ctx


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command",choices=("doctor","pilot","check","run","evaluate","summarize"))
    for key in ("run-dir","repo-dir","output-dir","source-fp64-dir","official-qera-root","harness-source","word-reference-dir"):
        p.add_argument("--"+key,required=True,type=Path)
    p.add_argument("--protocol",choices=("both","token","word"),default="both")
    p.add_argument("--max-hours",type=float,default=10)
    p.add_argument("--max-new-units",type=int)
    args = p.parse_args(argv)
    if not math.isfinite(args.max_hours) or args.max_hours<=0 or (args.max_new_units is not None and args.max_new_units<1):
        p.error("Budgets must be positive")
    stop = prev.Budget(args.max_hours,args.max_new_units)
    signal.signal(signal.SIGINT,stop.signal); signal.signal(signal.SIGTERM,stop.signal)
    ctx = setup(args)
    if args.command == "doctor":
        core.log(f"DOCTOR PASS: 672 source factor sets; ranks={RANKS}; output={ctx.output}; no evaluation performed")
        return 0
    if ctx.output.exists() and not (ctx.output/"experiment.json").exists() and any(ctx.output.iterdir()):
        raise RuntimeError("Refusing unowned nonempty output directory")
    ctx.output.mkdir(parents=True,exist_ok=True)
    payload_bytes = sum(sum(b["shape"]) for b in ctx.tasks.values())*len(METHODS)*64*2
    free = shutil.disk_usage(ctx.output).free
    core.log(f"storage new_BF16_factors_GiB={payload_bytes/2**30:.3f} free_GiB={free/2**30:.1f}")
    if free < max(8*2**30, 2*payload_bytes):
        raise RuntimeError("Need at least 8 GiB free for new factors, samples and checkpoints")
    with core.audit_lock(ctx.output):
        freeze_json(ctx.output/"experiment.json",ctx.experiment)
        try:
            if args.command == "summarize":
                summarize(ctx); return 0
            if args.command in ("run","check","pilot"):
                check_all(ctx,stop,list(parent.PILOT) if args.command=="pilot" else None)
            if args.command == "check":
                return 0
            if args.command == "pilot":
                # Also exercise actual 4096 full-test BF16 reference when selected.
                if ctx.word is not None:
                    state=word_evaluate(ctx,args,"teacher",None,stop); word_gate(ctx,"BF16",state)
                core.log("PILOT COMPLETE: two large modules checked; word BF16 control if selected. Use run.")
                return 0
            if write_check_summary(ctx)["status"] != "COMPLETE":
                raise RuntimeError("Run check/run before evaluate; no incomplete factor deployment")
            if args.protocol in ("token","both"):
                # Full r8 replay gates the new rank-aware path. Cheap enough to
                # re-evaluate rather than adopt source state under a new identity.
                for m in METHODS:
                    state=token_evaluate(ctx,ctx.legacy,m,8,stop); token_gate(ctx,m,state); summarize(ctx)
                for m,r in CONFIGS:
                    token_evaluate(ctx,ctx.legacy,m,r,stop); summarize(ctx)
            if ctx.word is not None:
                state=word_evaluate(ctx,args,"teacher",None,stop); word_gate(ctx,"BF16",state)
                state=word_evaluate(ctx,args,"full_gd",8,stop,old=True); word_gate(ctx,"OLD_FULL_GD_R8",state)
                for m,r in CONFIGS:
                    word_evaluate(ctx,args,m,r,stop); summarize(ctx)
            summarize(ctx)
            core.log("REQUESTED PROTOCOLS COMPLETE. Not an automatic scientific pass.")
        except prev.Paused:
            summarize(ctx)
            core.log("PAUSED: rerun SAME run command. Token: <=8 windows lost. Word: current configuration restarts.")
            return 75
        except Exception as exc:
            core.atomic_json(ctx.output/"last_failure.json",{"experiment_identity":ctx.identity,"type":type(exc).__name__,
                             "message":str(exc),"note":"No fallback, source rewrite, or protocol change performed."})
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
