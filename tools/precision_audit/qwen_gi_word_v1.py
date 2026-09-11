"""Same pinned 4096 document-level harness as Llama, with Qwen model/tokenizer.

Uses the frozen dual evaluator's result validator; no new tokenization,
detokenization, word counter, rolling likelihood or PPL reduction algorithm.
"""
from __future__ import annotations
import gc
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import time
import torch

PINS = {"full_a_all_ranks_dual_ppl_v1.py": "e7d331c417fbb1d114f93d62583580e7aa949e5179a557f2d6d14bc28572ebf7",
        "full_a_all_precision_r8_v1.py": "23ec871264e6d6312d3bd4c7fb08ddc89dbc745cd2b1f0ae9fa1289307833f4f"}
UTILS_SHA = "e8a781ddccb2dce08623a67efd16cac9c940ef78968fc73c2c2d33b1c2ee3967"


def prepare(args, manifest, output, identity, single, h):
    directory = Path(__file__).resolve().parent
    for name, digest in PINS.items():
        if h.sha256(directory/name) != digest:
            raise RuntimeError("Frozen word helper changed: "+name)
    dual = importlib.import_module("full_a_all_ranks_dual_ppl_v1")
    for module in (dual, dual.parent):
        if h.sha256(module.__file__) != PINS[Path(module.__file__).name]:
            raise RuntimeError("Wrong imported word helper")
    helper = dual.load_word_helper()
    helper.verify_harness(args.harness_source, check_import=True)
    from qera_original_a_isolation.common import import_official_qera
    official = import_official_qera({"qera_source_dir": str(args.official_qera_root),
                                     "official_qera_commit": "bd7fc86a2e44d41f95b9b0421f27f5624dd37064"})
    from qera.utils import QERA_SRC_DIR
    import qera.utils as qera_utils
    from lm_eval.tasks import TaskManager, get_task_dict
    if Path(QERA_SRC_DIR).resolve() not in ((args.official_qera_root/"src").resolve(), (args.official_qera_root/"src/qera").resolve()):
        raise RuntimeError("Unexpected QERA import")
    if hashlib.sha256(Path(qera_utils.__file__).read_bytes().replace(b"\r\n",b"\n")).hexdigest() != UTILS_SHA:
        raise RuntimeError("Official device-placement helper changed")
    manager = TaskManager(verbosity="INFO", include_path=str(Path(QERA_SRC_DIR).parent/"qera_harness_tasks"), include_defaults=True)
    task = get_task_dict(["wikitext"], manager)["wikitext"]
    data = helper.dataset_identity(task)
    ref = h.read_json(args.word_reference_dir/"protocol.json")
    fixed = ref["fixed_protocol"]
    for key, value in {"task":"wikitext", "context_length":4096, "max_position_embeddings":4096,
                       "dtype":"bfloat16", "attn_implementation":"eager", "device_map":"auto-balanced",
                       "lm_eval_batch_size":"auto", "num_fewshot":None}.items():
        if fixed.get(key) != value:
            raise RuntimeError("Llama word protocol mismatch: "+key)
    if (ref["harness_commit"] != helper.HARNESS_COMMIT or ref["harness_hashes"] != helper.HARNESS_HASHES
            or ref["official_qera_commit"] != official["commit"] or ref["packages"] != helper.core_versions()
            or ref["dataset"] != data or data["documents"] != 62):
        raise RuntimeError("Word harness / packages / actual document dataset differs from frozen Llama protocol")
    protocol = {"experiment": identity, "reference_protocol": single.file_record(args.word_reference_dir/"protocol.json"),
                "dataset": data, "packages": helper.core_versions(), "harness_hashes": helper.HARNESS_HASHES,
                "harness_commit": helper.HARNESS_COMMIT, "harness_path": str(args.harness_source.resolve()),
                "official_commit": official["commit"], "model_files": manifest["payload"]["model_files"],
                "model_tokenizer_change": "Qwen2.5-7B Base local model and its own tokenizer; no Llama PPL reference",
                "context": 4096, "batch_size": "auto", "chat_template": False, "limit": None,
                "bootstrap_iters": 0, "seeds": [0, 1234, 1234, 1234], "helpers": PINS,
                "qera_utils_sha256_lf": UTILS_SHA,
                "frozen_harness_helper": dual.WORD_HELPER_SHA}
    dual.freeze_json(output/"word_protocol.json", protocol)
    h.log("WORD PREFLIGHT PASS: same 62 documents / 4096 protocol; Qwen tokenizer; new BF16 reference required")
    return SimpleNamespace(identity=h.fingerprint(protocol), experiment=identity, dual=dual, helper=helper,
                           task=task, manager=manager, data=data, model_path=manifest["payload"]["source_config"]["model_path"])


def existing(folder, identity, factors, ctx, single, h):
    marker = folder/"complete.json"
    if not marker.exists():
        return None
    state = h.read_json(marker)
    if state.get("identity") != identity or state.get("factors") != factors or state.get("status") != "PASS":
        raise RuntimeError("Word checkpoint identity/status mismatch")
    for key, name in (("results_file", "results.json"), ("documents_file", "documents.json"), ("deployment_file", "deployment.json")):
        if Path(state[key]["path"]).resolve() != (folder/name).resolve():
            raise RuntimeError("Word checkpoint points elsewhere")
        h.verify(state[key])
    summary, documents = ctx.dual.word_documents(h.read_json(folder/"results.json"), ctx.task, ctx.helper)
    if summary != state["summary"] or documents != h.read_json(folder/"documents.json"):
        raise RuntimeError("Word results/summary/document mismatch")
    return state


def evaluate(ctx, entries, manifest, output, single, h, stop, main, pilot=False):
    root = output/"word_ppl"
    root.mkdir(parents=True, exist_ok=True)
    summaries, details = [], []
    configs = main.configurations()[:1] if pilot else main.configurations()
    for label, method, rank in configs:
        stop.check()
        factors = {}
        if method in main.METHODS:
            for entry in entries:
                obj = main.factor_record(output, method, entry, ctx.experiment, single, h)
                if obj is None:
                    raise RuntimeError("Complete FP64 factors before word-PPL")
                factors[entry["layer"]["name"]] = obj["file"]
        identity = h.fingerprint({"word_protocol": ctx.identity, "configuration": label, "factors": factors})
        folder = root/label
        folder.mkdir(parents=True, exist_ok=True)
        state = existing(folder, identity, factors, ctx, single, h)
        if state is None:
            from transformers import AutoModelForCausalLM
            from accelerate import dispatch_model
            from qera.utils import create_device_map
            from lm_eval.models.huggingface import HFLM
            from lm_eval.evaluator import simple_evaluate
            model, lm, handles = None, None, []
            started = time.monotonic()
            try:
                torch.manual_seed(1234)
                with h.progress("WORD load/install "+label):
                    model = AutoModelForCausalLM.from_pretrained(ctx.model_path, torch_dtype=torch.bfloat16,
                                local_files_only=True, _attn_implementation="eager", max_position_embeddings=4096)
                    model.eval(); model.config.use_cache = False
                    model = dispatch_model(model, device_map=create_device_map(model, "auto-balanced"))
                    ctx.dual.resident(model)
                    model.requires_grad_(False)
                    deployed = main.install(model, entries, factors, method, rank, single, h, handles)
                    single.core.atomic_json(folder/"deployment.json", {"identity":identity, "tensors":deployed, "device_map":model.hf_device_map})
                    lm = HFLM(model)
                    if lm.max_length != 4096 or Path(lm.tokenizer.name_or_path).resolve() != Path(ctx.model_path).resolve():
                        raise RuntimeError("Actual word context/tokenizer differs from Qwen protocol")
                    single.core.atomic_json(folder/"runtime.json", {"identity":identity, "context":lm.max_length,
                                            "tokenizer":lm.tokenizer.name_or_path, "batch_size":lm.batch_size})
                handles.append(model.register_forward_pre_hook(lambda _m,_a: stop.check()))
                with h.progress("WORD "+label+" 62 documents; checkpoint after whole configuration"), torch.no_grad():
                    result = simple_evaluate(model=lm, tasks=[ctx.task], task_manager=ctx.manager, num_fewshot=None,
                            batch_size="auto", limit=None, use_cache=None, bootstrap_iters=0, log_samples=True,
                            apply_chat_template=False, random_seed=0, numpy_random_seed=1234,
                            torch_random_seed=1234, fewshot_random_seed=1234)
                summary, documents = ctx.dual.word_documents(result, ctx.task, ctx.helper)
                if summary["documents"] != 62 or summary["scored_words"] != 241335:
                    raise RuntimeError("Word scoring coverage differs from frozen Llama documents")
                def convert(x):
                    item = getattr(x, "item", None)
                    return item() if callable(item) else str(x)
                single.core.atomic_json(folder/"results.json", json.loads(json.dumps(result, default=convert)))
                single.core.atomic_json(folder/"documents.json", documents)
                state = {"identity":identity, "status":"PASS", "factors":factors, "summary":summary,
                         "results_file":single.file_record(folder/"results.json"), "documents_file":single.file_record(folder/"documents.json"),
                         "deployment_file":single.file_record(folder/"deployment.json"), "elapsed_seconds":time.monotonic()-started}
                single.core.atomic_json(folder/"complete.json", state)
                h.log(f"COMMITTED WORD {label} ppl={summary['ppl']:.9f} documents=62")
            finally:
                for hook in handles:
                    hook.remove()
                handles.clear(); model = lm = None
                gc.collect(); torch.cuda.empty_cache()
        summaries.append({"configuration":label, "method":method or "teacher", "rank":rank or "",
                          "context":4096, "metric":"word_ppl", **state["summary"]})
        details.extend({"configuration":label, **r} for r in h.read_json(folder/"documents.json"))
        single.previous.write_csv(root/"ppl_summary.csv", summaries)
        single.previous.write_csv(root/"per_document.csv", details)
        single.core.atomic_json(root/"status.json", {"identity":ctx.identity, "status":"COMPLETE" if len(summaries)==10 else "INCOMPLETE",
                                "completed":len(summaries), "expected":10, "pilot":pilot})
