from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen25_base_isolation_v1 import data, protocol, stages, state, run
from qera_diag_g_isolation.storage import atomic_json, atomic_tensors, file_record, fingerprint, read_json
from qera_diag_g_isolation.full_g_v1.checkpoint import Stop, Paused


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(301)


def test_a_arithmetic_matches_frozen_hook_including_fp32_diagonal():
    full = torch.zeros(9, 9, dtype=torch.float64)
    diag = torch.zeros(9, dtype=torch.float32)
    rf, rd = full.clone(), diag.clone()
    for _ in range(10):
        x = torch.randn(2, 5, 9)
        stages.accumulate_a(x, full, diag)
        x = x.reshape(-1, 9).to(torch.float32)
        rf.add_((x.T @ x).to(device="cpu", dtype=torch.float64))
        rd.add_(x.square().sum(0).to(device="cpu", dtype=torch.float64))
    assert torch.equal(rf, full) and torch.equal(rd, diag)
    assert diag.dtype == torch.float32
    with pytest.raises(RuntimeError, match="dtype"):
        stages.accumulate_a(x, full, diag.double())


def test_isolation_refuses_shared_code_nested_output_and_protected_mutation(tmp_path):
    old = tmp_path / "old"
    new = tmp_path / "new"
    out = tmp_path / "results/new"
    protocol.assert_separate(new, out, [old], [tmp_path / "results/old"])
    for candidate in (old, old / "child", tmp_path):
        with pytest.raises(RuntimeError, match="separate code"):
            protocol.assert_separate(candidate, out, [old], [])
    with pytest.raises(RuntimeError, match="disjoint"):
        protocol.assert_separate(new, tmp_path / "results", [old], [tmp_path / "results/old"])
    path = tmp_path / "protected.py"
    path.write_text("original")
    record = file_record(path)
    protocol.verify_protected([record])
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="Immutable"):
        protocol.verify_protected([record])


@pytest.mark.parametrize("stage", ["tensor", "state", "pointer", "cleanup"])
def test_a_checkpoint_faults_preserve_whole_batch(tmp_path, monkeypatch, stage):
    schema = {"a": {"full": ((4, 4), torch.float64), "diag": ((4,), torch.float32)}}
    store = state.Store(tmp_path / "a", "manifest", schema, 2048)
    values = {"a": {"full": torch.eye(4, dtype=torch.float64), "diag": torch.ones(4)}}
    first = store.save(values, 4)
    original_json, original_tensor, original_cleanup = state.atomic_json, state.atomic_tensors, store.cleanup
    def json_fail(path, value):
        if (stage == "state" and Path(path).name == "STATE.json") or (stage == "pointer" and Path(path).name == "CURRENT.json"):
            raise RuntimeError("injected crash")
        original_json(path, value)
    def tensor_fail(*args):
        if stage == "tensor":
            raise RuntimeError("injected crash")
        original_tensor(*args)
    def cleanup_fail():
        if stage == "cleanup":
            raise RuntimeError("injected crash")
        original_cleanup()
    monkeypatch.setattr(state, "atomic_json", json_fail)
    monkeypatch.setattr(state, "atomic_tensors", tensor_fail)
    monkeypatch.setattr(store, "cleanup", cleanup_fail)
    with pytest.raises(RuntimeError, match="injected"):
        store.save(values, 8)
    assert store.current()["windows"] == (8 if stage == "cleanup" else 4)
    loaded, n, _ = store.load()
    assert loaded["a"]["diag"].dtype == torch.float32
    assert loaded["a"]["full"].dtype == torch.float64
    monkeypatch.setattr(state, "atomic_json", original_json)
    monkeypatch.setattr(state, "atomic_tensors", original_tensor)
    monkeypatch.setattr(store, "cleanup", original_cleanup)
    store.cleanup()
    if n == 4:
        store.save(values, 8)
    assert len(list(store.root.glob("gen_*"))) == 1


def test_checkpoint_rejects_bad_hash_count_and_foreign_cleanup(tmp_path):
    schema = {"x": {"diag": ((3,), torch.float64)}}
    store = state.Store(tmp_path, "m", schema, 2047)
    value = {"x": {"diag": torch.ones(3, dtype=torch.float64)}}
    first = store.save(value, 8, 10.)
    with pytest.raises(RuntimeError, match="commit"):
        store.save(value, 7)
    important = store.root / first["generation"] / "user.txt"
    important.write_text("keep")
    with pytest.raises(RuntimeError, match="Unexpected file"):
        store.save(value, 16, 20.)
    assert important.read_text() == "keep"
    latest = store.current()
    Path(latest["files"]["x"]["path"]).write_bytes(b"damaged")
    with pytest.raises(RuntimeError, match="Immutable"):
        store.load()


def test_data_replay_and_dynamic_windows():
    reference = {"input_ids": torch.ones(3, 2048, dtype=torch.int64), "attention_mask": torch.ones(3, 2048, dtype=torch.int64)}
    data.replay_gate(reference, deepcopy(reference))
    other = deepcopy(reference)
    other["input_ids"][0, 0] = 2
    with pytest.raises(RuntimeError, match="token replay"):
        data.replay_gate(other, reference)
    class Rows:
        def __len__(self):
            return 3
        def __getitem__(self, _):
            return {k: v.tolist() for k, v in reference.items()}
    result = data.windows({"test": Rows()}, "wikitext2")
    assert result["input_ids"].shape == (3, 2048)  # Not hard-coded to Llama's 138.


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.proj = torch.nn.Linear(8, 8, bias=True)
    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(last_hidden_state=self.proj(self.embed(input_ids)))


class Teacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Decoder()
        self.head = torch.nn.Linear(8, 16, bias=False)
        self.hf_device_map = {"model": 0, "head": 1}
    def get_input_embeddings(self):
        return self.model.embed
    def get_output_embeddings(self):
        return self.head
    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.head(self.model(input_ids).last_hidden_state))


@pytest.fixture
def toy_run(tmp_path, monkeypatch):
    config = {"run_dir": str(tmp_path / "qwen"), "ranks": [1, 2, 3, 4], "identity_product_tolerance": .001,
              "g_relative_floor": 1e-6, "solve_device": "cpu", "a_groups_per_shard": 1,
              "a_batch_size": 2, "a_checkpoint_windows": 8, "checkpoint_every_windows": 8,
              "ce_gradient_chunk_tokens": 128, "sqrtm_max_imaginary": 1e-6,
              "eval_batch_size": 2, "eval_ce_chunk_tokens": 256}
    model = Teacher()
    ids = torch.randint(0, 16, (256, 2048))
    p = tmp_path / "calibration.safetensors"
    e = tmp_path / "wikitext.safetensors"
    atomic_tensors(p, {"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    atomic_tensors(e, {"input_ids": ids[:3].clone(), "attention_mask": torch.ones_like(ids[:3])})
    name = "model.proj"
    group = {"target": name, "in_features": 8, "layers": [{"name": name, "shape": [8, 8], "bias": True}]}
    payload = {"config": config, "code": {}, "source_config": {}, "groups": [group],
               "data": {"calibration": file_record(p), "wikitext2": file_record(e)},
               "data_details": {"wikitext2": {"windows": 3}}}
    manifest = {"sha256": fingerprint(payload), "payload": payload}
    atomic_json(Path(config["run_dir"]) / "manifest.json", manifest)
    def load(*_):
        m = deepcopy(model)
        m.requires_grad_(False)
        return m
    monkeypatch.setattr(stages, "teacher", load)
    monkeypatch.setattr(stages, "verify_model", lambda *_: None)
    monkeypatch.setattr(stages, "_input_device", lambda *_: "cpu")
    monkeypatch.setattr(torch.autograd.graph, "save_on_cpu", lambda **_: nullcontext())
    monkeypatch.setattr(stages, "weight_tensor", lambda *_: model.model.proj.weight.detach().bfloat16())
    # Independent low-level reference, while separate existing tests exercise
    # actual pinned QERA quantizer/compute_ab on local official source.
    def inverse(scale, u):
        return torch.linalg.solve(torch.diag(scale) if scale.ndim == 1 else scale, u)
    def quantizer(weight, **kwargs):
        assert kwargs == {"width": 3, "block_size": 32, "block_axis": -1}
        return (weight.float()*8).round().div(8).to(weight.dtype)
    def compute_ab(name, layer, scale, cfg):
        assert cfg["w_quantizer"]["width"] == 3
        q = quantizer(layer.weight, width=3, block_size=32, block_axis=-1)
        error = (layer.weight-q).T
        weighted = (torch.diag(scale) if scale.ndim == 1 else scale) @ error
        u, s, vh = torch.linalg.svd(weighted, full_matrices=True)
        r = cfg["rank"]
        a, b = inverse(scale, u[:, :r]), torch.diag(s[:r]) @ vh[:r]
        return {name+".A": a, name+".B": b}, float((error-a@b).square().mean())
    import scipy.linalg
    official = {"mxint_quantizer": quantizer, "compute_ab": compute_ab, "sqrtm_scipy": scipy.linalg.sqrtm}
    monkeypatch.setattr(stages, "import_official_qera", lambda *_: official)
    mod = ModuleType("qera.approximate")
    mod._compute_scale_inv_dot_U = inverse
    monkeypatch.setitem(sys.modules, "qera.approximate", mod)
    return config, manifest, model


@pytest.mark.parametrize("kind", ["a", "dg"])
def test_real_collect_loop_resume_matches_uninterrupted(toy_run, tmp_path, kind):
    config, manifest, model = toy_run
    saved_bias = model.model.proj.bias.detach().clone()
    with pytest.raises(Paused):
        stages.collect(config, Stop(), kind, 4)
    with pytest.raises(Paused):
        stages.collect(config, Stop(), kind, 4)
    store = stages.a_store(config, manifest, 0) if kind == "a" else stages.g_store(config, manifest)
    resumed, n, nll = store.load()
    config2 = dict(config, run_dir=str(tmp_path / "continuous"))
    payload2 = deepcopy(manifest["payload"])
    payload2["config"] = config2
    manifest2 = {"sha256": fingerprint(payload2), "payload": payload2}
    atomic_json(Path(config2["run_dir"]) / "manifest.json", manifest2)
    with pytest.raises(Paused):
        stages.collect(config2, Stop(), kind, 8)
    other_store = stages.a_store(config2, manifest2, 0) if kind == "a" else stages.g_store(config2, manifest2)
    continuous, n2, nll2 = other_store.load()
    assert n == n2 == 8 and nll == nll2
    for k in resumed["model.proj"]:
        assert torch.equal(resumed["model.proj"][k], continuous["model.proj"][k])
    assert torch.equal(model.model.proj.bias, saved_bias)


def seed_statistics(config, manifest):
    x = torch.randn(100, 8)
    full, diag = torch.zeros(8, 8, dtype=torch.float64), torch.zeros(8)
    stages.accumulate_a(x, full, diag)
    stages.a_store(config, manifest, 0).save({"model.proj": {"full": full, "diag": diag}}, 256)
    stages.g_store(config, manifest).save({"model.proj": {"diag": torch.arange(1, 9, dtype=torch.float64)}}, 256, 100.)


def test_end_to_end_roots_gi_gd_eval_dynamic_rows_resume(toy_run):
    config, manifest, model = toy_run
    seed_statistics(config, manifest)
    stages.roots(config, Stop())
    stages.quantize(config, Stop())
    stages.solve(config, Stop(), "gi")
    stages.evaluate(config, Stop(), "gi")
    assert stages.summarize(config)["configurations"] == 10
    stages.solve(config, Stop(), "gd")
    stages.evaluate(config, Stop(), "gd")
    result = stages.summarize(config)
    assert result["status"] == "PASS" and result["configurations"] == 18
    assert result["windows_per_configuration"] == 3
    before = {p: p.stat().st_mtime_ns for p in Path(config["run_dir"]).rglob("*.safetensors")}
    stages.roots(config, Stop())
    stages.quantize(config, Stop())
    stages.solve(config, Stop(), "gi")
    stages.solve(config, Stop(), "gd")
    assert before == {p: p.stat().st_mtime_ns for p in before}


def test_solver_failed_identity_blocks_gi(toy_run, monkeypatch):
    config, manifest, _ = toy_run
    seed_statistics(config, manifest)
    stages.roots(config, Stop())
    stages.quantize(config, Stop())
    monkeypatch.setattr(stages, "correction_drift", lambda *_: {str(r): .003 for r in config["ranks"]})
    with pytest.raises(RuntimeError, match="Identity-G"):
        stages.solve(config, Stop(), "gi")
    assert not list((Path(config["run_dir"]) / "corrections").rglob("*.safetensors"))


def test_partial_backward_and_eval_pause_never_duplicate(toy_run, monkeypatch):
    config, manifest, _ = toy_run
    with pytest.raises(Paused):
        stages.collect(config, Stop(), "dg", 4)
    original = stages.diagonal_increment
    def crash(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("partial backward")
    monkeypatch.setattr(stages, "diagonal_increment", crash)
    with pytest.raises(RuntimeError, match="partial backward"):
        stages.collect(config, Stop(), "dg", 4)
    assert stages.g_store(config, manifest).current()["windows"] == 4
    original_save = stages.atomic_json
    stop = Stop()
    def save_stop(path, value):
        original_save(path, value)
        if Path(path).parent.name == "configurations":
            stop.signal_number = 15
    monkeypatch.setattr(stages, "atomic_json", save_stop)
    with pytest.raises(Paused):
        stages.evaluate(config, stop, "gi")
    saved = read_json(Path(config["run_dir"]) / "evaluation/configurations/BF16.json")
    assert len(saved["records"]) == 2
    assert [r["window"] for r in saved["records"]] == [0, 1]


def test_incomplete_evaluation_not_published(toy_run):
    config, manifest, _ = toy_run
    path = Path(config["run_dir"]) / "evaluation/configurations/BF16.json"
    digest = fingerprint({"manifest": manifest["sha256"], "name": "BF16", "files": {}})
    atomic_json(path, {"protocol": digest, "records": [{"window": 0, "tokens": 2047, "nll_sum": 12.}]})
    assert stages.summarize(config)["configurations"] == 0
    with pytest.raises(RuntimeError):
        stages.evaluation_rows(path, "wrong", 3)


def test_private_helper_copy_must_match_frozen_release(tmp_path):
    original = tmp_path / "old/experiments/qera_diag_g_isolation/math_ops.py"
    private = tmp_path / "new/experiments/qera_diag_g_isolation/math_ops.py"
    original.parent.mkdir(parents=True)
    private.parent.mkdir(parents=True)
    original.write_text("same")
    private.write_text("same")
    records = [file_record(original)]
    protocol.verify_private_helpers(tmp_path / "new", records)
    private.write_text("different")
    with pytest.raises(RuntimeError, match="Private legacy helper"):
        protocol.verify_private_helpers(tmp_path / "new", records)


def test_qwen_inventory_validates_base_shapes_bias_and_missing_weights(tmp_path, monkeypatch):
    config = {"model_type": "qwen2", "num_hidden_layers": 28, "hidden_size": 3584,
              "intermediate_size": 18944, "num_attention_heads": 28, "num_key_value_heads": 4,
              "vocab_size": 152064, "torch_dtype": "bfloat16", "tie_word_embeddings": False}
    atomic_json(tmp_path / "config.json", config)
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        atomic_json(tmp_path / filename, {})
    (tmp_path / "README.md").write_text("This repo contains the base 7B Qwen2.5 model.")
    shapes = {"model.embed_tokens.weight": [152064, 3584], "lm_head.weight": [152064, 3584]}
    shapes["model.norm.weight"] = [3584]
    shapes.update({f"model.layers.{i}.{kind}.weight": [3584]
                   for i in range(28) for kind in ("input_layernorm", "post_attention_layernorm")})
    for group in protocol._groups_from_config(SimpleNamespace(**config)):
        for name in group.all_layers:
            suffix = name.rsplit(".", 1)[-1]
            out = 512 if suffix in ("k_proj", "v_proj") else 18944 if suffix in ("gate_proj", "up_proj") else 3584
            shapes[name + ".weight"] = [out, group.in_features]
            if suffix in ("q_proj", "k_proj", "v_proj"):
                shapes[name + ".bias"] = [out]
    index = {k: "model.safetensors" for k in shapes}
    atomic_json(tmp_path / "model.safetensors.index.json", {"weight_map": index})
    (tmp_path / "model.safetensors").write_bytes(b"header fixture, not model weights")
    class Header:
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def keys(self): return shapes.keys()
        def get_slice(self, name):
            return SimpleNamespace(get_shape=lambda: shapes[name], get_dtype=lambda: "BF16")
    monkeypatch.setattr(protocol, "safe_open", lambda *_, **__: Header())
    _, _, groups = protocol.model_inventory(tmp_path)
    assert len(groups) == 112 and sum(len(g["layers"]) for g in groups) == 196
    assert sum(layer["bias"] for g in groups for layer in g["layers"]) == 84
    shapes["lm_head.weight"] = [1, 1]
    with pytest.raises(RuntimeError, match="unquantized embedding or lm_head"):
        protocol.model_inventory(tmp_path)
    shapes["lm_head.weight"] = [152064, 3584]
    missing = "model.layers.0.self_attn.q_proj.bias"
    del index[missing]
    atomic_json(tmp_path / "model.safetensors.index.json", {"weight_map": index})
    with pytest.raises(RuntimeError, match="bias coverage"):
        protocol.model_inventory(tmp_path)
    (tmp_path / "README.md").write_text("This repo contains an instruction-tuned model.")
    with pytest.raises(RuntimeError, match="Base-model README"):
        protocol.model_inventory(tmp_path)


def test_resolved_config_uses_actual_saved_a_not_qwen_template(tmp_path):
    a = {"calibration_batch_size": 4, "checkpoint_every_windows": 32, "num_workers": 8,
         "sqrtm_max_imaginary": 1e-6}
    dg = {"ranks": [8, 16, 32, 64], "teacher_dtype": "float32", "eval_batch_size": 8}
    value = protocol.resolved_config({"run_dir": str(tmp_path / "output"), "model_path": str(tmp_path / "model")},
                                     {"a_config": a, "dg_config": dg})
    assert value["a_batch_size"] == 4 and value["a_checkpoint_windows"] == 32
    assert value["teacher_dtype"] == "float32" and value["quantization"]["width"] == 3
    assert "a_batch_size" not in dg


@pytest.mark.parametrize("cache_exists", [False, True])
def test_prepare_accepts_import_created_cache_and_retry(tmp_path, monkeypatch, cache_exists):
    settings = {"run_dir": str(tmp_path)}
    config = {"run_dir": str(tmp_path), "model_path": "unused"}
    cache = tmp_path / "dataset_cache"
    if cache_exists:
        (cache / "transformers").mkdir(parents=True)
        (cache / "transformers/version.txt").write_text("1")
    def audit(_):
        # Reproduce Transformers' import side effect inside audit_reference.
        (cache / "transformers").mkdir(parents=True, exist_ok=True)
        return {}
    monkeypatch.setattr(protocol, "audit_reference", audit)
    monkeypatch.setattr(protocol, "resolved_config", lambda *_: config)
    monkeypatch.setattr(run, "environment", lambda: {"test": "fixed"})
    def reached_inventory(_):
        raise RuntimeError("reached model inventory")
    monkeypatch.setattr(protocol, "model_inventory", reached_inventory)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="reached model inventory"):
            run.prepare(settings)
        assert read_json(tmp_path / "initialization.json")["config"] == config
    assert cache.is_dir()


@pytest.mark.parametrize("name", ["statistics", "data", "manifest.json", "user.txt"])
def test_initial_output_still_rejects_unknown_results(tmp_path, name):
    (tmp_path / "dataset_cache").mkdir()
    (tmp_path / name).write_text("do not adopt")
    with pytest.raises(RuntimeError, match="Unknown output contents"):
        run.check_initial_output(tmp_path)
    assert (tmp_path / name).read_text() == "do not adopt"


def test_initial_cache_must_not_be_a_file(tmp_path):
    (tmp_path / "dataset_cache").write_text("not a directory")
    with pytest.raises(RuntimeError, match="real directory"):
        run.check_initial_output(tmp_path)
