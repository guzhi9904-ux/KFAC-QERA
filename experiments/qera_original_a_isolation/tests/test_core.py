from __future__ import annotations

import sys
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

EXPERIMENT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT.parent) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT.parent))

from qera_original_a_isolation.common import (  # noqa: E402
    InputGroup,
    discover_input_groups,
    qera_layer_config,
    safe_name,
    sha256_source_file,
    validate_config,
)
from qera_original_a_isolation.pipeline import (  # noqa: E402
    _chunked_window_nll,
    _evaluation_load_config,
    _evaluate_one,
    _groups_from_config,
    _read_jsonl,
    _real_sqrtm_root,
    _reset_evaluation_memory_peaks,
    _take_streaming_rows,
)


class ProjectionBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.self_attn.k_proj = torch.nn.Linear(8, 4, bias=False)
        self.self_attn.v_proj = torch.nn.Linear(8, 4, bias=False)
        self.self_attn.o_proj = torch.nn.Linear(8, 8, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.gate_proj = torch.nn.Linear(8, 24, bias=False)
        self.mlp.up_proj = torch.nn.Linear(8, 24, bias=False)
        self.mlp.down_proj = torch.nn.Linear(24, 8, bias=False)


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([ProjectionBlock(), ProjectionBlock()])


def test_discover_groups_matches_official_llama_sharing() -> None:
    groups = discover_input_groups(ToyModel())
    assert len(groups) == 8
    attention = next(group for group in groups if group.target == "model.layers.0.self_attn.k_proj")
    assert attention == InputGroup(
        "model.layers.0.self_attn.k_proj",
        ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.v_proj"),
        8,
    )
    gate = next(group for group in groups if group.target == "model.layers.0.mlp.gate_proj")
    assert gate.shares == ("model.layers.0.mlp.up_proj",)


def test_config_plan_has_four_groups_per_layer() -> None:
    model_config = type("Config", (), {"num_hidden_layers": 3, "hidden_size": 8, "intermediate_size": 24})()
    groups = _groups_from_config(model_config)
    assert len(groups) == 12
    assert sum(len(group.all_layers) for group in groups) == 21


def test_fixed_protocol_validation() -> None:
    config = {
        "ranks": [8, 16, 32, 64],
        "sequence_length": 2048,
        "num_calibration_samples": 256,
        "calibration_batch_size": 4,
        "eval_batch_size": 4,
        "checkpoint_every_windows": 32,
        "quantization": {"name": "mxint", "width": 4, "block_size": 32, "block_axis": -1},
    }
    validate_config(config)
    config["ranks"] = [32]
    with pytest.raises(ValueError, match="fixed"):
        validate_config(config)


def test_qera_layer_config_is_fresh_and_exact() -> None:
    first = qera_layer_config(64)
    second = qera_layer_config(64)
    first["w_quantizer"].pop("name")
    assert second["w_quantizer"] == {"name": "mxint", "width": 4, "block_size": 32, "block_axis": -1}


def test_safe_name_is_path_safe() -> None:
    assert safe_name("model.layers.0.self_attn.q_proj") == "model__layers__0__self_attn__q_proj"


def test_jsonl_resume_discards_only_incomplete_tail(tmp_path: Path) -> None:
    checkpoint = tmp_path / "partial.jsonl"
    checkpoint.write_text('{"window": 0}\n{"window":', encoding="utf-8")
    assert _read_jsonl(checkpoint) == [{"window": 0}]
    assert checkpoint.read_text(encoding="utf-8") == '{"window": 0}\n'


def test_source_hash_normalizes_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.py"
    crlf = tmp_path / "crlf.py"
    lf.write_bytes(b"x = 1\ny = 2\n")
    crlf.write_bytes(b"x = 1\r\ny = 2\r\n")
    assert sha256_source_file(lf) == sha256_source_file(crlf)


def test_streaming_prefix_stops_at_requested_rows() -> None:
    stream = ({"text": f"row-{index}", "meta": {"index": index}} for index in range(10))
    rows, digest = _take_streaming_rows(stream, 4)
    assert [row["text"] for row in rows] == ["row-0", "row-1", "row-2", "row-3"]
    assert len(digest) == 64


def test_complex_sqrtm_matches_official_real_cast_and_normalization() -> None:
    root = np.array([[4.0 + 0.5j, 0.0], [0.0, 2.0 - 0.25j]], dtype=np.complex128)
    real_root, raw_imaginary, normalized_imaginary = _real_sqrtm_root(root, 4)
    np.testing.assert_allclose(real_root, np.diag([2.0, 1.0]))
    assert raw_imaginary == pytest.approx(0.5)
    assert normalized_imaginary == pytest.approx(0.25)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("chunk_tokens", [1, 7, 256, 2048])
def test_chunked_ce_matches_original_masked_window_nll(batch_size, chunk_tokens) -> None:
    generator = torch.Generator().manual_seed(42)
    logits = torch.randn(batch_size, 19, 257, generator=generator)
    ids = torch.randint(257, (batch_size, 19), generator=generator)
    mask = torch.ones_like(ids)
    mask[-1, -3:] = 0
    shifted = logits[:, :-1, :].contiguous()
    original = torch.nn.functional.cross_entropy(
        shifted.view(-1, 257), ids[:, 1:].contiguous().view(-1), reduction="none"
    ).view(batch_size, -1)
    actual = _chunked_window_nll(logits, ids, mask, chunk_tokens)
    for row, (nll, tokens) in enumerate(actual):
        valid = mask[row, 1:].bool()
        assert tokens == int(valid.sum())
        assert nll == pytest.approx(float(original[row][valid].sum()), rel=1e-6)


def test_dual_gpu_loading_does_not_mutate_experiment_plan_config(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    config = {"max_memory": {"0": "16GiB", "cpu": "900GiB"}, "eval_device_map": "auto"}
    original = {"max_memory": dict(config["max_memory"]), "eval_device_map": "auto"}
    loading = _evaluation_load_config(config, True)
    assert config == original
    assert loading["max_memory"] == {0: "10GiB", 1: "10GiB", "cpu": "180GiB"}
    assert loading["eval_device_map"] == "balanced"


def test_dual_gpu_requires_two_visible_gpus(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(RuntimeError, match="two visible"):
        _evaluation_load_config({}, True)


def test_peak_statistics_initialize_each_device_before_reset(monkeypatch) -> None:
    initialized = set()
    reset = []

    def allocate(*args, device, **kwargs):
        initialized.add(int(str(device).split(":")[-1]))
        return object()

    def reset_peak(device):
        if device not in initialized:
            raise RuntimeError("Invalid device argument")
        reset.append(device)

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peak)
    _reset_evaluation_memory_peaks([0, 1])
    assert reset == [0, 1]


def test_peak_statistics_on_real_cuda_devices() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA hardware is required for the allocator integration check")
    devices = list(range(torch.cuda.device_count()))
    _reset_evaluation_memory_peaks(devices)
    for device in devices:
        assert torch.cuda.max_memory_allocated(device) >= 0


def test_evaluation_resumes_partial_batch_then_skips_completed(tmp_path, monkeypatch) -> None:
    import qera_original_a_isolation.pipeline as pipeline

    model = torch.nn.Module()
    model.embedding = torch.nn.Embedding(17, 17)
    model.get_input_embeddings = lambda: model.embedding
    model.forward = lambda input_ids, **kwargs: SimpleNamespace(logits=model.embedding(input_ids))
    loads = []

    def load(*args):
        loads.append(args)
        return model

    monkeypatch.setattr(pipeline, "load_model", load)
    config = {"run_dir": str(tmp_path), "eval_dtype": "float32", "eval_batch_size": 4}
    windows = {"input_ids": torch.arange(35).reshape(5, 7) % 17, "attention_mask": torch.ones(5, 7)}
    path = tmp_path / "evaluation" / "per_window" / "BF16.jsonl"
    path.parent.mkdir(parents=True)
    with torch.inference_mode():
        expected = _chunked_window_nll(model.embedding(windows["input_ids"]), windows["input_ids"], windows["attention_mask"], 256)
    prefix = [dict(configuration="BF16", window=i, nll_sum=nll, tokens=tokens) for i, (nll, tokens) in enumerate(expected[:2])]
    path.write_text("".join(json.dumps(row) + "\n" for row in prefix), encoding="utf-8")
    records = _evaluate_one(config, "BF16", None, None, windows, batch_size=4, ce_chunk_tokens=3)
    assert [record["window"] for record in records] == list(range(5))
    for record, (nll, tokens) in zip(records, expected):
        assert record["nll_sum"] == pytest.approx(nll, rel=1e-6)
        assert record["tokens"] == tokens
    assert _evaluate_one(config, "BF16", None, None, windows) == records
    assert len(loads) == 1
    runtime = json.loads(next((tmp_path / "evaluation" / "runtime").glob("*.json")).read_text())
    assert runtime["start_window"] == 2
    assert runtime["batch_size"] == 4
