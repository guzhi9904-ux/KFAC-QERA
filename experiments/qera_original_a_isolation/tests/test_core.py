from __future__ import annotations

import sys
from pathlib import Path

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
from qera_original_a_isolation.pipeline import _groups_from_config, _read_jsonl  # noqa: E402


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
