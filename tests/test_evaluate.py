from pathlib import Path

import torch

import qera_exp.evaluate as evaluation
from qera_exp.quantization import quant_paths
from qera_exp.utils import atomic_safetensors, read_csv, save_json, sha256_file, tensor_sha256


def test_evaluate_configuration_reports_new_and_cached_window_progress(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_nll_sum(*_args):
        calls.append(True)
        return 4.0, 2

    monkeypatch.setattr(evaluation, "nll_sum", fake_nll_sum)
    config = {"experiment": {"output_dir": str(tmp_path)}}
    row = {
        "configuration": "BF16_TEACHER",
        "method": "BF16_TEACHER",
        "a_level": "-",
        "g_level": "-",
        "rank": 0,
    }
    windows = [{"window_id": 0, "window_hash": "window-hash"}]
    metrics_path = tmp_path / "evaluation" / "wikitext2_per_window.csv"

    for _ in range(2):
        evaluation._evaluate_configuration(
            torch.nn.Identity(),
            config,
            "wikitext2",
            row,
            windows,
            torch.device("cpu"),
            metrics_path,
            "model-hash",
        )

    assert len(calls) == 1
    assert read_csv(metrics_path)[0]["complete"] == "True"


def test_install_quantized_weights_reports_progress(tmp_path: Path) -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)

    model = Model()
    reference = model.linear.weight.detach().cpu().to(torch.bfloat16).contiguous()
    quantized = torch.zeros_like(reference)
    artifact, metadata_path = quant_paths(tmp_path, "linear")
    artifact.parent.mkdir(parents=True)
    atomic_safetensors(artifact, {"wq_bf16": quantized})
    save_json(
        metadata_path,
        {
            "artifact_sha256": sha256_file(artifact),
            "reference_weight_bf16_sha256": tensor_sha256(reference),
        },
    )

    joint_hash = evaluation.install_quantized_weights(model, tmp_path, ["linear"])

    assert joint_hash
    assert torch.equal(model.linear.weight.detach().cpu(), quantized)
