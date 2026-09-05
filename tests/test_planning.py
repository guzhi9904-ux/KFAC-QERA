import json
from pathlib import Path

from qera_exp.statistics import make_shard_plan


def test_shard_plan_reports_retained_raw_disk_estimate(tmp_path: Path) -> None:
    modules = [
        {
            "module": "model.layers.0.self_attn.q_proj",
            "estimated_dense_accumulator_bytes": 400,
            "estimated_raw_statistics_bytes": 500,
            "weight_parameters": 25,
        },
        {
            "module": "model.layers.0.self_attn.k_proj",
            "estimated_dense_accumulator_bytes": 300,
            "estimated_raw_statistics_bytes": 380,
            "weight_parameters": 20,
        },
    ]
    (tmp_path / "state").mkdir()
    (tmp_path / "module_manifest.json").write_text(json.dumps({"modules": modules}), encoding="utf-8")
    config = {
        "experiment": {"output_dir": str(tmp_path)},
        "runtime": {"max_dense_ram_gib_per_shard": 1e-6, "cleanup_raw_after_solve": False},
    }
    plan = make_shard_plan(config)
    assert plan["raw_statistics_retained_after_solve"] is True
    assert plan["estimated_total_raw_statistics_bytes"] == 880
    assert plan["estimated_peak_raw_statistics_bytes_per_shard"] == 880
    assert plan["estimated_retained_raw_statistics_bytes"] == 880
    assert sum(shard["estimated_raw_statistics_bytes"] for shard in plan["shards"]) == 880

    config["runtime"]["cleanup_raw_after_solve"] = True
    cleanup_plan = make_shard_plan(config)
    assert cleanup_plan["raw_statistics_retained_after_solve"] is False
    assert cleanup_plan["estimated_total_raw_statistics_bytes"] == 880
    assert cleanup_plan["estimated_peak_raw_statistics_bytes_per_shard"] == 880
    assert cleanup_plan["estimated_retained_raw_statistics_bytes"] == 0
