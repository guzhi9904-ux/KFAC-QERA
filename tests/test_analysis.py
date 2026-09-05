from pathlib import Path

from qera_exp.analysis import analyze_all
from qera_exp.utils import ensure_layout, save_csv


def test_analysis_creates_focused_plots_and_tables(tmp_path: Path) -> None:
    methods = ["AD_GI", "AD_GD", "AD_GF", "AF_GI", "AF_GD", "AF_GF"]
    suffixes = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    ]
    config = {
        "experiment": {"output_dir": str(tmp_path)},
        "statistics": {"methods": methods, "ranks": [8, 16]},
        "model": {"target_suffixes": suffixes},
        "analysis": {"chart_dpi": 72, "focused_axis_margin_fraction": 0.08, "y_limits": {"wikitext2": None, "c4": None}},
    }
    ensure_layout(tmp_path)
    for role in ("wikitext2", "c4"):
        rows = [
            {"dataset": role, "configuration": "BF16_TEACHER", "method": "BF16_TEACHER", "a_level": "-", "g_level": "-", "rank": 0, "aggregate_mean_nll": 2.0, "perplexity": 7.38},
            {"dataset": role, "configuration": "MXINT4_WQ", "method": "MXINT4_WQ", "a_level": "-", "g_level": "-", "rank": 0, "aggregate_mean_nll": 2.2, "perplexity": 9.03},
        ]
        for rank in (8, 16):
            for index, method in enumerate(methods):
                rows.append(
                    {
                        "dataset": role,
                        "configuration": f"{method}_R{rank}",
                        "method": method,
                        "a_level": "diag" if method.startswith("AD") else "full",
                        "g_level": method.split("_")[1],
                        "rank": rank,
                        "aggregate_mean_nll": 2.18 - rank / 1000 + index / 10000,
                        "perplexity": 8.85 - rank / 100 + index / 1000,
                    }
                )
        save_csv(tmp_path / "evaluation" / f"ppl_summary_{role}.csv", rows)
    energy = []
    for suffix in suffixes:
        projection = suffix.split(".")[-1]
        for method in methods:
            for rank in range(1, 17):
                energy.append(
                    {
                        "module": f"model.layers.0.{suffix}",
                        "layer": 0,
                        "projection": projection,
                        "method": method,
                        "rank": rank,
                        "captured_energy_fraction": rank / 16,
                    }
                )
    save_csv(tmp_path / "rank_energy.csv", energy)
    result = analyze_all(config)
    assert result["status"] == "PASS"
    assert (tmp_path / "analysis" / "figures" / "ppl_vs_rank_wikitext2_focused.png").is_file()
    assert (tmp_path / "analysis" / "figures" / "rank_energy_by_projection.png").is_file()
