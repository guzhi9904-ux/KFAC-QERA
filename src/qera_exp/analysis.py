from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import output_root
from .utils import save_csv, save_json, utc_now


COLORS = {
    "AD_GI": "#4C78A8",
    "AD_GD": "#72B7B2",
    "AD_GF": "#54A24B",
    "AF_GI": "#F58518",
    "AF_GD": "#E45756",
    "AF_GF": "#B279A2",
}
MARKERS = {"GI": "o", "GD": "s", "GF": "^"}


def _focused_limits(frame: pd.DataFrame, config: Mapping[str, Any], role: str) -> tuple[float, float]:
    explicit = config.get("analysis", {}).get("y_limits", {}).get(role)
    if explicit:
        return float(explicit[0]), float(explicit[1])
    methods = set(config["statistics"]["methods"])
    selected = frame[frame["method"].isin(methods | {"MXINT4_WQ"})]["perplexity"].astype(float)
    if selected.empty:
        selected = frame["perplexity"].astype(float)
    lower, upper = float(selected.min()), float(selected.max())
    span = max(upper - lower, max(abs(lower), 1.0) * 0.005)
    margin = float(config.get("analysis", {}).get("focused_axis_margin_fraction", 0.08))
    return lower - margin * span, upper + margin * span


def plot_ppl(config: Mapping[str, Any], role: str) -> Path:
    root = output_root(config)
    source = root / "evaluation" / f"ppl_summary_{role}.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    ranks = [int(value) for value in config["statistics"]["ranks"]]
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    for method in config["statistics"]["methods"]:
        method_frame = frame[frame["method"] == method].sort_values("rank")
        g_level = str(method).split("_")[1]
        axis.plot(
            method_frame["rank"],
            method_frame["perplexity"],
            label=str(method),
            color=COLORS.get(str(method)),
            marker=MARKERS[g_level],
            linewidth=1.6,
            markersize=5,
        )
    for baseline, color, style in (("MXINT4_WQ", "#222222", "--"), ("BF16_TEACHER", "#999999", ":")):
        values = frame[frame["method"] == baseline]["perplexity"]
        if not values.empty:
            axis.axhline(float(values.iloc[0]), color=color, linestyle=style, linewidth=1.2, label=baseline)
    lower, upper = _focused_limits(frame, config, role)
    axis.set_ylim(lower, upper)
    axis.set_xticks(ranks)
    axis.set_xlabel("Rank")
    axis.set_ylabel("Perplexity")
    axis.set_title(f"{role}: PPL vs rank (focused y-axis)")
    axis.grid(True, alpha=0.22)
    axis.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    destination = root / "analysis" / "figures" / f"ppl_vs_rank_{role}_focused.png"
    fig.savefig(destination, dpi=int(config.get("analysis", {}).get("chart_dpi", 180)))
    plt.close(fig)
    return destination


def analyze_rank_energy(config: Mapping[str, Any]) -> tuple[Path, Path]:
    root = output_root(config)
    source = root / "rank_energy.csv"
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source)
    grouped = (
        frame.groupby(["projection", "method", "rank"], as_index=False)["captured_energy_fraction"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    csv_path = root / "analysis" / "rank_energy_by_projection.csv"
    save_csv(csv_path, grouped.to_dict(orient="records"))
    projections = [str(value).split(".")[-1] for value in config["model"]["target_suffixes"]]
    columns = 2
    rows = math.ceil(len(projections) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(11, 3.3 * rows), sharex=True, sharey=True)
    flat = np.atleast_1d(axes).reshape(-1)
    for axis, projection in zip(flat, projections):
        subset = grouped[grouped["projection"] == projection]
        for method in config["statistics"]["methods"]:
            values = subset[subset["method"] == method].sort_values("rank")
            axis.plot(values["rank"], values["mean"], color=COLORS.get(str(method)), linewidth=1.4, label=str(method))
        axis.set_title(projection)
        axis.set_ylim(0, 1.02)
        axis.grid(True, alpha=0.2)
    for axis in flat[len(projections) :]:
        axis.set_visible(False)
    for axis in flat[: len(projections)]:
        axis.set_xlabel("Rank")
        axis.set_ylabel("Mean captured energy")
    flat[0].legend(ncol=3, fontsize=7, frameon=False)
    fig.suptitle("Rank energy capture by projection")
    fig.tight_layout()
    figure_path = root / "analysis" / "figures" / "rank_energy_by_projection.png"
    fig.savefig(figure_path, dpi=int(config.get("analysis", {}).get("chart_dpi", 180)))
    plt.close(fig)
    return csv_path, figure_path


def factorial_contrasts(config: Mapping[str, Any], role: str) -> Path:
    root = output_root(config)
    frame = pd.read_csv(root / "evaluation" / f"ppl_summary_{role}.csv")
    frame = frame[frame["method"].isin(config["statistics"]["methods"])]
    rows: list[dict[str, Any]] = []
    for rank, group in frame.groupby("rank"):
        values = {row.method: float(row.aggregate_mean_nll) for row in group.itertuples()}
        required = {"AD_GI", "AD_GD", "AD_GF", "AF_GI", "AF_GD", "AF_GF"}
        if not required <= values.keys():
            continue
        rows.extend(
            [
                {"dataset": role, "rank": int(rank), "contrast": "A_full_minus_diag_at_GI", "delta_mean_nll": values["AF_GI"] - values["AD_GI"]},
                {"dataset": role, "rank": int(rank), "contrast": "A_full_minus_diag_at_GD", "delta_mean_nll": values["AF_GD"] - values["AD_GD"]},
                {"dataset": role, "rank": int(rank), "contrast": "A_full_minus_diag_at_GF", "delta_mean_nll": values["AF_GF"] - values["AD_GF"]},
                {"dataset": role, "rank": int(rank), "contrast": "G_full_minus_diag_at_AD", "delta_mean_nll": values["AD_GF"] - values["AD_GD"]},
                {"dataset": role, "rank": int(rank), "contrast": "G_full_minus_diag_at_AF", "delta_mean_nll": values["AF_GF"] - values["AF_GD"]},
            ]
        )
    destination = root / "analysis" / f"factorial_contrasts_{role}.csv"
    save_csv(destination, rows)
    return destination


def analyze_all(config: Mapping[str, Any]) -> dict[str, Any]:
    root = output_root(config)
    figures = []
    contrasts = []
    for role in ("wikitext2", "c4"):
        figures.append(str(plot_ppl(config, role)))
        contrasts.append(str(factorial_contrasts(config, role)))
    energy_csv, energy_figure = analyze_rank_energy(config)
    figures.append(str(energy_figure))
    result = {
        "status": "PASS",
        "completed_at_utc": utc_now(),
        "figures": figures,
        "factorial_contrasts": contrasts,
        "rank_energy_summary": str(energy_csv),
    }
    save_json(root / "state" / "analysis_complete.json", result)
    return result
