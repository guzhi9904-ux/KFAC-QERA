"""Protocol-prespecified classification; no tuning against MC agreement."""
from pathlib import Path
import math

import math_ops as mo
from storage import save_json,save_csv,atomic_bytes,read_json


def write_report(experiment):
    from run import MODULES,DIRECTIONS,slug
    root=experiment.root
    rows=[];mc_flat=[];kl_flat=[];convergence=[]
    for name in MODULES:
        records=experiment.mc_records(name)
        if len(set(r["window"] for r in records))!=8:raise RuntimeError("Incomplete MC windows")
        for record in records:
            for direction,values in record["directions"].items():
                mc_flat.append({k:v for k,v in record.items() if k!="directions"}|{"direction":direction,**values})
        for d in DIRECTIONS:
            stat=mo.monte_carlo(records,d)
            if stat["K"] not in (16,64):raise RuntimeError("Incomplete formal K")
            for k in (4,16,64):
                if k<=stat["K"]:convergence.append({"module":name,"direction":d,**mo.monte_carlo([r for r in records if r["replicate"]<k],d)})
            points=experiment.pooled_points(name,d);selected=mo.plateau(points)
            if not {0.05,0.1,0.2}.issubset({p["alpha"] for p in points}):raise RuntimeError("Initial KL grid incomplete")
            if selected is None:
                state="NUMERICAL_INVALID" if not any(p["valid"] for p in points) else "LOCAL_RANGE_UNRESOLVED"
                delta=None
            else:
                delta=abs(stat["q_hat"]-selected["q_KL"])/selected["q_KL"]
                state=("MC_INCONCLUSIVE" if stat["relative_halfwidth"]>.20 else
                       "PASS_LOCAL_ALIGNMENT" if delta<=.20 else "MISMATCH_TO_INVESTIGATE")
            row={"module":name,"direction":d,**stat,"local_alphas":selected["alphas"] if selected else [],
                 "q_KL":selected["q_KL"] if selected else None,"relative_alignment_error":delta,
                 "log_slope":selected["log_slope"] if selected else None,
                 "kappa_min":selected["kappa_min"] if selected else None,
                 "kappa_max":selected["kappa_max"] if selected else None,"status":state}
            rows.append(row)
    for path in sorted((root/"records/kl").glob("*.json")):
        r=experiment.read_record(path)
        kl_flat.append({k:v for k,v in r.items() if k not in ("weight_audit","output_audit")}
                       |{"weight_"+k:v for k,v in r["weight_audit"].items()}
                       |{"output_"+k:v for k,v in r["output_audit"].items()})
    save_csv(root/"mc_projection.csv",mc_flat);save_csv(root/"kl_path.csv",kl_flat)
    save_csv(root/"summary.csv",rows);save_csv(root/"mc_convergence.csv",convergence)
    save_json(root/"summary.json",rows)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(2,3,figsize=(14,7),constrained_layout=True)
    for row,ax in zip(rows,axes.flat):
        points=experiment.pooled_points(row["module"],row["direction"])
        for valid,marker,color in ((True,"o","#245ba7"),(False,"x","#b53b39")):
            subset=[p for p in points if p["valid"]==valid]
            if subset:ax.plot([p["alpha"] for p in subset],[p["KL_mean"]/p["alpha"]**2 for p in subset],marker=marker,color=color,linestyle="none" if not valid else "-",label="KL/alpha²" if valid else "invalid")
        ax.axhline(row["q_hat"],color="#b76a16",label=f"MC K={row['K']}")
        ax.axhspan(row["ci_low"],row["ci_high"],color="#b76a16",alpha=.16,label="MC approximate 95% interval")
        ax.set_xscale("log");ax.set_xlabel("alpha");ax.set_ylabel("Directional curvature")
        ax.set_title(row["module"].replace("model.layers.","L")+"\n"+row["direction"],fontsize=10)
        ax.grid(alpha=.2);ax.legend(fontsize=7)
    (root/"figures").mkdir(exist_ok=True)
    figure.savefig(root/"figures/curvature_alignment.png",dpi=180)
    figure.savefig(root/"figures/curvature_alignment.pdf");plt.close(figure)
    count=sum(r["status"]=="PASS_LOCAL_ALIGNMENT" for r in rows)
    text=["# QER teacher KL experiment 01 v2", "",f"Collection complete. {count}/6 directions passed the prespecified local alignment test.","",
          "The intervals condition on the eight fixed validation windows and reflect teacher-label Monte Carlo noise only. They are approximate intervals with adaptive K; no generalization or simultaneous coverage claim is made.","",
          "| Module | Direction | K | MC q | MC relative halfwidth | KL q | Relative difference | Status |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        def fmt(v):return "unresolved" if v is None else f"{v:.6g}"
        text.append(f"| {r['module']} | {r['direction']} | {r['K']} | {fmt(r['q_hat'])} | {fmt(r['relative_halfwidth'])} | {fmt(r['q_KL'])} | {fmt(r['relative_alignment_error'])} | {r['status']} |")
    text += ["","A numerical audit: see `a_audit.csv`, raw accumulators in `input_stats`, and decompositions in `solve_metrics`. Cholesky and symmetric-root objectives use the identical regularized metric. Regularization was selected before examining KL.","",
             "![Six direction alignment](figures/curvature_alignment.png)","",
             "K convergence: `mc_convergence.csv`. Every record retains signed directional projection, normalization, label hash and frozen direction hash. Invalid perturbation points remain in `kl_path.csv` and are excluded from plateau selection.","",
             f"Measured stage time (including setup and audits): {sum(r['seconds'] for r in experiment.resources):.1f} seconds. Peak RSS is a process-lifetime high-water mark, not a per-stage increment. GPU allocated and reserved peaks, UUIDs and recomputation settings are in `resource_usage.csv`.","",
             "Only two selected modules, six fixed directions, and eight same-input validation windows are covered. No full-model PPL, downstream evaluation, complete G collection, or superiority claim for A-weighted compensation is made."]
    atomic_bytes(root/"report.md",("\n".join(text)+"\n").encode())
    experiment.status("COLLECTION_COMPLETE",directions=6,passed=count,
                      scientific_states={r["module"]+":"+r["direction"]:r["status"] for r in rows})
