# Kronecker sketch / sample-budget experiment

One Llama-3.1-8B module (`model.layers.31.self_attn.q_proj`), fixed MXINT3 weight,
rank 64, FP32 teacher and FP64 statistics. No training, amplitude search or PPL.
The unchanged requested specification is in [protocol.md](protocol.md).
Implementation decisions and review are in [REVIEW.md](REVIEW.md).

## What is compared

Three nested budgets: S0 = 8 original articles × 4 saved labels; S1 = the same
8 articles × 16 labels; S2 = 32 articles × 4 labels. There are 224 distinct fit
gradients, not 288. S0 and S1 use the very same saved A8 input-statistics asset.
Each budget produces Marginal, three synchronous Token-joint rounds, independent
Sequence-one-step, and alternating Full-fit (at most 20 complete cycles).
Together with A8-only, A32-only and None this gives 15 deployments.

All candidates freeze before 16 new evaluation articles × 16 labels are sampled.
One backward pass per label is shared by all candidates: 256 evaluation backward
passes, 3,840 q scores, 240 main actual-KL values. Extra forward passes check every
candidate's repeated/direct/output-perturbation path on the first evaluation article.
Prior geometry's 48 articles are explicitly excluded in addition to parent history.

Reports include paired article and conditional-label bootstrap intervals, all nine
primary comparisons, equal-gradient budget contrasts, exact finite-sample curvature
cosines, damping effects and ideal-residual excess-loss bounds. Bounds are not claims
about population curvature or full-model downstream accuracy.

## Resources and safety

The approximately 70 GiB cache is **disk storage**, not VRAM or a requirement to
hold all samples in RAM: 28 GiB fit S, 32 GiB evaluation S, 7 GiB fit g and about
1.5 GiB input activations. Factors, corrections and iteration checkpoints need more.
Require 128 GiB free at entry, cap output at 112 GiB and retain 12 GiB free headroom.
CPU tensor LRU defaults to 8 GiB; Gram matrices stream blocks of at most 8 samples.
Teacher is unloaded before dense factor fitting. Do not run another GPU worker
alongside this model-parallel process. Two 24 GiB RTX4090s are required.

The requested lease was described as “十几个小时”. The registered working assumption
is at most **10 cumulative active hours**, including prepare/pilot/retries. Pilot
uses only two original fit labels, with caches reused by formal execution. It measures
all major stages and freezes a worst-case 20-cycle estimate with 35% safety and 10 min
overhead. Formal execution refuses an estimate exceeding the budget; it never silently
reduces labels, methods, precision or articles. This active-time cap is not a guarantee
of rental expiry: check remaining lease time before starting a delayed run.

## Commands on the existing dual4090 server

From `KFAC-QERA-teacher-kl`, activate the existing `qera-original-a` environment.
All paths below are examples for the already authenticated migrated parents.

```bash
python -B experiments/qer_kronecker_sketch_sample_v1/test_math.py
python -B experiments/qer_kronecker_sketch_sample_v1/test_pipeline.py

python -B experiments/qer_kronecker_sketch_sample_v1/configure.py \
  --from-geometry-config ../qer_geometry_v1.json \
  --geometry-history ../qera_runs/geometry_search_v1/run_01 \
  --output-parent ../qera_runs/kronecker_sketch_sample_v1 \
  --hours 10 --output ../qer_ksample_v1.json

CUDA_VISIBLE_DEVICES=0,1 bash experiments/qer_kronecker_sketch_sample_v1/run.sh \
  ../qer_ksample_v1.json ../qera_runs/kronecker_sketch_sample_v1/run_01 prepare
CUDA_VISIBLE_DEVICES=0,1 bash experiments/qer_kronecker_sketch_sample_v1/run.sh \
  ../qer_ksample_v1.json ../qera_runs/kronecker_sketch_sample_v1/run_01 pilot
cat ../qera_runs/kronecker_sketch_sample_v1/run_01/budget_freeze.json

# Only after pilot passed and fits_budget is true:
CUDA_VISIBLE_DEVICES=0,1 bash experiments/qer_kronecker_sketch_sample_v1/run.sh \
  ../qer_ksample_v1.json ../qera_runs/kronecker_sketch_sample_v1/run_01 run
```

Use `tmux` or your existing terminal supervisor for a long run. Output is unbuffered:
`START`, `COST`, `FIT_COMMITTED`, `EVAL_WINDOW_COMPLETE` and final status identify
the current stage. `resource_usage.json` records stage times and observed peaks.
Resume by repeating the same stage with identical code/config and the same run folder;
checksummed commits are reused. Code/config changes require a new, explicitly versioned
run, not editing frozen files. Never delete prior results to make a resume succeed.

CPU report regeneration (does not start a teacher):

```bash
bash experiments/qer_kronecker_sketch_sample_v1/run.sh \
  ../qer_ksample_v1.json ../qera_runs/kronecker_sketch_sample_v1/run_01 report
```

Completion requires `status.json: COMPLETE` and `verification.json: passed`.
Read `READOUT.md`, `RESULTS.md`, `summary/*.csv` and `summary/statistics.json`.
An incomplete run is explicitly reported as incomplete; missing values are never zeros.
No new experiments, remote login, lease renewal or administrator action is automated.
