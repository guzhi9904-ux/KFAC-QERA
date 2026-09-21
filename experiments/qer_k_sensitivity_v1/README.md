# K Sensitivity-weighted Marginal, dual4090

Implements the frozen v1.0 protocol in protocol.md. L0/L10/L20/L31 K only,
N256 x2048, original one sampled sum-NLL label sequence/window, full rank64.
Only A changes: w=||native full1024-dimensional K-gradient||²,
A=sum(w xx.T)/sum(w) with a single global normalization. G is the exact original
full1024x1024 G_canonical, verified against old Marginal gauge/correction hashes.
FP64 weights, accumulation, matrix roots and SVD; inherited etaA=etaG=.001.
Deployment uses Wq+float32(C64), with C64 formed in FP64.

No teacher load or autograd during statistics. One streaming pass loads and
validates each X/g pair against receipts, tensor/file hashes, samples, labels,
x_hash and original commits, then accumulates U,D. Required256 samples are never
silently reduced. Missing/invalid modules are reported individually; remaining
offline modules can finish, but four-layer evaluation/completion is blocked.
No automatic recollection or parent writes. G must have the canonical source
verified present by this deployment's resource preflight; missing G is reported.

Pilot first two frozen windows: L20 first, then remaining modules on two device
workers. Fixed first16 token/input-feature slice, full gradient norm. Explicit
weighted outer products, sqrt implementation, actual parent token_step with I,
and constant-weight control checked at FP64 tolerance1e-10. Pilot full-window
statistics count exactly once. Final D/(256*2047)=trace(parent G) is mandatory.

Two offline workers, one per GPU. Each module keeps current+previous atomic U,D
checkpoints (windows1,2 then every8); resume verifies checkpoint and all already
consumed parent file signatures/receipt/commit hashes. Uncommitted windows are
recomputed. Final U,D,A,G, per-window concentration, raw/regularized factors and
all four candidates remain inspectable. No forty-GiB extra copy.

Four candidates freeze before validation. Reuse320 old KL points, calculate64
new main points on the same16 windows, plus20 old replays and4 new repeats.
Teacher loaded only here. Same self-KL/replay tolerances; no TF32/autocast, no
test/PPL, ALS, Full-fit, head splitting, clipping, new labels or layer selector.
Direct Query history is optional_not_used; no dependency on its code or audit
completion. Original attention/RoPE audit is not rerun or imported.

Reports preserve every layer: absolute KL/recovery, differences to all four
baselines, paired2000-window bootstrap, weight effective count (not independent
sample size) and stage timings. Four-hour active cap is a safety bound, not ETA.
New-output cap8GiB; preserve inherited32GiB shared-storage recovery headroom.

Run/resume: `bash experiments/qer_k_sensitivity_v1/run.sh`.
Output: `../qera_runs/k_sensitivity_v1/run_01`.
Log: `<run>/logs/run.log` when launched with background redirection.
Success: `SENSITIVITY_EXPERIMENT_COMPLETE` then `SENSITIVITY_EXIT 0`.
