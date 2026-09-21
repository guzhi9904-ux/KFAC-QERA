# K structure audit A/B, dual4090

Implements only stages A/B of protocol.md. No new Query-aware method, no N256
fit, no new validation/test forward and no source/query mixing. Source run and
implementation identities remain frozen and read-only.

Layers0,10,20,31 K; fixed train windows0,32,64,96,128,160,192,224; one existing
sampled label sequence, sum-NLL, L2048/T2047. Existing five actual deployment
residuals W0.double()-Wdeploy.double(), including Wq baseline. Exact same
teacher/tokenizer/labels, FP32 native teacher, FP64 diagnostic contractions.

Reuses only required X/down/K/labels/factors/deployments and 320 original K
validation score records. Missing caches cause an explicit stop, not a silent
N256 recollection. Required assets are checked against receipts and commits.

Captures actual native q/q_rotated/k/k_rotated/cos/sin, probabilities and residual
inputs in one teacher forward/window. Uses existing down adjoint and local MLP
VJP for delta. Native torch softmax backward computes signed e from native
FP32 delta and V; p*(delta.V-delta.z), row sums, causal mask and last-position
zero gradients are checked. The RoPE transpose is y*c-J(y*s); it does not assume
the floating rotation inverse or regenerate position frequencies.

Two pilot windows share one connected full backward across all four K weights,
and verify cached gradients, score/autograd identity and full64 reordering.
No gate is weakened on failure. Remaining six windows start only after all eight
pilot layer commits pass. Pure64 cancellation safety is registered in PLAN;
nondegenerate comparisons retain relative1e-10, native comparisons1e-5.

Source/query diagnostics use the same FP64 edge terms, combining heads before
squaring. Cached-native source contractions are also saved for inspection.
Each window contributes scores normalized by1/(2*8*2047). Final report sums
those contributions. Five Rx are precomputed before the head loop. M is never
materialized at all positions; only specified 512 objects are processed.

8192 diagnostic F spectra via128x128 Gram eigensystems; no-RoPE algebraic rank1
control, 512 post-X objects for k1/4 and all five residual projections. Zero
energy is NA, not rho1=1. Reports energy weighting, quantiles, signed group
cross terms, paired residual score differences, and old validation KL separately.

Model stays sharded across two GPUs; head processing is serial on each owning
device to preserve VRAM headroom. No duplicate FP32 models or large persistent
attention cache. Output cap8GiB, cumulative active cap2h, atomic per-layer/window
commits and process lock. The largest uncommitted unit is one layer/window;
resuming may repeat its work, but never double-counts committed probes.

Run: `bash experiments/qer_k_structure_audit_v2/run.sh`.
Default independent output: `../qera_runs/k_structure_audit_v2/run_01`.
Finish: `K_STRUCTURE_AUDIT_AB_COMPLETE`, then `K_AUDIT_EXIT 0`.
Report: READOUT.md. Pilot measured estimate: resources/pilot_estimate.json.
