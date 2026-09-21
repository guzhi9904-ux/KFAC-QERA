# Direct Query-Marginal K, dual4090

Implements protocol v2.1 in protocol.md. One approximate candidate, all four K
layers0/10/20/31, frozen N256×2048, one original sampled label sequence/window,
rank64 over the complete1024x4096 weight. Original teacher and native gradients
retain RoPE/GQA. Only proxy construction ignores relative RoPE; query q is the
actual PRE-ROTATION projection. Surrogate S is never required to equal true S.

Native signed softmax edge gradients use the accepted audit-v3 implementation
and its documented probability-mass/FP32-reassociation checks. No absolute e,
elementwise e² approximation, diagonal B, ALS, Full-fit, spectrum audit, or test.

For each window/head: B+=D.T@D; G_block+=q.T@q/d for all L positions, including
zero-e rows. A_sum=X.T@B@X. Raw A=A_sum/(NLHq), G=G_sum/(NT), Gbar=G_sum/(NLHq).
Store both G normalizations, counts and factor definitions. One full-rank64
solver call/module, inherited etaA=etaG=.001, deployment Wq+float32(C64).
Raw and regularized full-size factors, solver audits and four corrections saved.

Pilot windows0/1: actual GQA/RoPE/cached inputs, local delta vs connected full
backward, real K chain vs cache and full weight autograd, native softmax/mask,
explicit-vs-reassociated FP64 Gram on predeclared heads0/4 and queries0:16 (full
source context), and both internal surrogate score arrangements. Each pilot
window's FULL statistics are counted exactly once toward256. No shortened
context or extra performance candidate. No RoPE eigenspectrum objects.

Per-window atomic checkpoint of ALL four A/G accumulators. Only current and
previous generations retained. Parent assets stay read-only; needed X/down and
labels checked against immutable receipts/commits, only two-window K caches read.
Receipt signatures enable verified resume without rehashing all64GiB. Per-window
attention and D/effective inputs are not saved. Typical new output a fewGiB;
cap16GiB, free-disk headroom32GiB. Conservative8h active cap is an operational
safety bound, NOT an estimate; measured first-two-window estimate is recorded.

All four new candidates freeze before validation. Reuse320 old scores, calculate
64 new scores; same teacher reference per window, full forward evaluation, first
window20 old-candidate replays and4 new-candidate repeats. No cached-prefix risk.
Final report: absolute KL, per-layer recovery, gain vs Marginal in pp, relative
remaining-KL reduction, paired comparisons to all four baselines, exploratory
2000 paired-window bootstrap. It evaluates the complete candidate construction,
not a causal grouping-only ablation. No automatic follow-on experiments.

Run: `bash experiments/qer_k_direct_query_v1/run.sh`.
Default output `../qera_runs/k_direct_query_v1/run_01`.
Progress: DIRECT_QUERY_PILOT_PASSED, DIRECT_QUERY_STATISTICS_COMMITTED n /256,
CHECK2_ALL_CANDIDATES_FROZEN, VALIDATION_WINDOW_COMPLETE n /16.
Finish: DIRECT_QUERY_EXPERIMENT_COMPLETE then DIRECT_QUERY_EXIT 0.
Resume: same command after prior process exits; source/data/candidate changes fail.
