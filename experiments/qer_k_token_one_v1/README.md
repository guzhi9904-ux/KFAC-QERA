# K Token-joint-one, dual4090

Frozen protocol v1.0 is in protocol.md. Exactly one new candidate per K at
layers0/10/20/31: original N256/L2048/T2047, sampled sum-NLL labels, rank64.
Original initialization A0=A_M (raw, undamped), G0=I1024. A1=U/(NTm) reuses
Sensitivity-A's full U; G1=sum((x.T A_M x) g g.T)/(NT ||A_M||F²) is the only new
full N256 accumulator. No using A1 in G update; no identity A0 or second round.

Parents KO/base and Sensitivity identities, frozen code, data/candidate records
and receipts are checked. The old Token3 factors must carry three synchronous
iterations and bind to the exact deployed candidate. U/D/A_sens, full canonical
G and original A_M are cross-checked. Existing per-module Sensitivity cache
evidence is reused only after current file signatures, receipts and commits
match. Streaming reads still check tensor bindings, dtype/shape/finiteness and
sample/label/x hashes. No parent writes or forty-GiB copy. Missing assets are
reported, never trigger automatic teacher-gradient recollection.

Pilot first uses L20 windows0/1, then all layers. Full raw4096x4096 A_M and full
windows/channels are used for the original parent token_step reference (one call
over two windows). Fixed first16 rows also check explicit quadratic forms and
weighted outer products. FP64 tolerance1e-10, no clamping negative weights.
Only temporary pilot A is calculated; formal A always comes from parent U.
The pilot's first two G contributions commit together and count exactly once.

Two offline device workers. Current+previous V checkpoints retained every8
windows (and first2); original sample order retained. Final input-sum and trace
identities checked against A_M. Raw A1/G1/V, gauge factors, regularized factors,
scales and hashes saved. Parent solver is called after the first-round gauge,
matching Token3's existing path including the solver's near-unit internal gauge.
Full FP64 roots/dense SVD, etaA=etaG=.001, rank64, Wq+float32(C64) deployment.

Teacher loads only after four candidates freeze. Reuse320 scores:
Marginal/Sensitivity-A/Token3/Sequence/None; new64 main KL points on same16
validation windows. First-window20 old replays and4 new repeats, unchanged
tolerances. No A-only/Direct Query evaluation, test, PPL, alpha sweep or new fit.

Reports contain absolute KL/recovery, Joint1−SensA, Joint1−Marginal and
Token3−Joint1 with explicit sign conventions, per-window paired differences and
2000 exploratory bootstrap replicates. All layers retained, no equivalence
claim merely from a confidence interval crossing zero. Timings are device-call
costs plus wall time; overlapping calls must not be summed as wall time.

Safety bounds: four active hours,8GiB new output, existing32GiB free-disk reserve.
These are operational caps, not duration estimates; no automatic deletions of
parent assets. All required work resumes with unchanged identity.

Run/resume: `bash experiments/qer_k_token_one_v1/run.sh`.
Default output: `../qera_runs/k_token_one_v1/run_01`.
Success: TOKEN_ONE_EXPERIMENT_COMPLETE then TOKEN_ONE_EXIT 0.
