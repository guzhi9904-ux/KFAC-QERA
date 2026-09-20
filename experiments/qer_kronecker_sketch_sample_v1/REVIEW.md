# Review and fixed implementation decisions

The plan is feasible as a single-module, finite-sample comparison, subject to measured
resource acceptance. It distinguishes curvature approximation, rank-constrained proxy
optimization, empirical projected-gradient loss and actual teacher KL. These need not
agree. No universal advantage, population error bound or global ALS optimum follows.

1. With full summed-NLL gradients, S = gᵀx contains all cross-position terms.
   H_full uses 1/(N T), with T=L−1, whereas marginal input moments use 1/(D L).
   Canonical Marginal G uses 1/(N T), equivalently (L/T)G_m. Input statistics count
   each article once; S0/S1 share one physical A8 asset.
2. Token-joint's two contractions use the old A and old G synchronously, for exactly
   three rounds, initialized at A_m and I. It approximates the position-diagonal
   sketch. It is not sequential ALS and has no monotone-objective claim.
3. Sequence-one-step computes both blocks from identity independently, dividing by
   the opposite identity's squared Frobenius norm (m or n). It is not two ALS steps.
4. Full-fit alternates G then A, from canonical Marginal. Save every cycle, verify
   nonincreasing J after each block, gauge after complete updates; stop only after
   two consecutive cycles satisfy both registered criteria, or at 20 cycles. J omits
   ||H||², so it can be negative and is not a relative error norm.
5. Raw factors are symmetrized, checked PSD and gauge-normalized. Both relative
   dampings are applied once, then FP64 roots and dense SVD solve the rank-64 problem.
   No eigenvalue clipping or result-dependent damping is introduced.
6. **Deployment ambiguity resolved against the actual parent implementation:**
   the parent's dense intervention is `Wq + (P64 @ Q64).float()` in FP32. The new
   experiment preserves that exact order. Computing `P32 @ Q32` would produce another
   deployment and is not silently substituted. Record ideal R64 and actual deployed
   residual separately; statistical q/KL uses the latter, the rank-constrained bound
   uses the former. Proxy deployment drift is checked against 1e-4.
7. All 15 candidate files are frozen and checksummed before evaluation labels or
   scores. History includes the completed geometry run's 48 articles, parent fit,
   validation and calibration. Exclusion checks article title/body and exact
   contiguous 64-token spans; new article ordering and label seeds are deterministic.
   No held-out values select methods, rounds, damping, candidates or budget.
8. Curvature inner products are exact empirical contractions. Sample Gram B gives
   ||H||² without forming a (mn)×(mn) matrix. The solve-factor contraction uses an
   algebraically exact damping expansion, cross-checked against direct contraction
   in both small-matrix tests and the real fit-only pilot. s* only diagnoses scale;
   it does not rescale a correction or choose a candidate.
9. The excess bound combines proxy improvement and the Frobenius approximation
   error times residual norms. Check it against empirical ideal-residual excess loss.
   A large valid bound can be uninformative. A related motivation is the curvature
   approximation discussion in [Model-Preserving Adaptive Rounding](https://arxiv.org/html/2505.22988v2#S3.SS2);
   its rounding setting does not supply a performance guarantee for this rank-constrained
   SVD experiment.
10. Use one common paired article bootstrap for all contrasts and a separate
    within-article label bootstrap for q conditional on fixed articles. These are
    different uncertainty sources, not additive variances. No refitting uncertainty
    or familywise multiple-comparison control is claimed. S1 vs S2 is a budget
    allocation contrast, not a pure change of just one statistical component.

The CPU checks explicitly form H in tiny dimensions, compare contraction identities,
normalization, synchronous versus alternating updates, damping expansion, rank-SVD
optimality and the excess bound, and check shared bootstrap indices and historical
exclusion. An integration fixture uses real summed-cross-entropy autograd through a
small nonlinear causal network with all 224/256 samples and 15 candidates; it exercises
freeze, resume, 3,840 q values, 240 KL records, diagnostics and independent reporting.
These tests do not replace the real GPU pilot or constitute real experiment results.

Resource acceptance uses a 10-hour working cap from the user's remaining-ten-plus-hour
lease statement. Approximately 70 GiB denotes disk caches; intermediate factors add
space. Formal sampling begins only after the fit-only pilot passes and its reserved
estimate fits the cap. If not, preserve measurements and report the shortfall rather
than weakening the experiment. The user's current request authorizes execution; the
attachment's generic final instruction not to auto-launch a subsequent experiment is
retained as a prohibition on starting further experiments after this one.
