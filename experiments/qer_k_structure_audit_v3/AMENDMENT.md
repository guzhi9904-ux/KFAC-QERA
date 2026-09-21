# Implementation correction v2: native softmax row identity

Original experiment protocol.md and v1 source/run remain unchanged. New v2 run
repeats the same two-window/four-layer full-autograd acceptance before proceeding
to the same eight-window A/B scope. No candidate, data, precision or label changes.

Observed failure: window32/L20/query-head25/row902. Native probability mass
0.9999990891178143; mu=sum(p*dp)=-0.03703952146144223. Measured sum(e)
-4.611937343135253e-8 exceeded the original zero-centered bound3.8344930839343464e-8.
For rounded native p the exact algebraic row sum is mu*(1-sum(p)), namely
-3.3738640265283254e-8. Remaining kernel rounding residual is
-1.2380733166069274e-8, below the original bound. Native backward vs FP64 formula
relative Frobenius error1.0457798331797793e-7. This is not evidence of a wrong
K gradient or a failure of the mathematical softmax identity for normalized p.

Correction: test abs(sum(e)-mu*(1-sum(p))) against the UNCHANGED bound
8*eps32*sum_FP64(p*abs(dp))+1e-20. Check finite/nonnegative probabilities and
probability mass error<=1e-5 independently. Save original row sums, predicted
rounding bias, residual, worst-row bound ratio, and number of old-check failures.
Keep native p/e intact: no renormalization, no edge centering, no gradient changes.
The existing centered-formula1e-5, full K/autograd1e-5, cache identity1e-5 and
pure FP64 reordering1e-10 gates are all unchanged.

Regression tests exercise rounded probability mass, true gradient corruption and
invalid probabilities. The exact failing native tensors are retained on server
in qera_runs/k_softmax_diagnosis_v1/head25.safetensors for a CUDA regression check.
