# v3: separate FP32 execution paths from FP64 algebra

v1/v2 sources and failed runs remain unchanged. This amendment supersedes v2's
claim that comparing the centered FP32 expression to native backward is the
appropriate algebraic gate. It preserves the original scope and gradients.

Reproduced window32/L31/head14 with original tensors:
- FP32 p*(dp-delta.z_forward) vs native backward: relative1.1628256901023669e-5.
- FP32 p*(dp-sum(p*dp)) vs native backward: relative5.907041237205919e-6.
- Same original P, delta, V, with both centered/direct contractions in FP64:
  relative1.723538389079119e-14.
- Native backward vs direct FP64 formula: relative3.879196561426921e-6.
- Native forward z vs FP64 P@V: relative2.811486013421259e-8.

Cause: z_forward=P@V and dp=delta@V.T follow distinct FP32 GEMM paths.
Subtracting similar terms amplifies their small rounding differences. The
centered formula is algebraically correct; expecting its independent FP32
execution to be the native backward reference confounds two checks.

v3 checks both algebraic expressions using the SAME original P, delta, V in
FP64 at1e-10. It independently checks native e, dp and z vs the FP64 reference
at1e-5. Original FP32-centered discrepancies are retained per head as observations.
Native e is still used unchanged by all K/source/query diagnostics. No P
renormalization, edge recentering, new labels or precision changes to teacher.
Full K/cache/autograd1e-5 and FP64 source/query1e-10 remain unchanged; the v2
probability-mass row-identity correction remains in force.

New independent run repeats all eight pilot layer checks before stage B.
