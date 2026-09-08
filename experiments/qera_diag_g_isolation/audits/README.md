# Identity-G numerical audit

This standalone diagnostic investigates the first-layer identity regression
without changing the production solver or invalidating its existing manifest.
It is an investigation, **not a fix or a gate override**. It does not restart G
collection, run evaluation, change tolerances, or generate replacement corrections.

The original run has already retained its complete 256-window raw G snapshot.
The audit verifies that checkpoint, the existing manifest's pinned code, and the
selected weight shard, A roots, frozen Wq, and historical GI factors. It runs under
the same output-directory lock as the pipeline, so they cannot execute together.
Only a uniquely named `diagnostics/identity_g_*/report.json` is written; partial
reports are saved between variants and a Python exception is recorded if possible.
The existing `.run.lock` is used. Original source artifacts remain read-only.

## Server command

In the **existing** `KFAC-QERA-98ad0c5` code directory, with the existing
`qera-original-a` environment active, update only the newly added audit files
(a normal fast-forward Git update also works if this directory is a clean checkout):

```bash
DIAG_G_RUN=/share/home/tm902089733300000/a913520780/chengkang/qera_runs/llama3.1-8b-diag-g-256
nohup bash experiments/qera_diag_g_isolation/scripts/audit_identity_g.sh >> "$DIAG_G_RUN/identity_audit.log" 2>&1 &
echo $!
tail -n 50 -F "$DIAG_G_RUN/identity_audit.log"
```

Default: `model.layers.0.self_attn.k_proj`, `diag`, `cuda:0`, ranks 8/16/32/64.
Use `--method full` or `--layer model.layers.0.self_attn.q_proj` for a separate
diagnostic later. `--skip-fp64-reference` skips only the slower optional reference;
the normal default includes it. This is a handful of one-layer SVDs, not a model
calibration pass. Runtime is hardware-dependent; 30-second heartbeats cover SVDs.
No dependencies are installed. The CLI enforces torch 2.3.0 / transformers 4.44.2.

The final log prints the unique report path. Send the report along with the log
for interpretation. `AUDIT_COMPLETE` means the diagnostic ran, **not** that the
production identity gate passed. Do not restart the pipeline until the discrepancy
has been explained; an unchanged solver can fail at the same gate again.

## What the report distinguishes

1. **Input identity:** freshly run official quantization versus frozen Wq, old/new
   quantization errors, and exact/relative differences between weighted SVD inputs.
   Shapes, dtypes and strides are recorded. Different strides alone are not proof
   of a numerical problem.
2. **Historical replay:** the actual pinned official QERA function versus saved
   GI factors; the actual current solver with G=I versus those same saved factors.
3. **Controlled SVD comparisons:** old/new input construction crossed with
   full/reduced SVD. All four variants use identical inverse and right-factor
   construction. Right-factor matrix multiplication versus elementwise scaling
   is additionally compared using the same singular vectors.
4. **FP64 reference:** rebuild the weighted matrix in FP64 and use CUDA `gesvd`.
   This is a diagnostic reference, not a replacement for the declared FP32 solver
   and not a guarantee of exact arithmetic. Its changes in both precision and
   driver are explicitly reported, so it cannot by itself attribute the cause.
5. **Interpretation evidence:** product discrepancies at each rank, weighted SSE,
   singular values and gaps at each truncation boundary, and scale range. A small
   spectral gap may motivate a sensitivity check; SVD sign flips alone cannot
   explain product drift. Similar objectives do not automatically waive the gate.

`vs_saved_production_gate` reproduces the existing product comparison arithmetic
for FP32 candidates, retaining the historical factor dtypes. The independent
`*_fp64_products` comparisons cast factors **before** multiplication. Both are
included to distinguish comparison rounding from solver discrepancies.

TF32 is disabled to match the failed new run. The process's inherited TF32 flags
are recorded but are **not** evidence of the historical run's flags. The first
audit does not intentionally enable lower-precision arithmetic to make a gate pass.

No automatic root-cause verdict or manifest migration is made. If official replay
matches history but the current solver does not, focus on the implementation
comparisons. If official replay also differs, inspect historical runtime and
artifact provenance before selecting any change.

## Local verification

```bash
python -m pytest experiments/qera_diag_g_isolation/audits/test_identity_g.py experiments/qera_diag_g_isolation/tests -o addopts='' -q
python experiments/qera_diag_g_isolation/audits/identity_g.py --help
bash -n experiments/qera_diag_g_isolation/scripts/audit_identity_g.sh
```

CPU tests cover comparison arithmetic, spectral/objective diagnostics, and a
synthetic end-to-end audit with immutable input files. They cannot reproduce the
server's Llama/CUDA discrepancy. Existing production `.py` files and configuration
are deliberately unchanged; audit Python files live in this subdirectory, outside
the production manifest's direct `*.py` code enumeration.
