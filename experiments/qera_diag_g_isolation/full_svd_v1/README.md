# Full-SVD continuation of the completed G256 collection

The 2026-09-08 first-layer audit found identical old/new SVD inputs, exact replay
of historical GI factors with the actual official function and full SVD, and
repeatable ~0.27–0.33% product drift with reduced SVD. This isolates SVD mode as
the trigger for that layer under the audited environment. It does not establish
which result is more accurate, why cuSOLVER differs internally, or that all layers
will pass. The FP64 reference also differs from history; objectives and spectral
gaps, rather than distance to history alone, are needed to interpret accuracy.

This version changes **only `full_matrices=False` to `full_matrices=True`** in the
numerical solver. Both the GI regression and the GD solve use this full-SVD
function, with the same FP32 dtype, default SVD driver, inverse callback, G floor,
and objective checks. All four identity gates retain `1e-3`, all ten PPL controls
retain `0.01`, and the usual 18 evaluations follow solve. No unconditional identity
fallback, threshold adjustment, or separate solver just for the gate is used.

## Explicit derived provenance

No existing production code or configuration is edited. An explicit process-local
binding selects the versioned solver for the unchanged pipeline, while the derived
manifest pins both this entry point and `solver.py` in addition to the parent code.
The binding is restored on exit, including exceptions; no torch function is patched.

New results are written below the **existing** G run:

```text
llama3.1-8b-diag-g-256/
  manifest.json                         # unchanged failed-run manifest
  statistics/checkpoints/window_*.safetensors  # retained original raw G
  quantized/*.safetensors                # retained original Wq
  diagnostics/identity_g_*/report.json    # retained audit evidence
  full_svd_v1/
    config.json                         # derived output path, same numerical settings
    manifest.json                       # parent/audit/code identities and solver transition
    initialization.json                 # immutable resumable setup identity
    statistics/collect_state.json       # G256 pointer bound to derived manifest
    statistics/checkpoints/window_0256.json
    quantized/*.json                    # verified references to parent Wq tensors
    reuse_complete.json
    corrections/{diag_gd,full_gd}/       # newly computed full-SVD GD factors
    evaluation/                         # 18 configurations under the derived manifest
```

Only metadata is created when reusing G and Wq; their tensor files are **not**
copied, hard-linked, or symlinked. The existing readers consume the explicit
hash-checked `file.path` records. KEEP the parent directory: the derived run is
not a self-contained export. All parent G prefix snapshots remain intact. No
parent GD corrections/evaluations/effective-G files are inherited into this run.

Startup verifies the complete committed G snapshot and its metadata, the parent
manifest and code, audit evidence, and each reused quantized weight. A compatible
completed first-layer audit is required; the latest matching report is selected
on first initialization and then pinned. Resume rejects changed code, parent
state, configuration, or pinned audit instead of repinning or overwriting them.
Partial initialization resumes idempotently. Both parent and child locks are held
throughout, preventing a legacy pipeline or a second continuation from writing at
the same time. An exception releases both locks normally.

## Run on the existing server

Keep the existing repository directory and `qera-original-a` environment. Download
only `full_svd_v1/run.py`, `full_svd_v1/solver.py`, and
`scripts/resume_full_svd.sh` from the published commit, or fast-forward a clean Git
checkout. No environment installation, model/data download, or G collection is needed.

From the existing code directory:

```bash
DIAG_G_RUN=/share/home/tm902089733300000/a913520780/chengkang/qera_runs/llama3.1-8b-diag-g-256
nohup bash experiments/qera_diag_g_isolation/scripts/resume_full_svd.sh >> "$DIAG_G_RUN/full_svd_resume.log" 2>&1 &
echo $!
tail -n 50 -F "$DIAG_G_RUN/full_svd_resume.log"
```

Default sequence: verify/reuse → solve → evaluate/summary. Startup rereads the
frozen Wq tensors for hashing; this is disk auditing, not requantization. It never
loads the FP32 teacher for backward. The full SVD allocates larger temporary U/V
matrices than the reduced path; runtime and actual GPU capacity for all layers
remain server checks. An OOM or any numerical gate failure stops without fallback.

The same command resumes completed factors and evaluation batches. Keep appending
to the log. Do **not** use the original `run_dual_4090.sh` to continue this derived
run, and do not edit the original YAML's `run_dir`. This entry point always takes
the original G256 config and selects its fixed `full_svd_v1` child internally.
Optional stages: `prepare`, `solve`, `evaluate`, `summary`. An explicit evidence
path can be supplied with `--audit-report /absolute/path/report.json`; otherwise
the retained matching report is found automatically. No `--only` bypass is exposed.

Final results: `full_svd_v1/evaluation/ppl_summary_wikitext2.csv`. A completed
first-layer identity check alone is not proof that all 224 projections or the
ten PPL controls have passed.

## Tests

```bash
python -m pytest experiments/qera_diag_g_isolation/full_svd_v1/test_resume.py experiments/qera_diag_g_isolation/audits/test_identity_g.py experiments/qera_diag_g_isolation/tests -o addopts='' -q
```

Tests exercise full SVD for both GI and GD, the weighted optimum, binding recovery
on failure, required audit evidence, preserved parent bytes, real checkpoint/Wq
reader compatibility, no tensor duplication, interrupted initialization, changed
code rejection, corrupt checkpoint rejection, and refusal to adopt unrelated files.
CPU tests do not certify the complete model's CUDA run.
