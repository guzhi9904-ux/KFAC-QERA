# Identity-G extension

`run_identity_g.py` evaluates the saved `DIAG_GI` and `FULL_GI` factors for
MXINT4 and MXINT3 with the same pinned QERA/harness 4096-context word-PPL
protocol as the completed diagonal-G run.

It deliberately uses a separate output directory:

```text
qera_runs/official-word-ppl-4096-existing-artifacts-identity-g
```

The existing BF16, W-only, and diagonal-G output remains unchanged. The
unchanged `run.py` engine and the new identity-G entrypoint are both bound into
the extension protocol by SHA-256.

After fetching the commit, use the same `qera-original-a` environment and cache
exports documented in `README.md`. First run the independent BF16 gate:

```bash
python experiments/qera_official_artifact_eval/run_identity_g.py \
  --config experiments/qera_official_artifact_eval/configs/llama3.1-8b-server-identity-g.yaml \
  evaluate --stage bf16
```

Then run all 16 identity-G configurations:

```bash
python experiments/qera_official_artifact_eval/run_identity_g.py \
  --config experiments/qera_official_artifact_eval/configs/llama3.1-8b-server-identity-g.yaml \
  evaluate --stage artifacts
```

When `--only` is omitted from the artifact stage, the entrypoint inserts exactly
the 16 `DIAG_GI`/`FULL_GI` selections. `Wq` remains in the configuration solely
because the engine must install each run's immutable quantized weights before
attaching its identity-G factors; W-only is not re-evaluated.

The final table is written to:

```text
qera_runs/official-word-ppl-4096-existing-artifacts-identity-g/summary.csv
```
