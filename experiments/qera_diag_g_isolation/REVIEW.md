# Round-one implementation review

Reviewed 2026-09-08 for the user's two-4090 / 224 GB server.

## Scope decisions

- Only new files under `experiments/qera_diag_g_isolation` are introduced.
- The previous 256-window A statistics and roots remain unchanged and retained.
- G means the second moment of the gradient of **sequence CE sum**, not an exact
  tokenwise Fisher. Full-G later must retain this definition to be comparable.
- Raw G is a separate vector for each of 224 projections. Input-A sharing is not
  reused for G. Counts for A and G deliberately differ (2048 versus 2047/window).
- G's relative floor is declared up front; no PPL-based tuning is implemented.
- The new experiment uses the original token-PPL protocol. The separate paper
  word-PPL discrepancy stays unresolved and is not hidden by changing thresholds.
- Weight, data, source-root and factor identities are pinned. The saved historical
  harness model checksum is also compared when present; its PPL is not reused.

## Gates and recovery

- FP32/BF16 official quantization equivalence is checked for every target before
  collecting G; one stored Wq is used for both GI and GD evaluation.
- Identity-G reduction is checked against original factor products at all four
  ranks for both input metrics. This is a numerical regression, not a sign check.
- Original ten control PPLs are re-evaluated first. Failure stops the normal run
  before GD evaluation. `--only` is an explicitly targeted diagnostic option.
- Atomic tensor/JSON writes, checksums and an OS lock protect checkpoints.
  Raw checkpoint files are retained, not cleaned after solve. Incomplete window
  work is replayed from the committed prefix rather than counted twice.
- Completed evaluation windows must be contiguous and have 2047 prediction
  tokens. Partial configurations are not published as final scores.
- Evaluation releases module dictionaries and hook closures before the next
  model load; a weak-reference integration test covers this memory-lifetime bug.

## Executed checks

Local CPU environment: Python 3.13.5, torch 2.9.1+cpu. The server entry explicitly
requires the existing torch 2.3.0 / transformers 4.44.2 environment.

```text
python -m pytest tests experiments/qera_original_a_isolation/tests \
  experiments/qera_diag_g_isolation/tests -o addopts='' -ra
85 passed, 1 skipped
```

The skipped original allocator integration test requires CUDA. Mathematical tests
include autograd comparisons, causal future-position dependence, G diagonal/full
moment consistency, exact weighted rank truncation and identity reduction.
Additional tests cover corrupt checkpoints, retained prefix snapshots, source
directory protection, control drift rejection, configuration coverage and model
release. `compileall`, the CLI help path, and Bash syntax checking also passed.

No actual Llama-3.1-8B GPU inference, two-device backward, server filesystem lock,
or server peak-memory test was performed locally. These remain runtime checks,
not claimed certifications. The first server command should be `doctor`, then
the logged resumable `run`; CUDA OOM stops without changing the scientific
protocol or deleting saved statistics. Ruff was unavailable in the local Python
environment; no package was installed to change either environment.
