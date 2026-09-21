# Four V modules, five methods, fixed test16

Evaluation-only extension. Keeps all previous source directories and run assets
read-only. L0/L10/L20/L31 v_proj, N256, full-matrix rank64; A-only, Marginal,
Token-joint, Sequence-one-step, Attention-aware, plus per-module Wq baseline.
Original parent WikiText2 test windows 0..15 (2048 tokens, 2047 scored each).
These windows were previously evaluated with joint12 compensation, so this is
held-out-from-fitting confirmation, not a claim of a pristine unseen test.

No new statistics, gradients, labels, SVD, fitting, tuning, or test-based selection.
Every weight and receipt is verified; learned candidates must match their source
candidate freeze. Teacher exact FP32 identity, native precision and stable KL
reuse the accepted implementation. Each intervention affects one module only.
All 384 candidate KLs are freshly evaluated on test; teacher logits are shared
within each window. First-window candidates are repeated for reproducibility.

Independent output directory, OS lock, frozen source/data/candidate manifest,
atomic per-candidate score commits, resume with identical command. Two-hour
cumulative cap. Completed records are checked, never silently overwritten.
Reports include absolute KL, ratio-of-aggregate recovery, per-window scores,
paired gains and exploratory 2000 paired-window bootstrap intervals. Intervals
do not cover text dependence, fit resampling or selection across experiments.

Run on the rental server:

```bash
bash experiments/qer_v_test16_v1/run.sh
```

Output: `../qera_runs/v_test16_v1/run_01/RESULTS.md`, `summary/`, `complete.json`.
Completion: `V_TEST16_COMPLETE` followed by `V_TEST16_EXIT 0`.
