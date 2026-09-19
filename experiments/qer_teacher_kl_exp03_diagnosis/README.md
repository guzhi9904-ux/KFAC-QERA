# Exp-3 frozen two-stage diagnosis

Only `cck`, inside one A6000 / 64 GiB labgpu lease. Frozen parent modules are imported read-only; PYTHONDONTWRITEBYTECODE is set. No training, fitting, SVD, new gradient Gram, or KL entry point is called.

Stage A streams each original S once for five deployed residual scores and four J contractions. All frozen file hashes and tensor identities are verified before and after; original J and q_K are replayed. Stage B uses exactly eight original windows and 16 new labels/window from base seed 2026091804. The first sample of each module gates both contraction paths and is retained once. Real labels, signed projections and paired statistics are saved.

The source manifest is frozen before either stage. Stage A is descriptive; stage B uses 2000 fixed-window paired bootstrap draws, PCG64 seed 2026091805. Ratio floor 1e-12, practical threshold 0.02. No new-S storage. New output directory required; no silent restart/resampling.

Run `python -B test_diagnosis.py`, then use the scheduler to execute `bash run.sh OUTPUT`. Results are independently checked using a separate local verifier after retrieval.
