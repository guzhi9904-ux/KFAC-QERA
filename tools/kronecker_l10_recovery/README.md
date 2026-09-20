# Cold-process CUDA linalg recovery for L10

Formal and pilot use different Python processes. In PyTorch 2.3, the formal
process's first linalg call can race between the two offline threads, raising
`lazy wrapper should be called at most once`. The accepted pilot had already
initialized linalg serially, which concealed the fresh-process condition.

This launcher initializes eigvalsh/eigh/gesvd/solve serially on both devices,
then validates three concurrent worker rounds before invoking the **unchanged**
controller in the **same Python process**. It does not change packages, precision,
sample/method counts, numerical tolerances or the frozen scientific sources.
It requires exact equality with the existing source/config manifest. The runtime
prelude is separately registered under runtime_recovery with its own source hash.
Committed gradients/factors are validated and reused by the original controller.

From the repository (with the original offline/thread environment exports):

```bash
python -B tools/kronecker_l10_recovery/resume.py CONFIG RUN_DIRECTORY run
```

Use this entrypoint for subsequent pilot/formal resumes of this frozen version.
An unrelated numerical failure remains fatal; no retry or tolerance relaxation.

Upstream related issue: https://github.com/pytorch/pytorch/issues/90613
