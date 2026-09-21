# Full-model Structure-aware Attention / SlimPajama v2

This independent experiment implements `protocol.md` without modifying frozen parent experiments.
The complete FP32 teacher is split across two RTX4090s. SlimPajama calibration is fixed at 256×2048,
with one newly frozen predictive-label vector per window. Q/K/V and gate/up share ordinary input Grams.
Pass A processes contiguous eight-layer groups; Pass B processes contiguous four-layer groups and uses
one connected, checkpointed backward pass for all registered targets in a group. Every group uses all
256 windows. The collector retains native probabilities only for its active group and saves no raw x/g/P/delta cache.

The scalar-G A-only solver is algebraically checked against the parent two-sided solver. All 416 unique
rank64 factors use FP64 symmetric roots, trace-relative damping 0.001 and dense SVD. Only raw statistics
and P64/Q64 are permanent; candidate manifests reference shared factors. Deployment multiplies P64/Q64
in FP64, casts the product to FP32, adds frozen Wq, and checks every target weight hash.

Run `python experiments/full_model_attention_slim_v2/test_full_model.py` for CPU regression checks.
On the authorized server, `bash experiments/full_model_attention_slim_v2/run.sh` executes ordered stages
and resumes only matching identities. Named stages are `audit`, `prepare-data`, `pilot`, `collect-a`,
`collect-functional`, `solve`, `freeze`, `eval-kl-ppl`, `eval-downstream`, `report`, and `resume`.
Use tmux for long execution, redirect to the run's logs/run.log, and perform only an initial health check.

Pilot verifies original/checkpointed hidden values, independent parent gradients, full-dimension K
contractions, V GQA reconstruction and explicit/effective Gram equivalence, interruption recovery,
parent/scalar-I solves, a 4096-token forward, head-chunked CE/KL/harness scoring, and the largest down solve.
Pilot labels are retained; its statistics are discarded and never counted in production.

Evaluation: six states on development KL16, all frozen historical WT2 test windows, C4-en fixed128;
five states (excluding C1) on the five specified full-split 0-shot tasks. The pinned harness handles
tokenization, task prompts, choices and metrics; a checked batch1 likelihood adapter chunks the LM head.
No response cache crosses candidates. Every task preserves per-question options/responses/results.
Missing data or failed gates produce a partial report, never successful completion or substituted scores.

Scientific settings are frozen in JSON-compatible `config.yaml`. Personal quota evidence is separate
from filesystem-wide free space. Checkpoints retain current/previous generations only inside this run's
temporary directory. The parent model, old results and parent code remain read-only.
