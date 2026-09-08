# Frozen-A / Diagonal-G Isolation — Round One

This is an **additional experiment**, not a change to `src/qera_exp` or
`experiments/qera_original_a_isolation`. The completed original run is read-only.
Its raw A, roots, frozen data and identity-G corrections must remain available.

## Reviewed experiment

| Item | Fixed setting |
|---|---|
| Model | The original local Meta-Llama-3.1-8B BF16 checkpoint |
| Calibration | The **existing** 256 SlimPajama windows, each 2048 tokens |
| A | Existing diagonal/full roots; no recomputation, new damping or new sqrtm policy |
| G teacher | Frozen, unquantized FP32 model, eager attention, cache disabled |
| G collection | One complete window at a time; all **224** projections in one traversal |
| G statistic | Squared output gradients of sequence **summed** next-token CE; FP64 accumulation |
| G positions | 2047 prediction positions/window = **524032**; A used **524288** inputs |
| G numerical policy | Divide raw diagonal by its mean, floor at `1e-6`, take square root |
| Quantization | Official pinned QERA MXINT4, block size 32, last weight axis |
| Correction rank | 8 / 16 / 32 / 64; retain rank-64 factors and use exact prefixes |
| Solve | FP32; unchanged QERA A-inverse fallback; exact economy SVD, not randomized SVD |
| Evaluation | Existing 138 WikiText-2 windows, length 2048, **282486** prediction tokens |
| GPU/RAM tier | 2 x RTX 4090 24 GB; 28 CPU cores; 224 GB RAM |
| Evaluation runtime | BF16, eager, two-GPU balanced placement, batch size 8 |
| Retention | Raw G sums/counts, all saved checkpoints, effective G roots, Wq and factors retained |

The eight new configurations are `DIAG_GD_R{8,16,32,64}` and
`FULL_GD_R{8,16,32,64}`. We also re-evaluate BF16, Quant-only and all eight
existing GI configurations using the same frozen Wq and evaluation path:
**18 evaluations**, but no new A/roots or GI-factor production.

The question is whether changing **only the output weighting** improves token
PPL at fixed A and calibration. No sample-count sweep, G-full matrix, new
calibration sampling, activation quantization, or module-scope ablation is included.

The QERA Table 16 **word PPL 7.55** discrepancy is deferred. It is not this
experiment's gate or metric. The original token-PPL BF16 baseline is approximately
6.24486. A paper table with a nearby number is not proof of identical protocols.

## Definition and limitations

For a projection output at position t, let

```text
L_sum = sum_t CE(next-token logits_t, observed next token_t)
g_t   = d L_sum / d z_t
d     = (1 / N_prediction_positions) sum_t (g_t * g_t)
d_eff = maximum(d / mean(d), 1e-6)
T     = diag(sqrt(d_eff))
```

The full decoder is differentiated: attention paths from later token losses are
included. This is a **sequence-loss gradient-second-moment proxy**, not exact
per-token Fisher, Hessian, or model-distribution-sampled Fisher. This distinction
must remain in subsequent Full-G experiments. Loss is never batch/token averaged
before backward. The final unscored position is excluded. Fully unpadded original
windows are required. A retains its original count and numerical computation;
we do not silently redefine A to match G's prediction-position count.

Each output projection has its **own** G vector. Shared input A for q/k/v and
gate/up does not imply shared G. Future Full-G must use the same definition,
then derive GD from the *same* accumulated full matrix for a strict comparison.
Saved diagonal vectors alone cannot reconstruct off-diagonal entries.

Using the repository's row-vector convention, E_T = (W - Wq).T and S is the
existing QERA input root. The new objective and solution are

```text
min_rank(LR)<=r || S (E_T - LR) T ||_F^2
U Sigma Vh = SVD(S E_T T)
L = S^-1 U_r
R = Sigma_r Vh_r T^-1
forward correction = (x @ L) @ R
```

Existing Full-A roots include the original real-cast sqrtm approximation; this
experiment preserves it instead of claiming the stored S is a perfect root.
Normalizing G by a scalar preserves the ideal minimizer; flooring changes it.
The floor is predeclared, and clipping counts/raw scales are saved. An all-zero
G or nonfinite statistics stop the run; there is no silent identity fallback.

## Gates before interpreting G

1. `prepare` hashes source A/raw roots, all model weights, frozen tokens and
   corrections; validates original completion markers, counts, shapes and the
   ten old token-PPL controls. It pins local helper code and the official QERA
   code. It does not change the source experiment. Initial hashing reads roughly
   100+ GiB; this is disk auditing, not recalibration. Logs identify each group.
2. The official quantizer is run on each target in both FP32 and BF16. We require
   **bitwise equality** after conversion to FP32, so solve and eval use one Wq.
   Wq is stored once and consumed by both GI and GD evaluation.
3. For every projection and both A variants, setting the new solver's G to I must
   reproduce the existing factor **products** at every requested rank, within
   relative Frobenius error `1e-3`. We compare products, not SVD signs. Results
   outside tolerance stop for review; this is not a knob to tune against PPL.
4. All ten control evaluations must match their original token-PPL values within
   absolute PPL `0.01`. This engineering regression bound is **not** a paper
   reproduction tolerance. The normal full run evaluates controls before GD.
5. Weighted objective, finite factors, source/checkpoint hashes, module coverage
   and token counts are checked. Partial evaluations are never reported as final
   PPL. No automatic dtype, window length, floor, or batch-size fallback occurs.

`G=I` numerical gates and CPU tests do not prove G will improve end-to-end PPL.
Report improvements and regressions across all four ranks, not only the best rank.

## Server: reuse the existing environment

Use `qera-original-a` (torch 2.3.0, transformers 4.44.2). **Do not run the root
`pip install -e .` again**: its general dependency bounds differ from this pinned
environment. No environment download, new dataset download or new A collection
is needed. QERA's original checkout must remain available at the path recorded
in the source run. This runner does not use the harness or network.

From the updated repository/archive directory:

```bash
conda activate /share/home/tm902089733300000/a913520780/chengkang/conda_envs/qera-original-a
export CUDA_VISIBLE_DEVICES=0,1
bash experiments/qera_diag_g_isolation/scripts/run_dual_4090.sh doctor
```

Then launch the full first round, with the entire nohup command on **one line**:

```bash
DIAG_G_RUN=/share/home/tm902089733300000/a913520780/chengkang/qera_runs/llama3.1-8b-diag-g-256
mkdir -p "$DIAG_G_RUN"
nohup bash experiments/qera_diag_g_isolation/scripts/run_dual_4090.sh > "$DIAG_G_RUN/pipeline.log" 2>&1 &
echo $!
tail -n 50 -F "$DIAG_G_RUN/pipeline.log"
```

The default YAML is independent of old shell `CONFIG` / `RUN_DIR` values.
For an intentionally different path, copy the new YAML, edit it before starting,
and set `DIAG_G_CONFIG=/absolute/path/to/new-config.yaml`.

Pipeline: `prepare -> quantize -> collect -> solve -> evaluate -> summary`.
There is no separate smoke dataset or reduced-layer default. Each stage can also
be called explicitly, for example:

```bash
bash experiments/qera_diag_g_isolation/scripts/run_dual_4090.sh collect
bash experiments/qera_diag_g_isolation/scripts/run_dual_4090.sh solve
bash experiments/qera_diag_g_isolation/scripts/run_dual_4090.sh evaluate
bash experiments/qera_diag_g_isolation/scripts/run_dual_4090.sh summary
```

The G forward/backward is heavier than A-only forward collection. Batch size 1
is deliberate even with two GPUs: the FP32 weights occupy about 30 GiB in total,
and backward needs saved activations. The runner places weights on the GPUs and
offloads autograd-saved tensors to CPU RAM. It verifies actual cgroup limits
where exposed; host RAM (e.g. a reported 1 TB) is not the rental memory limit.

The 128-token CE-gradient chunk and 256-token evaluation-CE chunk **only bound
temporary vocabulary computations**. Every transformer forward still receives
2048 tokens. No context is shortened, and the vocabulary is not truncated.
Two cards use model sharding, not two independently evaluated data subsets.
During evaluation, one verified Wq copy (about 14 GiB) is cached in host RAM to
avoid repeatedly reading the same weights from shared storage for every rank.

Logs appear per collection window, per solve method, and per eval batch; a
30-second heartbeat covers long hashes, model loading, backward and SVD calls.
Use the measured collection ETA after several windows; no unmeasured runtime
promise is made. If memory is insufficient, preserve the traceback and state;
do not silently change dtype or modify an active run's configuration.

## Resume and immutable inputs

After restart, activate the same environment and execute the same nohup command
with append redirection (`>>`) instead of truncating the previous log. Finished
quantized tensors/factors and evaluation batches are skipped after validation.
G resumes at the last fully committed snapshot, every 8 windows by default;
at most the work since that snapshot is repeated. Existing snapshots are kept.
An OS file lock prevents two processes writing the same run concurrently and
is automatically released on process death. Temporary files are never treated
as complete checkpoints.

Do not edit source/new code or YAML during a run: these are pinned in the manifest.
Resume rejects changed code/configuration. A code/protocol migration requires
review or a distinct output directory, not deletion of existing G. Source files
are rechecked when consumed; an explicit `prepare` repeats the complete source
audit without changing an existing matching manifest.

## Files and retention

```text
original-a-isolation/                 # read-only; KEEP this directory
  data/{calibration,wikitext2}.safetensors
  statistics/raw/                     # original A sums/counts
  statistics/roots/                   # original diagonal/full A roots
  corrections/{diag,full}/            # existing GI rank-64 factors
  evaluation/ppl_summary_wikitext2.csv

llama3.1-8b-diag-g-256/                # all new writes go here
  manifest.json                       # source identities, protocol, control PPL
  doctor.json
  quantized/*.safetensors              # frozen per-projection Wq, BF16
  statistics/collect_state.json        # pointer to last committed G checkpoint
  statistics/checkpoints/window_*.safetensors
  statistics/checkpoints/window_*.json # counts, raw-G hash and teacher NLL
  statistics/effective_g/*.safetensors # raw g_sum + stabilized sqrt_g
  statistics/effective_g/*.json        # normalization/floor diagnostics
  corrections/{diag_gd,full_gd}/        # rank-64 FP32 L/R saved as A/B keys
  evaluation/configurations/*.json     # atomic per-configuration window records
  evaluation/control_checks/*.json     # old versus new GI/baseline comparisons
  evaluation/ppl_summary_wikitext2.csv  # 18 rows on completion
  evaluation/wikitext2_per_window.csv
  evaluation/status.json
```

There is **no cleanup mode**. One raw G snapshot for this model is approximately
10.5 MiB (FP64 vectors for 224 projections); the retained 32 interval snapshots
are approximately 336 MiB, excluding metadata. Wq is about 14 GiB; factors and
other metadata add storage. Existing raw A/roots are referenced, not duplicated.
Full-G is not implemented in this first round and cannot be recovered from these
vectors. A later Full-G collector must permanently retain its matrix sums/counts.

## Verification

```bash
python -m pytest experiments/qera_diag_g_isolation/tests experiments/qera_original_a_isolation/tests -q
```

Tests cover analytic CE versus autograd, summed-loss batch invariance, future
attention-path semantics, diagonal/full moment consistency, weighted-SVD optimum,
identity-G reduction, source/output path separation, checkpoint retention and
corruption rejection, evaluation integrity, and release of the previous model
before loading the next configuration. CUDA capacity/performance requires an
actual server run; CPU tests are not a two-4090 end-to-end certification.
