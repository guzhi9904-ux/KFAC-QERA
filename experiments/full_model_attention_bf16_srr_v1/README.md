# Full-model attention: BF16 deployment and SRR evaluation

User-approved correction on 2026-09-22: retain full_model_attention_slim_v2/run_02's
FP32-teacher statistics, predictive labels, frozen MXINT3 Wq and 416 FP64 rank64 factors.
Evaluate every candidate using BF16 model parameters, BF16 activations, BF16 factors and
two BF16 correction GEMMs followed by a BF16 addition. The saved FP64 factors are read-only.
For parent P64[out,64], Q64[64,in], install A=Q64.T.to(BF16), B=P64.T.to(BF16),
then compute Linear(x,Wq_BF16)+(x@A)@B. Never merge the dense FP64 correction into Wq.
Source/candidate manifests freeze direct-cast BF16 bit hashes before scoring. Pilot checks
the real module path and all 224 projection executions, plus native harness at 4096 tokens.

The old FP32 run used FP32 merged weights, harness 0.4.3/3823cfe, and HellaSwag,
PIQA, Winogrande, ARC-Easy, ARC-Challenge. It is an internal FP32 comparison, not an
SRR-aligned BF16 evaluation. Its results remain in the old directory and are not relabeled.
The user's new instruction supersedes the old protocol's FP32 evaluation/five-task clauses.

SRR source reviewed at commit 46b245d2176f6f68df031aaed2069220e39540ba:
config quant_dtype=float32, eval_dtype=bfloat16; requirements lm_eval==0.4.7;
classic/hard task groups and ptq_parser.py define HellaSwag acc_norm, Winogrande acc,
BoolQ acc, MMLU acc (57 subjects, question-weighted), leaderboard_bbh acc_norm
(24 multiple-choice subtasks, unweighted subtask mean). All are 0-shot with no chat template,
full standard splits, context4096 and effective HFLM batch1. SRR configuration says batch4,
but its wrapper constructs HFLM(model) with default batch1. Its evaluation seed defaults
are 0/1234/1234/1234; use the same here. Bootstrap standard errors are omitted, not scores.
All five benchmark scores contribute equally to SRR_5_mean. Do not substitute generative BBH.
Our calibration estimators and FP64 solver remain method-specific, not an SRR algorithm reproduction.

The exact previously verified PyPI lm_eval 0.4.7 runtime is reused without changing shared
conda packages. All runtime checksums are verified. Native HFLM scoring, tokenization,
logits-cache optimization and aggregation are used unchanged. Requests that would truncate
are rejected instead of silently changing their context. Each question's raw results are saved.
Task contents/configuration must match the project's existing SRR-aligned frozen dataset records.
No response cache is shared between candidate models.

Six states (Teacher,C0,C1,C2,C3,C4): BF16 teacher-reference KL16, unchanged token-PPL on
138 WT2 and 128 C4 2048-token windows, plus native harness WikiText word_perplexity at4096.
Token losses/KL accumulate in FP64 for stability; this does not change BF16 model arithmetic.
Word-PPL is a distinct metric and is never mixed with token-PPL. C4 is an additional corpus,
not a claim about the SRR paper's C4 protocol. Downstream uses Teacher,C0,C2,C3,C4 (25 jobs).
All precision changes apply uniformly; no rank/gauge changes or outcome-based candidate selection.

Run CPU tests with `python experiments/full_model_attention_bf16_srr_v1/test_bf16.py`.
On the server, `bash experiments/full_model_attention_bf16_srr_v1/run.sh resume` performs
prepare, finite engineering pilot, evaluation and report. Resume checks identity and completed
files; token evaluation resumes per window, harness per completed benchmark. A single run lock
prevents duplicate writers. Logs print heartbeat/progress internally; no external monitor is created.
Results: tokens/<state>/<corpus>, harness/<state>/<task>, summary/results.csv and complete.json.
