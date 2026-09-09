# Official 4096-context word-PPL evaluation

This isolated evaluator applies the completed KFAC-QERA MXINT3/MXINT4 artifacts to
the original Llama-3.1-8B checkpoint, then calls the unmodified QERA `bd7fc86`
lm-eval path for the document-level WikiText word-perplexity metric.

It evaluates only:

- MXINT4 and MXINT3 RTN W-only;
- diagonal-A + diagonal-G (`DIAG_GD`) at ranks 8/16/32/64;
- full-A + diagonal-G (`FULL_GD`) at ranks 8/16/32/64.

The official checkout, harness checkout, model, and artifact runs are read-only.
The evaluator never invokes collection, quantization, solving, or the old
138-window token-PPL stage. Its only lock and writes are under `output_dir`.

## Server setup

Use the environment that already reproduced BF16 `word_perplexity = 7.5527`.
Do not reinstall the KFAC-QERA project or change its experiment YAML.

```bash
conda activate qera-original-a
cd /share/home/tm902089733300000/a913520780/chengkang/KFAC-QERA-98ad0c5

export QERA_OFFICIAL_ROOT=/share/home/tm902089733300000/a913520780/chengkang/QERA-official-bd7fc86
export PYTHONPATH="/share/home/tm902089733300000/a913520780/chengkang/qera_official_bf16_pydeps:$QERA_OFFICIAL_ROOT/src"
export HF_HOME=/share/home/tm902089733300000/a913520780/chengkang/huggingface_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Select two idle GPUs. Do not share them with the running Full-G process.
export CUDA_VISIBLE_DEVICES=0,1
```

Review the two artifact paths in
`configs/llama3.1-8b-server.yaml`, then run:

```bash
python experiments/qera_official_artifact_eval/run.py \
  --config experiments/qera_official_artifact_eval/configs/llama3.1-8b-server.yaml \
  doctor

python experiments/qera_official_artifact_eval/run.py \
  --config experiments/qera_official_artifact_eval/configs/llama3.1-8b-server.yaml \
  evaluate --stage bf16

python experiments/qera_official_artifact_eval/run.py \
  --config experiments/qera_official_artifact_eval/configs/llama3.1-8b-server.yaml \
  evaluate --stage artifacts
```

The artifact stage refuses to start unless the same protocol's BF16 result is
within `0.05` of `7.5527`. All stages are restartable. To evaluate one item:

```bash
python experiments/qera_official_artifact_eval/run.py \
  --config experiments/qera_official_artifact_eval/configs/llama3.1-8b-server.yaml \
  evaluate --stage artifacts --only mxint4:W4_MXINT
```

Outputs include `protocol.json`, the untouched harness `results.json` for every
configuration, checksummed `complete.json` records, `summary.csv`, and
`summary.json`. A protocol or artifact mismatch fails closed instead of
overwriting an existing result.
