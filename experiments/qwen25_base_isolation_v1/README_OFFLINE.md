# Offline prepare: exact SlimPajama prefix + read-only WikiText cache

Use only BEFORE the Qwen output has a frozen `manifest.json`. Do not patch a
prepared/running experiment. The provided bundle updates only Qwen entry/data
code, never the protected Llama/Full-G helpers. Keep the original model, YAML,
initialization marker, conda environment, dataset cache and Llama outputs intact.

## Verified acquisition

- Dataset: `DKYoon/SlimPajama-6B`
- Revision: `b5f90f419b7489cdba26fdbc8c022fcb5562f968`
- Rows: first 5120 training rows, original order, no text modification
- Original length-prefixed UTF-8 text SHA256:
  `84067fe5da1470cca552135d1ee3e2089231b249053336b842721877c1ba03ab`
- Gzip file SHA256:
  `00953238af838af285d07654f15da89378fec2a6f613ffb008f6db473d6e0dda`
- Compressed size: about 8.05 MiB.

`fetch_prefix.py` fetched the original first Parquet shard from the pinned public
Hugging Face revision, verified its SHA256, exported all original row fields and
checked the text hash against the user's frozen Llama metadata. The large source
Parquet is NOT needed on the server and is NOT included in the bundle.

This verifies raw text equality, NOT the server token replay yet. Offline prepare
repeats the hash/count/provenance checks against the SERVER's original metadata,
then runs the same mandatory Llama token replay for both datasets before accepting
Qwen-tokenized data. No thresholds, sampling, dtype, batch, or solve settings change.

## Deployment

Upload `qwen_offline_fix_b5f90f4.zip` to the shared `chengkang` directory. It contains
only Qwen-specific patched files under `KFAC-QERA-qwen25-base-v1/` and the small
prefix files under `qwen_offline_raw_b5f90f4/`. The previous prepare process must
have exited, as in the reported ConnectionError. Do not run multiple prepares.

```bash
cd /share/home/tm902089733300000/a913520780/chengkang
# Must print NOT_PREPARED. If it prints STOP, do not overwrite frozen code.
if [ -e qera_runs/qwen2.5-7b-base-mxint3-v1/manifest.json ]; then
  echo STOP_ALREADY_PREPARED
else
  echo NOT_PREPARED
  unzip -o qwen_offline_fix_b5f90f4.zip
fi
```

Only after successful extraction:

```bash
cd /share/home/tm902089733300000/a913520780/chengkang/KFAC-QERA-qwen25-base-v1
conda activate /share/home/tm902089733300000/a913520780/chengkang/conda_envs/qera-original-a
BASE=/share/home/tm902089733300000/a913520780/chengkang
WT="$BASE/huggingface_cache/datasets/Salesforce___wikitext/wikitext-2-raw-v1/0.0.0/b08601e04326c79dfdd32d625aee71d232d685c3"
LOG="$BASE/qera_runs/qwen25-base-v1.log"
nohup bash experiments/qwen25_base_isolation_v1/run_server.sh prepare \
  --offline-raw-dir "$BASE/qwen_offline_raw_b5f90f4" \
  --wikitext-cache-dir "$WT" >> "$LOG" 2>&1 &
tail -n 60 -F "$LOG"
```

Do NOT add `--allow-download`. All three HF offline flags are set before importing
official QERA/Transformers. No endpoint/mirror or authentication changes are needed.
The Transformers cache deprecation warning can remain; it is not a failed audit.

Expected milestones:

1. `offline SlimPajama raw rows=5120 ... PASS`
2. `calibration Llama token replay EXACT PASS`
3. `offline WikiText split=...`
4. `wikitext2 Llama token replay EXACT PASS`
5. `audit-qwen PASS`

The input audit completion may print evaluation status `INCOMPLETE` (zero PPL
configurations): this is expected before collecting/solving/evaluating the model.
After prepare exits successfully, use the original `pilot-a` command, then the
staged workflow. Offline directory options apply only to prepare; later stages
read the newly frozen Qwen token files and do not need raw data acquisition.

## Old cache protection

Only `wikitext-train.arrow`, `wikitext-validation.arrow`, `wikitext-test.arrow` are
accepted, each with a `text` column. Tokenized `cache-*.arrow` are not accepted.
Arrow is opened read-only and copied to an in-memory Dataset before official
mapping. Therefore transforms are not associated with the old Arrow directory.
Source file identities are recorded in the new manifest. No old cache is deleted,
no lock is removed, and no write lock is taken in any Llama run.
