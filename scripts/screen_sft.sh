#!/usr/bin/env bash
# d1-3B trained as the relevance screener (docs/06: screening), one GPU job, no API:
#   1. `jev controller screen-data`: DeepSeek's logged screening verdicts as d1 examples;
#   2. `jev controller train screen`: LoRA on the train tasks' verdicts, the epoch chosen on the
#      val tasks' (runs/controller/screen: adapter/, metrics.json);
#   3. `jev controller screen-eval` with the adapter on the held-out June dev tasks and on val
#      (runs/screen/sft-{dev,val}.md), to compare with runs/screen/zero-shot-dev.md.
# Usage (from the repository):
#   sbatch scripts/screen_sft.sh
#SBATCH --job-name=jev-screen-sft
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=screen-sft-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

DATA=data/derived/controller/screen
OUT=runs/screen
ADAPTER=runs/controller/screen/adapter
mkdir -p "$OUT"

uv run jev controller screen-data "$DATA"
scripts/train_env.sh jev controller train screen
for part in dev val; do
  scripts/train_env.sh jev controller screen-eval "$DATA/$part.jsonl" \
    --out "$OUT/sft-$part.jsonl" --adapter "$ADAPTER" | tee "$OUT/sft-$part.md"
done
sha256sum "$ADAPTER/adapter_model.safetensors"
echo "reports: $OUT/sft-dev.md, $OUT/sft-val.md"
