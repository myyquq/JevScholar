#!/usr/bin/env bash
# d1-3B as the relevance screener, zero-shot (docs/06: screening), one short GPU job, no API:
#   1. `jev controller screen-data`: every DeepSeek screening verdict the runs logged, as d1
#      examples (train and val tasks: train.jsonl; June dev tasks: dev.jsonl; never test tasks);
#   2. `jev controller screen-eval` on both: d1's P(relevant) against DeepSeek's verdicts and the
#      target papers' real citations, alone and as a cascade with DeepSeek
#      (runs/screen/<NAME>-<part>.md, per-example probabilities in .jsonl).
# Usage (from the repository):
#   sbatch scripts/screen_eval.sh
# Settings (environment): ADAPTER (none: d1-3B as released), NAME (zero-shot).
#SBATCH --job-name=jev-screen
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=screen-eval-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ADAPTER=${ADAPTER:-}
NAME=${NAME:-zero-shot}
DATA=data/derived/controller/screen
OUT=runs/screen
mkdir -p "$OUT"

uv run jev controller screen-data "$DATA"
for part in dev train; do
  scripts/train_env.sh jev controller screen-eval "$DATA/$part.jsonl" \
    --out "$OUT/$NAME-$part.jsonl" ${ADAPTER:+--adapter "$ADAPTER"} | tee "$OUT/$NAME-$part.md"
done
echo "reports: $OUT/$NAME-dev.md, $OUT/$NAME-train.md"
