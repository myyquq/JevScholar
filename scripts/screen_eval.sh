#!/usr/bin/env bash
# d1-3B as the relevance screener, zero-shot (docs/06: screening), one short GPU job, no API:
#   1. `jev controller screen-data`: every DeepSeek screening verdict the runs logged, as d1
#      examples (train, val and June dev tasks: train.jsonl, val.jsonl, dev.jsonl; never test);
#   2. `jev controller screen-eval` on each of PARTS: d1's P(relevant) against DeepSeek and the
#      target papers' real citations, alone and as a cascade with DeepSeek
#      (runs/screen/<NAME>-<part>.md, per-example probabilities in .jsonl).
# Usage (from the repository):
#   sbatch scripts/screen_eval.sh
# Settings (environment): ADAPTER (none: d1-3B as released), NAME (zero-shot), PARTS (dev train).
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
PARTS=${PARTS:-"dev train"}
DATA=data/derived/controller/screen
OUT=runs/screen
mkdir -p "$OUT"

uv run jev controller screen-data "$DATA"
for part in $PARTS; do
  scripts/train_env.sh jev controller screen-eval "$DATA/$part.jsonl" \
    --out "$OUT/$NAME-$part.jsonl" ${ADAPTER:+--adapter "$ADAPTER"} | tee "$OUT/$NAME-$part.md"
done
echo "reports: $OUT/$NAME-<part>.md for $PARTS"
