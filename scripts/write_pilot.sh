#!/usr/bin/env bash
# The writing-decision pilot (docs/06) as one CPU job: no GPU, the fixed rule decides.
#   1. `jev run writing --variant train|val`: each task's submitted draft of rollouts-train and
#      rollouts-val, submitted unchanged, grounded, expanded, or expanded then grounded, two
#      revision samples each (DeepSeek; holds during DeepSeek peak hours);
#   2. `jev eval`, then the report against submitting the draft unchanged and the per-task
#      headroom of choosing among the four (cross-fitted on the two samples).
# Resumable: resubmit and finished tasks and judgments are kept.
# Usage (from the repository):
#   sbatch scripts/write_pilot.sh
# Settings (environment): VARIANTS ("train val"), PREFIX (write-pilot: runs/<PREFIX>-<variant>).
#SBATCH --job-name=jev-write
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=write-pilot-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

VARIANTS=${VARIANTS:-"train val"}
PREFIX=${PREFIX:-write-pilot}

for variant in $VARIANTS; do
  run=$PREFIX-$variant
  uv run jev run writing --variant "$variant" --run-id "$run"
  uv run jev eval "$run"
  uv run jev report "$run" --baseline w_submit
  uv run jev controller headroom "$run" --metric utility --metric nugget_coverage \
    --metric cite_p --metric claim_support | tee "runs/$run/headroom.md"
  echo "report: runs/$run/report.md, runs/$run/headroom.md"
done
