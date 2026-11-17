#!/usr/bin/env bash
# The reading pilot (docs/06) as one CPU job: no GPU, the fixed rule decides.
#   1. `jev data fulltext`: the full text of rollouts-train's and rollouts-val's writer documents
#      (arXiv PDFs, one request per 3 s; about 800 documents, under an hour; cached, so a
#      resubmission fetches only what is missing);
#   2. `jev run reading --variant train|val`: each task's submitted draft submitted unchanged,
#      expanded, read (revised from the full text) or redrafted from it, two revision samples
#      each (DeepSeek; holds during DeepSeek peak hours);
#   3. `jev eval`, the report against submitting the draft unchanged and the per-task headroom
#      of choosing among the four (cross-fitted on the two samples).
# Resumable: resubmit and finished tasks and judgments are kept.
# Usage (from the repository):
#   sbatch scripts/read_pilot.sh
# Settings (environment): VARIANTS ("train val"), PREFIX (read-pilot: runs/<PREFIX>-<variant>).
#SBATCH --job-name=jev-read
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=read-pilot-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

VARIANTS=${VARIANTS:-"train val"}
PREFIX=${PREFIX:-read-pilot}

uv run jev data fulltext rollouts-train rollouts-val
for variant in $VARIANTS; do
  run=$PREFIX-$variant
  uv run jev run reading --variant "$variant" --run-id "$run"
  uv run jev eval "$run"
  uv run jev report "$run" --baseline r_submit
  uv run jev controller headroom "$run" --metric utility --metric nugget_coverage \
    --metric cite_p --metric claim_support | tee "runs/$run/headroom.md"
  echo "report: runs/$run/report.md, runs/$run/headroom.md"
done
