#!/usr/bin/env bash
# Judge a finished run and write its reports, as a CPU job: judging needs no GPU, and a GPU job
# that holds an idle GPU for an hour is cancelled by the cluster. Pair it with a GPU job run
# with JUDGE=0 (e.g. scripts/screen_dev.sh), started after it:
#   sbatch --dependency=afterok:<GPU job id> --export=ALL,RUN=<run> scripts/judge.sh
#   1. `jev eval RUN` (DeepSeek; holds during DeepSeek peak hours);
#   2. `jev report RUN --baseline BASELINE`;
#   3. the matched-cost comparison (`jev controller frontier`) when the run has one
#      (FRONTIER=1), into runs/<RUN>/frontier.md.
# Resumable: resubmit and finished judgments are kept.
# Settings (environment): RUN (required), BASELINE (ctl_fixed), FRONTIER (1).
#SBATCH --job-name=jev-judge
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=judge-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

RUN=${RUN:?set RUN to the run to judge}
BASELINE=${BASELINE:-ctl_fixed}
FRONTIER=${FRONTIER:-1}
[ -f "runs/$RUN/manifest.json" ] || { echo "no run at runs/$RUN" >&2; exit 2; }

uv run jev eval "$RUN"
uv run jev report "$RUN" --baseline "$BASELINE"
if [ "$FRONTIER" = 1 ]; then
  uv run jev controller frontier "$RUN" | tee "runs/$RUN/frontier.md"
fi
echo "report: runs/$RUN/report.md"
