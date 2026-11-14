#!/usr/bin/env bash
# June dev: the fixed rule with d1 as its relevance screener against the fixed rule's own
# cost-quality curve (docs/06: screening), one GPU job:
#   1. serve the screening adapter (port 9943, .venv-train);
#   2. `jev run evaluate --variant dev-screen`: the fixed rule at 3-6 batches (the same cells as
#      eval-dev-trained, so DeepSeek's answers replay) and screened at 3-6 and 8 batches, on the
#      20 June dev tasks, two writer samples each (DeepSeek; holds during DeepSeek peak hours);
#   3. stop the server, judge (`jev eval`), the report against the fixed rule and the
#      matched-cost comparison (`jev controller frontier`, runs/<RUN>/frontier.md).
# Resumable: resubmit with the same RUN and finished tasks and judgments are kept.
# Usage (from the repository):
#   sbatch scripts/screen_dev.sh
# Settings (environment): RUN (eval-dev-screen), ADAPTER (runs/controller/screen/adapter),
# JUDGE (1; 0 = generate only).
#SBATCH --job-name=jev-screen-dev
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=screen-dev-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

RUN=${RUN:-eval-dev-screen}
ADAPTER=${ADAPTER:-runs/controller/screen/adapter}
JUDGE=${JUDGE:-1}
[ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "no adapter at $ADAPTER" >&2; exit 2; }

# 1. The server; the loop checks its adapter hash against the config.
PID=
stop_server() {  # uv may not pass the signal on: stop the python process by port too
  [ -n "$PID" ] && kill "$PID" 2>/dev/null || true
  pkill -u "$USER" -f "jev controller serve .*--port 9943" 2>/dev/null || true
  [ -n "$PID" ] && wait "$PID" 2>/dev/null || true
  PID=
}
trap stop_server EXIT
scripts/train_env.sh jev controller serve --adapter "$ADAPTER" --port 9943 & PID=$!
for _ in $(seq 180); do
  curl -sf http://127.0.0.1:9943/info >/dev/null && break
  kill -0 "$PID" 2>/dev/null || { echo "the d1 server exited" >&2; exit 1; }
  sleep 5
done
curl -sf http://127.0.0.1:9943/info; echo

# 2. Generation.
uv run jev run evaluate --variant dev-screen --run-id "$RUN"

# 3. Judging and the report (no GPU needed).
stop_server
trap - EXIT
[ "$JUDGE" = 1 ] || exit 0
uv run jev eval "$RUN"
uv run jev report "$RUN" --baseline ctl_fixed
uv run jev controller frontier "$RUN" | tee "runs/$RUN/frontier.md"
echo "report: runs/$RUN/report.md, runs/$RUN/frontier.md"
