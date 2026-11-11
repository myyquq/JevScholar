#!/usr/bin/env bash
# June dev comparison of the controllers (docs/06: which d1 carries the main claim), one GPU job:
#   1. serve d1 (base, port 9940), d1-sft (9941) and d1-rl (9942), each in .venv-train;
#   2. `jev run evaluate --variant dev-trained`: the fixed rule (3 to 6 batches), d1, d1-sft and
#      d1-rl on the 20 June dev tasks, two writer samples each (DeepSeek; holds during DeepSeek
#      peak hours);
#   3. stop the servers, judge (`jev eval`), write the report against the fixed rule and the
#      matched-cost comparison (`jev controller frontier`, runs/<RUN>/frontier.md).
# Resumable: resubmit with the same RUN and finished tasks and judgments are kept.
# Usage (from the repository):
#   sbatch scripts/june_dev.sh
# Settings (environment): RUN (eval-dev-trained), SFT (runs/controller/sft/adapter),
# RL (runs/controller/rl/first/r1/adapter), JUDGE (1; 0 = generate only).
#SBATCH --job-name=jev-june
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=june-dev-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

RUN=${RUN:-eval-dev-trained}
SFT=${SFT:-runs/controller/sft/adapter}
RL=${RL:-runs/controller/rl/first/r1/adapter}
JUDGE=${JUDGE:-1}
for dir in "$SFT" "$RL"; do
  [ -f "$dir/adapter_model.safetensors" ] || { echo "no adapter at $dir" >&2; exit 2; }
done

# 1. Three servers; the policies check each server's adapter hash against the config.
PIDS=()
stop_servers() {  # uv may not pass the signal on: stop the python processes by port too
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  pkill -u "$USER" -f "jev controller serve .*--port 994[012]" 2>/dev/null || true
  for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
  PIDS=()
}
trap stop_servers EXIT
scripts/train_env.sh jev controller serve --port 9940 & PIDS+=($!)
scripts/train_env.sh jev controller serve --adapter "$SFT" --port 9941 & PIDS+=($!)
scripts/train_env.sh jev controller serve --adapter "$RL" --port 9942 & PIDS+=($!)
for port in 9940 9941 9942; do
  for _ in $(seq 180); do
    curl -sf "http://127.0.0.1:$port/info" >/dev/null && break
    for pid in "${PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null || { echo "a d1 server exited" >&2; exit 1; }
    done
    sleep 5
  done
  curl -sf "http://127.0.0.1:$port/info"; echo
done

# 2. Generation.
uv run jev run evaluate --variant dev-trained --run-id "$RUN"

# 3. Judging and the report (no GPU needed).
stop_servers
trap - EXIT
[ "$JUDGE" = 1 ] || exit 0
uv run jev eval "$RUN"
uv run jev report "$RUN" --baseline ctl_fixed
uv run jev controller frontier "$RUN" | tee "runs/$RUN/frontier.md"
echo "report: runs/$RUN/report.md, runs/$RUN/frontier.md"
