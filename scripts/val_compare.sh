#!/usr/bin/env bash
# Compare d1 adapters on the 9 val tasks (docs/06), judge-free, as a single GPU job:
#   1. decisions: regret on the branched val points (dev.jsonl), d1 only, no executor calls;
#   2. episodes: every val task from the start with each adapter choosing (its own server,
#      one at a time), runs/compare-<SOURCE>-<name>;
#   3. the comparison table against the fixed rule's own val run.
# Usage (from the repository):
#   sbatch scripts/val_compare.sh
#   ADAPTERS="name=dir ..." sbatch scripts/val_compare.sh   (default: d1-sft, and d1-rl as d1_first1)
# Settings (environment): ADAPTERS (name=dir ...; missing dirs are skipped), SOURCE
# (rollouts-val), DATA (data/derived/controller/sft/dev.jsonl), RUN_USD (0.6 per run),
# EPISODES (1; 0 = decisions only, no executor calls), PORT (9940).
#SBATCH --job-name=jev-val
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=val-compare-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ADAPTERS=${ADAPTERS:-"d1_sft=runs/controller/sft/adapter d1_first1=runs/controller/rl/first/r1/adapter"}
SOURCE=${SOURCE:-rollouts-val}
DATA=${DATA:-data/derived/controller/sft/dev.jsonl}
RUN_USD=${RUN_USD:-0.6}
EPISODES=${EPISODES:-1}
PORT=${PORT:-9940}
[ "$PORT" = 9940 ] || { echo "the d1 policy's url is port 9940" >&2; exit 2; }

names=() dirs=()
for item in $ADAPTERS; do
  name=${item%%=*} dir=${item#*=}
  if [ -f "$dir/adapter_model.safetensors" ]; then
    names+=("$name") dirs+=("$dir")
  else
    echo "skipping $name: no adapter at $dir"
  fi
done
[ ${#names[@]} -gt 0 ] || { echo "no adapter to compare" >&2; exit 2; }
echo "comparing ${names[*]} on $SOURCE"

# 1. Decisions (no executor calls).
args=()
for dir in "${dirs[@]}"; do args+=(--adapter "$dir"); done
scripts/train_env.sh jev controller regret "$DATA" "${args[@]}" | tee "val-regret-${SLURM_JOB_ID:-local}.jsonl"
[ "$EPISODES" = 1 ] || exit 0

# 2. Episodes, one adapter (and server) at a time.
SERVER=
stop_server() {  # uv may not pass the signal on: stop the python process by its port too
  [ -n "$SERVER" ] || return 0
  kill "$SERVER" 2>/dev/null || true
  pkill -u "$USER" -f "jev controller serve .*--port $PORT" 2>/dev/null || true
  wait "$SERVER" 2>/dev/null || true
  SERVER=
  for _ in $(seq 30); do curl -sf "http://127.0.0.1:$PORT/info" >/dev/null || break; sleep 2; done
}
trap stop_server EXIT
cells=("$SOURCE:ctl_fixed")
for i in "${!names[@]}"; do
  name=${names[$i]} dir=${dirs[$i]}
  scripts/train_env.sh jev controller serve --adapter "$dir" --port "$PORT" &
  SERVER=$!
  for _ in $(seq 180); do
    curl -sf "http://127.0.0.1:$PORT/info" >/dev/null && break
    kill -0 "$SERVER" 2>/dev/null || { echo "the d1 server exited" >&2; exit 1; }
    sleep 5
  done
  curl -sf "http://127.0.0.1:$PORT/info"; echo
  uv run jev controller episodes "$SOURCE" --name "$name" --adapter "$dir" --run-usd "$RUN_USD"
  stop_server
  cells+=("compare-$SOURCE-$name:$name")
done

# 3. The comparison (the fixed rule first).
uv run jev controller compare "${cells[@]}" | tee "val-compare-${SLURM_JOB_ID:-local}.md"
