#!/usr/bin/env bash
# One bounded RL round of d1 (docs/06) as a single GPU job:
#   serve the round's adapter (.venv-train) -> sample the groups -> stop the server ->
#   build the RL data -> one RL update (KL to the SFT adapter; reverted above 0.05 nats).
# Usage (from the repository):
#   sbatch scripts/rl_round.sh                 # round 1, from the SFT adapter
#   ROUND=2 sbatch scripts/rl_round.sh         # round 2, from round 1's adapter
#   ROUND=2 NGROUPS=16 LR=2e-5 RUN_USD=0.4 sbatch scripts/rl_round.sh   # 64 trajectories
# Settings (environment): ROUND (1|2), NGROUPS (4), SIZE (4), RUN_USD (0.15 per group run),
# LR (5e-5), STEPS (4 optimiser steps), SOURCE (rollouts-train), SFT (runs/controller/sft/adapter), PORT (9940).
#SBATCH --job-name=jev-rl
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=rl-round-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

ROUND=${ROUND:-1}
NGROUPS=${NGROUPS:-4}
SIZE=${SIZE:-4}
RUN_USD=${RUN_USD:-0.15}
LR=${LR:-5e-5}
STEPS=${STEPS:-4}
SOURCE=${SOURCE:-rollouts-train}
SFT=${SFT:-runs/controller/sft/adapter}
PORT=${PORT:-9940}
OUT=runs/controller/rl/r$ROUND
DATA=data/derived/controller/rl/r$ROUND.jsonl
PREFIX=groups-$SOURCE-r$ROUND

case $ROUND in
  1) ADAPTER=$SFT ;;
  2) ADAPTER=runs/controller/rl/r1/adapter ;;
  *) echo "ROUND is 1 or 2 (at most two update rounds)" >&2; exit 2 ;;
esac
[ -f "$ADAPTER/adapter_model.safetensors" ] || { echo "no adapter at $ADAPTER" >&2; exit 2; }
[ "$PORT" = 9940 ] || { echo "the d1 policy's url is port 9940" >&2; exit 2; }
echo "round $ROUND: adapter $ADAPTER, $NGROUPS groups x $SIZE, reference $SFT"

# 1. Serve the adapter in the background; stop it however the job ends.
scripts/train_env.sh jev controller serve --adapter "$ADAPTER" --port "$PORT" &
SERVER=$!
stop_server() {  # uv may not pass the signal on: stop the python process by its port too
  kill "$SERVER" 2>/dev/null || true
  pkill -u "$USER" -f "jev controller serve .*--port $PORT" 2>/dev/null || true
  wait "$SERVER" 2>/dev/null || true
}
trap stop_server EXIT
for _ in $(seq 180); do
  curl -sf "http://127.0.0.1:$PORT/info" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "the d1 server exited" >&2; exit 1; }
  sleep 5
done
curl -sf "http://127.0.0.1:$PORT/info"; echo

# 2. Sample the groups (main environment; DeepSeek via the gateway).
uv run jev controller groups "$SOURCE" --adapter "$ADAPTER" --round "$ROUND" \
  --groups "$NGROUPS" --size "$SIZE" --run-usd "$RUN_USD" --prefix "$PREFIX"

# 3. Stop the server: the update needs the GPU.
stop_server
trap - EXIT
for _ in $(seq 30); do curl -sf "http://127.0.0.1:$PORT/info" >/dev/null || break; sleep 2; done

# 4. Rewards, advantages and the RL examples.
mkdir -p "$(dirname "$DATA")"
uv run jev controller rl-data runs/"$PREFIX"-b* --out "$DATA"

# 5. The update.
scripts/train_env.sh jev controller rl-update "$DATA" --adapter "$ADAPTER" --reference "$SFT" \
  --out "$OUT" --lr "$LR" --steps "$STEPS"
