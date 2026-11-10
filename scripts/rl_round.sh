#!/usr/bin/env bash
# One bounded RL round of d1 (docs/06) as a single GPU job:
#   serve the round's adapter (.venv-train) -> sample the groups -> stop the server ->
#   build the RL data -> one RL update (KL to the SFT adapter; reverted above 0.05 nats).
# Its defaults are d1-rl's recipe (docs/06, bounded RL): d1 samples the restored point only, at
# the train points whose options' branch returns differ by more than 0.01, 8 per group, lr 1e-5.
# Usage (from the repository):
#   sbatch scripts/rl_round.sh
# Settings (environment): ROUND (1|2), NGROUPS (64), SIZE (8), RUN_USD (0.3 per group run),
# LR (1e-5), STEPS (4), SOURCE (rollouts-train), SFT (runs/controller/sft/adapter), PORT (9940),
# NAME (first: runs/controller/rl/<NAME>/r<ROUND>), FIRST_ONLY (1), MIN_SPREAD (0.01),
# PER_TASK (0: no limit).
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
NGROUPS=${NGROUPS:-64}
SIZE=${SIZE:-8}
RUN_USD=${RUN_USD:-0.3}
LR=${LR:-1e-5}
STEPS=${STEPS:-4}
SOURCE=${SOURCE:-rollouts-train}
SFT=${SFT:-runs/controller/sft/adapter}
PORT=${PORT:-9940}
NAME=${NAME-first}
FIRST_ONLY=${FIRST_ONLY:-1}
MIN_SPREAD=${MIN_SPREAD-0.01}
PER_TASK=${PER_TASK:-0}
VARIANT=${NAME:+$NAME/}
OUT=runs/controller/rl/${VARIANT}r$ROUND
DATA=data/derived/controller/rl/${VARIANT}r$ROUND.jsonl
PREFIX=groups-$SOURCE-${NAME:+$NAME-}r$ROUND
EXTRA=(--per-task "$PER_TASK")
[ "$FIRST_ONLY" = 1 ] && EXTRA+=(--first-only)
[ -n "$MIN_SPREAD" ] && EXTRA+=(--min-spread "$MIN_SPREAD")

case $ROUND in
  1) ADAPTER=$SFT ;;
  2) ADAPTER=runs/controller/rl/${VARIANT}r1/adapter ;;
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
  --groups "$NGROUPS" --size "$SIZE" --run-usd "$RUN_USD" --prefix "$PREFIX" "${EXTRA[@]}"

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
