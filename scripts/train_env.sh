#!/usr/bin/env bash
# Run a command in the d1-3B environment, .venv-train (the `train` dependency group, which
# conflicts with `deepscholar`: LOTUS pins transformers < 5). Example:
#   scripts/train_env.sh jev controller train sanity      (a recipe of configs/training.yaml)
set -euo pipefail
cd "$(dirname "$0")/.."
UV_PROJECT_ENVIRONMENT=.venv-train exec uv run --no-default-groups --group train --group dev "$@"
