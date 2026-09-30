#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
LOGDIR="${1:-../outputs/convnextv2_base/logs/tensorboard}"
PORT="${2:-6006}"

tensorboard --logdir "$LOGDIR" --host 0.0.0.0 --port "$PORT"
