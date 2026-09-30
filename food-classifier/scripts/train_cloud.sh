#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
CONFIG_PATH="${1:-configs/convnextv2_base.yaml}"
LOG_DIR="${2:-$(dirname "$CONFIG_PATH")/../cloud_logs}"
mkdir -p "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"

echo "Using config: $CONFIG_PATH"
echo "Streaming logs to: $LOG_FILE"

python src/train.py --config "$CONFIG_PATH" 2>&1 | tee "$LOG_FILE"
