#!/usr/bin/env bash
# Launch FastAPI dev server (auto-reload). Linux/macOS.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
else
    PY="python"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

echo "Using interpreter: $PY"
echo "Starting uvicorn at http://${HOST}:${PORT}"
echo

"$PY" -m uvicorn server:app --reload --host "$HOST" --port "$PORT"
