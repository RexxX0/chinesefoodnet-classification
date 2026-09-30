#!/usr/bin/env bash
# Start a Cloudflare Quick Tunnel pointing at the local FastAPI server.
# Prereq:
#   1) Run scripts/install_cloudflared.sh once to download bin/cloudflared
#   2) Run scripts/run_dev.sh in another terminal to start the FastAPI
#      server on port 8000
#
# This script prints a public URL like:
#   https://something-random-words.trycloudflare.com
# Anyone with that URL can use your model from the public internet, with no
# account required. The URL is throwaway - it disappears when you Ctrl+C.

set -euo pipefail
cd "$(dirname "$0")/.."

CF="bin/cloudflared"
LOCAL_URL="http://localhost:8000"

if [[ ! -x "$CF" ]]; then
    echo "[ERROR] $CF not found or not executable."
    echo "        Please run scripts/install_cloudflared.sh first."
    exit 1
fi

cat <<EOF
============================================================
 Cloudflare Quick Tunnel - free, ephemeral public URL
============================================================
 Local target:  $LOCAL_URL
 Make sure your FastAPI server is already running there!
 (Run scripts/run_dev.sh in another terminal if not.)

 Look for a line like:
   Your quick Tunnel has been created! Visit it at:
   https://xxxxx-xxxxx-xxxxx-xxxxx.trycloudflare.com

 Press Ctrl+C to stop the tunnel.
============================================================

EOF

exec "$CF" tunnel --url "$LOCAL_URL" --no-autoupdate
