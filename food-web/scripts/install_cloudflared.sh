#!/usr/bin/env bash
# Download cloudflared binary for Linux/macOS into bin/
set -euo pipefail

cd "$(dirname "$0")/.."
BIN_DIR=bin
mkdir -p "$BIN_DIR"
TARGET="$BIN_DIR/cloudflared"

if [[ -x "$TARGET" ]]; then
    echo "[INFO] $TARGET already exists. Delete it first to redownload."
    "$TARGET" --version
    exit 0
fi

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case "$ARCH" in
    x86_64|amd64) ARCH=amd64 ;;
    aarch64|arm64) ARCH=arm64 ;;
    *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac

case "$OS" in
    linux)
        URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH}"
        ;;
    darwin)
        # Mac releases are .tgz - extract on the fly
        URL="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-${ARCH}.tgz"
        ;;
    *)
        echo "Unsupported OS: $OS" >&2; exit 1 ;;
esac

MIRROR="https://ghproxy.com/$URL"

echo "[INFO] Downloading cloudflared from: $URL"
if ! curl -fL --connect-timeout 15 -o "$TARGET.tmp" "$URL"; then
    echo "[WARN] Direct download failed, retry via ghproxy: $MIRROR"
    curl -fL --connect-timeout 30 -o "$TARGET.tmp" "$MIRROR"
fi

if [[ "$OS" == "darwin" ]]; then
    tar -xzf "$TARGET.tmp" -C "$BIN_DIR"
    rm -f "$TARGET.tmp"
else
    mv "$TARGET.tmp" "$TARGET"
fi
chmod +x "$TARGET"

echo "[OK] Installed at $(pwd)/$TARGET"
"$TARGET" --version
