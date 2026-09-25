#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ -f "$DIR/build/soar_engine" ]; then
    exec "$DIR/build/soar_engine" "$@"
elif [ -f "$DIR/build/soar_engine.exe" ]; then
    exec "$DIR/build/soar_engine.exe" "$@"
else
    echo "[ERROR] soar_engine binary not found in build directory. Please build first using cmake."
    exit 1
fi
