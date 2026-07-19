#!/usr/bin/env bash
# Re-sync the vendored inference core from the research repo.
# Usage: tools/vendor_core.sh [path-to-thought-bank]
set -euo pipefail
SRC="${1:-../thought-bank}/deepseek_v4_mini"
DST="$(dirname "$0")/../fractale/_core"
for f in config.py mhc.py attention.py moe.py memory.py model.py; do
  cp "$SRC/$f" "$DST/$f"
done
echo "vendored from $SRC -> $DST (do not edit _core in place)"
