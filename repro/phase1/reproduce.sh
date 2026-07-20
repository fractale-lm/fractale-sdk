#!/usr/bin/env bash
# Reproduce the Fractale-350M-base phase-1 pretrain (19,600 steps, 8x A100).
# See README.md in this folder. Usage:
#   export HF_TOKEN=hf_...   # bigcode/the-stack is gated
#   [WORKDIR=/workspace] [NPROC=8] ./reproduce.sh [--resume]
set -euo pipefail

COMMIT=073bb67                 # provenance commit of the released checkpoint
WORKDIR="${WORKDIR:-/workspace}"
NPROC="${NPROC:-8}"
REPO_DIR="$WORKDIR/thought-bank"
CFG=deepseek_v4_mini/configs/v350_phase1_10b.yaml

[ -n "${HF_TOKEN:-}" ] || { echo "HF_TOKEN is required (the-stack is gated)"; exit 1; }
python -c "import torch; assert tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) >= (2, 6), torch.__version__" \
  || { echo "PyTorch >= 2.6 required (compile + grad checkpointing)"; exit 1; }

mkdir -p "$WORKDIR/data_cache" "$WORKDIR/checkpoints" "$WORKDIR/runs"

# 1. training code, pinned at the provenance commit
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone https://github.com/kkuette/thought-bank "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --all --quiet
git -C "$REPO_DIR" checkout --quiet "$COMMIT"

# 2. environment
pip install -q -r "$REPO_DIR/requirements.txt"

# 3. pre-tokenized data cache (cache-keyed; a second call is a no-op)
( cd "$REPO_DIR" && python scripts/farm/prebuild_data.py --cache-dir "$WORKDIR/data_cache" "$CFG" )

# 4. the run (config reads/writes under /workspace; symlink if WORKDIR differs)
if [ "$WORKDIR" != "/workspace" ] && [ ! -e /workspace ]; then
  ln -s "$WORKDIR" /workspace 2>/dev/null || \
    echo "WARN: cannot symlink /workspace -> $WORKDIR; edit cache_dir/save_dir/metrics_file in $CFG"
fi
cd "$REPO_DIR"
if [ "$NPROC" -gt 1 ]; then
  torchrun --nproc_per_node="$NPROC" -m deepseek_v4_mini.code_defer_native "$CFG" "$@"
else
  python -m deepseek_v4_mini.code_defer_native "$CFG" "$@"
fi
