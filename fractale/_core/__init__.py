# Vendored inference core of the Fractale models.
# Source of truth: https://github.com/kkuette/thought-bank (deepseek_v4_mini/).
# Re-sync with tools/vendor_core.sh — do not edit these files here.
# IMPORTANT: vendor from the branch that TRAINED the target checkpoint —
# currently claude/status-check-2fa903 @ 13f5998 (the 350M phase-1 code);
# main lags it (e.g. cfg.mem_seed_init) and cannot load phase-1 checkpoints.
from .config import ThoughtBankConfig
from .model import ThoughtBankLM, TrunkLM

__all__ = ["ThoughtBankConfig", "ThoughtBankLM", "TrunkLM"]
