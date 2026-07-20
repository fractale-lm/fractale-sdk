# Reproducing the Fractale-350M-base phase-1 pretrain

This folder reproduces the exact run that produced
[`fractale-lm/Fractale-350M-base`](https://huggingface.co/fractale-lm/Fractale-350M-base):
19,600 steps ≈ 10.8B tokens on 8× A100-80GB (~30 h of pod time, ≈ $320 of
rented compute in 2026).

The training code is **not** duplicated here. The single source of truth is
the research repo pinned at the provenance commit of the released checkpoint:
[`kkuette/thought-bank@073bb67`](https://github.com/kkuette/thought-bank/commit/073bb67)
(branch `claude/status-check-2fa903` — config and stability patches exactly
as run). `reproduce.sh` clones that commit and drives it.

## Requirements

- 8× A100-80GB (or equivalent; 1 GPU works for a debug run — see below)
- PyTorch **≥ 2.6** with CUDA (2.5.x crashes under `compile` +
  gradient checkpointing; the run used the
  `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel` image)
- A Hugging Face token: [`bigcode/the-stack`](https://huggingface.co/datasets/bigcode/the-stack)
  is **gated** — accept its terms first, then `export HF_TOKEN=...`
- Disk: ~10 GB for the pre-tokenized data cache, ~3.3 GB per checkpoint

## Run it

```bash
export HF_TOKEN=hf_...          # the-stack is gated
./reproduce.sh                  # defaults: WORKDIR=/workspace, 8 GPUs
# or:
WORKDIR=$HOME/fractale-repro NPROC=8 ./reproduce.sh
```

The script does four things, each skippable if already done:

1. clone `thought-bank` at `073bb67`
2. `pip install -r requirements.txt`
3. pre-tokenize the 13-source data mix into `$WORKDIR/data_cache`
   (`scripts/farm/prebuild_data.py`; ~2.4B-token unique pool, sampled with
   replacement to 10B)
4. launch: `torchrun --nproc_per_node=8 -m deepseek_v4_mini.code_defer_native
   deepseek_v4_mini/configs/v350_phase1_10b.yaml`

Checkpoints land in `$WORKDIR/checkpoints/v350_phase1_10b/` every 500 steps;
`final.pt` at step 19,600. To resume after an interruption, re-run with
`--resume` (the script passes it through: `./reproduce.sh --resume`).

**Debug on one GPU:** `NPROC=1 ./reproduce.sh` runs the same config
single-process (expect ~8× the wall clock; lower `data.batch_size` in the
yaml if you have less than 80 GB).

## What to expect

- **Statistical, not bit-exact, reproduction.** The MoE routing is not
  bit-deterministic across runs/hardware; you reproduce the curves and the
  endpoint quality, not the byte-identical weights.
- The deferred-continuation **GAP** (reset − carried CE on held-out
  documents) should rise from ~+1 nat early to **≈ +9 nats on code and ≈ +7
  on web** by the end, flat in depth (d2 ≈ d8). The released checkpoint
  measured +9.42 / +7.27 on a held-out eval (±0.3 nats re-run noise).
- **The NaN guard will fire.** This run skips updates when the all-reduced
  grad norm is non-finite (~16% of updates late in the run, escalating
  despite LR decay; `[nan-guard]` lines in the log are expected, not a
  crash). See the "Training incident, disclosed" section of the
  [model card](https://huggingface.co/fractale-lm/Fractale-350M-base) for
  the full story.

## Evaluate what you trained

The GAP eval replays the trainer's `evaluate()`/`evaluate_by_depth()` on
held-out per-source views; the qualitative decode is
`python -m deepseek_v4_mini.code_defer_sample_deep <cfg> <ckpt>`. Or load
your checkpoint straight into the usage kit from this repo:

```python
from fractale import BankSession
sess = BankSession.load("checkpoints/v350_phase1_10b/final.pt")
```
