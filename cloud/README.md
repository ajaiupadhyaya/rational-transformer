# Cloud runs (Thunder Compute)

Facts read from thundercompute.com on 2026-11-17 (pricing page, docs, students page).
Prices change; re-read before every new instance.

## What the credit buys

- The $20 student credit is usable on **RTX A6000 (48 GB) at $0.35/GPU-hour** and
  **L40 (48 GB) at $0.79/GPU-hour** only. A100 80GB ($1.09) and H100 ($3.20) are not covered.
- A6000: 6 vCPUs included, up to 8 at $0.04 per extra vCPU-hour. RAM scales with vCPUs.
- Disk: first 100 GB per GPU included; billed only while the instance runs.
- Billing is per minute while the instance exists. **There is no stopped state**: deleting the
  instance ends all billing; "stopping" means snapshot ($0.05/GB-month) then delete.
- No CLI or API command reads the balance; the console billing page is the only spend view.
- Project cap: **$15 of spend, $5 reserve**; the sweep's budget guard runs with `--budget-usd 12`.

## Image

`base` template: Ubuntu, CUDA 13.0, driver 580, PyTorch preinstalled (we ignore it and use the
locked uv environment instead).

## Commands

```bash
tnr status --no-wait                                   # list instances
tnr create --gpu a6000 --num-gpus 1 --vcpus 8 --template base --disk 100
tnr connect 0                                          # first connect writes ~/.ssh/config
tnr scp <src> 0:<dst>                                  # or plain ssh/rsync via the alias
tnr delete 0                                           # ends billing; removes all data
```

`tnr create --help` crashes when not attached to a terminal (CLI 2.0.51); the flags above are
from the docs.

## Workflow

Launch from the repo root, with `RT_CONFIG` and `RT_EVAL_DIR` unset.

1. `cloud/push.sh <host>`: copies the committed tree (`git archive HEAD`), so the code hash on
   the instance equals the tagged one.
2. On the instance: `bash cloud/setup.sh`: uv, locked deps, tests, learning check and gate on
   the GPU. The sweep refuses to start without a green gate at the current hash.
3. Pilot: `uv run python -m rt.sweep --time-only --cell <id> --steps <n> --device cuda` per
   size, at 1, 4 and 8 concurrent jobs; project hours and dollars; the owner approves.
4. Sweep in tmux: `uv run python -m rt.sweep --device cuda --budget-usd 12
   --price-per-hour <P> --jobs <J>`.
5. `cloud/pull.sh <host>` every 30 minutes and at the end; it never deletes local files and
   puts the instance's gate files under `results/cloud/`.
6. Verify locally (every metrics CSV, all checkpoints, ledger rows), then `tnr delete` and
   confirm `tnr status` shows nothing.
