# Slurm on Gilbreth

This project's compute (LLM API calls via `fast_llm.py`, pandas/sklearn analysis, GDELT/BigQuery
pulls) is CPU + network, not GPU, except where noted otherwise. Gilbreth has no CPU-only
partition, and **the scheduler rejects any job that requests zero GPUs**
(`Job rejected: No GPUs requested`) — confirmed by testing, not documented anywhere obvious. So
every job, even pure-CPU ones, must request `--gres=gpu:1` (or more); it just won't use it.
Nodes are not exclusive, so the rest of a GPU node's CPU/RAM is still shared with other jobs.

Everything (account, partition, QOS, time, cpus, mem, GPU) is passed as `sbatch` flags at submit
time — nothing is hardcoded in `job.sbatch` — because which account/partition is free changes job
to job. Check first:

```
slist                 # GPUs total/queued/running/free per account and GPU type
sinfo                 # node state by partition
squeue --qos=standby  # cluster-wide standby usage (cap is 24 GPUs in use at once, shared by everyone)
```

Submit from the repo root (the job script `cd`s to `$SLURM_SUBMIT_DIR`):

```
sbatch [sbatch flags...] slurm/job.sbatch <command to run>
```

Output/error go to `slurm/logs/<job-name>-<job-id>.out`.

## Your accounts

| Account    | GPU type   | Total GPUs |
|------------|------------|------------|
| `ai-forge` | A10        | 4          |
| `csml`     | A30        | 7          |
| `dgoldwas` | A100-80gb  | 1          |

## Two lanes

**Short / experimental (free, no allocation charged):** `--qos=standby`. Max walltime 4h.
Cluster-wide cap of 24 GPUs in use at once across *all* standby users — not per-account. Every
partition allows every account (`AllowAccounts=ALL`), so standby also reaches the much bigger
A100 pools (a100-40gb: 104 GPUs, a100-80gb: 114 GPUs), not just `dgoldwas`'s own 1 GPU. Lower
priority and preemptible — don't rely on it for anything that can't be rerun.

```
# CPU-bound work (API calls, pandas/sklearn) — still must request a GPU, just doesn't use it
sbatch --account=csml --partition=a30 --qos=standby --time=01:00:00 \
       --cpus-per-task=4 --mem=16G --gres=gpu:1 \
       slurm/job.sbatch python code/run_direction_baseline_set.py --run

# actually needs a GPU
sbatch --account=ai-forge --partition=a100-80gb --qos=standby --time=02:00:00 \
       --cpus-per-task=4 --mem=32G --gres=gpu:1 \
       slurm/job.sbatch python code/some_gpu_script.py
```

**Long-running (charged against that account's allocation):** `--qos=normal`. Max walltime 14
days. Pick whichever of the three accounts has free capacity for the GPU type you need (or none,
for CPU-only).

```
sbatch --account=dgoldwas --partition=a100-80gb --qos=normal --time=2-00:00:00 \
       --cpus-per-task=8 --mem=32G --gres=gpu:1 \
       slurm/job.sbatch python code/some_long_gpu_job.py
```

## GPU requests

Use generic `--gres=gpu:<n>` — the partition you pick already fixes the GPU type (a10/a30/etc.),
there's no separate type-qualified gres name to set. Always include it, even for CPU-only jobs
(see above).

## Notes

- `--time` has no default under either QOS worth relying on (partition default is 30 min) —
  always set it explicitly.
- `job.sbatch` activates the `wvs` conda env and `exec`s whatever command you pass after it.
- `slurm/logs/` must exist before submission (tracked via `.gitkeep`); Slurm won't create it.
- Verified end-to-end 2026-10-26: submitted a real `--qos=standby` job (csml/a30/gres=gpu:1) that
  activated `wvs` and imported `pandas`; completed in 12s, exit 0.
- 2026-10-27: `sklearn` (1.8.0) imports fine in `wvs` under Slurm (csml/a30/standby, job 12032555)
  — the "not installed" note above was stale, `sklearn` must have been added to the env since.
