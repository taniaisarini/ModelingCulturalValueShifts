#!/bin/bash
# Generic Slurm job wrapper for ModelingCulturalValueShifts. Activates the `wvs` conda env and
# runs the command passed as arguments. Account, partition, QOS, time, cpus, mem and GPU are
# NOT set here — pass them as sbatch flags at submit time. See slurm/README.md for the two
# lanes (standby = short/free, normal = long/charged) and example invocations.
#
# Usage (from repo root):
#   sbatch --account=<acct> --partition=<part> --qos=<standby|normal> --time=<HH:MM:SS> \
#          --cpus-per-task=<n> --mem=<size> [--gres=gpu:<n>] \
#          slurm/job.sbatch <command...>

#SBATCH --job-name=wvs-job
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.out

set -eo pipefail

cd "$SLURM_SUBMIT_DIR"

module load anaconda/2025.06-py313
source activate wvs   # conda's own activate/deactivate hooks reference unbound vars, so -u is off

echo "job $SLURM_JOB_ID ($SLURM_JOB_NAME) on $SLURM_NODELIST | account=$SLURM_JOB_ACCOUNT partition=$SLURM_JOB_PARTITION qos=$SLURM_JOB_QOS"
echo "running: $*"

exec "$@"
