#!/usr/bin/env bash
# Generic single-GPU Slurm wrapper:  sbatch scripts/slurm_job.sh <command ...>
#SBATCH -p gpus
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH -o logs/%x-%j.out
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rethoi
cd "${SLURM_SUBMIT_DIR:-.}"
echo "[$(date)] $(hostname) GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "+ $*"
"$@"
