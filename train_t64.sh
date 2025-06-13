#!/bin/bash
#SBATCH --account=pr_126_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --job-name=T64
#SBATCH --mail-type=END
#SBATCH --mail-user=wh2528@nyu.edu
#SBATCH --output=logs/T64/%j/slurm.out
#SBATCH --error=logs/T64/%j/slurm.err

cd /scratch/wh2528/SakanaAI/nanoGPT
source .venv/bin/activate

python train_t64.py --log_dirname="logs/T64/${SLURM_JOB_ID}"
