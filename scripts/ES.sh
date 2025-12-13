#!/bin/bash
#SBATCH --job-name=avg_solvetime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=128
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%A_%a.out
#SBATCH --partition=cpu,cpu-preempt
#SBATCH --array=0-2

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

module load python/3.12  

source venv/bin/activate  

srun python python/train.py  