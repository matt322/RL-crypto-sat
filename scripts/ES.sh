#!/bin/bash
#SBATCH --job-name=avg_solvetime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --partition=cpu


module load python/3.12  

source venv/bin/activate  

srun python python/evol_strategies.py  