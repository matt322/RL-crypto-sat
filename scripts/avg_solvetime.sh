#!/bin/bash
#SBATCH --job-name=avg_solvetime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j.out

module load python/3.12  

source venv/bin/activate  

python python/avg_solvetime.py  