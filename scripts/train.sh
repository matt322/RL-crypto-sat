#!/bin/bash
#SBATCH --job-name=avg_solvetime
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint=sm_70
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/%j.out


module load python/3.12  
module load cuda/12.6
module load cudnn/8.9.7.29-12-cuda12.6

source venv/bin/activate  

python python/ppo.py  