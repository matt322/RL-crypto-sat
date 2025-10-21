#!/bin/bash
#SBATCH --job-name=test_solver_interface
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=slurm_logs/%j.out
#SBATCH --partition=cpu

glucose_modified/simp/glucose cnf/instance_21_rounds_20.cnf -decisions=50000 -model -verb=1 -cpu-lim=40 