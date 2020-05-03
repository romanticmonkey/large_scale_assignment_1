#!/bin/bash

#SBATCH --job-name=q3_mpi
#SBATCH --output=q3_mpi.out
#SBATCH --ntasks=20
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

module load mpi4py/3.0.1a0_py3

python ./q3_mpi.py
python ./q3_plot.py q3_result.csv 
