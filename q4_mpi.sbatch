#!/bin/bash

#SBATCH --job-name=q4_mpi
#SBATCH --output=q4_mpi.out
#SBATCH --ntasks=20
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --error=q4_mpi.err  # error file

module load mpi4py/3.0.1a0_py3

python ./q4_mpi.py
