#!/bin/bash
#SBATCH --job-name=q4_cl   # job name
#SBATCH --output=q4_cl.out # output log file
#SBATCH --error=q4_cl.err  # error file
#SBATCH --time=00:20:00  # 20 minutes of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU

module load cuda
module load mpi4py/3.0.1a0_py3

python ./q4_cl.py
