#!/bin/bash

#SBATCH --job-name=q1
#SBATCH --output=q1.out
#SBATCH --ntasks=20
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr

module load mpi4py/3.0.1a0_py3

for i in {1..20}
do
  mpirun -n $i python ./q1.py &
done
wait

python ./plot.py q1_result.csv 
