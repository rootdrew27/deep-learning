#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --gpus=1
#SBATCH --time=500:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=10G
#SBATCH --job-name="a3q2"
#SBATCH --output=out/a3q2.out
#SBATCH --error=error/a3q2.err
#SBATCH --mail-user=ROOTA5351@UWEC.EDU
#SBATCH --mail-type=ALL

module load python-libs

python ./scripts/Assign3_Q2.py
