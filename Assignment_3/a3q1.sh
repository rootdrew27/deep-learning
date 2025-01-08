#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --gpus=1
#SBATCH --time=5:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --mem=10G
#SBATCH --job-name="a3q1"
#SBATCH --output=./out/a3q1.out
#SBATCH --error=./error/a3q1.err
#SBATCH --mail-user=ROOTA5351@UWEC.EDU
#SBATCH --mail-type=ALL

module load python-libs

python ./scripts/Assign3_Q1.py
