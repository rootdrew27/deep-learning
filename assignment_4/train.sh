#!/bin/bash



#SBATCH --partition=GPU

#SBATCH --gpus=1

#SBATCH --time=5:00:00

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=10

#SBATCH --mem=15G

#SBATCH --job-name="UNET1"

#SBATCH --output=output.txt

#SBATCH --error=error.txt

#SBATCH --mail-user=ROOTA5351@UWEC.EDU

#SBATCH --mail-type=ALL

module load python-libs

conda activate tensorflow-2.9-gpu

python train.py
