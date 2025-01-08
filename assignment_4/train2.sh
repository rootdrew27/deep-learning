#!/bin/bash



#SBATCH --partition=GPU

#SBATCH --gpus=1

#SBATCH --time=5:00:00

#SBATCH --nodes=1

#SBATCH --ntasks-per-node=10

#SBATCH --mem=15G

#SBATCH --job-name="UNET2T"

#SBATCH --output=output2.txt

#SBATCH --error=error2.txt

#SBATCH --mail-user=ROOTA5351@UWEC.EDU

#SBATCH --mail-type=END

module load python-libs

#conda activate tensorflow-2.9-gpu

python train2.py
