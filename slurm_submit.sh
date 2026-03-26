#!/bin/sh
#SBATCH --job-name=FOD_FasterRCNN
#SBATCH --partition=insy,general
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpumem32
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/out_%j.txt
#SBATCH --error=slurm_logs/err_%j.txt
#SBATCH --mail-type=END

echo "=========================================================="
echo "Start FL Training on DAIC"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "=========================================================="

# apptainer exec voert jouw bash script uit BINNEN de gemaakte omgeving
# We binden de huidige directory ($PWD) aan /app in de container
apptainer exec --nv --bind $PWD:/app apptainer/fod_container.sif bash /app/run_slurm_train.sh

echo "=========================================================="
echo "Training completed!"
echo "=========================================================="
