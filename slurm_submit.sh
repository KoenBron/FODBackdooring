#!/bin/sh
#SBATCH --job-name=FOD_FasterRCNN
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm_logs/out_%j.txt
#SBATCH --error=slurm_logs/err_%j.txt

echo "=========================================================="
echo "Start FL Training on DAIC"
echo "Job ID: $SLURM_JOB_ID | Node: $SLURMD_NODENAME"
echo "=========================================================="

# apptainer exec voert jouw bash script uit BINNEN de gemaakte omgeving
# We binden de huidige directory ($PWD) aan /app in de container
apptainer exec --nv --bind $PWD:/app apptainer/fl_container.sif bash /app/run_training.sh

echo "=========================================================="
echo "Training completed!"
echo "=========================================================="
