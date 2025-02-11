#!/bin/bash -e
#SBATCH --job-name=animate_diff                   # Create a short name for your job
#SBATCH --output=/home/dungnt206/workspace/code/AnimateDiff/output_file.txt
#SBATCH --error=/home/dungnt206/workspace/code/AnimateDiff/error_file.txt
#SBATCH --partition=research                # Choose partition
#SBATCH --gpus=1                         # GPU count
#SBATCH --nodes=1                        # Node count
#SBATCH --cpus-per-gpu=12                 # CPU cores per task/ or GPU with --cpu-per-gpu
#SBATCH --mem=12GB
# Your commands here
conda activate animated
cd /home/dungnt206/workspace/code/AnimateDiff/
torchrun --nnodes=1 --nproc_per_node=1 train.py --config configs/training/v1/training_smth_20.yaml
