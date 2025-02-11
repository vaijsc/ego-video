#!/bin/bash -e
#SBATCH --job-name=animate_diff                   # Create a short name for your job
#SBATCH --output=/home/dungnt206/workspace/code/AnimateDiff/output_file.txt
#SBATCH --error=/home/dungnt206/workspace/code/AnimateDiff/error_file.txt
#SBATCH --partition=research                # Choose partition
#SBATCH --gpus=4                         # GPU count
#SBATCH --nodes=1                        # Node count
#SBATCH --cpus-per-task=32                 # CPU cores per task/ or GPU with --cpu-per-gpu
#SBATCH --mem=120GB
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.dungnt206@vinai.io
# Your commands here

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate animated
cd /home/dungnt206/workspace/code/AnimateDiff/
# Start training
torchrun --nnodes=1 --nproc_per_node=4 train.py --config configs/training/v1/training_smth_20.yaml > logfile.log 2>&1 &
# Get the PID of the training process
train_pid=$!
# Start monitoring GPU usage in background
watch -n 1 nvidia-smi > nvidia-smi.log 2>&1 &
# Get the PID of the nvidia-smi monitoring process
watch_pid=$!
# Wait for training to finish
wait $train_pid
# Kill the nvidia-smi monitoring process
kill $watch_pid
echo "Training completed, nvidia-smi monitoring stopped."

