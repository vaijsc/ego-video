#!/bin/bash -e
#SBATCH --job-name=depth                   # Create a short name for your job
#SBATCH --output=/home/dungnt206/workspace/code/Depth-Anything-V2/output_file.txt
#SBATCH --error=/home/dungnt206/workspace/code/Depth-Anything-V2/error_file.txt
#SBATCH --partition=research                # Choose partition
#SBATCH --gpus=1                         # GPU count
#SBATCH --nodes=1                        # Node count
#SBATCH --cpus-per-task=32                 # CPU cores per task/ or GPU with --cpu-per-gpu
#SBATCH --mem=40GB
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.dungnt206@vinai.io
# Your commands here

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate depth-any
cd /home/dungnt206/workspace/code/Depth-Anything-V2/
# Start training
python -u run_video.py \
  --encoder vitb \
  --video-path /home/dungnt206/workspace/data/something_something_v2/data/20bn-something-something-v2 --outdir /home/dungnt206/workspace/data/something_something_v2/data/20bn-something-something-v2/depth-20class-20bn-something-something-v2 \
  --input-size 256 \
  --pred-only > logfile.log 2>&1 &


# Get the PID of the training process
train_pid=$!
# Start monitoring GPU usage in background
watch -n 1 nvidia-smi > nvidia-smi.log 2>&1 &
# Get the PID of the nvidia-smi monitoring process
watch_pid=$!
# Wait for training to finish
wait $train_pid

# kill $watch_pid
echo "Training completed, nvidia-smi monitoring stopped."




# python run_video.py \
#   --encoder <vits| vitb | vitl | vitg> \
#   --video-path ~/workspace/data/something_something_v2/data/20class-20bn-something-something-v2/ --outdir ~/workspace/data/something_something_v2/data/depth-20class-20bn-something-something-v2 \
#   --input-size 256 
#   --pred-only --grayscale