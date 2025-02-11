#!/bin/bash -e
#SBATCH --job-name=animate_diff                   
#SBATCH --output=/home/dungnt206/workspace/code/AnimateDiff/output_file.txt
#SBATCH --error=/home/dungnt206/workspace/code/AnimateDiff/error_file.txt
#SBATCH --partition=research                
#SBATCH --gpus=2                         
#SBATCH --nodes=1                        
#SBATCH --cpus-per-task=24                
#SBATCH --mem=120GB
#SBATCH --mail-type=begin        
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail          
#SBATCH --mail-user=v.dungnt206@vinai.io

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate animated
cd /home/dungnt206/workspace/code/AnimateDiff/
echo "Using Python from $(which python)" >> /home/dungnt206/workspace/code/AnimateDiff/output_file.txt

# Start training
$(conda run -n animated which torchrun) --nnodes=1 --nproc_per_node=2 --master_port=29593 train.py --config configs/training/v3/training_smth_curated_20.yaml --tensorboard > logfile.log 2>&1 &

train_pid=$!

# Start GPU monitoring
while true; do
    nvidia-smi >> nvidia-smi.log
    sleep 1
done &
watch_pid=$!

# Trap to ensure cleanup
trap "kill -9 $train_pid $watch_pid; exit 1" SIGINT SIGTERM EXIT

# Wait for training to complete
wait $train_pid
kill -9 $watch_pid
echo "Training completed, GPU monitoring stopped."
