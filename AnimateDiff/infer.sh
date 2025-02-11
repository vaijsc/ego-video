#!/bin/bash -e
#SBATCH --job-name=animate_diff                   
#SBATCH --output=/home/dungnt206/workspace/code/AnimateDiff/output_file.txt
#SBATCH --error=/home/dungnt206/workspace/code/AnimateDiff/error_file.txt
#SBATCH --partition=research                
#SBATCH --gpus=1                      
#SBATCH --nodes=1                        
#SBATCH --cpus-per-task=16                
#SBATCH --mem=40GB
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
$(conda run -n animated which torchrun) --nnodes=1 --nproc_per_node=1 --master_port=29595 train.py --config configs/training/inference/full/v2/inference_magvit.yaml

# train_pid=$!s

# # Start GPU monitoring
# while true; do
#     nvidia-smi >> nvidia-smi.log
#     sleep 1
# done &
# watch_pid=$!

# # Trap to ensure cleanup
# trap "kill -9 $train_pid $watch_pid; exit 1" SIGINT SIGTERM EXIT

# # Monitoring parameters
# max_idle_time=3900  # 65 phút
# check_interval=60
# idle_counter=0

# while true; do
#     sleep $check_interval
    
#     # Check if training process is still running
#     if ! ps -p $train_pid > /dev/null; then
#         echo "$(date): Training process has stopped. Exiting monitoring loop."
#         kill -9 $watch_pid
#         exit 0
#     fi

#     # Check GPU utilization
#     gpu_utils=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null || echo "error")
#     if [[ $gpu_utils == "error" ]]; then
#         echo "$(date): Error querying GPU utilization. Exiting."
#         kill -9 $watch_pid
#         exit 1
#     fi

#     all_idle=true
#     for util in $gpu_utils; do
#         if [[ $util -ne 0 ]]; then
#             all_idle=false
#             break
#         fi
#     done

#     # Update idle counter
#     if $all_idle; then
#         idle_counter=$((idle_counter + check_interval))
#     else
#         idle_counter=0
#     fi

#     # Stop if GPU idle for too long
#     if [[ $idle_counter -ge $max_idle_time ]]; then
#         echo "$(date): GPU has been idle for $((max_idle_time / 60)) minutes. Terminating training job."
#         kill -9 $train_pid $watch_pid
#         exit 1
#     fi
# done

# # Wait for training to complete
# wait $train_pid
# kill -9 $watch_pid
# echo "Training completed, GPU monitoring stopped."
