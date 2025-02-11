#!/bin/bash -e
#SBATCH --job-name=f_dynamic                  
#SBATCH --output=/home/dungnt206/workspace/code/DynamiCrafter/logs/dummy/output_file.txt
#SBATCH --error=/home/dungnt206/workspace/code/DynamiCrafter/logs/dummy/error_file.txt
#SBATCH --partition=research                
#SBATCH --gpus=2                         
#SBATCH --nodes=1                        
#SBATCH --cpus-per-task=48                
#SBATCH --mem=240GB
#SBATCH --mail-type=begin        
#SBATCH --mail-type=end          
#SBATCH --mail-type=fail          
#SBATCH --mail-user=v.dungnt206@vinai.io
#SBATCH --exclude=sdc2-hpc-dgx-a100-015

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate dynamic
cd /home/dungnt206/workspace/code/DynamiCrafter

# Start training
# sh '/home/dungnt206/workspace/code/DynamiCrafter/configs/training_256_v1.0/run.sh' > /home/dungnt206/workspace/code/DynamiCrafter/logs/output.txt 2>&1 &

name="training_256_v1.0"
config_file=configs/${name}/config.yaml

# save root dir for logs, checkpoints, tensorboard record, etc.
save_root="/home/dungnt206/workspace/code/DynamiCrafter/logs/train"

mkdir -p $save_root/$name
export HOST_GPU_NUM=2
## run
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
python3 -m torch.distributed.launch \
    --nproc_per_node=$HOST_GPU_NUM --nnodes=1 --master_addr=127.0.0.1 --master_port=12356 --node_rank=0 \
    ./main/trainer.py \
    --base $config_file \
    --train \
    --name $name \
    --logdir $save_root \
    --devices $HOST_GPU_NUM \
    lightning.trainer.num_nodes=1 #> /home/dungnt206/workspace/code/DynamiCrafter/logs/output.txt 2>&1 &

# train_pid=$!

## End of training
