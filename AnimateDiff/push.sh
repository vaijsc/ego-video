sshpass -p $SSHPASS rsync -aPv \
    --exclude='*.ckpt' \
    --exclude='*.safetensors' \
    --exclude='*.bin' \
    --exclude='*.h5' \
    --exclude='*.tiff' \
    --exclude='*.pth' \
    --exclude='*.pkl' \
    --exclude='*.log' \
    --exclude='outputs' \
    ~/VinAI/code/AnimateDiff superpod:~/workspace/code