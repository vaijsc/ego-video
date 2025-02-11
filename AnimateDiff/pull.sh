sshpass -p $SSHPASS rsync -aPv \
    --exclude='*.ckpt' \
    --exclude='*.safetensors' \
    --exclude='*.bin' \
    --exclude='*.h5' \
    --exclude='*.tiff' \
    --exclude='*.pth' \
    --exclude='*.pkl' \
    --exclude='*.tar' \
    --exclude='*.gz' \
    --exclude='*.nii' \
    superpod:~/workspace/code/AnimateDiff ~/VinAI/code