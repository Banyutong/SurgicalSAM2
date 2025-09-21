#!/bin/bash

export http_proxy="http://localhost:20171"
export https_proxy="http://localhost:20171"

# >>> 1. 统计 GPU
GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
[[ $GPU_COUNT -eq 0 ]] && { echo "No GPU found"; exit 1; }

# >>> 2. 启动与主脚本同生命周期的 agent
for ((i=0; i<GPU_COUNT; i++)); do
    /bd_byta6000i0/users/surgicaldinov2/miniforge3/condabin/conda run \
        --live-stream -n sam2 \
        env CUDA_VISIBLE_DEVICES=$i \
        wandb agent sjtu-edu-cn/sam2-sweep/izmhc4xv  &
done

# >>> 3. 关键：阻塞，直到所有后台子进程退出
wait
echo "All agents finished."
