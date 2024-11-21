#!/bin/bash
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export DS_SKIP_CUDA_CHECK=1
export FLUX_DEV=/mnt/petrelfs/zhaoshitian/alpha_vl/zhaoshitian/FLUX.1-dev/flux1-dev.safetensors
export AE=/mnt/petrelfs/zhaoshitian/alpha_vl/zhaoshitian/FLUX.1-dev/ae.safetensors
export TORCH_DISTRIBUTED_DEBUG=DETAIL
# srun -p Gvlab-S1-32  --gres=gpu:4 --cpus-per-task 8 -n1 --ntasks-per-node=1 --quotatype=spot --job-name flux \
# accelerate launch train_flux_deepspeed_controlnet.py --config /mnt/petrelfs/zhaoshitian/x-flux-forge/train_configs/controlnet.yaml

#SBATCH -J wubaodong-job
#SBATCH -o /mnt/petrelfs/zhaoshitian/x-flux-forge/logs/train-%j.out
#SBATCH -e /mnt/petrelfs/zhaoshitian/x-flux-forge/logs/train-%j.err
srun accelerate launch train_flux_deepspeed_controlnet.py --config /mnt/petrelfs/zhaoshitian/x-flux-forge/train_configs/controlnet_load_data_from_json.yaml

# sbatch -p Gvlab-S1-32 --gres=gpu:8 --cpus-per-task 8 -n1 --ntasks-per-node=1 --quotatype=spot --job-name train /mnt/petrelfs/zhaoshitian/x-flux-forge/run_train.sh