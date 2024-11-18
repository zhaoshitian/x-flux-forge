export CUDA_VISIBLE_DEVICES=4,5,6,7
export DS_SKIP_CUDA_CHECK=1
export FLUX_DEV=/data2/stzhao/model_weight/FLUX.1-dev/flux1-dev.safetensors
export AE=/data2/stzhao/model_weight/FLUX.1-dev/ae.safetensors
accelerate launch train_flux_deepspeed_controlnet.py --config "/data2/stzhao/x-flux/train_configs/controlnet.yaml"