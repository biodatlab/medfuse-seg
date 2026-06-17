#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -c 64
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH -J medfuse_eval
#SBATCH --mem=256G

# Load your environment
# e.g., ml Mamba; module load cuda/12.6; conda activate your-env
conda activate your-conda-env

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Path to your fine-tuned checkpoint
export CKPT_PATH="path/to/ckpt_model"

echo "Starting evaluation on 4 GPUs with DeepSpeed..."

deepspeed --num_gpus=4 --master_port=29501 evaluate.py \
  --version="google/medgemma-4b-it" \
  --model_name="google/medgemma-4b-it" \
  --vision-tower="google/medgemma-4b-it" \
  --dataset_dir="./dataset" \
  --vision_pretrained="medsam_vit_b.pth" \
  --val_dataset="hf_refseg|test" \
  --precision="bf16" \
  --workers=48 \
  --lora_r=64 \
  --lora_alpha=128 \
  --lora_target_modules="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj,out_proj,fc1,fc2" \
  --model_path=$CKPT_PATH
