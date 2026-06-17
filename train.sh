#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -c 64
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH -J medfuseseg
#SBATCH --mem=256G

# Load your environment
# e.g., ml Mamba; module load cuda/12.6; conda activate your-env
conda activate your-conda-env

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting distributed training on 4 GPUs with DeepSpeed..."

deepspeed --num_gpus=4 --master_port=24302 train_ds.py \
  --version="google/medgemma-4b-it" \
  --vision-tower="google/medgemma-4b-it" \
  --dataset_dir="./dataset" \
  --vision_pretrained="medsam_vit_b.pth" \
  --val_dataset="hf_refseg|test" \
  --exp_name="medfuseseg" \
  --epochs=5 \
  --steps_per_epoch=13371 \
  --batch_size=8 \
  --grad_accumulation_steps=1 \
  --lr=1e-4 \
  --precision="bf16" \
  --lora_r=64 \
  --lora_alpha=128 \
  --workers=16 \
  --gradient_checkpointing
