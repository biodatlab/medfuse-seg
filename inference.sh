#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH -c 64
#SBATCH -t 24:00:00
#SBATCH -p gpu
#SBATCH -J medfuse_infer
#SBATCH --mem=256G

# Load your environment
# e.g., ml Mamba; module load cuda/12.6; conda activate your-env
conda activate your-conda-env

export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

echo "Starting inference on 4 GPUs with DeepSpeed..."

deepspeed --num_gpus=4 --master_port=29500 inference.py \
  --version="google/medgemma-4b-it" \
  --model_name="google/medgemma-4b-it" \
  --vision-tower="google/medgemma-4b-it" \
  --model_path="path/to/ckpt_model" \
  --dataset_dir="./dataset" \
  --vision_pretrained="medsam_vit_b.pth" \
  --val_dataset="hf_refseg|test" \
  --results_dir="./inference_results" \
  --image_size=1024 \
  --precision="bf16" \
  --workers=8 \
  --lora_r=32 \
  --lora_alpha=64 \
  --val_batch_size=8
