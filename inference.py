#!/usr/bin/env python
# -*- coding: utf-8 -*-
# inference_multi_gpu.py

import argparse
import os
import sys
import cv2
import pandas as pd
import numpy as np
import torch
import tqdm
import deepspeed  
from peft import LoraConfig, get_peft_model
from functools import partial
# --- [CRITICAL FIX] Remove PIL image pixel limit ---
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None 
ImageFile.LOAD_TRUNCATED_IMAGES = True
# ----------------------------------------------

from transformers import AutoProcessor
from torch.utils.data.distributed import DistributedSampler 
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from model.MedFuseSeg import MedFuseSegForCausalLM
from utils.dataset import ValDataset, collate_fn
from utils.utils import dict_to_cuda

def parse_args(args):
    parser = argparse.ArgumentParser(description="MedFuseSeg Multi-GPU Inference Script")
    
    # Model Arguments
    parser.add_argument("--version", default="google/medgemma-4b-it")
    parser.add_argument("--model_name", default="google/medgemma-4b-it", type=str)
    parser.add_argument("--model_path", default="", type=str, help="Path to DeepSpeed checkpoint")
    parser.add_argument("--vision_pretrained", default="medsam_vit_b.pth", type=str)
    parser.add_argument("--vision-tower", default="google/medgemma-4b-it", type=str)

    # Config
    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    
    # Loss weights (Required for model init, though not used in inference)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=4.0, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)

    # LoRA
    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj,out_proj,fc1,fc2", type=str)
    
    # Data & Output
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--val_dataset", default="hf_refseg|test", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--results_dir", default="./inference_results_multi_gpu", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int, help="Keep 1 for inference to match filenames easily")
    parser.add_argument("--workers", default=4, type=int)
    
    return parser.parse_args(args)

def find_linear_layers(m, lora_target_modules):
    cls = torch.nn.Linear
    keep = set()
    for name, mod in m.named_modules():
        if isinstance(mod, cls):
            if all(x not in name for x in [
                "visual_model", "vision_tower", "multi_modal_projector", "projector"
            ]) and any(x in name for x in lora_target_modules):
                keep.add(name)
    return sorted(list(keep))

def run_inference(val_loader, model_engine, tokenizer, args): 
    """Inference function: Generates text & masks and saves them."""
    
    # Create Output Dir
    if args.local_rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
    
    model_engine.eval() 
    
    # List to store results for CSV
    results_list = []
    
    # Progress bar only for Rank 0
    iterator = tqdm.tqdm(val_loader) if args.local_rank == 0 else val_loader

    for input_dict in iterator:
        # torch.cuda.empty_cache()

        # 1. Extract Metadata
        image_paths = input_dict.get("image_paths", [])
        dataset_names = input_dict.pop("dataset_names", [args.val_dataset] * len(image_paths))
        
        # 2. Prepare Inputs
        input_dict = dict_to_cuda(input_dict)
        
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["pixel_values"] = input_dict["pixel_values"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["pixel_values"] = input_dict["pixel_values"].float()

        # 3. Generate (Model Inference)
        # Use module.evaluate for actual generation, not just a forward pass
        with torch.no_grad():
            output_ids, pred_masks = model_engine.module.evaluate(
                pixel_values=input_dict["pixel_values"],
                images=input_dict["images"],
                input_ids=input_dict["input_ids"],
                attention_mask=input_dict["attention_masks"],
                resize_list=input_dict["resize_list"],
                original_size_list=[(1024, 1024)] * len(image_paths), # Assuming 1024 output
                max_new_tokens=32,
                tokenizer=tokenizer,
            )

        # 4. Save Results
        for i in range(len(image_paths)):
            fake_path = image_paths[i]
            ds_name = dataset_names[i]
            
            # Resolve Filename
            original_filename = f"image_{fake_path.replace('/', '_')}.png"
            if "hf_index" in fake_path:
                try:
                    # Attempt to extract dataset ID from synthetic path
                    idx_str = fake_path.split("/")[-1]
                    original_filename = f"hf_{idx_str}.png"
                except:
                    pass

            # Save Mask
            save_folder = os.path.join(args.results_dir, ds_name)
            os.makedirs(save_folder, exist_ok=True)
            
            # Check whether any masks were predicted
            if i < len(pred_masks) and pred_masks[i].numel() > 0:
                mask_tensor = pred_masks[i]
                if mask_tensor.dim() > 2: mask_tensor = mask_tensor[0] # Take first mask if multiple
                
                mask_np = (mask_tensor.sigmoid().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
                save_path = os.path.join(save_folder, original_filename)
                try:
                    cv2.imwrite(save_path, mask_np)
                except Exception as e:
                    print(f"[Rank {args.local_rank}] Error saving {save_path}: {e}")

            # Decode Text
            if i < len(output_ids):
                generated_text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
                generated_text = generated_text.replace("\n", " ").strip()
                
                results_list.append({
                    "dataset": ds_name,
                    "filename": original_filename,
                    "generated_text": generated_text
                })

    # 5. Save Partial CSV (Per Rank)
    if len(results_list) > 0:
        df = pd.DataFrame(results_list)
        csv_name = f"results_rank{args.local_rank}.csv"
        df.to_csv(os.path.join(args.results_dir, csv_name), index=False)
        print(f"[Rank {args.local_rank}] Saved {len(df)} records to {csv_name}")

def main():
    args = parse_args(sys.argv[1:])

    # Init DeepSpeed Distributed
    deepspeed.init_distributed()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)

    if args.local_rank == 0:
        print(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    if "[SEG]" not in tokenizer.get_vocab():
        tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16": torch_dtype = torch.bfloat16
    elif args.precision == "fp16": torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}

    # Model Configuration
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
    }
    model = MedFuseSegForCausalLM.from_pretrained(args.version, low_cpu_mem_usage=True, **model_args, **kwargs)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_med_fuse_seg_modules(model.get_model().config)

    # LoRA Setup
    if args.lora_r > 0:
        targets = find_linear_layers(model, args.lora_target_modules.split(","))
        lconf = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=targets, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lconf)

    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(device=args.local_rank, dtype=torch_dtype)

    # DeepSpeed Configuration
    ds_config = {
        "train_micro_batch_size_per_gpu": args.val_batch_size,
        "fp16": { "enabled": args.precision == "fp16" },
        "bf16": { "enabled": args.precision == "bf16" },
    }

    model_engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

    # Load Checkpoint
    if args.model_path and os.path.isdir(args.model_path):
        if args.local_rank == 0: print(f"Loading checkpoint: {args.model_path}")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(args.model_path)
        model_engine.module.load_state_dict(state_dict, strict=True)

    # Dataloader Setup (The Robust Part)
    model_engine.eval()
    val_dataset = ValDataset(args.dataset_dir, processor, args.vision_tower, args.val_dataset, args.image_size)
    
    # Use DistributedSampler to handle Multi-GPU Logic
    val_sampler = DistributedSampler(val_dataset, shuffle=False) 
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True, 
        sampler=val_sampler, 
        collate_fn=partial(collate_fn, processor=processor, local_rank=args.local_rank),
    )
    
    # Run Inference Loop
    run_inference(val_loader, model_engine, tokenizer, args)

if __name__ == "__main__":
    main()