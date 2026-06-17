#!/usr/bin/env python
# -*- coding: utf-8 -*-
# evaluate.py

import argparse
import os
import sys
from functools import partial

import numpy as np
import torch
import tqdm
import deepspeed  
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor
from torch.utils.data.distributed import DistributedSampler 

from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from monai.metrics import compute_hausdorff_distance
from model.MedFuseSeg import MedFuseSegForCausalLM
from utils.dataset import ValDataset, collate_fn
from utils.utils import (AverageMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
import pandas as pd

def parse_args(args):
    parser = argparse.ArgumentParser(description="MedFuseSeg Model Evaluation Script (Multi-GPU)")
    
    parser.add_argument("--version", default="google/medgemma-4b-it")
    parser.add_argument("--model_name", default="google/medgemma-4b-it", type=str)
    parser.add_argument("--model_path", default="", type=str, help="Path to DeepSpeed checkpoint")
    parser.add_argument("--vision_pretrained", default="medsam_vit_b.pth", type=str)
    parser.add_argument("--vision-tower", default="google/medgemma-4b-it", type=str)

    parser.add_argument("--precision", default="bf16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=2.0, type=float)
    parser.add_argument("--bce_loss_weight", default=4.0, type=float)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)

    parser.add_argument("--lora_r", default=16, type=int)
    parser.add_argument("--lora_alpha", default=32, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj,out_proj,fc1,fc2", type=str)
    
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--val_dataset", default="hf_refseg|test", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    
    return parser.parse_args(args)

def find_linear_layers(m, lora_target_modules):
    cls = torch.nn.Linear
    keep = set()
    for name, mod in m.named_modules():
        if isinstance(mod, cls):
            if all(x not in name for x in [
                "visual_model", "projector", "fusion_adapter"
            ]) and any(x in name for x in lora_target_modules):
                keep.add(name)
    return sorted(list(keep))

# [MODIFIED] เพิ่ม task argument เพื่อแยก key
def get_or_create_meters(dataset_name, task_name, all_meters_dict):
    """Create/retrieve metric meters for a specific (dataset, task) pair."""
    key = (dataset_name, task_name)
    if key not in all_meters_dict:
        all_meters_dict[key] = {
            "intersection_meter": AverageMeter("Intersec", ":6.3f", Summary.SUM),
            "union_meter": AverageMeter("Union", ":6.3f", Summary.SUM),
            "acc_iou_meter": AverageMeter("gIoU", ":6.3f", Summary.AVERAGE),
            "dice_meter": AverageMeter("Dice", ":6.3f", Summary.AVERAGE),
            "hd95_meter": AverageMeter("HD95", ":6.4f", Summary.AVERAGE),
        }
    return all_meters_dict[key]

def validate(val_loader, model_engine, epoch, writer, args): 
    """Validation function handling per-dataset AND per-task metrics."""
    # Dictionary to store meters per (dataset, task)
    all_meters = {}

    per_sample_results = []

    model_engine.eval() 
    
    if args.local_rank == 0:
        val_loader_tqdm = tqdm.tqdm(val_loader)
    else:
        val_loader_tqdm = val_loader

    for input_dict in val_loader_tqdm:
        torch.cuda.empty_cache()

        # [MODIFIED] ดึง dataset_names และ tasks จาก batch
        # หมายเหตุ: ต้องแก้ dataset.py ให้ส่ง 'tasks' มาด้วย ถ้าไม่มีจะใส่ 'unknown'
        batch_dataset_names = input_dict.pop("dataset_names", [args.val_dataset] * len(input_dict["images"]))
        batch_tasks = input_dict.pop("tasks", ["unknown"] * len(input_dict["images"]))
        batch_image_paths = input_dict.pop("image_paths", ["unknown"] * len(input_dict["images"]))

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

        with torch.no_grad():
            output_dict = model_engine(**input_dict) 

        pred_masks = output_dict["pred_masks"]
        gt_masks = output_dict["gt_masks"]
        
        # วนลูปตามจำนวน sample ใน batch เพื่อเก็บคะแนนลงถัง (Meter) ที่ถูกต้อง
        for i in range(len(pred_masks)):
            ds_name = batch_dataset_names[i]
            task_name = batch_tasks[i] # 'semantic' or 'referring'
            img_path = batch_image_paths[i]
            # ดึง Meter ของ (Dataset, Task) นั้นๆ
            meters = get_or_create_meters(ds_name, task_name, all_meters)

            # ดึง mask ของ sample นี้
            output_i = (pred_masks[i] > 0).int() 
            mask_i = gt_masks[i].int()           
            
            num_masks_in_sample = mask_i.shape[0]

            # --- Metrics Calculation per sample ---
            intersection, union = 0.0, 0.0
            acc_iou_sum_per_sample = 0.0 
            dice_score_sum_per_sample = 0.0 

            hd95_sum_per_sample = 0.0
            eps = 1e-6 

            sample_iou_list = []
            sample_dice_list = []
            sample_hd95_list = []

            for m_idx in range(num_masks_in_sample):
                out_m = output_i[m_idx]
                gt_m = mask_i[m_idx]

                # IoU
                intersection_i, union_i, _ = intersectionAndUnionGPU(
                    out_m.contiguous().clone(), gt_m.contiguous(), 2, ignore_index=255
                )
                intersection += intersection_i
                union += union_i
                
                iou_val = intersection_i / (union_i + eps)
                
                # Use tensor indexing instead of conditional statements
                iou_val[union_i == 0] = 1.0 
                
                acc_iou_sum_per_sample += iou_val

                current_iou = iou_val[1].item() if len(iou_val) > 1 else iou_val[0].item()
                sample_iou_list.append(current_iou)

                # Dice
                intersection_d = (out_m & gt_m).sum().float()
                total_pred = out_m.sum().float()
                total_gt = gt_m.sum().float()
                
                dice_val = (2. * intersection_d + eps) / (total_pred + total_gt + eps)
                if (total_pred + total_gt) == 0: dice_val = 1.0 
                
                dice_score_sum_per_sample += dice_val
                sample_dice_list.append(dice_val.item())


                # HD95
                # Prepare tensors in MONAI format (N, C, D, H, W)
                pred_tensor = out_m.float().unsqueeze(0).unsqueeze(0)
                gt_tensor = gt_m.float().unsqueeze(0).unsqueeze(0)

                # Handle edge cases before calling MONAI
                if gt_tensor.sum() == 0 and pred_tensor.sum() == 0:
                    # True Negative: both empty, award full score
                    hd95_val = 0.0
                elif gt_tensor.sum() > 0 and pred_tensor.sum() == 0:
                    # False Negative: GT has lesion but prediction is empty, penalize
                    hd95_val = 100.0 
                else:
                    # Compute HD95 using MONAI
                    hd95_tensor = compute_hausdorff_distance(
                        y_pred=pred_tensor, 
                        y=gt_tensor, 
                        include_background=False, 
                        percentile=95
                    )
                    hd95_val = hd95_tensor.item()
                    
                    # Guard: treat NaN/Inf as failure
                    if np.isnan(hd95_val) or np.isinf(hd95_val):
                        hd95_val = 100.0

                hd95_sum_per_sample += hd95_val
                sample_hd95_list.append(hd95_val)

            intersection = intersection.cpu().numpy()
            union = union.cpu().numpy()
            
            avg_iou_per_mask = acc_iou_sum_per_sample.cpu().numpy() / num_masks_in_sample
            avg_dice_per_mask = dice_score_sum_per_sample.cpu().numpy() / num_masks_in_sample
            avg_hd95_per_mask = hd95_sum_per_sample / num_masks_in_sample

            # Update Meters
            meters["intersection_meter"].update(intersection)
            meters["union_meter"].update(union)
            meters["acc_iou_meter"].update(avg_iou_per_mask, n=num_masks_in_sample) 
            meters["dice_meter"].update(avg_dice_per_mask, n=num_masks_in_sample)
            meters["hd95_meter"].update(avg_hd95_per_mask, n=num_masks_in_sample)


            final_sample_iou = sum(sample_iou_list) / len(sample_iou_list)
            final_sample_dice = sum(sample_dice_list) / len(sample_dice_list)
            final_sample_hd95 = sum(sample_hd95_list) / len(sample_hd95_list)

            per_sample_results.append({
                "image_path": img_path,
                "dataset": ds_name,
                "task": task_name,
                "iou": final_sample_iou,
                "dice": final_sample_dice,
                "hd95": final_sample_hd95
            })



    df = pd.DataFrame(per_sample_results)
    save_path = f"16_Exp_MedSAM_B-Frz_NoCross_NoMultiScale_results_rank_{args.local_rank}.csv"
    df.to_csv(save_path, index=False)
    print(f"[Rank {args.local_rank}] Saved per-sample results to {save_path} ({len(df)} samples)")

    # --- Summary ---
    final_results = {}
    # Sort keys for clean alphabetical output
    dataset_keys_sorted = sorted(all_meters.keys())

    if args.local_rank == 0:
        print(f"\n{'='*20} Evaluation Summary (Epoch {epoch}) {'='*20}")

    for key in dataset_keys_sorted:
        ds_name, task_name = key
        meters = all_meters[key]
        
        meters["intersection_meter"].all_reduce()
        meters["union_meter"].all_reduce()
        meters["acc_iou_meter"].all_reduce()
        meters["dice_meter"].all_reduce()
        meters["hd95_meter"].all_reduce()

        iou_class = meters["intersection_meter"].sum / (meters["union_meter"].sum + 1e-10)
        ciou = iou_class[1] 
        giou = meters["acc_iou_meter"].avg[1] 
        dice_score = meters["dice_meter"].avg
        hd95_score = meters["hd95_meter"].avg

        
        result_key = f"{ds_name}_{task_name}"
        final_results[result_key] = {"giou": giou, "ciou": ciou, "dice": dice_score, "hd95": hd95_score}

        if args.local_rank == 0:
            print(f"Dataset: [{ds_name}] | Task: [{task_name}]")
            print(f"  Count: {meters['dice_meter'].count} samples")
            print("  gIoU: {:.4f} | cIoU: {:.4f} | Dice: {:.4f} | HD95: {:.2f}".format(giou, ciou, dice_score, hd95_score))
            print("-" * 60)

    if args.local_rank == 0:
        print("=" * 80)

    return final_results

def main():
    args = parse_args(sys.argv[1:])

    deepspeed.init_distributed()
    args.local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(args.local_rank)

    if args.local_rank == 0:
        print(f"Loading processor: {args.model_name}")
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    
    if "[SEG]" not in tokenizer.get_vocab():
        tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = torch.float32
    if args.precision == "bf16": torch_dtype = torch.bfloat16
    elif args.precision == "fp16": torch_dtype = torch.half

    kwargs = {"torch_dtype": torch_dtype}

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

    ds_config = {
        "train_micro_batch_size_per_gpu": args.val_batch_size,
        "fp16": { "enabled": args.precision == "fp16" },
        "bf16": { "enabled": args.precision == "bf16" },
    }

    model_engine, _, _, _ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=ds_config)

    if args.model_path and os.path.isdir(args.model_path):
        if args.local_rank == 0: print(f"Loading checkpoint: {args.model_path}")
        state_dict = get_fp32_state_dict_from_zero_checkpoint(args.model_path)
        model_engine.module.load_state_dict(state_dict, strict=True)

    model_engine.eval()
    val_dataset = ValDataset(args.dataset_dir, processor, args.vision_tower, args.val_dataset, args.image_size)
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
    
    validate(val_loader, model_engine, epoch=0, writer=None, args=args)

if __name__ == "__main__":
    main()