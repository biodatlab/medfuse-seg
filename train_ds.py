import argparse
import os
import shutil
import sys
import time
from functools import partial
import random

import deepspeed
import numpy as np
import torch
import tqdm
from peft import LoraConfig, get_peft_model
from monai.metrics import compute_hausdorff_distance
from torch.utils.tensorboard import SummaryWriter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

from model.MedFuseSeg import MedFuseSegForCausalLM
from utils.dataset import HybridDataset, ValDataset, collate_fn
from utils.utils import (AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)


def parse_args(args):
    parser = argparse.ArgumentParser(description="MedFuseSeg Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="google/medgemma-4b-it"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=4096, type=int)
    parser.add_argument("--lora_r", default=64, type=int)
    parser.add_argument(
        "--vision-tower", default="google/medgemma-4b-it", type=str
    )
    parser.add_argument(
        "--model_name", default="google/medgemma-4b-it", type=str,
        help="Model name for AutoProcessor"
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--val_dataset", default="hf_refseg|test", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./experiments", type=str)
    parser.add_argument("--exp_name", default="lisa", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=32, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=2.0, type=float)
    parser.add_argument("--bce_loss_weight", default=4.0, type=float)
    parser.add_argument("--lora_alpha", default=128, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--use_dora", action="store_true", default=False, help="Use DoRA instead of LoRA")
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj,out_proj,fc1,fc2", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="medsam_vit_b.pth", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    return parser.parse_args(args)


def main(args):
    args = parse_args(args)

    import deepspeed
    import datetime

    deepspeed.init_distributed(dist_init_required=True, timeout=datetime.timedelta(hours=4))
    
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if torch.distributed.get_rank() == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    # Create processor
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer

    num_added_tokens = tokenizer.add_tokens("[SEG]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = MedFuseSegForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_med_fuse_seg_modules(model.get_model().config)



    lora_r = args.lora_r
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "projector",
                                "fusion_adapter",
                                "multi_modal_projector"
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
            model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            use_dora=args.use_dora
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    # make "lm_head", "embed_tokens", "mask_decoder", "projector",'multi_modal_projector', 'fusion_adapter' trainable
    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["lm_head", "embed_tokens", "mask_decoder", "projector",'multi_modal_projector', 'fusion_adapter']

            ]
        ):
            print("n: ", n, "p.shape: ", p.shape)
            p.requires_grad = True

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    train_dataset = HybridDataset(
        base_image_dir=args.dataset_dir,
        processor=processor,
        vision_tower=args.vision_tower,
        samples_per_epoch=args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
        precision=args.precision,
        image_size=args.image_size,
    )

    if args.no_eval == False:
        val_dataset = ValDataset(
            args.dataset_dir,
            processor,
            args.vision_tower,
            args.val_dataset,
            args.image_size,
        )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")

    total_steps = args.epochs * args.steps_per_epoch
    warmup_steps = int(total_steps * 0.03)
    
    print(f"Auto-configured Warmup: {warmup_steps} steps (3% of {total_steps})")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "torch_adam": True,
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_ratio": 0.0,   
                "warmup_num_steps": warmup_steps,
                "warmup_type": "linear",
                "cos_min_ratio": 0.01      
            }
        },
        "fp16": {
            "enabled": args.precision == "fp16",
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }


    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(), 
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn,
            processor=processor,
            local_rank=args.local_rank,
        ),
        config=ds_config,
    )

    # resume deepspeed checkpoint
    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False, drop_last=False
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn,
                processor=processor,
                local_rank=args.local_rank,
            ),
        )

    train_iter = iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        giou, ciou, dice_score, hd95_score = validate(val_loader, model_engine, 0, writer, args)
        exit()

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )

        is_best = False

        if args.no_eval == False:
            giou, ciou, dice_score, hd95_score = validate(val_loader, model_engine, epoch, writer, args)
            is_best = dice_score > best_score
            best_score = max(dice_score, best_score)
            cur_ciou = ciou if is_best else cur_ciou
            cur_giou = giou if is_best else cur_giou


        if args.no_eval or is_best:
            save_dir = os.path.join(args.log_dir, "ckpt_model")
            
            # Ensure only one rank saves
            if torch.distributed.get_rank() == 0: 
                torch.save(
                    {"epoch": epoch},
                    os.path.join(
                        args.log_dir,
                        "meta_log_dice{:.3f}_giou{:.3f}.pth".format(
                            best_score, cur_ciou
                        ),
                    ),
                )
                
                # Remove old checkpoint to save disk space
                if os.path.exists(save_dir):
                    try:
                        shutil.rmtree(save_dir)
                    except OSError as e:
                        print(f"Error removing directory: {e}")

            torch.distributed.barrier() 
            
            # Save model (DeepSpeed handles the saving)
            model_engine.save_checkpoint(save_dir)


def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
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

            output_dict = model(**input_dict)

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()

            if torch.distributed.get_rank() == 0:
                progress.display(global_step + 1)
                if writer is not None:
                    writer.add_scalar("train/loss", losses.avg, global_step)
                    writer.add_scalar("train/ce_loss", ce_losses.avg, global_step)
                    writer.add_scalar(
                        "train/mask_bce_loss", mask_bce_losses.avg, global_step
                    )
                    writer.add_scalar(
                        "train/mask_dice_loss", mask_dice_losses.avg, global_step
                    )
                    writer.add_scalar("train/mask_loss", mask_losses.avg, global_step)
                    writer.add_scalar(
                        "metrics/total_secs_per_batch", batch_time.avg, global_step
                    )
                    writer.add_scalar(
                        "metrics/data_secs_per_batch", data_time.avg, global_step
                    )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if writer is not None:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

    return train_iter


def validate(val_loader, model_engine, epoch, writer, args):
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.AVERAGE)
    dice_meter = AverageMeter("Dice", ":6.3f", Summary.AVERAGE)
    hd95_meter = AverageMeter("HD95", ":6.2f", Summary.AVERAGE)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

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
        masks_list = output_dict["gt_masks"][0].int()
        
        # Handle case where model predicts no masks (negative samples)
        if len(pred_masks) == 0:
            # Create empty tensor for comparison with GT
            output_list = torch.zeros_like(masks_list)
        else:
            output_list = (pred_masks[0] > 0).int()

        num_masks_in_sample = masks_list.shape[0]
        
        # Prevent division by zero (no ground truth masks in this sample)
        if num_masks_in_sample == 0:
            # Skip scoring for True Negative samples (0/0 is undefined)
            continue

        # Compute gIoU, Dice, and HD95 together
        intersection, union = 0.0, 0.0
        acc_iou_sum_per_sample = 0.0 
        dice_score_sum_per_sample = 0.0 
        hd95_sum_per_sample = 0.0
        eps = 1e-6 

        for mask_i, output_i in zip(masks_list, output_list):
            # IoU computation
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            
            iou_i = intersection_i / (union_i + eps)
            iou_i[union_i == 0] = 1.0 
            acc_iou_sum_per_sample += iou_i

            # Hard Dice
            intersection_d = (output_i & mask_i).sum().float()
            total_pred = output_i.sum().float()
            total_gt = mask_i.sum().float()
            
            dice_i = (2. * intersection_d + eps) / (total_pred + total_gt + eps)
            if (total_pred + total_gt) == 0:
                dice_i = 1.0
            
            dice_score_sum_per_sample += dice_i

            # HD95 using MONAI
            pred_tensor = output_i.float().unsqueeze(0).unsqueeze(0)
            gt_tensor = mask_i.float().unsqueeze(0).unsqueeze(0)

            if gt_tensor.sum() == 0 and pred_tensor.sum() == 0:
                hd95_val = 0.0
            elif gt_tensor.sum() > 0 and pred_tensor.sum() == 0:
                hd95_val = 100.0
            else:
                hd95_tensor = compute_hausdorff_distance(
                    y_pred=pred_tensor,
                    y=gt_tensor,
                    include_background=False,
                    percentile=95
                )
                hd95_val = hd95_tensor.item()
                if np.isnan(hd95_val) or np.isinf(hd95_val):
                    hd95_val = 100.0

            hd95_sum_per_sample += hd95_val

        intersection = intersection.cpu().numpy()
        union = union.cpu().numpy()
        
        # [Safe Division]
        avg_iou_per_mask = acc_iou_sum_per_sample.cpu().numpy() / num_masks_in_sample
        avg_dice_per_mask = dice_score_sum_per_sample.cpu().numpy() / num_masks_in_sample
        avg_hd95_per_mask = hd95_sum_per_sample / num_masks_in_sample

        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(avg_iou_per_mask, n=num_masks_in_sample)
        dice_meter.update(avg_dice_per_mask, n=num_masks_in_sample)
        hd95_meter.update(avg_hd95_per_mask, n=num_masks_in_sample)

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    dice_meter.all_reduce()
    hd95_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    dice_score = dice_meter.avg
    hd95_score = hd95_meter.avg

    if writer is not None:
        writer.add_scalar("val/giou", giou, epoch)
        writer.add_scalar("val/ciou", ciou, epoch)
        writer.add_scalar("val/dice", dice_score, epoch)
        writer.add_scalar("val/hd95", hd95_score, epoch)
        print("giou: {:.4f}, ciou: {:.4f}, dice: {:.4f}, hd95: {:.2f}".format(
            giou, ciou, dice_score, hd95_score))

    return giou, ciou, dice_score, hd95_score


if __name__ == "__main__":
    main(sys.argv[1:])