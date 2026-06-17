# app_lisa_plus.py (LoRA + apply_chat_template)

import argparse
import os
import re
import sys

import bleach
import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Gemma3ImageProcessor
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

from model.MedFuseSeg import MedFuseSegForCausalLM
from model.segment_anything.utils.transforms import ResizeLongestSide
from utils.utils import IMAGE_TOKEN_INDEX

from peft import LoraConfig, get_peft_model


def parse_args(args):
    parser = argparse.ArgumentParser(description="MedFuseSeg chat (plus, with LoRA)")
    # base / model
    parser.add_argument("--version", default="google/medgemma-4b-it")
    parser.add_argument("--model_name", default="google/medgemma-4b-it", type=str,
                        help="For AutoProcessor (must match the chat_template tokenizer)")
    parser.add_argument("--model_path", default="", type=str,
                        help="Path to DeepSpeed ckpt folder (e.g. .../ckpt_model/global_stepXXXX)")
    parser.add_argument("--vision_pretrained", default="medsam_vit_b.pth", type=str,
                        help="path .pth for SAM ViT-H")
    parser.add_argument("--vision-tower", default="google/medgemma-4b-it", type=str)

    # precision / runtime
    parser.add_argument("--precision", default="fp16", type=str, choices=["fp32", "bf16", "fp16"])
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    # MedFuseSeg heads (must match training)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)

    # LoRA (must match training)
    parser.add_argument("--lora_r", default=32, type=int)
    parser.add_argument("--lora_alpha", default=64, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules",
                        default="q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj", type=str)

    # UI / preprocess
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)

    return parser.parse_args(args)


def preprocess(
    x,
    pixel_mean=torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std=torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size=1024,
) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    x = F.pad(x, (0, img_size - w, 0, img_size - h))
    return x


args = parse_args(sys.argv[1:])
os.makedirs(args.vis_save_path, exist_ok=True)

# ---- Processor/tokenizer (for chat_template) ----
processor = AutoProcessor.from_pretrained(args.model_name)
tokenizer = processor.tokenizer
# add [SEG]
if "[SEG]" not in tokenizer.get_vocab():
    tokenizer.add_tokens("[SEG]")
seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

# add <image> start/end if using

# ---- dtype / quant ----
torch_dtype = torch.float32
if args.precision == "bf16":
    torch_dtype = torch.bfloat16
elif args.precision == "fp16":
    torch_dtype = torch.half

kwargs = {"torch_dtype": torch_dtype}
if args.load_in_4bit:
    kwargs.update(
        {
            "torch_dtype": torch.half,
            "load_in_4bit": True,
            "quantization_config": BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_skip_modules=["visual_model"],
            ),
        }
    )
elif args.load_in_8bit:
    kwargs.update(
        {
            "torch_dtype": torch.half,
            "quantization_config": BitsAndBytesConfig(
                llm_int8_skip_modules=["visual_model"],
                load_in_8bit=True,
            ),
        }
    )

# ---- Build MedFuseSeg (same heads/flags as training) ----
model = MedFuseSegForCausalLM.from_pretrained(
    args.version,
    low_cpu_mem_usage=True,
    train_mask_decoder=args.train_mask_decoder,
    out_dim=args.out_dim,
    ce_loss_weight=args.ce_loss_weight,
    dice_loss_weight=args.dice_loss_weight,
    bce_loss_weight=args.bce_loss_weight,
    seg_token_idx=seg_token_idx,
    vision_pretrained=args.vision_pretrained,
    vision_tower=args.vision_tower,
    **kwargs,
)
model.config.eos_token_id = tokenizer.eos_token_id
model.config.bos_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# ---- Init SAM + projection (MedFuseSeg modules) ----
model.get_model().initialize_med_fuse_seg_modules(model.get_model().config)

# ---- Plug LoRA in (module names must match training) ----
def find_linear_layers(m, lora_target_modules):
    cls = torch.nn.Linear
    keep = set()
    for name, mod in m.named_modules():
        if isinstance(mod, cls):
            if all(x not in name for x in [
                "visual_model", "vision_tower", "multi_modal_projector", "text_hidden_fcs"
            ]) and any(x in name for x in lora_target_modules):
                keep.add(name)
    return sorted(list(keep))

if args.lora_r > 0:
    targets = find_linear_layers(model, args.lora_target_modules.split(","))
    lconf = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha,
        target_modules=targets, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lconf)

# ---- Resize embeddings after adding tokens/LoRA ----
model.resize_token_embeddings(len(tokenizer))

# ---- Load DeepSpeed checkpoint (same as app_lisa) ----
if args.model_path and os.path.isdir(args.model_path):
    print(f"Loading checkpoint from {args.model_path}")
    state_dict = get_fp32_state_dict_from_zero_checkpoint(args.model_path)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print("missing:", len(missing), "unexpected:", len(unexpected))
else:
    print("[Warn] --model_path is empty or not a folder; running with base weights only")

# ---- Device / dtype ----
if hasattr(model.model, "visual_model"):
    model.model.visual_model.to(dtype=torch_dtype)

model = model.to("cuda")
if args.precision == "bf16":
    model = model.bfloat16()
elif args.precision == "fp16":
    model = model.half()

# vision tower (Gemma vision) device
vision_tower = model.get_model().get_vision_tower()
vision_tower.to(device=args.local_rank, dtype=model.dtype)

# preprocessors
clip_image_processor = Gemma3ImageProcessor.from_pretrained(model.config.vision_tower)
transform = ResizeLongestSide(args.image_size)
model.eval()


title = "MedFuseSeg (plus): Reasoning Segmentation via LLM"

def inference(input_str, input_image):
    input_str = bleach.clean(input_str)
    if not isinstance(input_str, str) or len(input_str.strip()) == 0:
        output_image = cv2.imread("./resources/error_happened.png")[:, :, ::-1]
        return output_image, "[Error] Invalid input."

    # ---- Build messages for AutoProcessor (chat_template) ----
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are an expert radiologist."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": input_image},
                {"type": "text", "text": input_str}
            ]
        }
    ]

    # Processor produces: input_ids, attention_mask, pixel_values from messages
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    )
    # move to cuda
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    attention_mask = inputs["attention_mask"]

    # Prepare SAM image
    image_np = cv2.imread(input_image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    original_size_list = [image_np.shape[:2]]

    image = transform.apply_image(image_np)
    resize_list = [image.shape[:2]]
    image = (
        preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), img_size=args.image_size)
        .unsqueeze(0)
        .to("cuda", dtype=model.dtype)
    )

    with torch.no_grad():
        output_ids, pred_masks = model.evaluate(
            pixel_values=pixel_values,
            images=image,
            input_ids=input_ids,
            attention_mask=attention_mask,
            resize_list=resize_list,
            original_size_list=original_size_list,
            max_new_tokens=512,
            tokenizer=tokenizer,
        )

    # decode; strip the template prefixes nicely
    out_ids = output_ids[0][output_ids[0] != IMAGE_TOKEN_INDEX] if output_ids.dim() > 1 else output_ids
    text_output = tokenizer.decode(out_ids, skip_special_tokens=False)
    text_output = text_output.replace("\n", " ").replace("  ", " ")
    s_marker = "<start_of_turn>model"
    e_marker = "<end_of_turn>"
    if s_marker in text_output:
        text_output = text_output.split(s_marker, 1)[1]
    if e_marker in text_output:
        text_output = text_output.split(e_marker, 1)[0]

    # =================================================================================
    # VISUALIZATION LOGIC UPDATED: MULTI-COLOR SUPPORT
    # 1. Red, 2. Blue, 3. Green, 4. Yellow, >4. Random
    # =================================================================================
    save_img = image_np.copy()
    found_any_mask = False

    # Define fixed colors (BGR format for OpenCV)
    # Red: [0, 0, 255], Blue: [255, 0, 0], Green: [0, 255, 0], Yellow: [0, 255, 255]
    fixed_colors = [
        np.array([0, 0, 255]),
        np.array([255, 0, 0]),
        np.array([0, 255, 0]),
        np.array([0, 255, 255])
    ]

    for pm in pred_masks:
        # pm shape: [N_SEG_Tokens, H, W]
        if pm.shape[0] == 0:
            continue
        
        pm_np = pm.detach().float().cpu().numpy()
        
        # Iterate over every [SEG] mask found in the response
        for i in range(pm_np.shape[0]):
            mask = (pm_np[i] > 0)
            
            if mask.any():
                found_any_mask = True
                
                # Select color based on index
                if i < len(fixed_colors):
                    color = fixed_colors[i]
                else:
                    # Random color for 5th mask onwards
                    color = np.random.randint(0, 255, size=3)

                # Apply mask overlay with alpha blending (0.5)
                save_img[mask] = (
                    save_img[mask].astype(float) * 0.5 + 
                    color.astype(float) * 0.5
                ).astype(np.uint8)

        # Only process the first batch item (usually batch size is 1 for inference)
        break 

    if not found_any_mask:
        save_img = None
    # =================================================================================

    if save_img is None:
        vis = cv2.imread("./resources/no_seg_out.png")[:, :, ::-1] if os.path.exists("./resources/no_seg_out.png") else image_np
    else:
        vis = save_img

    return Image.fromarray(vis), text_output


demo = gr.Interface(
    inference,
    inputs=[
        gr.Textbox(lines=1, label="Text Instruction"),
        gr.Image(type="filepath", label="Input Image"),
    ],
    outputs=[
        gr.Image(type="pil", label="Segmentation Output"),
        gr.Textbox(lines=3, label="Text Output"),
    ],
    title=title,
    allow_flagging="auto",
)
demo.queue()
demo.launch()
# demo.launch(share=True)