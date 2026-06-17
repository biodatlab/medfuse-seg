"""MedFuseSeg: Multi-Level Visual and Semantic Context Fusion for Medical Image Segmentation.

Usage:
    >>> from medfuseseg import MedFuseSegPipeline
    >>> pipe = MedFuseSegPipeline(model="google/medgemma-4b-it", checkpoint="path/to/ckpt")
    >>> result = pipe(image="scan.png", prompt="Segment the abnormality")
    >>> print(result.text)
    >>> result.show()
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Union, List, Optional
from transformers import AutoProcessor
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
from peft import LoraConfig, get_peft_model

# Add project root to path when running as package
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model.MedFuseSeg import MedFuseSegForCausalLM
from model.segment_anything.utils.transforms import ResizeLongestSide


# ──────────────────────────────────────────────────────────────
# Result object
# ──────────────────────────────────────────────────────────────

class MedFuseSegResult:
    """Container for a single MedFuseSeg inference result."""

    def __init__(self, image: np.ndarray, mask: np.ndarray, text: str):
        self._image = image          # original image (H, W, C) numpy uint8
        self.mask = mask             # binary mask (H, W), uint8 {0, 255}
        self.text = text             # generated text output

    @property
    def overlay(self) -> Image.Image:
        """Return the input image with the predicted mask overlaid in semi-transparent green."""
        rgb = self._image.astype(np.float32)
        mask_bool = self.mask > 0
        color = np.array([0, 255, 0], dtype=np.float32)  # green
        alpha = 0.45
        rgb[mask_bool] = rgb[mask_bool] * (1 - alpha) + color * alpha
        return Image.fromarray(rgb.astype(np.uint8))

    def show(self) -> None:
        """Display the overlay in the default image viewer."""
        self.overlay.show()

    def save_mask(self, path: str) -> None:
        """Save the binary mask as a PNG image."""
        Image.fromarray(self.mask).save(path)

    def save_overlay(self, path: str) -> None:
        """Save the overlaid image."""
        self.overlay.save(path)

    def __repr__(self) -> str:
        return f"MedFuseSegResult(text={self.text!r}, mask_shape={self.mask.shape})"


# ──────────────────────────────────────────────────────────────
# Helper: model initialisation (shared with train/inference)
# ──────────────────────────────────────────────────────────────

def _build_model(
    version: str,
    vision_pretrained: str,
    checkpoint: Optional[str],
    precision: str,
    lora_r: int,
    lora_alpha: int,
):
    """Build and load the MedFuseSegForCausalLM model."""
    tokenizer_adds = []
    processor = AutoProcessor.from_pretrained(version)
    tokenizer = processor.tokenizer

    if "[SEG]" not in tokenizer.get_vocab():
        tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    torch_dtype = getattr(torch, {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(precision, "bfloat16"))

    model_args = {
        "train_mask_decoder": True,
        "out_dim": 256,
        "ce_loss_weight": 1.0,
        "dice_loss_weight": 0.5,
        "bce_loss_weight": 2.0,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": vision_pretrained,
        "vision_tower": version,
    }

    model = MedFuseSegForCausalLM.from_pretrained(
        version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.get_model().initialize_med_fuse_seg_modules(model.get_model().config)

    # LoRA
    if lora_r > 0:
        def find_linear_layers(m, target_modules):
            cls = torch.nn.Linear
            keep = set()
            for name, mod in m.named_modules():
                if isinstance(mod, cls) and all(x not in name for x in ["visual_model", "projector", "fusion_adapter"]):
                    if any(x in name for x in target_modules):
                        keep.add(name)
            return sorted(list(keep))

        target = find_linear_layers(
            model, "q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj,out_proj,fc1,fc2".split(",")
        )
        lconf = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=target, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lconf)

    model.resize_token_embeddings(len(tokenizer))
    model.get_model().get_vision_tower()  # ensure vision tower exists

    if checkpoint and os.path.isdir(checkpoint):
        state_dict = get_fp32_state_dict_from_zero_checkpoint(checkpoint)
        model = model.to("cpu")  # load on CPU first to avoid GPU OOM
        model.load_state_dict(state_dict, strict=False)

    return model


def _preprocess_sam(
    x: torch.Tensor,
    pixel_mean: torch.Tensor = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1),
    pixel_std: torch.Tensor = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1),
    img_size: int = 1024,
) -> torch.Tensor:
    x = (x - pixel_mean) / pixel_std
    h, w = x.shape[-2:]
    x = F.pad(x, (0, img_size - w, 0, img_size - h))
    return x


# ──────────────────────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────────────────────

class MedFuseSegPipeline:
    """Easy-to-use pipeline for MedFuseSeg inference.

    Example:
        >>> pipe = MedFuseSegPipeline(checkpoint="path/to/ckpt")
        >>> result = pipe(image="xray.png", prompt="Segment the pneumonia region")
        >>> result.show()
    """

    BASE_MODEL = "google/medgemma-4b-it"
    SAM_CHECKPOINT = "medsam_vit_b.pth"

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: Optional[str] = None,
        precision: str = "bf16",
        lora_r: int = 8,
        lora_alpha: int = 16,
    ):
        """
        Args:
            checkpoint: Path to a DeepSpeed checkpoint directory (e.g. "path/to/ckpt_model").
                        If None, loads the base model without fine-tuned LoRA weights.
            device: "cuda", "cpu", or None for auto-detect.
            precision: "bf16", "fp16", or "fp32".
            lora_r: LoRA rank (should match the rank used during training).
            lora_alpha: LoRA alpha (should match the alpha used during training).
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = precision
        self.image_size = 1024
        self.sam_transform = ResizeLongestSide(self.image_size)

        # Build model with hardcoded base model
        self.model = _build_model(
            version=self.BASE_MODEL,
            vision_pretrained=self.SAM_CHECKPOINT,
            checkpoint=checkpoint,
            precision=precision,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )

        torch_dtype = getattr(torch, {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}.get(precision, "bfloat16"))
        if hasattr(self.model.model, "visual_model"):
            self.model.model.visual_model.to(dtype=torch_dtype)

        self.model = self.model.to(self.device)
        if precision == "bf16":
            self.model = self.model.bfloat16()
        elif precision == "fp16":
            self.model = self.model.half()

        vision_tower = self.model.get_model().get_vision_tower()
        vision_tower.to(device=self.device, dtype=self.model.dtype)
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(model)
        self.tokenizer = self.processor.tokenizer
        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]

    def _load_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Accept filepath, URL, numpy array, or PIL Image and return RGB numpy."""
        if isinstance(image, str):
            if image.startswith(("http://", "https://")):
                import requests
                from io import BytesIO
                res = requests.get(image, headers={"User-Agent": "Mozilla/5.0"})
                image = Image.open(BytesIO(res.content))
            else:
                image = Image.open(image)
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        return image

    def __call__(
        self,
        image: Union[str, np.ndarray, Image.Image],
        prompt: str,
        max_new_tokens: int = 512,
    ) -> MedFuseSegResult:
        """Run inference on a single image.

        Args:
            image: Path to image file, URL, numpy array, or PIL Image.
            prompt: Text instruction (e.g. "Segment the pneumonia region").
            max_new_tokens: Maximum text tokens to generate.

        Returns:
            A MedFuseSegResult with text, mask, and visualization.
        """
        np_img = self._load_image(image)
        original_size = np_img.shape[:2]

        # Build messages for MedGemma
        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
            {"role": "user", "content": [
                {"type": "image", "image": Image.fromarray(np_img)},
                {"type": "text", "text": prompt},
            ]},
        ]

        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        attention_mask = inputs["attention_mask"]

        # Run evaluation (generate + mask)
        with torch.no_grad():
            output_ids, pred_masks = self.model.evaluate(
                pixel_values=pixel_values,
                images=self._prepare_sam_images([np_img]),
                input_ids=input_ids,
                attention_mask=attention_mask,
                resize_list=[np_img.shape[:2]],
                original_size_list=[original_size],
                max_new_tokens=max_new_tokens,
                tokenizer=self.tokenizer,
            )

        # Decode text
        out_ids = output_ids[0][output_ids[0] != 255999] if output_ids.dim() > 1 else output_ids
        text = self.tokenizer.decode(out_ids, skip_special_tokens=False).strip()
        # Strip Gemma3 conversation template markers
        s_marker = "<start_of_turn>model"
        e_marker = "<end_of_turn>"
        if s_marker in text:
            text = text.split(s_marker, 1)[1]
        if e_marker in text:
            text = text.split(e_marker, 1)[0]
        text = text.strip().replace("\n", " ")

        # Extract mask
        mask = np.zeros(original_size, dtype=np.uint8)
        if len(pred_masks) > 0 and pred_masks[0].numel() > 0:
            pm = pred_masks[0]
            if pm.dim() > 2:
                pm = pm[0]   # take first mask if batch
            if pm.dim() == 3:
                pm = pm[0]   # take first mask if multi-instance
            mask_bin = (pm.sigmoid().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
            if mask_bin.shape[:2] != original_size:
                mask_bin = np.array(Image.fromarray(mask_bin).resize(
                    original_size[::-1], Image.NEAREST
                ))
            mask = mask_bin

        return MedFuseSegResult(image=np_img, mask=mask, text=text)

    def _prepare_sam_images(self, images: List[np.ndarray]) -> torch.Tensor:
        """Prepare images for SAM encoder (same as training pipeline)."""
        sam_list = []
        for img in images:
            img_sam = self.sam_transform.apply_image(img)
            tensor = _preprocess_sam(
                torch.from_numpy(img_sam).permute(2, 0, 1).contiguous().unsqueeze(0),
                img_size=self.image_size,
            )
            sam_list.append(tensor)
        return torch.cat(sam_list, dim=0).to(self.device, dtype=self.model.dtype)
