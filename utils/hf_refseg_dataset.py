# utils/hf_refseg_dataset.py
"""Reference segmentation dataset loader for HuggingFace Med-ReasonSeg."""

from datasets import load_dataset, DatasetDict
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from model.segment_anything.utils.transforms import ResizeLongestSide
import os
import albumentations as A
import cv2


class HFRefSegDataset(torch.utils.data.Dataset):
    """Reference segmentation dataset that loads Med-ReasonSeg from HF Hub."""

    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    HF_REPO = "biodatlab/Med-ReasonSeg"

    def __init__(self, processor, vision_tower, split="train", image_size=1024):
        self.processor = processor
        self.transform = ResizeLongestSide(image_size)
        self.image_size = image_size
        self.split = split

        # Load dataset from HuggingFace Hub
        print(f"Loading {self.HF_REPO} split '{split}' from HuggingFace Hub...")
        ds_loaded = load_dataset(self.HF_REPO, trust_remote_code=True)

        # Select the appropriate split from the loaded dataset
        if isinstance(ds_loaded, (dict, DatasetDict)):
            if split in ds_loaded:
                print(f"Selecting split '{split}' from loaded DatasetDict.")
                self.ds_raw = ds_loaded[split]
            else:
                raise KeyError(f"Split '{split}' not found. Available: {list(ds_loaded.keys())}")
        else:
            self.ds_raw = ds_loaded

        print(f"[{split} split] Total: {len(self.ds_raw)}")

        # Define augmentation pipeline for training
        self.aug = A.Compose([
            A.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.2,
                rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT,
            ),
            A.GaussianBlur(blur_limit=(3, 3), p=0.15),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.15),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
            ], p=0.5),
        ])

    def __len__(self):
        return len(self.ds_raw)

    def _load_pil(self, value, mode="RGB"):
        """Load a PIL image from a filepath string or return existing PIL Image."""
        if isinstance(value, str):
            img = Image.open(value)
        else:
            img = value
        return img.convert("L") if mode == "L" else img.convert("RGB")

    def _preprocess_sam(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize and pad image tensor for SAM encoder input."""
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.img_size - w, 0, self.img_size - h))
        return x

    def get_dataset_name(self, idx):
        """Return the dataset name for a given index."""
        try:
            return self.ds_raw[idx]["dataset"]
        except Exception:
            return "unknown_subdataset"

    def __getitem__(self, idx):
        ex = self.ds_raw[idx]
        dataset_name = ex.get("dataset", "unknown")
        task_name = ex.get("task", "unknown")

        # Load image from path and convert to numpy array
        pil_img = self._load_pil(ex["image_path"], mode="RGB")
        np_img = np.array(pil_img)

        # Handle positive and negative samples differently
        if str(dataset_name).endswith("_negative"):
            # Negative sample: no mask, zero masks
            masks = torch.zeros(0, self.img_size, self.img_size)
            labels = torch.zeros(0, self.img_size, self.img_size)

            if self.split == "train":
                augmented = self.aug(image=np_img)
                np_img = augmented["image"]
        else:
            # Positive sample: load and process mask
            pil_mk = self._load_pil(ex["mask_path"], mode="L")
            np_mk = np.array(pil_mk)

            if self.split == "train":
                augmented = self.aug(image=np_img, mask=np_mk)
                np_img = augmented["image"]
                np_mk = augmented["mask"]

            # Resize mask and convert to binary tensors
            mk_resized = self.transform.apply_image(np_mk)
            mk = (mk_resized > 127).astype(np.uint8)
            masks = torch.from_numpy(mk[None, ...])
            labels = torch.ones(mk.shape, dtype=torch.float32) * self.ignore_label

        # Convert augmented image back to PIL for VLM input
        aug_pil_img = Image.fromarray(np_img)

        # Prepare image for SAM encoder with preprocessing
        img_sam = self.transform.apply_image(np_img)
        resize = img_sam.shape[:2]
        img_sam = self._preprocess_sam(torch.from_numpy(img_sam).permute(2, 0, 1).contiguous())

        # Build conversation template
        question = ex["question"]
        answer = ex["answer"]

        conversations = [[
            {"role": "system", "content": [{"type": "text", "text": "You are an expert radiologist."}]},
            {"role": "user", "content": [
                {"type": "image", "image": aug_pil_img},
                {"type": "text", "text": question},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": answer}]},
        ]]

        # Return tuple matching expected collate format
        image_path = f"hf_index/{idx}"
        return ((image_path, dataset_name, task_name), img_sam, conversations, masks, labels, resize, None, None)
