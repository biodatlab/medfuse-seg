# utils/dataset.py
import numpy as np
import torch
import torch.nn.functional as F

from model.segment_anything.utils.transforms import ResizeLongestSide
from .hf_refseg_dataset import HFRefSegDataset


class HybridDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        processor,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 1024,
        **kwargs,
    ):
        self.samples_per_epoch = samples_per_epoch
        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.processor = processor
        self.precision = precision
        self.hf_dataset = HFRefSegDataset(processor, vision_tower, "train", image_size)

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        sample = list(self.hf_dataset[idx % len(self.hf_dataset)])
        try:
            dataset_name = self.hf_dataset.get_dataset_name(idx % len(self.hf_dataset))
        except Exception:
            dataset_name = "hf_refseg_fallback"
        sample[0] = (sample[0], dataset_name)
        sample.append(False)  # inference flag
        return tuple(sample)


class ValDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(self, base_image_dir, processor, vision_tower, val_dataset, image_size=1024):
        splits = val_dataset.split("|")
        ds_name, split = splits[0], splits[1] if len(splits) > 1 else "test"
        if ds_name == "hf_refseg":
            self.hf_val = HFRefSegDataset(processor, vision_tower, split, image_size)
            self.dataset_name_val = f"hf_refseg_{split}"
        else:
            raise ValueError(f"Unsupported val dataset: {ds_name}")
        self.image_size = image_size
        self.processor = processor
        self.transform = ResizeLongestSide(image_size)

    def __len__(self):
        return len(self.hf_val)

    def preprocess(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        x = F.pad(x, (0, self.img_size - w, 0, self.img_size - h))
        return x

    def __getitem__(self, idx):
        sample = list(self.hf_val[idx])
        dataset_name = self.hf_val.get_dataset_name(idx)
        sample[0] = (sample[0], dataset_name)
        sample.append(True)
        return tuple(sample)


def collate_fn(batch, processor=None, local_rank=-1):
    image_path_list = []
    dataset_name_list = []
    images_list = []
    conversation_list = []
    masks_list = []
    label_list = []
    resize_list = []
    offset_list = [0]
    cnt = 0
    inferences = []

    for (
        image_path_data,
        images,
        conversations,
        masks,
        label,
        resize,
        questions,
        sampled_classes,
        inference,
    ) in batch:
        if isinstance(image_path_data, tuple):
            image_path = image_path_data[0]
            dataset_name = image_path_data[1]
        else:
            image_path = image_path_data
            dataset_name = "unknown"

        image_path_list.append(image_path)
        dataset_name_list.append(dataset_name)
        images_list.append(images)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        resize_list.append(resize)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    all_input_ids = []
    all_attention_masks = []
    all_pixel_values = []

    for messages in conversation_list:
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=True,
            return_dict=True, return_tensors="pt"
        )
        all_input_ids.append(inputs['input_ids'].squeeze(0))
        all_attention_masks.append(inputs['attention_mask'].squeeze(0))
        all_pixel_values.append(inputs['pixel_values'].squeeze(0))

    input_ids = torch.nn.utils.rnn.pad_sequence(
        all_input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id
    )
    attention_masks = torch.nn.utils.rnn.pad_sequence(
        all_attention_masks, batch_first=True, padding_value=0
    )
    pixel_values = torch.stack(all_pixel_values, dim=0)

    IGNORE_INDEX = -100

    if inferences[0] is False:
        truncate_len = 4096
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]

    labels = input_ids.clone()
    labels[:] = IGNORE_INDEX
    tok = processor.tokenizer

    sot_id = tok.vocab.get("<start_of_turn>") if hasattr(tok, "vocab") else None
    eot_id = tok.vocab.get("<end_of_turn>") if hasattr(tok, "vocab") else None
    model_id = tok.vocab.get("model") if hasattr(tok, "vocab") else None

    try:
        NL_IDS = tok.encode("\n", add_special_tokens=False)
    except Exception:
        NL_IDS = []
    try:
        SP_IDS = tok.encode(" ", add_special_tokens=False)
    except Exception:
        SP_IDS = []
    SKIP_IDS = set(NL_IDS + SP_IDS)

    for b in range(input_ids.size(0)):
        ids = input_ids[b]
        starts = (ids == sot_id).nonzero(as_tuple=True)[0] if sot_id is not None else []
        ends = (ids == eot_id).nonzero(as_tuple=True)[0] if eot_id is not None else []
        n = min(len(starts), len(ends))
        for i in range(n):
            s = int(starts[i].item())
            e = int(ends[i].item())
            if s + 1 >= ids.size(0):
                continue
            if model_id is not None and int(ids[s + 1].item()) == model_id:
                k = s + 2
                while k < e and int(ids[k].item()) in SKIP_IDS:
                    k += 1
                if k < e:
                    labels[b, k:e] = ids[k:e]

    if tok.pad_token_id is not None:
        labels[labels == tok.pad_token_id] = IGNORE_INDEX

    return {
        "image_paths": image_path_list,
        "dataset_names": dataset_name_list,
        "images": torch.stack(images_list, dim=0),
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "labels": labels,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
    }
