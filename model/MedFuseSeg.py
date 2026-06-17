from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import Gemma3ForConditionalGeneration, Gemma3Model                                            
from .segment_anything import build_sam_vit_b

from scipy.optimize import linear_sum_assignment
import numpy as np

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,  # 100000.0,
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    alpha: float = 0.25,
    gamma: float = 2.0,
    eps: float = 1e-6,
):
    """
    Compute the sigmoid focal loss between `inputs` and the ground truth `targets`.

    Args:
        inputs: A float tensor of arbitrary shape. Raw logits.
        targets: A float tensor with the same shape. Binary labels (0 or 1).
        num_masks: Normalization factor (usually number of masks or positive samples).
        alpha: Focal loss alpha weighting factor.
        gamma: Focal loss focusing parameter.
        eps: Numerical stability.

    Returns:
        Scaled focal loss normalized by num_masks.
    """
    prob = inputs.sigmoid()
    prob = torch.clamp(prob, min=eps, max=1.0 - eps)

    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    modulating_factor = (1 - p_t) ** gamma
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_factor * modulating_factor * ce_loss
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss


# =========================================================================
# [NEW] Multi-scale Feature Fusion Modules (SigLIP + SAM + ConvNeXt)
# =========================================================================

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) 
        x = input + x 
        return x

class SiglipSamFusionAdapter(nn.Module):
    def __init__(self, siglip_dim=1152, sam_dim=256, selected_layers=[6, 12, 18, 24]):
        super().__init__()
        self.selected_layers = selected_layers
        
        # Projectors for each layer
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(siglip_dim, sam_dim, kernel_size=1),
                nn.GroupNorm(8, sam_dim),
                nn.GELU()
            )
            for _ in selected_layers
        ])
        
        # Fusion Stage
        total_in_dim = sam_dim + (sam_dim * len(selected_layers))
        self.reduce_conv = nn.Conv2d(total_in_dim, sam_dim, kernel_size=1)
        self.refine_blocks = nn.Sequential(
            ConvNeXtBlock(sam_dim),
            ConvNeXtBlock(sam_dim)
        )

    def forward(self, sam_features, vision_tower_outputs):
        siglip_feats_processed = []
        target_size = sam_features.shape[-2:]
        
        for i, layer_idx in enumerate(self.selected_layers):
            # SigLIP hidden state: [Batch, Seq, Dim]
            raw_feat = vision_tower_outputs.hidden_states[layer_idx]
            
            # Reshape to [Batch, Dim, H, W]
            B, N, C = raw_feat.shape
            H = W = int(N**0.5) 
            feat_img = raw_feat.permute(0, 2, 1).reshape(B, C, H, W)
            
            # Project
            feat_proj = self.projections[i](feat_img)
            
            # Resize if needed
            if feat_proj.shape[-2:] != target_size:
                feat_proj = F.interpolate(
                    feat_proj, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=False
                )
            
            siglip_feats_processed.append(feat_proj)
            
        # Concatenate SAM + SigLIP Layers
        concat_feat = torch.cat([sam_features] + siglip_feats_processed, dim=1)
        
        # Reduce & Refine
        x = self.reduce_conv(concat_feat)
        fused_output = self.refine_blocks(x)
        
        return fused_output

# [UPDATED] MLP Projector with mid_dim
class MedFuseSegProjector(nn.Module):
    def __init__(self, config, out_dim):
        super().__init__()
        self.config = config
        self.out_dim = out_dim
        
        in_dim = config.hidden_size
        
        mid_dim = 1024 
        
        self.projector = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x):
        return self.projector(x)


class MedFuseSegMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(MedFuseSegMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_med_fuse_seg_modules(self.config)

    def initialize_med_fuse_seg_modules(self, config):
        # SAM
        self.visual_model = build_sam_vit_b(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                param.requires_grad = True

        # [CHANGED] Use MLP Projector instead of CrossAttn
        self.projector = MedFuseSegProjector(config.text_config, config.out_dim)

        self.projector.train()
        for param in self.projector.parameters():
            param.requires_grad = True

        # Fusion Adapter injects multi-level MedSigLIP features into MedSAM
        self.fusion_adapter = SiglipSamFusionAdapter(
            siglip_dim=1152,  # MedGemma SigLIP dim
            sam_dim=config.out_dim,  # 256
            selected_layers=[6, 12, 18, 24] 
        )
        self.fusion_adapter.train()
        for param in self.fusion_adapter.parameters():
            param.requires_grad = True


class MedFuseSegModel(MedFuseSegMetaModel, Gemma3Model):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(MedFuseSegModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False

    def get_vision_tower(self):
        """Return the vision tower from the parent Gemma3Model"""
        return self.vision_tower


class MedFuseSegForCausalLM(Gemma3ForConditionalGeneration):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        if not hasattr(config, "train_mask_decoder"):
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "google/medgemma-4b-it"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
        else:
            config.mm_vision_tower = kwargs.get("vision_tower", config.vision_tower)
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")

        super().__init__(config)

        self.model = MedFuseSegModel(config, **kwargs)

        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_model(self):
        """Return the underlying MedFuseSegModel"""
        return self.model

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        # Handle attention mask naming
        if 'attention_masks' in kwargs and 'attention_mask' not in kwargs:
            kwargs['attention_mask'] = kwargs.pop('attention_masks')

        # Check if this is a generation call (has past_key_values or no labels)
        is_generation = 'past_key_values' in kwargs or 'labels' not in kwargs
        
        if is_generation:
            generation_kwargs = {}
            for key in ['input_ids', 'attention_mask', 'pixel_values', 'past_key_values', 
                       'use_cache', 'output_hidden_states', 'return_dict']:
                if key in kwargs:
                    generation_kwargs[key] = kwargs[key]
            
            return super().forward(**generation_kwargs)
        else:
            # For training, use the custom MedFuseSeg forward pass
            return self.model_forward(**kwargs)

    def batch_dice_loss(self, inputs, targets):
        inputs = inputs.sigmoid()
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
        denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss
    

    def batch_sigmoid_ce_loss(self, inputs: torch.Tensor, targets: torch.Tensor):
        hw = inputs.shape[1]

        pos = F.binary_cross_entropy_with_logits(
            inputs, torch.ones_like(inputs), reduction="none"
        )
        neg = F.binary_cross_entropy_with_logits(
            inputs, torch.zeros_like(inputs), reduction="none"
        )

        loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
            "nc,mc->nm", neg, (1 - targets)
        )

        return loss / hw

    def batch_sigmoid_focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0):
            inputs = inputs.flatten(1)
            targets = targets.flatten(1)
            
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
            
            pt = torch.exp(-bce_loss) 
            focal_term = (1 - pt) ** gamma
            alpha_term = torch.where(targets == 1, alpha, 1 - alpha)
            loss = alpha_term * focal_term * bce_loss
            return loss.mean(1).sum()


    def adjust_indices_order(self, pred_indices, gt_indices):
        adjusted_gt_indices = np.empty_like(gt_indices)
        sorted_pred_indices = np.argsort(pred_indices)
        for i, sorted_idx in enumerate(sorted_pred_indices):
            adjusted_gt_indices[i] = gt_indices[sorted_idx]
        return np.arange(len(pred_indices)), adjusted_gt_indices


    def hungarian_matcher(self, pred_masks, gt_masks):
        pred_masks = torch.stack([m.squeeze(0) for m in pred_masks]).flatten(1)
        gt_masks = torch.stack([m.squeeze(0) for m in gt_masks]).flatten(1)

        dice_loss_cur = self.batch_dice_loss(pred_masks, gt_masks)
        sigmoid_ce_loss_cur = self.batch_sigmoid_ce_loss(pred_masks, gt_masks)
        
        cost_matrix =  dice_loss_cur + sigmoid_ce_loss_cur

        pred_indices, gt_indices = linear_sum_assignment(cost_matrix.detach().cpu())
        adjust_pred_indices, adjust_gt_indices = self.adjust_indices_order(pred_indices, gt_indices)

        return adjust_pred_indices, adjust_gt_indices


    def hungarian_matcher_batch(self, pred_masks, gt_masks, change_list):
        reordered_gt_masks = []
        for batch_idx, groups in enumerate(change_list):
            batch_pred_masks = pred_masks[batch_idx]
            batch_gt_masks = gt_masks[batch_idx]
            reordered_batch_gt_masks = batch_gt_masks.clone()
            for group in groups:
                group_pred_masks = batch_pred_masks[group, :, :]
                group_gt_masks = batch_gt_masks[group, :, :]
                group_pred_masks = group_pred_masks.unsqueeze(1).flatten(1)
                group_gt_masks = group_gt_masks.unsqueeze(1).flatten(1)
                _, group_gt_indices = self.hungarian_matcher(group_pred_masks, group_gt_masks)
                for idx, gt_idx in enumerate(group_gt_indices):
                    reordered_batch_gt_masks[group[idx]] = batch_gt_masks[group[gt_idx]]
            reordered_gt_masks.append(reordered_batch_gt_masks)
        return reordered_gt_masks

    def create_padded_gt(self, pred_masks, gt_masks):
        pred_flat = pred_masks.flatten(1) 
        gt_flat = gt_masks.flatten(1)
        with torch.no_grad():
            dice_cost = self.batch_dice_loss(pred_flat, gt_flat)
            ce_cost = self.batch_sigmoid_ce_loss(pred_flat, gt_flat)
            cost_matrix = dice_cost + ce_cost
        pred_indices, gt_indices = linear_sum_assignment(cost_matrix.cpu().numpy())
        target_masks = torch.zeros_like(pred_masks)
        target_masks[pred_indices] = gt_masks[gt_indices]
        return target_masks


    def model_forward(
            self,
            images: torch.FloatTensor,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor,
            labels: torch.LongTensor,
            attention_masks: torch.LongTensor,
            offset: torch.LongTensor,
            masks_list: List[torch.FloatTensor],
            label_list: List[torch.Tensor],
            resize_list: List[tuple],
            inference: bool = False,
            change_list: List[torch.Tensor] = [],
            **kwargs,
        ):
            image_embeddings = self.get_visual_embs(images)
            batch_size = image_embeddings.shape[0]
            assert batch_size == len(offset) - 1

            if not inference and labels is not None:
                seg_token_mask = (labels == self.seg_token_idx)
            else:
                seg_token_mask = (input_ids == self.seg_token_idx)

            seg_token_mask = torch.cat(
                [seg_token_mask[:, 1:], torch.zeros_like(seg_token_mask[:, :1])],
                dim=1
            ).to(torch.bool)

            fused_image_embeddings = None 

            if inference:
                length = input_ids.shape[0]
                assert pixel_values.shape[0] == 1
                pixel_values_extend = pixel_values.expand(length, -1, -1, -1).contiguous()

                with torch.no_grad():
                    siglip_outputs = self.model.vision_tower(
                        pixel_values_extend, 
                        output_hidden_states=True
                    )
                
                fused_image_embeddings = self.model.fusion_adapter(
                    image_embeddings, 
                    siglip_outputs
                )

                out = super().forward(
                    pixel_values=pixel_values_extend,
                    attention_mask=attention_masks,
                    input_ids=input_ids,
                    output_hidden_states=True,
                    return_dict=True,
                )

                output_hidden_states = out.hidden_states
                output = None

            else: # Training Mode
                pixel_values_list = []
                for i in range(len(offset) - 1):
                    start_i, end_i = offset[i], offset[i + 1]
                    pixel_values_i = (
                        pixel_values[i]
                        .unsqueeze(0)
                        .expand(end_i - start_i, -1, -1, -1)
                        .contiguous()
                    )
                    pixel_values_list.append(pixel_values_i)
                pixel_values_processed = torch.cat(pixel_values_list, dim=0)

                with torch.no_grad(): 
                    siglip_outputs = self.model.vision_tower(
                        pixel_values_processed, 
                        output_hidden_states=True
                    )

                fused_image_embeddings = self.model.fusion_adapter(image_embeddings, siglip_outputs)

                output = super().forward(
                    pixel_values=pixel_values_processed,
                    attention_mask=attention_masks,
                    input_ids=input_ids,
                    labels=labels,
                    output_hidden_states=True,
                )
                output_hidden_states = output.hidden_states

            last_hidden_state = output_hidden_states[-1]
            
            # [CHANGED] Use MLP approach: Project first, then select
            # Legacy approach: feed hidden state to projector first, then mask [SEG] tokens
            # Unlike CrossAttn, which required masking before projection
            
            # Step 1: Project all tokens first (consistent with original MedFuseSeg approach)
            # For memory efficiency, selecting before projecting works equivalently for MLP
            
            # For safety, follow the original MedFuseSeg implementation:
            # hidden_states.append(self.model.text_hidden_fcs[0](last_hidden_state)) 
            # last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            
            # Our approach:
            pred_embeddings = self.model.projector(last_hidden_state) # [Batch, Seq, Dim]
            
            # Select only [SEG] token embeddings
            pred_embeddings = pred_embeddings[seg_token_mask] # [Total_SEGs, Dim]
            
            # Edge case: guard against zero SEG tokens in the batch
            if pred_embeddings.shape[0] == 0:
                 pred_embeddings = torch.zeros(0, self.config.out_dim).to(last_hidden_state.device)

            seg_token_counts = seg_token_mask.int().sum(-1) 
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )
            seg_token_offset = seg_token_offset[offset]

            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=fused_image_embeddings[i].unsqueeze(0), 
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                if low_res_masks.shape[0] > 0:
                    pred_mask = self.model.visual_model.postprocess_masks(
                        low_res_masks,
                        input_size=resize_list[i],
                        original_size=label_list[i].shape,
                    )
                else:
                    target_h, target_w = label_list[i].shape[-2:]
                    pred_mask = torch.zeros(
                        (0, 1, target_h, target_w),
                        dtype=low_res_masks.dtype,
                        device=low_res_masks.device
                    )
                pred_masks.append(pred_mask[:, 0])

            model_output = output
            gt_masks = masks_list

            for list_id in range(len(change_list)):
                if isinstance(change_list[list_id], list):
                    gt_masks_cur = self.hungarian_matcher_batch([pred_masks[list_id]], [gt_masks[list_id]], [change_list[list_id]])
                    gt_masks[list_id] = gt_masks_cur[0]
                else:
                    gt_masks[list_id] = gt_masks[list_id] 

            if inference:
                return {
                    "pred_masks": pred_masks,
                    "gt_masks": gt_masks,
                }

            output = model_output.logits

            ce_loss = model_output.loss
            ce_loss = ce_loss * self.ce_loss_weight
            mask_bce_loss = 0
            mask_dice_loss = 0
            num_masks = 0
            for batch_idx in range(len(pred_masks)):
                gt_mask = gt_masks[batch_idx]
                pred_mask = pred_masks[batch_idx]

                assert (
                    gt_mask.shape[0] == pred_mask.shape[0]
                ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                    gt_mask.shape, pred_mask.shape
                )
                current_focal_loss = self.batch_sigmoid_focal_loss(
                    pred_mask, 
                    gt_mask, 
                    alpha=0.25, 
                    gamma=2.0
                )
                mask_bce_loss += current_focal_loss

                mask_dice_loss += (
                    dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                    * gt_mask.shape[0]
                )
                num_masks += gt_mask.shape[0]

            mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
            mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
            mask_loss = mask_bce_loss + mask_dice_loss
            loss = ce_loss + mask_loss

            return {
                "loss": loss,
                "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "mask_loss": mask_loss,
            }

    def evaluate(
        self,
        pixel_values,
        images,
        input_ids,
        attention_mask,
        resize_list,
        original_size_list,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            # Step 1: Generate text response
            outputs = self.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
                do_sample=False,
                eos_token_id=[self.seg_token_idx],
                no_repeat_ngram_size=3,
                num_beams=5,
                repetition_penalty=1.1
            )
            output_ids = outputs.sequences   # [B, L]

            # Step 2: Build attention mask covering the generated sequence
            if self.config.pad_token_id is not None:
                full_attn = (output_ids != self.config.pad_token_id).long()
            else:
                full_attn = torch.ones_like(output_ids, dtype=torch.long)

            # Step 3: Re-run forward pass to obtain per-token hidden states
            # Using super().forward to call Gemma3 directly (returns [B, L, H])
            full_out = super().forward(
                pixel_values=pixel_values,
                input_ids=output_ids,
                attention_mask=full_attn,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden_tensor = full_out.hidden_states[-1]   # [B, L, H]

            # Step 4: Locate [SEG] token positions
            seg_token_mask = (output_ids == self.seg_token_idx)
            seg_token_mask = torch.cat(
                [seg_token_mask[:, 1:], torch.zeros_like(seg_token_mask[:, :1])],
                dim=1
            ).to(torch.bool)

            # [CHANGED] Use MLP Projection Logic for Inference
            pred_embeddings = self.model.projector(last_hidden_tensor)
            pred_embeddings = pred_embeddings[seg_token_mask]

            if pred_embeddings.shape[0] == 0:
                 pred_embeddings = torch.zeros(0, self.config.out_dim, dtype=torch.bfloat16).to(last_hidden_tensor.device)

            # Step 6: Group embeddings by number of [SEG] tokens per sample
            seg_token_counts = seg_token_mask.int().sum(-1)
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat([torch.zeros(1, device=seg_token_offset.device, dtype=torch.long), seg_token_offset], dim=0)
            pred_embeddings_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
            pred_embeddings = pred_embeddings_

            # Step 7: SAM mask decoding
            image_embeddings = self.get_visual_embs(images)
            siglip_outputs = self.model.vision_tower(
                pixel_values, 
                output_hidden_states=True
            )

            fused_image_embeddings = self.model.fusion_adapter(
                image_embeddings, 
                siglip_outputs
            )

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                sparse_embeddings, dense_embeddings = self.model.visual_model.prompt_encoder(
                    points=None, boxes=None, masks=None, text_embeds=pred_embeddings[i].unsqueeze(1),
                )
                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, _ = self.model.visual_model.mask_decoder(
                    image_embeddings=fused_image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks, input_size=resize_list[i], original_size=original_size_list[i],
                )
                pred_masks.append(pred_mask[:, 0])

        return output_ids, pred_masks