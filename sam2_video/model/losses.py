# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from loguru import logger

# Local constant to avoid circular import
CORE_LOSS_KEY = "total_loss"


def dice_loss(inputs, targets, num_objects, loss_on_multimask=False):
    inputs = inputs.sigmoid()
    if loss_on_multimask:
        assert inputs.dim() == 4 and targets.dim() == 4
        inputs = inputs.flatten(2)
        targets = targets.flatten(2)
        numerator = 2 * (inputs * targets).sum(-1)
    else:
        inputs = inputs.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


def sigmoid_focal_loss(
    inputs,
    targets,
    num_objects,
    alpha: float = 0.25,
    gamma: float = 2,
    loss_on_multimask=False,
):
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    prob = inputs.sigmoid()
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if loss_on_multimask:
        assert loss.dim() == 4
        return loss.flatten(2).mean(-1) / num_objects
    return loss.mean(1).sum() / num_objects


def iou_loss(
    inputs, targets, pred_ious, num_objects, loss_on_multimask=False, use_l1_loss=False
):
    assert inputs.dim() == 4 and targets.dim() == 4
    pred_mask = inputs.flatten(2) > 0
    gt_mask = targets.flatten(2) > 0
    area_i = torch.sum(pred_mask & gt_mask, dim=-1).float()
    area_u = torch.sum(pred_mask | gt_mask, dim=-1).float()
    actual_ious = area_i / torch.clamp(area_u, min=1.0)

    if use_l1_loss:
        loss = F.l1_loss(pred_ious, actual_ious, reduction="none")
    else:
        loss = F.mse_loss(pred_ious, actual_ious, reduction="none")
    if loss_on_multimask:
        return loss / num_objects
    return loss.sum() / num_objects


class MultiStepMultiMasksAndIous(nn.Module):
    def __init__(
        self,
        weight_dict,
        focal_alpha=0.25,
        focal_gamma=2.0,
        supervise_all_iou=False,
        iou_use_l1_loss=False,
        pred_obj_scores=False,
        focal_gamma_obj_score=0.0,
        focal_alpha_obj_score=-1,
        logit_temperature: float = 1.0,
    ):
        super().__init__()
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        assert "loss_mask" in self.weight_dict
        assert "loss_dice" in self.weight_dict
        assert "loss_iou" in self.weight_dict
        if "loss_class" not in self.weight_dict:
            self.weight_dict["loss_class"] = 0.0

        self.focal_alpha_obj_score = focal_alpha_obj_score
        self.focal_gamma_obj_score = focal_gamma_obj_score
        self.supervise_all_iou = supervise_all_iou
        self.iou_use_l1_loss = iou_use_l1_loss
        self.pred_obj_scores = pred_obj_scores
        if not (isinstance(logit_temperature, (int, float)) and logit_temperature > 0):
            raise ValueError("logit_temperature must be a positive float")
        self.logit_temperature = float(logit_temperature)

    @logger.catch(onerror=lambda _: sys.exit(1))
    def forward(self, outs_batch: List[Dict], targets_batch: torch.Tensor):
        assert len(outs_batch) == len(targets_batch)
        num_objects = float(targets_batch.shape[1])
        losses = defaultdict(int)
        for outs, targets in zip(outs_batch, targets_batch):
            cur_losses = self._forward(outs, targets, num_objects)
            for k, v in cur_losses.items():
                losses[k] += v

        return losses

    @logger.catch(onerror=lambda _: sys.exit(1))
    def _forward(self, outputs: Dict, targets: torch.Tensor, num_objects):
        target_masks = targets.unsqueeze(1).float()
        src_masks_list = outputs["multistep_pred_multimasks_high_res"]
        ious_list = outputs["multistep_pred_ious"]
        object_score_logits_list = outputs["multistep_object_score_logits"]

        assert len(src_masks_list) == len(ious_list)
        assert len(object_score_logits_list) == len(ious_list)

        losses = {"loss_mask": 0, "loss_dice": 0, "loss_iou": 0, "loss_class": 0}
        for src_masks, ious, object_score_logits in zip(
            src_masks_list, ious_list, object_score_logits_list
        ):
            self._update_losses(
                losses, src_masks, target_masks, ious, num_objects, object_score_logits
            )
        losses[CORE_LOSS_KEY] = self.reduce_loss(losses)
        return losses

    def _update_losses(
        self, losses, src_masks, target_masks, ious, num_objects, object_score_logits
    ):
        target_masks = target_masks.expand_as(src_masks)

        # Filter valid masks (channels with foreground pixels)
        valid = target_masks.sum(
            dim=(2, 3)
        ).bool()  # [N] which channels have foreground

        if not valid.any():
            logger.warning("DEBUG: no valid masks")
            logger.warning(f"DEBUG: src_masks.shape = {src_masks.shape}")
            zero_loss = src_masks.sum() * 0.0
            losses["loss_mask"] += zero_loss
            losses["loss_dice"] += zero_loss
            losses["loss_iou"] += zero_loss
            losses["loss_class"] += zero_loss
            raise ValueError("No valid masks")  # 抛出异常，不再继续计算

        # Filter tensors to only include valid masks
        src_masks = src_masks[valid].unsqueeze(1)
        # Temperature scale mask logits before losses
        src_masks = src_masks / self.logit_temperature
        target_masks = target_masks[valid].unsqueeze(1)
        ious = ious[valid].unsqueeze(1)

        if object_score_logits is not None:
            object_score_logits = object_score_logits[valid]

        # Update num_objects for filtered tensors
        num_objects = float(src_masks.shape[0])
        loss_multimask = sigmoid_focal_loss(
            src_masks,
            target_masks,
            num_objects,
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
            loss_on_multimask=True,
        )
        loss_multidice = dice_loss(
            src_masks, target_masks, num_objects, loss_on_multimask=True
        )
        if not self.pred_obj_scores:
            loss_class = loss_multimask.sum() * 0.0
            target_obj = torch.ones(
                loss_multimask.shape[0],
                1,
                dtype=loss_multimask.dtype,
                device=loss_multimask.device,
            )
        else:
            target_obj = torch.any((target_masks[:, 0] > 0).flatten(1), dim=-1)[
                ..., None
            ].float()
            loss_class = sigmoid_focal_loss(
                object_score_logits,
                target_obj,
                num_objects,
                alpha=self.focal_alpha_obj_score,
                gamma=self.focal_gamma_obj_score,
            )

        loss_multiiou = iou_loss(
            src_masks,
            target_masks,
            ious,
            num_objects,
            loss_on_multimask=True,
            use_l1_loss=self.iou_use_l1_loss,
        )
        assert loss_multimask.dim() == 2
        assert loss_multidice.dim() == 2
        assert loss_multiiou.dim() == 2
        if loss_multimask.size(1) > 1:
            loss_combo = (
                loss_multimask * self.weight_dict["loss_mask"]
                + loss_multidice * self.weight_dict["loss_dice"]
            )
            best_loss_inds = torch.argmin(loss_combo, dim=-1)
            batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
            loss_mask = loss_multimask[batch_inds, best_loss_inds].unsqueeze(1)
            loss_dice = loss_multidice[batch_inds, best_loss_inds].unsqueeze(1)
            if self.supervise_all_iou:
                loss_iou = loss_multiiou.mean(dim=-1).unsqueeze(1)
            else:
                loss_iou = loss_multiiou[batch_inds, best_loss_inds].unsqueeze(1)
        else:
            loss_mask = loss_multimask
            loss_dice = loss_multidice
            loss_iou = loss_multiiou

        losses["loss_mask"] += loss_mask.sum()
        losses["loss_dice"] += loss_dice.sum()
        losses["loss_iou"] += loss_iou.sum()
        losses["loss_class"] += loss_class

    def reduce_loss(self, losses):
        reduced_loss = 0.0
        for loss_key, weight in self.weight_dict.items():
            if loss_key not in losses:
                raise ValueError(f"{type(self)} doesn't compute {loss_key}")
            if weight != 0:
                reduced_loss += losses[loss_key] * weight

        return reduced_loss


class BCECategoryLoss(nn.Module):
    """
    Binary cross-entropy loss across category channels for segmentation masks.

    Expects per-frame predictions aggregated at category level with logits of
    shape [C, 1, H, W] or [C, H, W] and ground-truth masks of shape [C, H, W]
    with values in {0, 1} (bool or float). Applies BCE with logits independently
    per category and averages over pixels and frames.

    This follows the simple formulation:

        BCE(pred_logits, soft_mask, pos_weight=None, reduction='mean')

    where pos_weight optionally addresses class imbalance.
    """

    def __init__(
        self,
        pos_weight: Optional[Union[List[float], torch.Tensor]] = None,
        reduction: str = "mean",
        logit_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        # Store as tensor if provided; device/dtype adjusted in forward
        if isinstance(pos_weight, list):
            self.register_buffer(
                "_pos_weight",
                torch.tensor(pos_weight, dtype=torch.float32),
                persistent=False,
            )
        elif isinstance(pos_weight, torch.Tensor):
            self.register_buffer(
                "_pos_weight", pos_weight.to(dtype=torch.float32), persistent=False
            )
        else:
            self._pos_weight = None  # type: ignore
        self.reduction = reduction
        if not (isinstance(logit_temperature, (int, float)) and logit_temperature > 0):
            raise ValueError("logit_temperature must be a positive float")
        self.logit_temperature = float(logit_temperature)

    @staticmethod
    def _bce_loss(
        pred_logits: torch.Tensor,
        soft_mask: torch.Tensor,
        pos_weight: Optional[torch.Tensor],
        reduction: str,
    ):
        """Helper mirroring the requested API for BCE with logits."""
        return F.binary_cross_entropy_with_logits(
            pred_logits,
            soft_mask,
            pos_weight=pos_weight,
            reduction=reduction,
        )

    @logger.catch(onerror=lambda _: sys.exit(1))
    def forward(
        self, outs_batch: List[Dict], targets_batch: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        assert len(outs_batch) == len(targets_batch), (
            f"Mismatched sequence lengths: outs={len(outs_batch)} vs targets={len(targets_batch)}"
        )

        total_loss = 0.0
        num_frames = len(outs_batch)

        for frame_idx, (outs, targets) in enumerate(zip(outs_batch, targets_batch)):
            # Select prediction logits at category level
            logits = outs.get("pred_masks_high_res")
            if logits is None:
                logits = outs.get("pred_masks")
            if logits is None:
                raise KeyError(
                    "BCECategoryLoss expects 'pred_masks_high_res' or 'pred_masks' in outputs"
                )

            # logits: [C, 1, H, W] or [C, H, W]
            if logits.dim() == 4 and logits.shape[1] == 1:
                logits = logits.squeeze(1)
            elif logits.dim() != 3:
                raise ValueError(
                    f"Unexpected logits shape for BCECategoryLoss: {tuple(logits.shape)}"
                )
            # targets: [C, H, W] (bool or float)
            if targets.dim() != 3:
                raise ValueError(
                    f"Unexpected target shape for BCECategoryLoss: {tuple(targets.shape)}"
                )
            soft_mask = targets.to(dtype=logits.dtype)

            valid = targets.sum(dim=(1, 2)).bool()  # [C] 哪些通道有前景
            logits = logits[valid]
            soft_mask = soft_mask[valid]
            # Temperature scale logits before BCE
            logits = logits / self.logit_temperature
            pos_w = None
            # Prepare optional pos_weight
            if self._pos_weight is not None:
                # Accept either [C] or [C,1,1] and move to device/dtype
                if self._pos_weight.dim() == 1:
                    pos_w = self._pos_weight.to(
                        device=logits.device, dtype=logits.dtype
                    ).view(-1, 1, 1)
                else:
                    pos_w = self._pos_weight.to(
                        device=logits.device, dtype=logits.dtype
                    )
                if pos_w.shape[0] != logits.shape[0]:
                    raise ValueError(
                        f"pos_weight length {pos_w.shape[0]} does not match number of classes {logits.shape[0]}"
                    )
            pos_w = pos_w[valid] if pos_w is not None else None
            loss = self._bce_loss(logits, soft_mask, pos_w, self.reduction)
            total_loss = total_loss + loss

        # Average over frames for stable scaling
        total_loss = total_loss / max(num_frames, 1)
        return {
            "loss_bce": total_loss,
            CORE_LOSS_KEY: total_loss,
        }
