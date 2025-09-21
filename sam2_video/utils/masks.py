"""
Mask and merging utilities for SAM2 training.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import cv2
from loguru import logger
import sys


@logger.catch(onerror=lambda _: sys.exit(1))
def find_connected_components(mask: torch.Tensor) -> List[torch.Tensor]:
    mask_np = mask.cpu().numpy().astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened_mask = cv2.dilate(
        cv2.erode(mask_np, kernel, iterations=1), kernel, iterations=1
    )
    num_components, labeled_mask = cv2.connectedComponents(opened_mask)
    connected_areas: List[torch.Tensor] = []
    for component_id in range(1, num_components):
        component_mask = (labeled_mask == component_id).astype(np.uint8)
        component_tensor = torch.from_numpy(component_mask.astype(np.float32)).to(
            mask.device
        )
        connected_areas.append(component_tensor)
    return connected_areas


@logger.catch(onerror=lambda _: sys.exit(1))
def cat_to_obj_mask(
    cat_frame_masks: torch.Tensor,
) -> Tuple[torch.Tensor, List[int], int]:
    N = int(cat_frame_masks.shape[0])
    obj_to_cat: List[int] = []
    obj_masks_list: List[torch.Tensor] = []
    for catergory_idx in range(N):
        category_mask = (cat_frame_masks[catergory_idx][0] > 0).to(torch.float32)
        if category_mask.sum() == 0:
            continue
        connected_areas = find_connected_components(category_mask)
        for area_mask in connected_areas:
            obj_masks_list.append(area_mask)
            obj_to_cat.append(catergory_idx)
    if not obj_masks_list:
        raise ValueError(
            "cat_to_obj_mask: no objects found in category masks (fail-fast)"
        )
    return torch.stack(obj_masks_list).unsqueeze(1), obj_to_cat, N


@logger.catch(onerror=lambda _: sys.exit(1))
def merge_object_results_to_category(
    previous_stages_out: List[Dict[str, Any]],
    obj_to_cat: List[int],
    num_categories: int,
) -> List[Dict[str, Any]]:
    """
    Merge per-object outputs back to per-category outputs for each frame.

    Rules
    - Mask-like tensors (logits) are merged by pixelwise max across objects in the same category
      to approximate logical OR at the probability level.
    - Score/IoU-like tensors are merged by weighted average using per-object mask area
      (sum of sigmoid(logits)) as weights. If weights sum to 0 for a category, fall back to mean.
    - Null (None) values are ignored/preserved as None.
    - Mask memory related items are ignored (not included in merged output).

    Args:
        previous_stages_out: List of per-frame dicts as produced by forward_tracking
        obj_to_cat: List mapping object index -> category index

    Returns:
        List of per-frame dicts aggregated to category level.
    """

    if not previous_stages_out:
        return []

    # Determine category groups from obj_to_cat (ensure contiguous categories 0..max)
    if len(obj_to_cat) == 0:
        return [
            {k: None for k in ("pred_masks", "pred_masks_high_res")}
            for _ in previous_stages_out
        ]
    # Categories are indexed by their id in obj_to_cat; use max id + 1
    cat_to_indices: List[List[int]] = [[] for _ in range(num_categories)]
    for obj_idx, cat_idx in enumerate(obj_to_cat):
        cat_to_indices[int(cat_idx)].append(int(obj_idx))

    def _area_weights_from_masks(mask_logits: torch.Tensor) -> torch.Tensor:
        """Compute per-object weights from mask logits as probability mass area.

        mask_logits: [N, 1, H, W]
        returns: [N]
        """
        probs = torch.sigmoid(mask_logits)  # [N, 1, H, W]
        areas = probs.sum(dim=(1, 2, 3))  # [N]
        return areas

    def _grouped_max(
        tensor: torch.Tensor,  # [N, ...]
        groups: List[List[int]],
    ) -> torch.Tensor:
        """Pixelwise max across objects within the same category."""
        if tensor.numel() == 0:
            # Create an empty tensor with category dim if no objects
            return tensor.new_zeros((len(groups),) + tuple(tensor.shape[1:]))
        out_per_cat: List[torch.Tensor] = []
        for idxs in groups:
            if len(idxs) == 0:
                out_per_cat.append(tensor.new_zeros(tensor.shape[1:]))
            else:
                out_per_cat.append(tensor[idxs].max(dim=0).values)
        return torch.stack(out_per_cat, dim=0)

    def _grouped_weighted_avg(
        tensor: torch.Tensor,  # [N, ...]
        groups: List[List[int]],
        weights: torch.Tensor,  # [N]
    ) -> torch.Tensor:
        """Weighted average across objects within the same category.

        Broadcasting: weights will be broadcast across remaining dims of `tensor`.
        """
        if tensor.numel() == 0:
            return tensor.new_zeros((len(groups),) + tuple(tensor.shape[1:]))
        # Prepare weights broadcast shape: [N, 1, 1, ...]
        w = weights.view(weights.shape[0], *([1] * (tensor.dim() - 1)))
        out_per_cat: List[torch.Tensor] = []
        for idxs in groups:
            if len(idxs) == 0:
                out_per_cat.append(tensor.new_zeros(tensor.shape[1:]))
                continue
            sub = tensor[idxs]
            sub_w = w[idxs]
            denom = sub_w.sum(dim=0)
            if torch.all(denom == 0):
                out_per_cat.append(sub.mean(dim=0))
            else:
                out_per_cat.append((sub * sub_w).sum(dim=0) / denom)
        return torch.stack(out_per_cat, dim=0)

    merged_per_frame: List[Dict[str, Any]] = []

    for frame_out in previous_stages_out:
        merged: Dict[str, Any] = {}

        # Determine weights from available masks
        if isinstance(frame_out.get("pred_masks_high_res"), torch.Tensor):
            weights_source = frame_out["pred_masks_high_res"]  # [N, 1, H, W]
        elif isinstance(frame_out.get("pred_masks"), torch.Tensor):
            weights_source = frame_out["pred_masks"]  # [N, 1, H, W]
        else:
            raise ValueError(
                "Weights source not found in frame outputs; expected 'pred_masks' or 'pred_masks_high_res'"
            )

        weights: torch.Tensor = _area_weights_from_masks(weights_source)

        # 1) Merge mask-like keys by pixelwise max (union)
        for key in (
            "pred_masks",
            "pred_masks_high_res",
            "multistep_pred_masks",
            "multistep_pred_masks_high_res",
        ):
            val = frame_out.get(key)
            if isinstance(val, torch.Tensor):  # [N, ...]
                merged[key] = _grouped_max(val, cat_to_indices)

        # 2) Merge multi-mask proposals per step by pixelwise max
        for key in ("multistep_pred_multimasks", "multistep_pred_multimasks_high_res"):
            val = frame_out.get(key)
            if (
                isinstance(val, list)
                and len(val) > 0
                and isinstance(val[0], torch.Tensor)
            ):
                step_out: List[torch.Tensor] = []
                for step_tensor in val:  # [N, K, H, W]
                    step_out.append(_grouped_max(step_tensor, cat_to_indices))
                merged[key] = step_out

        # 3) Merge IoUs and scores per step by weighted average
        for key in ("multistep_pred_ious", "multistep_object_score_logits"):
            val = frame_out.get(key)
            if (
                isinstance(val, list)
                and len(val) > 0
                and isinstance(val[0], torch.Tensor)
            ):
                step_out: List[torch.Tensor] = []
                for step_tensor in val:  # [N, K]
                    step_out.append(
                        _grouped_weighted_avg(step_tensor, cat_to_indices, weights)
                    )
                merged[key] = step_out

        merged["point_inputs"] = frame_out["point_inputs"]
        merged["mask_inputs"] = frame_out["mask_inputs"]
        if isinstance(frame_out.get("multistep_point_inputs"), list):
            merged["multistep_point_inputs"] = [
                None for _ in frame_out["multistep_point_inputs"]
            ]

        # Explicitly ignore mask memory related items
        # (maskmem_features, maskmem_pos_enc) are not included

        merged_per_frame.append(merged)

    return merged_per_frame
