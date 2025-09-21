"""
Prompt generation utilities (points and boxes) for SAM2 training.
"""

from typing import Tuple
import torch
import numpy as np
from scipy import ndimage
from loguru import logger
import sys


@logger.catch(onerror=lambda _: sys.exit(1))
def generate_point_prompt(
    mask: torch.Tensor,
    num_pos_points: int = 1,
    num_neg_points: int = 0,
    include_center: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _, H, W = mask.shape
    device = mask.device
    dtype = torch.float32

    mask = (mask.squeeze(1) > 0).to(torch.uint8)  # [B, H, W]
    total_points = num_pos_points + num_neg_points
    points_all = torch.empty(B, total_points, 2, dtype=dtype, device=device)
    labels_all = torch.empty(B, total_points, dtype=torch.int32, device=device)

    for b in range(B):
        m = mask[b]
        pos_coords = torch.stack(torch.where(m == 1), dim=1)  # (y, x)
        num_pos_available = pos_coords.shape[0]

        if num_pos_points > 0 and num_pos_available == 0:
            raise ValueError("generate_point_prompt: no positive pixels available for sampling")

        if num_pos_available > 0:
            cy, cx = ndimage.center_of_mass(m.cpu().numpy())
            center = torch.tensor([cx, cy], dtype=dtype, device=device)
        else:
            center = torch.empty(2, dtype=dtype, device=device)

        if include_center and num_pos_points > 0:
            pos_pts = [center.unsqueeze(0)]
            need_extra = max(0, num_pos_points - 1)
        else:
            pos_pts = []
            need_extra = num_pos_points

        if need_extra > 0:
            idx = torch.randperm(num_pos_available, device=device)[:need_extra]
            sampled = pos_coords[idx].flip(-1).to(dtype)  # (x, y)
            pos_pts.append(sampled)

        pos_pts = torch.cat(pos_pts, dim=0) if num_pos_points > 0 else torch.empty(0, 2, dtype=dtype, device=device)

        neg_coords = torch.stack(torch.where(m == 0), dim=1)
        num_neg_available = neg_coords.shape[0]
        if num_neg_points > 0 and num_neg_available > 0:
            idx = torch.randperm(num_neg_available, device=device)[:num_neg_points]
            neg_pts = neg_coords[idx].flip(-1).to(dtype)
        else:
            neg_pts = torch.empty(0, 2, dtype=dtype, device=device)

        pts = torch.cat([pos_pts, neg_pts], 0)
        lbls = torch.cat(
            [
                torch.ones(pos_pts.shape[0], dtype=torch.int32, device=device),
                torch.zeros(neg_pts.shape[0], dtype=torch.int32, device=device),
            ]
        )
        points_all[b] = pts
        labels_all[b] = lbls

    return points_all, labels_all


@logger.catch(onerror=lambda _: sys.exit(1))
def generate_box_prompt(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, _, H, W = mask.shape
    points = torch.empty((B, 2, 2), dtype=torch.float32, device=mask.device)
    labels = torch.empty((B, 2), dtype=torch.int32, device=mask.device)

    for i in range(B):
        m = (mask[i, 0] > 0).cpu().numpy()
        ys, xs = np.where(m > 0)
        if xs.size == 0:
            raise ValueError("generate_box_prompt: no positive pixels to form a bounding box")
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        points[i, 0] = torch.tensor([float(x_min), float(y_min)], device=mask.device)
        labels[i, 0] = 2
        points[i, 1] = torch.tensor([float(x_max), float(y_max)], device=mask.device)
        labels[i, 1] = 3

    return points, labels

