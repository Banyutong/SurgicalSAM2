import tempfile
import json
import os
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from icecream import ic
import cv2
import numpy as np
from copy import deepcopy
import uuid
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from loguru import logger
from typing import List
import random
import albumentations as A
from typing import List, Dict, TypedDict, NamedTuple, Generator, Tuple, Set


# class PromptObj(TypedDict):
#     """Typed dictionary for storing prompt object."""

#     mask: np.ndarray
#     bbox: List[float]
#     points: List[List[float]]
#     obj_id: int
#     pos_or_neg_label: List[int]


# class PromptInfo(TypedDict):
#     """Typed dictionary for storing prompt information."""

#     prompt_objs: List[Dict]
#     frame_idx: int
#     prompt_type: str
#     video_id: str
#     path: str


# class ClipRange(NamedTuple):
#     """Named tuple for storing clip range."""

#     start_idx: int
#     end_idx: int


def show_mask(mask, ax, obj_id=None, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([1])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def mask_to_masks(mask: np.ndarray) -> list:
    kernel = np.ones((5, 5), np.uint8)  # 可以调整核的大小来控制闭运算程度

    # 对 mask 进行闭运算
    closed_mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        closed_mask.astype(np.uint8)
    )
    binary_masks = []
    min_area = 10  # 设置最小连通区域面积
    for i in range(1, num_labels):  # 从 1 开始，因为 0 表示背景
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:  # 过滤面积过小的连通区域
            # 生成只包含当前连通区域的二值mask
            binary_mask = labels == i
            binary_masks.append(binary_mask)

    return binary_masks


def mask_to_points(mask, num_points=1):
    # 确保mask是一个二值化的numpy数组
    if not isinstance(mask, np.ndarray) or mask.dtype != bool:
        print(type(mask))
        raise ValueError("mask must be a binary numpy array")

    # 找到掩码中的所有True点的坐标
    points = np.argwhere(mask)
    points = points[:, [1, 0]]
    # 如果num_points为1，返回掩码的中心点
    if num_points == 1:
        # 计算中心点
        center = np.mean(points, axis=0).astype(int)
        return center.reshape(1, -1)

    # 如果num_points大于1，从掩码中随机采样指定数量的点
    # logger.info(f"points.shape: {points.shape}")
    if num_points > points.shape[0]:
        raise ValueError("num_points is greater than the number of points in the mask")

    sampled_points = points[
        np.random.choice(points.shape[0], num_points, replace=False)
    ]
    return sampled_points


def mask_to_bbox(mask):
    """
    Extracts the bounding box from a binary mask.
    """
    pos = np.where(mask)
    if len(pos[0]) == 0:
        return None
    xmin, ymin = np.min(pos[1]), np.min(pos[0])
    xmax, ymax = np.max(pos[1]), np.max(pos[0])
    return [float(xmin), float(ymin), float(xmax), float(ymax)]


# def dilate_mask(mask, **kwargs):
#     kernel_size = random.randrange(3, 21, 2)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
#     noised_mask = cv2.dilate(mask, kernel)
#     return noised_mask.astype(bool)


# def erode_mask(mask, **kwargs):
#     kernel_size = random.randrange(3, 21, 2)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
#     noised_mask = cv2.erode(mask, kernel)
#     return noised_mask.astype(bool)


# MASK_TRANSFORM = transform = A.Compose(
#     [
#         A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5),
#         A.OneOf([A.Lambda(mask=dilate_mask), A.Lambda(mask=erode_mask)], p=0.5),
#     ]
# )


# def add_noise_to_mask(obj: PromptObj):
#     mask = obj["mask"]
#     image = np.zeros(mask.shape, dtype=np.uint8)
#     mask = mask.astype(np.uint8)
#     transformed = MASK_TRANSFORM(image=image, mask=mask)
#     transformed_image = transformed["image"]
#     transformed_mask = transformed["mask"]
#     obj["mask"] = transformed_mask.astype(bool)
#     return obj


# BBOX_TRANSFORM = A.Compose(
#     [A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.5, rotate_limit=10, p=0.5)],
#     bbox_params=A.BboxParams(format="pascal_voc"),
# )


# def add_noise_to_bbox(obj: PromptObj):
#     bbox = obj["bbox"]
#     image = np.zeros(obj["mask"].shape, dtype=np.uint8)
#     bbox.append("bbox")
#     transformed = BBOX_TRANSFORM(image=image, bboxes=[bbox])
#     if len(transformed["bboxes"]) == 0:
#         return None
#     new_bbox = list(transformed["bboxes"][0])[:-1]
#     obj["bbox"] = new_bbox
#     return obj
