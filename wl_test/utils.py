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
