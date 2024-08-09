import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import colorsys
from itertools import groupby
import pycocotools.mask as mask_util
import numpy as np
from itertools import groupby

def get_model_cfg(checkpoint_name):
    if 'tiny' in checkpoint_name:
        return 'sam2_hiera_t.yaml'
    elif 'small' in checkpoint_name:
        return 'sam2_hiera_s.yaml'
    elif 'base' in checkpoint_name:
        return 'sam2_hiera_b.yaml'
    elif 'large' in checkpoint_name:
        return 'sam2_hiera_l.yaml'
    else:
        raise ValueError(f"Unable to determine model configuration from checkpoint name: {checkpoint_name}")

def rle_to_binary_mask(rle):
    """
    Converts RLE to a binary mask.
    """
    if isinstance(rle, dict):
        return mask_util.decode(rle)
    elif isinstance(rle, list):
        # If RLE is in COCO format (list of counts)
        h, w = rle.get('size', (0, 0))
        if h == 0 or w == 0:
            raise ValueError("Invalid mask size in RLE data")
        rle_dict = {'counts': rle, 'size': [h, w]}
        return mask_util.decode(rle_dict)
    else:
        raise ValueError("Unsupported RLE format")

def mask_to_rle(binary_mask):
    """
    Converts a binary mask to Run-Length Encoding (RLE).
    """
    rle = mask_util.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def mask_to_bbox(mask):
    """
    Extracts the bounding box from a binary mask.
    """
    pos = np.where(mask)
    if len(pos[0]) == 0:
        return None
    xmin, ymin = np.min(pos[1]), np.min(pos[0])
    xmax, ymax = np.max(pos[1]), np.max(pos[0])
    return [float(xmin), float(ymin), float(xmax - xmin + 1), float(ymax - ymin + 1)]


def create_coco_annotation(mask, image_id, category_id, annotation_id):
    contours, _ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour = contour.flatten().tolist()
        if len(contour) > 4:  # Avoid single points or lines
            segmentation.append(contour)
    if not segmentation:
        return None
    x, y, w, h = cv2.boundingRect(contours[0])
    area = int(np.sum(mask))
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": segmentation,
        "area": area,
        "bbox": [x, y, w, h],
        "iscrowd": 0
    }


