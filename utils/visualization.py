import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import colorsys
from itertools import groupby
from skimage import measure
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   


def get_color_map(num_classes):
    """Generate a color map for visualizing different objects."""
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(rgb)  # Keep as float values in range 0-1
    return colors



def visualize_first_frame(image, bboxes, sampled_points, output_path):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image)
    
    colors = get_color_map(len(bboxes))
    
    for i, (bbox, points) in enumerate(zip(bboxes, sampled_points)):
        color = colors[i]
        
        # Draw bounding box
        x, y, w, h = bbox
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        
        # Plot sampled points
        points = np.array(points)
        ax.scatter(points[:, 0], points[:, 1], c=[color], s=50)
    
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"First frame visualization (bboxes and points) saved to {output_path}")

import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import random

def visualize_first_frame_mask(image, masks, sampled_points, output_path):
    # Convert image to RGB if it's not already
    if isinstance(image, str):
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif isinstance(image, Image.Image):
        image = np.array(image)

    # Create a copy of the image for drawing contours
    contour_image = image.copy()

    # Generate a color map for the masks
    colors = get_color_map(len(masks))

    # Loop over each mask and draw the contour
    for i, (mask, points) in enumerate(zip(masks, sampled_points)):
        color = [int(c * 255) for c in colors[i]]

        # Find contours of the mask
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw contours on the image
        cv2.drawContours(contour_image, contours, -1, color, 2)

        # Plot sampled points
        for point in points:
            cv2.circle(contour_image, tuple(map(int, point)), 5, color, -1)

    # Visualize the image with the contours
    plt.figure(figsize=(10, 10))
    plt.imshow(contour_image)
    plt.title("First Frame with Mask Contours and Sampled Points")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"First frame visualization saved to {output_path}")

def get_color_map(n):
    def hsv2rgb(h, s, v):
        return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))
    
    return [hsv2rgb(i / n, 1, 1) for i in range(n)]

def rle_to_binary_mask(rle):
    if isinstance(rle, dict):
        return mask_util.decode(rle)
    elif isinstance(rle, list):
        # If RLE is in COCO format (list of counts)
        h, w = rle['size'] if 'size' in rle else (0, 0)
        rle_dict = {'counts': rle, 'size': [h, w]}
        return mask_util.decode(rle_dict)
    else:
        raise ValueError("Unsupported RLE format")

def mask_to_rle(binary_mask):
    rle = mask_util.encode(np.asfortranarray(binary_mask))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def mask_to_bbox(mask):
    pos = np.where(mask)
    if len(pos[0]) == 0:
        return None
    xmin, ymin = np.min(pos[1]), np.min(pos[0])
    xmax, ymax = np.max(pos[1]), np.max(pos[0])
    return [float(xmin), float(ymin), float(xmax - xmin + 1), float(ymax - ymin + 1)]