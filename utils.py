import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import colorsys
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

def rle_to_binary_mask(rle):
    """
    Convert RLE to binary mask.
    """
    binary_array = np.zeros(rle['size'], dtype=np.uint8)
    starts, lengths = rle['counts'][::2], rle['counts'][1::2]
    current_position = 0
    for start, length in zip(starts, lengths):
        current_position += start
        binary_array[current_position:current_position + length] = 1
        current_position += length
    return binary_array
