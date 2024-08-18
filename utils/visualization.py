import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import colorsys
from itertools import groupby

from scipy.constants import point
from skimage import measure
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import random
import re
from tqdm import tqdm


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
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def get_color_map(num_classes):
    """Generate a color map for visualizing different objects."""
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(rgb)  # Keep as float values in range 0-1
    return colors


def ensure_color_range(color):
    """Ensure color values are within 0-1 range."""
    return tuple(max(0, min(c, 1)) for c in color)


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


def get_color_map_255(num_classes):
    """Generate a color map for visualizing different objects, with values in 0-255 range."""
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def show_mask_keep_color(mask, ax, color_id, class_to_color_mapper, transparent=1.0):

    color = np.array([class_to_color_mapper[color_id][0] / 255.0, class_to_color_mapper[color_id][1] / 255.0, class_to_color_mapper[color_id][2] / 255.0,
                      transparent])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image


def visualize_first_frame_comprehensive(image, first_valid_gt, sampled_points, prediction, output_path, gt_type,
                                        class_to_color_mapper,point_class_labels):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
    # 1. Original Image
    ax1.imshow(image)
    ax1.set_title("Original Image")
    ax1.axis('off')

    # 2. Image with ground truth and point prompts
    ax2.imshow(image)
    if gt_type == 'pixel_mask':
        ax2.imshow(first_valid_gt, alpha=0.7)
        if sampled_points is not None:
            for i, points in enumerate(sampled_points):
                class_id = point_class_labels[i]
                color = class_to_color_mapper.get(class_id)  # Default to white if color not found
                color = [c / 255 for c in color]  # Normalize to 0-1 range
                points = np.array(points)
                ax2.scatter(points[:, 0], points[:, 1], c=[color], s=200, marker='*', edgecolor='white', linewidth=1.25)
    else:
        ax2.imshow(image)
        colors = get_color_map_255(len(first_valid_gt))
        for i, (gt, points) in enumerate(zip(first_valid_gt, sampled_points)):
            color = tuple(c / 255 for c in colors[i])  # Normalize color to 0-1 range for matplotlib

            if gt_type == 'bbox':
                # Draw bounding box
                x, y, w, h = gt
                rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
                ax2.add_patch(rect)
            elif gt_type == 'mask':
                # Draw filled mask contour
                contours, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    ax2.add_patch(plt.Polygon(contour.reshape(-1, 2), fill=True, alpha=0.4, color=color))

            # Plot sampled points
            points = np.array(points)
            ax2.scatter(points[:, 0], points[:, 1], c=[color], s=100, marker='*')

    ax2.set_title(f"Ground Truth ({gt_type.capitalize()}) and Point Prompts")
    ax2.axis('off')

    # 3. Image with predictions
    if gt_type == 'pixel_mask':
        ax3.imshow(image)
        for out_obj_id, out_mask in prediction.items():
            show_mask_keep_color(out_mask, ax3, out_obj_id, class_to_color_mapper, transparent=0.7)
    else:
        mask_image = np.zeros_like(image[:, :, 0])
        for out_obj_id, out_mask in prediction.items():
            mask_image[out_mask] = out_obj_id
        ax3.imshow(mask_image, cmap='tab20')
    ax3.set_title("Predicted Segmentation")
    ax3.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Comprehensive first frame visualization saved to {output_path}")


def visualize_all_frames(video_segments, frame_names, video_dir, output_dir, gt_data,
                         prompt_frame_index=None, prompt_points=None, gt_type="pixel_mask",  class_to_color_mapper=None,
                         show_first_frame=True, show_points=True, point_class_labels=None):
    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)

    # Load the prompt frame once
    if show_first_frame:
        prompt_frame = np.array(Image.open(os.path.join(video_dir, frame_names[prompt_frame_index])))
    else:
        prompt_frame = None

    for frame_idx, frame_name in tqdm(enumerate(frame_names), desc="visualize frames"):
        current_frame = np.array(Image.open(os.path.join(video_dir, frame_name)))
        # prediction = np.zeros_like(current_frame[:, :, 0])
        # for obj_id, mask in video_segments[frame_idx].items():
        #     prediction[mask] = obj_id
        prediction = video_segments[frame_idx]
        if gt_type == 'pixel_mask' or gt_type == 'mask':
            current_gt = gt_data[frame_idx] if frame_idx < len(gt_data) else np.zeros_like(current_frame[:, :, 0])
        else:  # bbox
            current_gt = gt_data[frame_idx] if frame_idx < len(gt_data) else []


        visualize_frame(
            current_frame,
            frame_name,
            current_gt,
            prediction,
            output_dir,
            prompt_frame,
            prompt_points,
            gt_type,
            class_to_color_mapper,
            show_first_frame,
            show_points,
            point_class_labels,


        )
    print(f"All frame visualizations saved to {vis_dir}")


def visualize_frame(current_frame, frame_name, ground_truth, prediction, output_path, prompt_frame=None, prompt_points=None,
                    gt_type='pixel_mask', class_to_color_mapper= None, show_first_frame=True, show_points=True, point_class_labels=None):
    if show_first_frame:
        fig, axes = plt.subplots(1, 4, figsize=(40, 10))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    axes = axes.flatten()  # Flatten the axes array for easier indexing

    if show_first_frame:
        # Prompt frame with plotted points
        axes[0].imshow(prompt_frame)

        axes[0].set_title("Prompt Frame")
        if show_points:
            for i, points in enumerate(prompt_points):
                class_id = point_class_labels[i]
                color = class_to_color_mapper.get(class_id)  # Default to white if color not found
                color = [c / 255 for c in color]  # Normalize to 0-1 range
                points = np.array(points)
                axes[0].scatter(points[:, 0], points[:, 1], c=[color], s=200, marker='*', edgecolor='white', linewidth=1.25)

        axes[0].axis('off')

    # Adjust index based on whether we're showing the first frame
    idx_offset = 0 if show_first_frame else -1

    # Current frame
    axes[1 + idx_offset].imshow(current_frame)
    axes[1 + idx_offset].set_title("Current Frame")
    axes[1 + idx_offset].axis('off')

    # Ground truth
    if ground_truth is None or len(ground_truth) == 0:
        axes[2 + idx_offset].imshow(current_frame)
        axes[2 + idx_offset].text(0.5, 0.5, 'No GT', ha='center', va='center', transform=axes[2 + idx_offset].transAxes,
                                  fontsize=20, color='white',
                                  bbox=dict(facecolor='black', alpha=0.5))
    elif gt_type == 'mask':
        axes[2 + idx_offset].imshow(current_frame)
        for mask in ground_truth:
            axes[2 + idx_offset].imshow(mask, cmap='tab20', alpha=0.7)
    elif gt_type == 'pixel_mask':
        axes[2 + idx_offset].imshow(current_frame)
        axes[2 + idx_offset].imshow(ground_truth, alpha=0.7)
    elif gt_type == 'bbox':
        axes[2 + idx_offset].imshow(current_frame)
        for bbox in ground_truth:
            rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='r', linewidth=2)
            axes[2 + idx_offset].add_patch(rect)
    axes[2 + idx_offset].set_title(f"Ground Truth ({gt_type.capitalize()})")
    axes[2 + idx_offset].axis('off')

    # Prediction
    if gt_type == 'pixel_mask':
        axes[3 + idx_offset].imshow(current_frame)
        for out_obj_id, out_mask in prediction.items():
            mask_image = show_mask_keep_color(out_mask, axes[3 + idx_offset], out_obj_id, class_to_color_mapper, transparent=0.7)
    else:
        mask_image = np.zeros_like(current_frame[:, :, 0])
        for out_obj_id, out_mask in prediction.items():
            mask_image[out_mask] = out_obj_id
        axes[3 + idx_offset].imshow(mask_image, cmap='tab20')

    axes[3 + idx_offset].set_title("Predicted Segmentation")
    axes[3 + idx_offset].axis('off')
    # os.makedirs(os.path.join(output_path, 'merged_pixel_masks'), exist_ok=True)
    # extent = axes[3 + idx_offset].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig(os.path.join(output_path, 'merged_pixel_masks', f"frame_{frame_name}"), bbox_inches=extent, pad_inches=0)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def get_color_map(num_classes):
    """Generate a color map for visualizing different objects."""
    colors = []
    for i in range(num_classes):
        hue = i / num_classes
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append(rgb)  # Keep as float values in range 0-1
    return colors