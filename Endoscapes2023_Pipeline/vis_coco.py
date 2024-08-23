import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import json
import pickle
import cv2
import colorsys
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

MARKER_SIZE = 200
MARKER_LIST = [
    '*', 'o', 's', 'p', 'h', '8', '^', 'v', '<', '>', 'D', 'd', 'P', 'X', '.', ',', '_', '|', 'd', 'P', 'X', '.', ',',
    '_'
]


def create_class_to_color_mapper(num_classes=51):
    cmap_20b = plt.get_cmap('tab20b')
    cmap_20c = plt.get_cmap('tab20c')
    colors = []
    colors.extend([cmap_20b(i) for i in range(cmap_20b.N)])
    colors.extend([cmap_20c(i) for i in range(cmap_20c.N)])

    while len(colors) < num_classes:
        colors.extend(colors[:num_classes - len(colors)])

    np.random.shuffle(colors)

    class_to_rgb_tuple = {}
    class_to_hex_color = {}

    for i in range(num_classes):
        rgb = colors[i][:3]
        rgb_tuple = tuple(int(x * 255) for x in rgb)
        class_to_rgb_tuple[i] = rgb_tuple
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb_tuple)
        class_to_hex_color[i] = hex_color

    return class_to_hex_color, class_to_rgb_tuple


COLOR_MAPPER, class_to_color_mapper = create_class_to_color_mapper(num_classes=51)


def show_mask_keep_color(mask, ax, color_id, class_to_color_mapper, transparent=1.0):
    color = np.array([class_to_color_mapper[color_id][0] / 255.0,
                      class_to_color_mapper[color_id][1] / 255.0,
                      class_to_color_mapper[color_id][2] / 255.0,
                      transparent])
    print(f'color_id {color_id}, color {color}')
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return mask_image


def rle_to_binary_mask(rle):
    if isinstance(rle, dict):
        return mask_utils.decode(rle)
    elif isinstance(rle, list):
        h, w = rle.get('size', (0, 0))
        if h == 0 or w == 0:
            raise ValueError("Invalid mask size in RLE data")
        rle_dict = {'counts': rle, 'size': [h, w]}
        return mask_utils.decode(rle_dict)
    else:
        raise ValueError("Unsupported RLE format")


def visualize_frame(prompt_frame, current_frame, ground_truth, prediction, output_path, prompt_objs,
                    show_first_frame=True):
    if show_first_frame:
        fig, axes = plt.subplots(1, 4, figsize=(50, 10))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(40, 10))

    axes = axes.flatten()

    if show_first_frame:
        axes[0].imshow(prompt_frame)
        for obj in prompt_objs:
            obj_id = obj['obj_id']
            marker_color = COLOR_MAPPER[obj_id]
            for point, label in zip(obj['points'], obj['pos_or_neg_label']):
                marker = 'o' if label == 1 else 'x'
                linewidth = 1.25 if label == 1 else 5
                axes[0].scatter(point[0], point[1], c=marker_color, s=250, marker=marker, alpha=1.0, edgecolor='white',
                                linewidth=linewidth)
        axes[0].set_title("Prompt Frame")
        axes[0].axis('off')

    idx_offset = 0 if show_first_frame else -1

    axes[1 + idx_offset].imshow(current_frame)
    axes[1 + idx_offset].set_title("Current Frame")
    axes[1 + idx_offset].axis('off')

    axes[2 + idx_offset].imshow(current_frame)
    unique_classes = np.unique(ground_truth)
    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue
        mask = (ground_truth == class_id)
        print(f"gt {class_id}")
        show_mask_keep_color(mask, axes[2 + idx_offset], class_id, class_to_color_mapper, transparent=0.7)
    axes[2 + idx_offset].set_title("Ground Truth Segmentation")
    axes[2 + idx_offset].axis('off')

    axes[3 + idx_offset].imshow(current_frame)
    unique_classes = np.unique(prediction)
    for class_id in unique_classes:
        if class_id == 0:  # Skip background
            continue
        mask = (prediction == class_id)
        print(f"prediction {class_id}")
        show_mask_keep_color(mask, axes[3 + idx_offset], class_id, class_to_color_mapper, transparent=0.7)
    axes[3 + idx_offset].set_title("Predicted Segmentation")
    axes[3 + idx_offset].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()


def visualize_all_frames(pred_pkl_file, pred_json_file, gt_json_file, output_dir, gt_type):
    vis_dir = os.path.join(output_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)

    with open(pred_pkl_file, 'rb') as f:
        pkl_data = pickle.load(f)
    prompt_objs = pkl_data[0]['prompt_objs']

    with open(pred_json_file, 'r') as f:
        pred_data = json.load(f)
    # coco = COCO(pred_json_file)
    coco_gt = COCO(gt_json_file)
    coco_dt = coco_gt.loadRes(pred_json_file)
    image_size = pred_data[0]['segmentation']['size']

    # todo customize
    def frame_number_to_target(frame_number):
        return f"{frame_number[0]}_{frame_number[2:]}"

    prompt_frame = None
    for i, frame_data in enumerate(tqdm(pred_data, desc="visualize frames")):
        frame_idx = frame_data['image_id']

        ann_gt_ids = coco_gt.getAnnIds(imgIds=frame_idx)
        anns_gt = coco_gt.loadAnns(ann_gt_ids)
        ann_dt_ids = coco_dt.getAnnIds(imgIds=frame_idx)
        anns_dt = coco_dt.loadAnns(ann_dt_ids)

        if not anns_gt:
            print(f"Warning: No ground truth data found for frame {frame_idx}")
            continue

        source_frame_name = f"{frame_idx}"
        target_frame_name = frame_number_to_target(source_frame_name)
        frame_path = os.path.join(output_dir, f"{target_frame_name}.jpg")

        if not os.path.exists(frame_path):
            print(f"Warning: Frame not found: {frame_path}")
            continue

        current_frame = np.array(Image.open(frame_path))
        if i == 0:
            prompt_frame = current_frame

        prediction = np.zeros(image_size, dtype=np.uint8)
        for ann in anns_dt:
            pred_mask = coco_dt.annToMask(ann)
            category_id = ann['category_id']
            prediction[pred_mask > 0] = category_id
        # mask = rle_to_binary_mask(frame_data['segmentation'])
        # category_id = frame_data['category_id']
        # prediction[mask > 0] = category_id

        ground_truth = np.zeros(image_size, dtype=np.uint8)
        for ann in anns_gt:
            gt_mask = coco_gt.annToMask(ann)
            gt_category_id = ann['category_id']
            ground_truth[gt_mask > 0] = gt_category_id

        output_path = os.path.join(vis_dir, f'frame_{frame_idx:04d}.png')

        visualize_frame(
            prompt_frame,
            current_frame,
            ground_truth,
            prediction,
            output_path,
            prompt_objs,
        )

    print(f"All frame visualizations saved to {vis_dir}")


random.seed(999)

# gt_type : mask_point, bbox_point, mask, bbox


# Example usage:
visualize_all_frames('/data/proj/SurgicalSAM2/Endoscapes2023_Pipeline/test/output/points/prompt.pkl',
                     '/data/proj/SurgicalSAM2/Endoscapes2023_Pipeline/test/output/points/predict.json',
                     '/data/proj/SurgicalSAM2/Endoscapes2023_Pipeline/sample_annotations.json',
                     '/data/proj/SurgicalSAM2/Endoscapes2023_Pipeline/mini_sample', gt_type='bbox_point')