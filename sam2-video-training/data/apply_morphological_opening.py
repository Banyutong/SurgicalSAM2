#!/usr/bin/env python3
"""
Script to apply 10x10 morphological closing to COCO annotation segmentation masks.
"""

import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
import os
from tqdm import tqdm


def rle_to_mask(rle, height, width):
    """Convert RLE to binary mask."""
    if isinstance(rle, dict):
        # RLE is already in the right format
        mask = mask_utils.decode(rle)
    else:
        # Handle string RLE format if needed
        mask = mask_utils.decode({"size": [height, width], "counts": rle})
    return mask


def mask_to_rle(mask):
    """Convert binary mask to RLE format."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    # Convert bytes to string for JSON serialization
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def apply_morphological_closing_to_annotations(coco_data, kernel_size=5):
    """Apply 10x10 morphological closing to all annotations in COCO data."""
    print(f"Processing {len(coco_data['annotations'])} annotations...")

    # Create morphological kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Keep track of annotations to remove
    annotations_to_remove = []

    for i, ann in enumerate(tqdm(coco_data["annotations"])):
        if "segmentation" not in ann or ann["segmentation"] is None:
            continue

        seg = ann["segmentation"]

        # Get image dimensions
        if "size" in seg:
            height, width = seg["size"]
        else:
            # Use default dimensions if not specified
            height, width = 1080, 1920

        # Convert RLE to binary mask
        mask = rle_to_mask(seg, height, width)

        # Apply morphological closing
        opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Calculate new area
        new_area = int(opened_mask.sum())

        # Skip if area becomes zero after closing
        if new_area == 0:
            annotations_to_remove.append(i)
            continue

        # Convert back to RLE
        new_rle = mask_to_rle(opened_mask)

        # Update annotation
        if isinstance(new_rle, dict):
            ann["segmentation"] = {"size": new_rle["size"], "counts": new_rle["counts"]}
        else:
            ann["segmentation"] = new_rle

        # Update area
        ann["area"] = new_area

    # Remove annotations with zero area (in reverse order to maintain indices)
    print(
        f"Removing {len(annotations_to_remove)} annotations with zero area after closing..."
    )
    for i in reversed(annotations_to_remove):
        del coco_data["annotations"][i]

    return coco_data


def process_coco_file(input_path, output_path, kernel_size=10):
    """Process a single COCO annotation file."""
    print(f"Loading {input_path}...")

    # Load COCO data
    with open(input_path, "r") as f:
        coco_data = json.load(f)

    original_count = len(coco_data["annotations"])
    print(f"Original data has {original_count} annotations")

    # Apply morphological closing
    coco_data = apply_morphological_closing_to_annotations(coco_data, kernel_size)

    final_count = len(coco_data["annotations"])
    print(f"Final data has {final_count} annotations")
    print(f"Removed {original_count - final_count} annotations with zero area")

    print(f"Saving processed data to {output_path}...")

    # Save processed data
    with open(output_path, "w") as f:
        json.dump(coco_data, f, indent=2)

    print(f"Successfully processed and saved {final_count} annotations")


def main():
    """Main function to process both train and validation annotation files."""
    # File paths
    data_dir = "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2-video-training/data"

    val_input = os.path.join(data_dir, "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2-video-training/data/endovis18_coco_annotations_val.json")
    val_output = os.path.join(data_dir, "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2-video-training/data/endovis18_coco_annotations_val_opened.json")

    train_input = os.path.join(data_dir, "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2-video-training/data/endovis18_coco_annotations_train.json")
    train_output = os.path.join(
        data_dir, "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2-video-training/data/endovis18_coco_annotations_train_openend.json"
    )

    kernel_size = 10

    print("=" * 50)
    print("Processing validation annotations...")
    print("=" * 50)
    process_coco_file(val_input, val_output, kernel_size)

    print("\n" + "=" * 50)
    print("Processing training annotations...")
    print("=" * 50)
    process_coco_file(train_input, train_output, kernel_size)

    print("\n" + "=" * 50)
    print("All files processed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
