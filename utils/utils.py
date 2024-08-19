import re
import os
import numpy as np
from PIL import Image


def extract_frame_number(filename):
    # Find all sequences of digits in the filename
    matches = re.findall(r'\d+', filename)

    if matches:
        # Combine all sequences of digits into a single string
        combined_digits = ''.join(matches)
        # Convert the combined string to an integer
        return int(combined_digits)

    return 0  # Default value if no number is found


def find_frames(frame_dir):
    valid_extensions = [".jpg", ".jpeg", ".png"]
    frame_names = [
        p for p in os.listdir(frame_dir)
        if os.path.splitext(p)[-1].lower() in valid_extensions
           and "mask" not in p.lower()  # An special case to exclude files with "mask" in the name (case-insensitive)
    ]

    # frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    frame_names.sort(key=extract_frame_number)

    # print(f"------------{frame_names[0]}")

    return frame_names

def extract_frame_number(filename):
    # Find all sequences of digits in the filename
    matches = re.findall(r'\d+', filename)

    if matches:
        # Combine all sequences of digits into a single string
        combined_digits = ''.join(matches)
        # Convert the combined string to an integer
        return int(combined_digits)

    return 0  # Default value if no number is found


def get_class_to_color_mapping(gt_mask):
    # Flatten the mask if it's multi-dimensional
    if gt_mask.ndim > 2:
        gt_mask = gt_mask.reshape(-1, gt_mask.shape[-1])
    else:
        gt_mask = gt_mask.flatten()

    # Get unique values (colors) in the mask
    unique_values = np.unique(gt_mask, axis=0)

    # Create the class-to-color mapping
    class_to_color_mapper = {}
    color_to_class_mapper = {}
    # for class_label, color in enumerate(unique_values, start=1):
    #     if gt_mask.ndim > 1:
    #         color = tuple(color)  # Convert to tuple for multi-channel masks
    #     else:
    #         color = int(color)  # Keep as int for single-channel masks
    #     class_to_color_mapper[class_label] = color
    #     color_to_class_mapper[color] = class_label
    for class_label, color in enumerate(unique_values, start=1):
        if gt_mask.ndim > 1:
            color = tuple(color)  # Convert to tuple for multi-channel masks
        else:
            color = int(color)  # Keep as int for single-channel masks

        if np.all(np.array(color) == 0) or np.all(np.array(color) == 255):  # Black or white will be skipped
            continue
        else:  # Other colors

            class_to_color_mapper[class_label] = color
            color_to_class_mapper[color] = class_label
    return class_to_color_mapper, color_to_class_mapper


def process_gt_pixel_mask(frame_names, gt_path):
    gt_data = []

    # Extract mask_dir from gt_path
    mask_dir = os.path.dirname(gt_path)

    # Extract the basename and find all digit sequences in it
    gt_basename = os.path.basename(gt_path)
    digit_sequences = re.findall(r'\d+', gt_basename)

    for frame_name in frame_names:
        # Extract digits from frame_name
        frame_digits = re.findall(r'\d+', frame_name)

        # Construct mask_name by replacing digit sequences
        mask_name = gt_basename
        for old_seq, new_seq in zip(digit_sequences, frame_digits):
            mask_name = mask_name.replace(old_seq, new_seq)

        # Get the full mask path
        mask_path = os.path.join(mask_dir, mask_name)

        # Load the mask as numpy array
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            gt_data.append(mask)
        else:
            print(f"Warning: No corresponding mask found for {frame_name}")

    return gt_data