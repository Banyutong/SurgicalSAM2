import numpy as np

import numpy as np
import torch
from scipy.ndimage import label, find_objects
import numpy as np
from skimage.measure import label
# from typing import Any, Dict, Generator, ItemsView, List, Tuple
from sam2.utils.amg import  remove_small_regions
def convert_coco_to_points(bbox):
	"""
	Convert COCO format bounding box to [x_min, y_min, x_max, y_max] format.

	Args:
	bbox (list): Bounding box in COCO format [x, y, width, height]

	Returns:
	list: Bounding box in [x_min, y_min, x_max, y_max] format
	"""
	x, y, width, height = bbox
	return [x, y, x + width, y + height]


def sample_points_from_bboxes(bboxes, num_points=3, include_center=True):
	"""
	Sample a fixed number of points from each bounding box.

	Args:
	bboxes (list of list): List of bounding boxes in COCO format [x, y, width, height]
	num_points (int): Number of points to sample for each bounding box
	include_center (bool): Whether to always include the center point as one of the points

	Returns:
	list of list: List of sampled points for each bounding box
	"""
	sampled_points = []
	for bbox in bboxes:
		x_min, y_min, x_max, y_max = convert_coco_to_points(bbox)
		points = []

		if include_center:
			center_x = (x_min + x_max) / 2
			center_y = (y_min + y_max) / 2
			points.append((center_x, center_y))

		remaining_points = num_points - len(points)
		for _ in range(remaining_points):
			x = np.random.uniform(x_min, x_max)
			y = np.random.uniform(y_min, y_max)
			points.append((x, y))

		sampled_points.append(points)

	return sampled_points


def sample_points_from_masks(masks, num_points=3, include_center=True):
	"""
	Sample a fixed number of points from each binary mask.

	Args:
	masks (list of np.array): List of binary masks, where each mask is a 2D numpy array
	num_points (int): Number of points to sample for each mask
	include_center (bool): Whether to always include the center point as one of the points

	Returns:
	list of list: List of sampled points for each mask
	"""
	sampled_points = []
	for mask in masks:
		y_indices, x_indices = np.nonzero(mask)
		points = []

		if len(x_indices) > 0 and len(y_indices) > 0:
			if include_center:
				center_x = np.mean(x_indices)
				center_y = np.mean(y_indices)
				points.append((center_x, center_y))

			remaining_points = num_points - len(points)
			if remaining_points > 0:
				# Sample random indices
				sample_indices = np.random.choice(len(x_indices), remaining_points, replace=True)
				for idx in sample_indices:
					points.append((x_indices[idx], y_indices[idx]))

		# If we couldn't sample enough points (e.g., not enough non-zero pixels),
		# fill the remaining with None or the center point
		while len(points) < num_points:
			points.append(None if not include_center else points[0])

		sampled_points.append(points)

	return sampled_points


import numpy as np
from skimage import measure

import numpy as np
from skimage import measure

import numpy as np
from skimage import measure


def get_connected_components(mask):
	N, _, H, W = mask.shape
	labels = np.zeros_like(mask)
	counts = np.zeros_like(mask)
	for i in range(N):
		label_image = measure.label(mask[i, 0], connectivity=2)  # 8-connectivity
		region_properties = measure.regionprops(label_image)
		for region in region_properties:
			counts[i, 0][label_image == region.label] = region.area
		labels[i, 0] = label_image
	return labels, counts


# def sample_points_from_pixel_mask(mask, num_points, include_center=True):
#     original_mask = mask.copy()  # Keep a copy of the original mask
#     if mask.ndim == 2:
#         mask = mask[np.newaxis, np.newaxis, ...]  # (H,W) -> (1,1,H,W)
#     elif mask.ndim == 3:
#         mask = np.transpose(mask, (2, 0, 1))[np.newaxis, ...]  # (H,W,C) -> (1,C,H,W)
#     labels, counts = get_connected_components(mask)
#     unique_labels = np.unique(labels)[1:]  # Exclude background (0)
#     points = []
#     class_labels = []
#     class_to_color_mapper = {}
#     original_value_to_class_label = {}
#     next_class_label = 0
#     for component_label in unique_labels:
#         component_mask = (labels[0, 0] == component_label)
#         indices = np.argwhere(component_mask)
#         if include_center:
#             center = indices.mean(axis=0).round().astype(int)
#             sampled_points = [[(int(center[1]), int(center[0]))]]  # Ensure (x, y) order
#             num_additional = min(num_points - 1, indices.shape[0] - 1)
#         else:
#             sampled_points = []
#             num_additional = min(num_points, indices.shape[0])
#         if num_additional > 0:
#             additional_indices = np.random.choice(indices.shape[0], num_additional, replace=False)
#             additional_points = [[(int(idx[1]), int(idx[0]))] for idx in indices[additional_indices]]  # Ensure (x, y) order and wrap in list
#             sampled_points.extend(additional_points)
#         points.extend(sampled_points)
#         # Get the original mask value for this component
#         x, y = sampled_points[0][0]
#         original_value = tuple(original_mask[y, x]) if original_mask.ndim == 3 else int(original_mask[y, x])
#         if original_value not in original_value_to_class_label:
#             original_value_to_class_label[original_value] = next_class_label
#             next_class_label += 1
#         class_label = original_value_to_class_label[original_value]
#         class_labels.extend([class_label] * len(sampled_points))
#         class_to_color_mapper[class_label] = original_value
#     return points, class_labels, class_to_color_mapper

import numpy as np
from skimage import measure


def get_connected_components(mask):
	N, _, H, W = mask.shape
	labels = np.zeros_like(mask)
	counts = np.zeros_like(mask)
	for i in range(N):
		label_image = measure.label(mask[i, 0], connectivity=2)  # 8-connectivity
		region_properties = measure.regionprops(label_image)
		for region in region_properties:
			counts[i, 0][label_image == region.label] = region.area
		labels[i, 0] = label_image
	return labels, counts


def remove_small_regions(mask, area_thresh=30, mode="islands"):
	labels, counts = get_connected_components(mask)
	small_regions = counts < area_thresh

	if mode == "islands":
		mask[small_regions] = 0
	else:  # mode == "holes"
		mask[small_regions] = 1

	return mask


import numpy as np
from skimage import measure

import numpy as np
from skimage import measure

def get_connected_components(mask):
    N, _, H, W = mask.shape
    labels = np.zeros_like(mask)
    counts = np.zeros_like(mask)
    for i in range(N):
        label_image = measure.label(mask[i, 0], connectivity=2)  # 8-connectivity
        region_properties = measure.regionprops(label_image)
        for region in region_properties:
            counts[i, 0][label_image == region.label] = region.area
        labels[i, 0] = label_image
    return labels, counts

def remove_small_regions(mask, area_thresh=20, mode="islands"):
    labels, counts = get_connected_components(mask)
    small_regions = counts < area_thresh
    if mode == "islands":
        mask[small_regions] = 0
    else:  # mode == "holes"
        mask[small_regions] = 1
    return mask




def sample_points_from_pixel_mask(mask, num_points, include_center=True, area_thresh=10, remove_small=True):
    original_mask = mask.copy()  # Keep a copy of the original mask
    if mask.ndim == 2:
        mask = mask[np.newaxis, np.newaxis, ...]  # (H,W) -> (1,1,H,W)
    elif mask.ndim == 3:
        mask = np.transpose(mask, (2, 0, 1))[np.newaxis, ...]  # (H,W,C) -> (1,C,H,W)

    if remove_small:
        # Remove small regions
        mask = remove_small_regions(mask, area_thresh=area_thresh, mode="hole")

    # Get connected components
    labels, _ = get_connected_components(mask)
    unique_labels = np.unique(labels)#[1:]  # Exclude background (0)

    points = []
    class_labels = []
    class_to_color_mapper = {}
    original_value_to_class_label = {}
    next_class_label = 0

    H, W = original_mask.shape[:2]  # Get the height and width of the original mask

    for component_label in unique_labels:
        component_mask = (labels[0, 0] == component_label)
        indices = np.argwhere(component_mask)

        if len(indices) == 0:
            continue  # Skip empty components

        # Ensure the sampled point is within the bounds of the original mask
        for _ in range(len(indices)):
            idx = indices[np.random.randint(len(indices))]
            y, x = idx
            if 0 <= y < H and 0 <= x < W:
                break
        else:
            continue  # Skip this component if no valid point is found

        # Get the original mask value for this component
        if original_mask.ndim == 2:  # Grayscale
            original_value = int(original_mask[y, x])
        else:  # Color
            original_value = tuple(original_mask[y, x])

        # Assign or get the class label for this original value
        if original_value not in original_value_to_class_label:
            original_value_to_class_label[original_value] = next_class_label
            next_class_label += 1
        class_label = original_value_to_class_label[original_value]

        # Store the mapping of class label to original color
        class_to_color_mapper[class_label] = original_value

        # Sample points for this component
        sampled_points = []
        if include_center:
            center = indices.mean(axis=0).round().astype(int)
            if 0 <= center[0] < H and 0 <= center[1] < W:
                sampled_points.append([(int(center[1]), int(center[0]))])
            num_additional = min(num_points - 1, indices.shape[0] - 1)
        else:
            num_additional = min(num_points, indices.shape[0])

        if num_additional > 0:
            additional_indices = np.random.choice(indices.shape[0], num_additional, replace=False)
            for idx in indices[additional_indices]:
                if 0 <= idx[0] < H and 0 <= idx[1] < W:
                    sampled_points.append([(int(idx[1]), int(idx[0]))])

        points.extend(sampled_points)
        class_labels.extend([class_label] * len(sampled_points))

    return points, class_labels, class_to_color_mapper


def ori_sample_points_from_pixel_mask(mask, num_points, include_center=True):
    original_mask = mask.copy()  # Keep a copy of the original mask
    if mask.ndim == 2:
        mask = mask[np.newaxis, np.newaxis, ...]  # (H,W) -> (1,1,H,W)
    elif mask.ndim == 3:
        mask = np.transpose(mask, (2, 0, 1))[np.newaxis, ...]  # (H,W,C) -> (1,C,H,W)
    labels, _ = get_connected_components(mask)
    unique_labels = np.unique(labels)[1:]  # Exclude background (0)
    points = []
    class_labels = []
    class_to_color_mapper = {}
    original_value_to_class_label = {}
    next_class_label = 0
    for component_label in unique_labels:
        component_mask = (labels[0, 0] == component_label)
        indices = np.argwhere(component_mask)
        if include_center:
            center = indices.mean(axis=0).round().astype(int)
            sampled_points = [[(int(center[1]), int(center[0]))]]  # Ensure (x, y) order
            num_additional = min(num_points - 1, indices.shape[0] - 1)
        else:
            sampled_points = []
            num_additional = min(num_points, indices.shape[0])
        if num_additional > 0:
            additional_indices = np.random.choice(indices.shape[0], num_additional, replace=False)
            additional_points = [[(int(idx[1]), int(idx[0]))] for idx in indices[additional_indices]]  # Ensure (x, y) order and wrap in list
            sampled_points.extend(additional_points)
        points.extend(sampled_points)
        # Get the original mask value for this component
        x, y = sampled_points[0][0]  # Note the change here to access the inner tuple
        if original_mask.ndim == 2:  # Grayscale
            original_value = int(original_mask[y, x])
        else:  # Color
            original_value = tuple(original_mask[y, x])
        # Assign or get the class label for this original value
        if original_value not in original_value_to_class_label:
            original_value_to_class_label[original_value] = next_class_label
            next_class_label += 1
        class_label = original_value_to_class_label[original_value]
        class_labels.extend([class_label] * len(sampled_points))
        # Store the mapping of class label to original color
        class_to_color_mapper[class_label] = original_value
    return points, class_labels, class_to_color_mapper


if __name__ == "__main__":
	# Extract bounding boxes from the given data.
	bboxes = [
		[280, 90, 550, 300],  # Object 1: [x_min, y_min, x_max, y_max]
		[360, 110, 410, 290]  # Object 2: [x_min, y_min, x_max, y_max]
	]

	# Sample points from bounding boxes
	bbox_points = sample_points_from_bboxes(bboxes, num_points=3, include_center=True)
	print("Sampled points from bounding boxes:")
	for i, points in enumerate(bbox_points):
		print(f"Object {i + 1}: {points}")