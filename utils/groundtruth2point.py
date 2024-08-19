import numpy as np

import numpy as np


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
from scipy.ndimage import label, center_of_mass


def generate_color_list(gt_mask):
    """
    Generate a list of unique colors from a ground truth mask.

    Args:
    gt_mask (np.ndarray): A 3D numpy array representing the ground truth mask.

    Returns:
    list: List of unique colors in the mask, each color as an RGB tuple.
    """
    # Reshape the mask to a 2D array of RGB tuples
    reshaped_mask = gt_mask.reshape(-1, gt_mask.shape[-1])

    # Find unique colors, excluding the background (assumed to be [0, 0, 0])
    unique_colors = np.unique(reshaped_mask, axis=0)
    color_list = [tuple(color) for color in unique_colors if np.any(color != 0)]

    return color_list


def sample_points_from_pixel_mask(gt_mask, num_points=2, include_center=True, use_top_bottom=False,
                                  area_threshold=900, exclude_extreme_colors={"white":True, "black":True,"gray":True}):
    """
    Sample points from a color image based on a ground truth mask, returning points in (x, y) format.
    Args:
    gt_mask (np.ndarray): A 3D numpy array representing the ground truth mask.
    num_points (int): Number of points to sample for each object.
    include_center (bool): Whether to include the center point of each object.
    use_top_bottom (bool): Whether to use top and bottom points instead of random selection.
    area_threshold (int): Minimum area for a region to be considered.
    Returns:
    tuple: (sampled_points, classes_points, class_to_color_mapper, color_to_class_mapper)
        sampled_points (list): List of all sampled points [x, y] across all classes.
        classes_points (list): List containing [color_id, [x, y]] for each point.
        class_to_color_mapper (dict): A dictionary mapping class IDs to colors.
        color_to_class_mapper (dict): A dictionary mapping colors to class IDs.
    """
    color_list = generate_color_list(gt_mask)
    if (255, 255, 255) in color_list and exclude_extreme_colors["white"]:  # remove white
        print("there are white pixels but not considered for segmentation. they are usually background if you want to consider, please set exclude_extreme_colors ")
        color_list.remove((255, 255, 255))
    if (0, 0, 0) in color_list and exclude_extreme_colors["black"]:  # remove black
        color_list.remove((0, 0, 0))
        print("there are black pixels but not considered for segmentation. if you want to consider, please set exclude_extreme_colors ")
    if (127, 127, 127) in color_list and exclude_extreme_colors["gray"]:  # remove black
        color_list.remove((127, 127, 127))
        print("there are gray pixels but not considered for segmentation. if you want to consider, please set exclude_extreme_colors ")
    # Generate class_to_color_mapper and color_to_class_mapper
    class_to_color_mapper = {i: color for i, color in enumerate(color_list)}
    color_to_class_mapper = {color: i for i, color in enumerate(color_list)}

    sampled_points = []
    classes_points = []

    for color_id, color in enumerate(color_list):
        # Generate a binary mask for the current color
        mask = np.all(gt_mask == color, axis=-1)
        # Label the masked regions
        labeled_mask, num_features = label(mask)
        if num_features > 0:
            # Process each labeled region
            for i in range(1, num_features + 1):
                object_mask = (labeled_mask == i)
                area = np.sum(object_mask)
                if area < area_threshold:
                    continue
                points = []
                # Include center if requested
                if include_center:
                    center = center_of_mass(object_mask)
                    point = [int(center[1]), int(center[0])]
                    points.append(point)
                    classes_points.append(color_id)
                if use_top_bottom:
                    # Find top and bottom points
                    y_indices, x_indices = np.where(object_mask)
                    top_point = [x_indices[np.argmin(y_indices)], np.min(y_indices)]
                    bottom_point = [x_indices[np.argmax(y_indices)], np.max(y_indices)]
                    points.extend([top_point, bottom_point])
                    classes_points.extend([color_id, color_id])
                else:
                    # Random selection
                    remaining_points = num_points - len(points)
                    if remaining_points > 0:
                        object_coords = np.argwhere(object_mask)
                        sampled_indices = np.random.choice(len(object_coords), remaining_points, replace=False)
                        for coord in object_coords[sampled_indices]:
                            point = coord[::-1].tolist()  # [x, y] format
                            points.append(point)
                            classes_points.append(color_id)

                # Add the sampled points to sampled_points
                sampled_points.extend(points)
    info = {
        'sampled_points': sampled_points,
        'classes_points': classes_points,
        'class_to_color_mapper': class_to_color_mapper,
        'color_to_class_mapper': color_to_class_mapper
    }
    return info


# Example usage:
# sampled_points, classes_points, class_to_color_mapper, color_to_class_mapper = sample_points_from_pixel_mask(gt_mask)



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