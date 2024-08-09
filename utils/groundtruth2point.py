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


if __name__ == "__main__":
    # Extract bounding boxes from the given data
    bboxes = [
        [280, 90, 550, 300],  # Object 1: [x_min, y_min, x_max, y_max]
        [360, 110, 410, 290]  # Object 2: [x_min, y_min, x_max, y_max]
    ]
    
    # Sample points from bounding boxes
    bbox_points = sample_points_from_bboxes(bboxes, num_points=3, include_center=True)
    print("Sampled points from bounding boxes:")
    for i, points in enumerate(bbox_points):
        print(f"Object {i + 1}: {points}")