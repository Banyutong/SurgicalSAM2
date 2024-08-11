import numpy as np


def rgb_to_class_id(rgb_mask, color_map):
    """
    Converts an RGB segmentation mask to a class ID mask.
    """
    class_id_mask = np.zeros(rgb_mask.shape[:2], dtype=np.uint8)
    for color, class_id in color_map.items():
        color_mask = np.all(rgb_mask == color, axis=-1)
        class_id_mask[color_mask] = class_id
    return class_id_mask


def calculate_iou(ground_truth, prediction, class_id):
    """
    Calculates the Intersection over Union (IoU) for a specific class.
    """
    gt_mask = ground_truth == class_id
    pred_mask = prediction == class_id
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 0.0
    iou = intersection / union
    return iou


def calculate_miou(ground_truth, prediction, class_map):
    """
    Calculates the mean Intersection over Union (mIoU) over multiple classes
    and records IoU for each class with its name.

    Args:
      ground_truth: A numpy array representing the ground truth segmentation mask (RGB).
      prediction: A numpy array representing the predicted segmentation mask (RGB).
      class_map: A list of dictionaries, each containing "name", "color", and "classid"
                 for a class.

    Returns:
      A tuple containing:
        - The mIoU score.
        - A dictionary mapping class names to their IoU scores.
    """

    color_map_gt = {tuple(c["color"]): c["classid"] for c in class_map}
    color_map_pred = {tuple(c["color"]): c["classid"] for c in class_map}
    gt_class_id_mask = rgb_to_class_id(ground_truth, color_map_gt)
    pred_class_id_mask = rgb_to_class_id(prediction, color_map_pred)

    iou_scores = {}
    for class_info in class_map:
        class_id = class_info["classid"]
        class_name = class_info["name"]
        iou = calculate_iou(gt_class_id_mask, pred_class_id_mask, class_id)
        iou_scores[class_name] = iou

    miou = np.mean(list(iou_scores.values()))
    return miou, iou_scores


def calculate_dice(ground_truth, prediction, class_id):
    """
    Calculates the Dice score for a specific class.
    """
    gt_mask = ground_truth == class_id
    pred_mask = prediction == class_id
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    dice = (2 * intersection) / (gt_mask.sum() + pred_mask.sum())
    return dice


def calculate_mdice(ground_truth, prediction, class_map):
    """
    Calculates the mean Dice score (mDice) over multiple classes
    and records Dice for each class with its name.

    Args:
      ground_truth: A numpy array representing the ground truth segmentation mask (RGB).
      prediction: A numpy array representing the predicted segmentation mask (RGB).
      class_map: A list of dictionaries, each containing "name", "color", and "classid"
                 for a class.

    Returns:
      A tuple containing:
        - The mDice score.
        - A dictionary mapping class names to their Dice scores.
    """

    color_map_gt = {tuple(c["color"]): c["classid"] for c in class_map}
    color_map_pred = {tuple(c["color"]): c["classid"] for c in class_map}
    gt_class_id_mask = rgb_to_class_id(ground_truth, color_map_gt)
    pred_class_id_mask = rgb_to_class_id(prediction, color_map_pred)

    dice_scores = {}
    for class_info in class_map:
        class_id = class_info["classid"]
        class_name = class_info["name"]
        dice = calculate_dice(gt_class_id_mask, pred_class_id_mask, class_id)
        dice_scores[class_name] = dice

    mdice = np.mean(list(dice_scores.values()))
    return mdice, dice_scores



