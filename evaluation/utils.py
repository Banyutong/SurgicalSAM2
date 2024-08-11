import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


cholec_thing_classes = [
    "Black Background",
    "Abdominal Wall",
    "Liver",
    "Gastrointestinal Tract",
    "Fat",
    "Grasper",
    "Connective Tissue",
    "Blood",
    "Cystic Duct",
    "L-hook Electrocautery",
    "Gallbladder",
    "Hepatic Vein",
    "Liver Ligament",
]


class IoUScore:
    def __init__(self, class_map):
        """
        Initializes the IoUCalculator with the class map.

        Args:
            class_map: A list of dictionaries, each containing "name" and "classid"
                       for a class.
        """
        self.class_map = class_map

    def calculate_iou(self, ground_truth, prediction, class_id):
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

    def calculate_miou(self, ground_truth, prediction):
        """
        Calculates the mean Intersection over Union (mIoU) over multiple classes
        and records IoU for each class with its name for a batch of masks.

        Args:
            ground_truth: A numpy array representing a batch of ground truth segmentation masks (class IDs).
                          Shape: (batch_size, height, width)
            prediction: A numpy array representing a batch of predicted segmentation masks (class IDs).
                        Shape: (batch_size, height, width)

        Returns:
            A tuple containing:
              - A list of mIoU scores for each mask in the batch.
              - A list of dictionaries, each mapping class names to their IoU scores for a mask in the batch.
        """

        batch_size = ground_truth.shape[0]
        miou_scores = []
        all_iou_scores = []

        for i in range(batch_size):
            gt_mask = ground_truth[i]
            pred_mask = prediction[i]

            iou_scores = {}
            for class_info in self.class_map:
                class_id = class_info["classid"]
                class_name = class_info["name"]
                iou = self.calculate_iou(gt_mask, pred_mask, class_id)
                iou_scores[class_name] = iou

            miou = np.mean(list(iou_scores.values()))
            miou_scores.append(miou)
            all_iou_scores.append(iou_scores)

        return miou_scores, all_iou_scores


class DiceScore:
    def __init__(self, class_map):
        """
        Initializes the DiceCalculator with the class map.

        Args:
            class_map: A list of dictionaries, each containing "name" and "classid"
                       for a class.
        """
        self.class_map = class_map

    def calculate_dice(self, ground_truth, prediction, class_id):
        """
        Calculates the Dice score for a specific class.
        """
        gt_mask = ground_truth == class_id
        pred_mask = prediction == class_id
        intersection = np.logical_and(gt_mask, pred_mask).sum()
        dice = (2 * intersection) / (gt_mask.sum() + pred_mask.sum())
        return dice

    def calculate_mdice(self, ground_truth, prediction):
        """
        Calculates the mean Dice score (mDice) over multiple classes
        and records Dice for each class with its name for a batch of masks.

        Args:
            ground_truth: A numpy array representing a batch of ground truth segmentation masks (class IDs).
                          Shape: (batch_size, height, width)
            prediction: A numpy array representing a batch of predicted segmentation masks (class IDs).
                        Shape: (batch_size, height, width)

        Returns:
            A tuple containing:
              - A list of mDice scores for each mask in the batch.
              - A list of dictionaries, each mapping class names to their Dice scores for a mask in the batch.
        """

        batch_size = ground_truth.shape[0]
        mdice_scores = []
        all_dice_scores = []

        for i in range(batch_size):
            gt_mask = ground_truth[i]
            pred_mask = prediction[i]

            dice_scores = {}
            for class_info in self.class_map:
                class_id = class_info["classid"]
                class_name = class_info["name"]
                dice = self.calculate_dice(gt_mask, pred_mask, class_id)
                dice_scores[class_name] = dice

            mdice = np.mean(list(dice_scores.values()))
            mdice_scores.append(mdice)
            all_dice_scores.append(dice_scores)

        return mdice_scores, all_dice_scores


# class_map = [
#     {'name': 'cystic plate', 'id': 1},
#     {'name': 'polyps', 'id': 2},
#     # ... other classes
# ]

# # Create an APScore object
# ap_calculator = APScore(annotation_file, class_map)

# # Get your model predictions in COCO format (list of dictionaries)
# predictions = [
#     {'image_id': 1, 'category_id': 1, 'bbox': [10, 20, 30, 40], 'score': 0.9},
#     {'image_id': 1, 'category_id': 2, 'bbox': [50, 60, 70, 80], 'score': 0.8},
#     # ... other predictions
# ]


class APScore:
    def __init__(self, annotation_file, class_map=None):
        """
        Initializes the APScore calculator.

        Args:
            annotation_file: Path to the COCO annotation file (.json).
            class_map (optional): A list of dictionaries, each containing "name" and "id"
                                 for a class. If None, all classes in the annotation file
                                 will be used.
        """
        self.coco_gt = COCO(annotation_file)
        if class_map is None:
            self.class_map = [
                {"name": cat["name"], "id": cat["id"]}
                for cat in self.coco_gt.loadCats(self.coco_gt.getCatIds())
            ]
        else:
            self.class_map = class_map
        self.class_ids = [c["id"] for c in self.class_map]
        self.coco_dt = None

    def update(self, predictions):
        """
        Updates the predictions for evaluation.

        Args:
            predictions: A list of dictionaries in COCO prediction format.
                         Each dictionary should have keys like "image_id", "category_id",
                         "bbox", and "score".
        """
        # Create a COCO object for the predictions
        self.coco_dt = self.coco_gt.loadRes(predictions)

    def calculate_ap(self):
        """
        Calculates the average precision (AP) and AP for each class.

        Returns:
            A dictionary containing:
              - "AP": The overall AP.
              - "AP50": AP at IoU threshold 0.50.
              - "AP75": AP at IoU threshold 0.75.
              - "AP_small": AP for small objects.
              - "AP_medium": AP for medium objects.
              - "AP_large": AP for large objects.
              - "class_ap": A dictionary mapping class names to their AP scores.
              - "class_ap50": A dictionary mapping class names to their AP50 scores.
        """
        if self.coco_dt is None:
            raise ValueError("Predictions have not been updated yet.")

        coco_eval = COCOeval(self.coco_gt, self.coco_dt, "bbox")
        coco_eval.params.catIds = self.class_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        ap_scores = {
            "mAP5-90": coco_eval.stats[0],
            "AP50": coco_eval.stats[1],
            "AP75": coco_eval.stats[2],
            "AP_small": coco_eval.stats[3],
            "AP_medium": coco_eval.stats[4],
            "AP_large": coco_eval.stats[5],
        }

        class_ap = {}
        class_ap50 = {}
        for class_info in self.class_map:
            class_id = class_info['id']
            class_name = class_info['name']

            # Calculate AP (mAP 50-95)
            precisions = coco_eval.eval['precision'][0, :, class_id - 1, 0, 2]
            ap = np.mean(precisions[precisions > -1])
            class_ap[class_name] = ap

            # Calculate AP50
            precisions_50 = coco_eval.eval['precision'][0, :, class_id - 1, 0, 0]
            ap50 = precisions_50[0]  # Precision at IoU threshold 0.50
            class_ap50[class_name] = ap50

        ap_scores["class_ap"] = class_ap
        ap_scores["class_ap50"] = class_ap50
        
        return ap_scores
    
    
    
    
class MaeScore:
    def __init__(self, class_map):
        """
        Initializes the MaeScore calculator.

        Args:
            class_map: A list of dictionaries, each containing "name" and "id" 
                       for a class.
        """
        self.class_map = class_map
        self.class_ids = [c['id'] for c in self.class_map]

    def calculate_mae(self, ground_truth, prediction):
        """
        Calculates the Mean Absolute Error (MAE) for a batch of class ID masks.

        Args:
            ground_truth: A numpy array representing a batch of ground truth segmentation masks (class IDs).
                          Shape: (batch_size, height, width)
            prediction: A numpy array representing a batch of predicted segmentation masks (class IDs).
                        Shape: (batch_size, height, width)

        Returns:
            A tuple containing:
              - A list of MAE scores for each mask in the batch.
              - A list of dictionaries, each mapping class names to their MAE scores for a mask in the batch.
        """

        batch_size = ground_truth.shape[0]
        mae_scores = []
        all_class_mae_scores = []

        for i in range(batch_size):
            gt_mask = ground_truth[i]
            pred_mask = prediction[i]

            class_mae_scores = {}
            for class_info in self.class_map:
                class_id = class_info['id']
                class_name = class_info['name']

                gt_class_mask = gt_mask == class_id
                pred_class_mask = pred_mask == class_id

                mae = np.mean(np.abs(gt_class_mask.astype(np.float32) - pred_class_mask.astype(np.float32)))
                class_mae_scores[class_name] = mae

            mae_scores.append(np.mean(list(class_mae_scores.values())))
            all_class_mae_scores.append(class_mae_scores)

        return mae_scores, all_class_mae_scores

# Example usage:

# Define the class map
# class_map = [
#   {
#     "name": "instrument-clasper",
#     "id": 2
#   },
#   {
#     "name": "instrument-wrist",
#     "id": 3
#   },
#   {
#     "name": "kidney-parenchyma",
#     "id": 4
#   }
# ]

# Assuming ground_truth and prediction are now numpy arrays with shape 
# (batch_size, height, width) - containing class IDs
