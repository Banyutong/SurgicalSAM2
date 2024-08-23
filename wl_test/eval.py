from pycocotools.coco import COCO
import numpy as np
import pickle
from loguru import logger

CAT_IDS = None


def get_dicts_by_field_value(data, field_name, target_value):
    return [item for item in data if item.get(field_name) == target_value]


def caculate_iou(pred, gt):
    """
    Calculate the Intersection over Union (IoU) between two masks.

    Args:
        pred (np.ndarray): Predicted mask.
        gt (np.ndarray): Ground truth mask.

    Returns:
        float: IoU value.
    """
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum() + 1e-7
    iou = intersection / union
    return iou


def caculate_dice(pred, gt):
    intersection = np.sum(pred * gt)
    dice_score = (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + 1e-7)
    return dice_score


def caculate_mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def get_image_scores(cocoDT, cocoGT):

    imgids = cocoGT.getImgIds()
    imgs = cocoGT.loadImgs(imgids)
    video_id_set = set()

    img_scores = []

    for img in imgs:
        anns_dt = cocoDT.loadAnns(cocoDT.getAnnIds(imgIds=img["id"]))
        anns_gt = cocoGT.loadAnns(cocoGT.getAnnIds(imgIds=img["id"]))

        all_iou = {cat_id: [] for cat_id in CAT_IDS}
        all_mae = {cat_id: [] for cat_id in CAT_IDS}
        all_dice = {cat_id: [] for cat_id in CAT_IDS}

        img_score = {
            "video_id": img["video_id"],
            "order_in_video": img["order_in_video"],
            "cat_scores": {
                cat_id: {"iou": None, "mae": None, "dice": None} for cat_id in CAT_IDS
            },
            "avg_scores": {"iou": None, "mae": None, "dice": None},
        }

        for ann_dt in anns_dt:
            for ann_gt in anns_gt:

                if ann_dt["category_id"] == ann_gt["category_id"]:

                    mask_dt = cocoDT.annToMask(ann_dt)
                    mask_gt = cocoGT.annToMask(ann_gt)
                    iou = caculate_iou(mask_dt, mask_gt)
                    mae = caculate_mae(mask_dt, mask_gt)
                    dice = caculate_dice(mask_dt, mask_gt)

                    all_iou[ann_dt["category_id"]].append(iou)
                    all_mae[ann_dt["category_id"]].append(mae)
                    all_dice[ann_dt["category_id"]].append(dice)

        avg_iou = []
        avg_mae = []
        avg_dice = []

        for cat_id in CAT_IDS:
            img_score["cat_scores"][cat_id]["iou"] = (
                np.nanmean(all_iou[cat_id]) if len(all_iou[cat_id]) > 0 else np.nan
            )
            img_score["cat_scores"][cat_id]["mae"] = (
                np.nanmean(all_mae[cat_id]) if len(all_mae[cat_id]) > 0 else np.nan
            )
            img_score["cat_scores"][cat_id]["dice"] = (
                np.nanmean(all_dice[cat_id]) if len(all_dice[cat_id]) > 0 else np.nan
            )

            avg_iou.append(img_score["cat_scores"][cat_id]["iou"])
            avg_mae.append(img_score["cat_scores"][cat_id]["mae"])
            avg_dice.append(img_score["cat_scores"][cat_id]["dice"])

        img_score["avg_scores"]["iou"] = (
            np.nanmean(avg_iou) if len(avg_iou) > 0 else np.nan
        )
        img_score["avg_scores"]["mae"] = (
            np.nanmean(avg_mae) if len(avg_mae) > 0 else np.nan
        )
        img_score["avg_scores"]["dice"] = (
            np.nanmean(avg_dice) if len(avg_dice) > 0 else np.nan
        )

        video_id_set.add(img["video_id"])
        img_scores.append(img_score)

    return video_id_set, img_scores


def get_video_scores(video_id_set, img_scores):
    video_scores = []
    for video_id in video_id_set:
        frames = get_dicts_by_field_value(img_scores, "video_id", video_id)

        video_score = {
            "video_id": video_id,
            "frames": frames,
            "cat_scores": {
                cat_id: {"iou": None, "mae": None, "dice": None} for cat_id in CAT_IDS
            },
            "avg_scores": {"iou": None, "mae": None, "dice": None},
        }

        all_iou = {cat_id: [] for cat_id in CAT_IDS}
        all_mae = {cat_id: [] for cat_id in CAT_IDS}
        all_dice = {cat_id: [] for cat_id in CAT_IDS}

        for frame in frames:
            for category_id, category_data in frame["cat_scores"].items():
                all_iou[category_id].append(category_data["iou"])
                all_mae[category_id].append(category_data["mae"])
                all_dice[category_id].append(category_data["dice"])

        for key in CAT_IDS:
            video_score["cat_scores"][key]["iou"] = np.nanmean(all_iou[key])
            video_score["cat_scores"][key]["mae"] = np.nanmean(all_mae[key])
            video_score["cat_scores"][key]["dice"] = np.nanmean(all_dice[key])

        # print(all_iou)

        avg_iou = []
        avg_mae = []
        avg_dice = []

        for cat_id in CAT_IDS:
            video_score["cat_scores"][cat_id]["iou"] = (
                np.nanmean(all_iou[cat_id]) if len(all_iou[cat_id]) > 0 else np.nan
            )
            video_score["cat_scores"][cat_id]["mae"] = (
                np.nanmean(all_mae[cat_id]) if len(all_mae[cat_id]) > 0 else np.nan
            )
            video_score["cat_scores"][cat_id]["dice"] = (
                np.nanmean(all_dice[cat_id]) if len(all_dice[cat_id]) > 0 else np.nan
            )

            avg_iou.append(video_score["cat_scores"][cat_id]["iou"])
            avg_mae.append(video_score["cat_scores"][cat_id]["mae"])
            avg_dice.append(video_score["cat_scores"][cat_id]["dice"])

        video_score["avg_scores"]["iou"] = (
            np.nanmean(avg_iou) if len(avg_iou) > 0 else np.nan
        )
        video_score["avg_scores"]["mae"] = (
            np.nanmean(avg_mae) if len(avg_mae) > 0 else np.nan
        )
        video_score["avg_scores"]["dice"] = (
            np.nanmean(avg_dice) if len(avg_dice) > 0 else np.nan
        )

        video_scores.append(video_score)

    return video_scores


def get_result(video_scores):
    result = {
        "videos": video_scores,
        "cat_scores": {
            cat_id: {"iou": None, "mae": None, "dice": None} for cat_id in CAT_IDS
        },
        "avg_scores": {"iou": None, "mae": None, "dice": None},
    }

    all_iou = {cat_id: [] for cat_id in CAT_IDS}
    all_mae = {cat_id: [] for cat_id in CAT_IDS}
    all_dice = {cat_id: [] for cat_id in CAT_IDS}

    for video_data in result["videos"]:
        for category_id, category_data in video_data["cat_scores"].items():
            all_iou[category_id].append(category_data["iou"])
            all_mae[category_id].append(category_data["mae"])
            all_dice[category_id].append(category_data["dice"])

    for key in CAT_IDS:
        result["cat_scores"][key]["iou"] = np.nanmean(all_iou[key])
        result["cat_scores"][key]["mae"] = np.nanmean(all_mae[key])
        result["cat_scores"][key]["dice"] = np.nanmean(all_dice[key])

    # print(all_iou)

    avg_iou = []
    avg_mae = []
    avg_dice = []

    for cat_id in CAT_IDS:
        result["cat_scores"][cat_id]["iou"] = (
            np.nanmean(all_iou[cat_id]) if len(all_iou[cat_id]) > 0 else np.nan
        )
        result["cat_scores"][cat_id]["mae"] = (
            np.nanmean(all_mae[cat_id]) if len(all_mae[cat_id]) > 0 else np.nan
        )
        result["cat_scores"][cat_id]["dice"] = (
            np.nanmean(all_dice[cat_id]) if len(all_dice[cat_id]) > 0 else np.nan
        )

        avg_iou.append(result["cat_scores"][cat_id]["iou"])
        avg_mae.append(result["cat_scores"][cat_id]["mae"])
        avg_dice.append(result["cat_scores"][cat_id]["dice"])

    result["avg_scores"]["iou"] = np.nanmean(avg_iou) if len(avg_iou) > 0 else np.nan
    result["avg_scores"]["mae"] = np.nanmean(avg_mae) if len(avg_mae) > 0 else np.nan
    result["avg_scores"]["dice"] = np.nanmean(avg_dice) if len(avg_dice) > 0 else np.nan

    return result


def eval(predict_path, coco_path, output_path):

    global CAT_IDS

    cocoGT = COCO(coco_path)
    cocoDT = cocoGT.loadRes(predict_path)
    output_path = f"{output_path}/eval.pkl"

    CAT_IDS = cocoGT.getCatIds()

    video_id_set, img_scores = get_image_scores(cocoDT, cocoGT)
    video_scores = get_video_scores(video_id_set, img_scores)
    result = get_result(video_scores)

    with open(output_path, "wb") as f:
        pickle.dump(result, f)


if __name__ == "__main__":

    eval(
        predict_path="/bd_byta6000i0/users/sam2/kyyang/SurgicalSAM2/wl_test/test/output/mask/predict.json",
        coco_path="gt1_coco_annotations.json",
        output_path="test/output/bbox/",
    )
