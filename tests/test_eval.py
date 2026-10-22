import importlib.util
import os

import numpy as np
import pycocotools.mask as maskUtils
import pytest
from pycocotools.coco import COCO

ROOT = os.path.join(os.path.dirname(__file__), "..")
EVAL_FILES = [
    "Endoscapes2023_Pipeline/eval.py",
    "evaluation/eval.py",
    "sam2-video-training/sam2_video/eval/eval.py",
    "sam2-video-training/notebooks/eval.py",
]


def load_module(rel_path):
    name = rel_path.replace("/", "_").replace("-", "_")[:-3]
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rel_path", EVAL_FILES)
def test_mae_uint8_masks_do_not_wrap_around(rel_path):
    caculate_mae = load_module(rel_path).caculate_mae
    gt = np.array([[1, 1], [0, 0]], dtype=np.uint8)
    pred = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    # 2 of 4 pixels differ by 1 -> 0.5 (uint8 subtraction gave (255 + 1) / 4 = 64)
    assert caculate_mae(pred, gt) == pytest.approx(0.5)
    assert caculate_mae(gt, pred) == pytest.approx(0.5)
    assert caculate_mae(gt, gt) == 0.0
    assert caculate_mae(np.zeros((2, 2), np.uint8), np.ones((2, 2), np.uint8)) == 1.0


def _rle(mask):
    rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def test_absent_categories_are_scored(monkeypatch):
    eval_module = load_module("evaluation/eval.py")
    h, w = 8, 8
    mask_a = np.zeros((h, w), np.uint8)
    mask_a[:4, :4] = 1
    mask_b = np.zeros((h, w), np.uint8)
    mask_b[4:, 4:] = 1

    coco_gt = COCO()
    coco_gt.dataset = {
        "images": [
            {"id": 1, "height": h, "width": w, "video_id": 0, "order_in_video": 0},
            {"id": 2, "height": h, "width": w, "video_id": 0, "order_in_video": 1},
        ],
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "segmentation": _rle(mask_a),
             "area": float(mask_a.sum()), "bbox": [0, 0, 4, 4], "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "segmentation": _rle(mask_a),
             "area": float(mask_a.sum()), "bbox": [0, 0, 4, 4], "iscrowd": 0},
        ],
    }
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes([
        # image 1: perfect category 1 + false-positive category 2 (absent in GT)
        {"image_id": 1, "category_id": 1, "segmentation": _rle(mask_a), "score": 0.9},
        {"image_id": 1, "category_id": 2, "segmentation": _rle(mask_b), "score": 0.9},
        # image 2: category 1 is in GT but not predicted (missed)
    ])
    monkeypatch.setattr(eval_module, "CAT_IDS", coco_gt.getCatIds())

    _, img_scores = eval_module.get_image_scores(coco_dt, coco_gt)
    img1, img2 = sorted(img_scores, key=lambda s: s["order_in_video"])

    assert img1["cat_scores"][1]["iou"] == pytest.approx(1.0)
    assert img1["cat_scores"][2]["iou"] == pytest.approx(0.0)  # was NaN (ignored)
    assert img1["cat_scores"][2]["mae"] == pytest.approx(mask_b.mean())
    assert img1["avg_scores"]["iou"] == pytest.approx(0.5)
    assert img2["cat_scores"][1]["iou"] == pytest.approx(0.0)  # was NaN (ignored)
    assert np.isnan(img2["cat_scores"][2]["iou"])  # absent in both GT and prediction
