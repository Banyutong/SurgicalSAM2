import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from loguru import logger
from pycocotools.coco import COCO


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = np.logical_and(pred, gt).sum(dtype=np.float64)
    denom = pred.sum(dtype=np.float64) + gt.sum(dtype=np.float64) + 1e-7
    return float(2.0 * inter / denom)


def load_meta(probs_dir: str) -> Dict:
    meta_path = os.path.join(probs_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {probs_dir}")
    with open(meta_path, "r") as f:
        return json.load(f)


def grid_search(
    probs_dir: str,
    coco_path: str,
    t_min: float = 0.2,
    t_max: float = 0.8,
    t_step: float = 0.05,
    exclude_background: bool = False,
) -> Tuple[float, float, List[Tuple[float, float]]]:
    coco = COCO(coco_path)
    meta = load_meta(probs_dir)
    mod = int(meta["mod"])
    image_ids = meta.get("image_ids")
    if not image_ids:
        # Fallback: scan directory
        image_ids = [
            int(Path(p).stem)
            for p in os.listdir(probs_dir)
            if p.endswith(".npz") and Path(p).stem.isdigit()
        ]

    thresholds = []
    t = t_min
    while t <= t_max + 1e-9:
        thresholds.append(round(t, 5))
        t += t_step

    sum_dice = np.zeros(len(thresholds), dtype=np.float64)
    count = np.zeros(len(thresholds), dtype=np.int64)

    for image_id in image_ids:
        npz_path = os.path.join(probs_dir, f"{image_id}.npz")
        if not os.path.exists(npz_path):
            logger.warning(f"Missing probs file: {npz_path}")
            continue
        data = np.load(npz_path)
        probs: np.ndarray = data["probs"]  # [N,H,W], float16
        obj_ids: np.ndarray = data["obj_ids"]  # [N]
        H = int(data["height"]) if "height" in data else probs.shape[1]
        W = int(data["width"]) if "width" in data else probs.shape[2]

        # Categories present in predictions for this image
        pred_cat_ids = set((obj_ids % mod).tolist()) if obj_ids.size > 0 else set()

        # Categories present in GT for this image
        ann_ids = coco.getAnnIds(imgIds=int(image_id))
        anns = coco.loadAnns(ann_ids)
        gt_cat_ids = set([a["category_id"] for a in anns])

        categories = sorted(pred_cat_ids.union(gt_cat_ids))
        if exclude_background and 0 in categories:
            categories.remove(0)

        # Precompute GT masks per category
        gt_masks: Dict[int, np.ndarray] = {}
        for c in categories:
            cat_anns = [a for a in anns if a["category_id"] == c]
            if not cat_anns:
                # keep empty mask to preserve size
                gt_masks[c] = np.zeros((H, W), dtype=bool)
                continue
            merged = np.zeros((H, W), dtype=bool)
            for a in cat_anns:
                merged |= coco.annToMask(a).astype(bool)
            gt_masks[c] = merged

        # For each threshold compute merged prediction per category and Dice
        for ti, thr in enumerate(thresholds):
            has_any = False
            for c in categories:
                # Skip if neither pred objects nor gt objects exist for this cat
                pred_indices = np.where((obj_ids % mod) == c)[0]
                gt_nonempty = gt_masks[c].any()
                if pred_indices.size == 0 and not gt_nonempty:
                    continue

                has_any = True
                if pred_indices.size == 0:
                    pred_mask = np.zeros((H, W), dtype=bool)
                else:
                    pred_mask = np.any(
                        probs[pred_indices, :, :] >= thr, axis=0
                    )
                d = dice_score(pred_mask, gt_masks[c])
                sum_dice[ti] += d
                count[ti] += 1

            # nothing to aggregate on this image for this thr
            _ = has_any

    # Compute mean dice per threshold
    valid = count > 0
    if not valid.any():
        raise RuntimeError("No valid categories found for Dice computation.")
    mean_dice = np.full_like(sum_dice, fill_value=-np.inf, dtype=np.float64)
    mean_dice[valid] = sum_dice[valid] / count[valid]

    best_idx = int(np.argmax(mean_dice))
    # Tie-breaker: prefer closest to 0.5 if multiple equal
    best_candidates = np.where(mean_dice == mean_dice[best_idx])[0]
    if len(best_candidates) > 1:
        best_idx = min(best_candidates, key=lambda i: abs(thresholds[i] - 0.5))

    best_thr = float(thresholds[best_idx])
    best_dice = float(mean_dice[best_idx])
    per_thr = [(float(thresholds[i]), float(mean_dice[i])) for i in range(len(thresholds)) if valid[i]]
    return best_thr, best_dice, per_thr


@logger.catch(onerror=lambda _: __import__("sys").exit(1))
def main():
    parser = argparse.ArgumentParser(
        description="Grid search probability threshold for best Dice using saved probs"
    )
    parser.add_argument("--probs-dir", required=True, type=str)
    parser.add_argument("--coco-path", required=True, type=str)
    parser.add_argument("--min", dest="t_min", type=float, default=0.2)
    parser.add_argument("--max", dest="t_max", type=float, default=0.8)
    parser.add_argument("--step", dest="t_step", type=float, default=0.05)
    parser.add_argument("--exclude-background", action="store_true")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to save best_threshold.json (defaults to probs_dir/../best_threshold.json)",
    )
    args = parser.parse_args()

    best_thr, best_dice, per_thr = grid_search(
        probs_dir=args.probs_dir,
        coco_path=args.coco_path,
        t_min=args.t_min,
        t_max=args.t_max,
        t_step=args.t_step,
        exclude_background=args.exclude_background,
    )

    parent = Path(args.probs_dir).parent
    out_path = args.output_json or str(parent / "best_threshold.json")
    payload = {
        "best_threshold": best_thr,
        "best_dice": best_dice,
        "threshold_curve": per_thr,
        "exclude_background": bool(args.exclude_background),
        "range": {
            "min": float(args.t_min),
            "max": float(args.t_max),
            "step": float(args.t_step),
        },
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"Saved best threshold {best_thr:.3f} (Dice={best_dice:.4f}) to {out_path}")


if __name__ == "__main__":
    main()

