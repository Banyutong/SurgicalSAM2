import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from loguru import logger
from pycocotools import mask as maskUtils

from .utils import mask_to_bbox


def load_meta(probs_dir: str) -> Dict:
    meta_path = os.path.join(probs_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found in {probs_dir}")
    with open(meta_path, "r") as f:
        return json.load(f)


def export_predict(
    probs_dir: str,
    threshold: float,
    output_predict: str | None = None,
    exclude_background: bool = False,
) -> str:
    meta = load_meta(probs_dir)
    mod = int(meta["mod"])
    image_ids = meta.get("image_ids")
    if not image_ids:
        image_ids = [
            int(Path(p).stem)
            for p in os.listdir(probs_dir)
            if p.endswith(".npz") and Path(p).stem.isdigit()
        ]

    annotations: List[Dict] = []

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

        # Map category_id -> indices of objects
        cat_to_indices: Dict[int, np.ndarray] = {}
        for idx, oid in enumerate(obj_ids.tolist()):
            cat_id = int(oid % mod)
            if exclude_background and cat_id == 0:
                continue
            if cat_id not in cat_to_indices:
                cat_to_indices[cat_id] = []
            cat_to_indices[cat_id].append(idx)

        for cat_id, indices in cat_to_indices.items():
            if len(indices) == 0:
                continue
            indices = np.asarray(indices, dtype=np.int64)
            # Threshold and merge objects for this category
            merged = np.any(probs[indices, :, :] >= threshold, axis=0)
            if merged.sum() == 0:
                continue

            # Score: max probability across all objects for this category
            cat_scores = [float(probs[i].max()) for i in indices.tolist()]
            score = float(np.max(cat_scores)) if len(cat_scores) > 0 else 0.0

            rle = maskUtils.encode(np.asfortranarray(merged.astype(np.uint8)))
            rle["counts"] = rle["counts"].decode("utf-8")
            ann = {
                "image_id": int(image_id),
                "category_id": int(cat_id),
                "segmentation": rle,
                "bbox": mask_to_bbox(merged),
                "iscrowd": 0,
                "score": score,
            }
            annotations.append(ann)

    if output_predict is None:
        parent = Path(probs_dir).parent
        output_predict = str(parent / f"predict_t{threshold:.2f}.json")
    with open(output_predict, "w") as f:
        json.dump(annotations, f, indent=2)
    logger.info(f"Wrote predictions to {output_predict}")
    return output_predict


@logger.catch(onerror=lambda _: __import__("sys").exit(1))
def main():
    parser = argparse.ArgumentParser(
        description="Export predict.json from saved per-frame probability maps"
    )
    parser.add_argument("--probs-dir", required=True, type=str)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--output-predict", type=str, default=None)
    parser.add_argument("--exclude-background", action="store_true")
    args = parser.parse_args()

    export_predict(
        probs_dir=args.probs_dir,
        threshold=args.threshold,
        output_predict=args.output_predict,
        exclude_background=args.exclude_background,
    )


if __name__ == "__main__":
    main()

