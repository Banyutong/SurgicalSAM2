import importlib.util
import json
import os
import sys
import types
from contextlib import nullcontext

import numpy as np
import pytest
import torch
from pycocotools.coco import COCO

PIPELINE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "Endoscapes2023_Pipeline")
)


@pytest.fixture
def inference(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # inference.py adds a loguru sink at dense_points/log.log
    # stub the SAM2 model builder and prompt-noise module, only save_as_coco_format is tested
    build_sam = types.ModuleType("sam2.build_sam")
    build_sam.build_sam2_video_predictor = None
    noise = types.ModuleType("PromptObjNoiseAdder")
    noise.PromptObjNoiseAdder = None
    # inference.py sets up CUDA autocast / TF32 at import time; make that a no-op on CPU
    monkeypatch.setattr(torch, "autocast", lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *args: types.SimpleNamespace(major=0)
    )
    monkeypatch.setitem(sys.modules, "sam2", types.ModuleType("sam2"))
    monkeypatch.setitem(sys.modules, "sam2.build_sam", build_sam)
    monkeypatch.setitem(sys.modules, "PromptObjNoiseAdder", noise)
    monkeypatch.delitem(sys.modules, "utils", raising=False)
    monkeypatch.syspath_prepend(PIPELINE_DIR)
    spec = importlib.util.spec_from_file_location(
        "endoscapes_inference", os.path.join(PIPELINE_DIR, "inference.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_category_score_is_max_over_its_objects(inference, monkeypatch, tmp_path):
    coco = COCO()
    coco.dataset = {
        "images": [{"id": 7, "height": 4, "width": 4, "video_id": 0, "order_in_video": 0}],
        "categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}],
        "annotations": [],
    }
    coco.createIndex()
    monkeypatch.setattr(inference, "COCO_INFO", coco)
    monkeypatch.setattr(inference, "MOD", 10)
    monkeypatch.setattr(inference, "OUTPUT_PATH", str(tmp_path))

    def obj(rows, score):
        mask = np.zeros((1, 4, 4), dtype=bool)
        mask[0, rows] = True
        return {"mask": mask, "score": score}

    # object ids 1 and 11 are category 1 (id % MOD), object 2 is category 2
    segments = {0: {0: {1: obj(0, 0.9), 11: obj(1, 0.3), 2: obj(2, 0.6)}}}
    predict_path, _ = inference.save_as_coco_format(segments, [0])

    with open(predict_path) as f:
        anns = {ann["category_id"]: ann for ann in json.load(f)}
    assert anns[1]["score"] == pytest.approx(0.9)  # was 0.6, the last object's score
    assert anns[2]["score"] == pytest.approx(0.6)
    assert coco.annToMask(dict(anns[1], image_id=7)).sum() == 8  # both objects merged
