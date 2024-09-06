import os
import pickle
from typing import List

import imageio
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from natsort import natsorted
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

from utils import PromptInfo, get_dicts_by_field_value, sort_dicts_by_field

COCO_GT = None
COCO_PREDICT = None
MOD = None
FRAMES = None
OUTPUT_PATH = None
CMAP = plt.get_cmap("tab20")
IMAGE_PATH_FOR_GIF = None

logger.add("logs/visualize.log")


def show_box(box, ax, category_id):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]

    # 从 color_map 中获取指定类别的颜色
    edgecolor = CMAP(category_id)  # 如果未找到类别，则默认为绿色
    facecolor = (edgecolor[0], edgecolor[1], edgecolor[2], 0.2)
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=facecolor, lw=2)
    )


def show_mask(mask, ax, category_id):
    color = np.array([*CMAP(category_id)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(points, labels, ax, category_id):
    for point, label in zip(points, labels):
        marker = "o" if label == 1 else "x"
        linewidth = 1 if label == 1 else 2
        ax.scatter(
            point[0],
            point[1],
            facecolor=CMAP(category_id),
            s=50,
            marker=marker,
            alpha=1.0,
            edgecolor="white",
            linewidth=linewidth,
        )


def visualize_prompt_frame(prompt_info: PromptInfo, ax):
    ax.set_title(f"Prompt Frame {prompt_info.frame_idx}")
    image_path = prompt_info.path
    image = np.array(Image.open(image_path))
    ax.imshow(image)
    for obj in prompt_info.prompt_objs:
        obj_id = obj.obj_id % MOD
        match prompt_info.prompt_type:
            case "points":
                show_points(obj.points, obj.pos_or_neg_label, ax, obj_id)
            case "bbox":
                show_box(obj.bbox, ax, obj_id)
            case "mask":
                show_mask(obj.mask, ax, obj_id)


def visualize_current_frame(frame_id, axs):
    frame = COCO_GT.loadImgs(frame_id)[0]
    image = np.array(Image.open(frame["path"]))
    for ax in axs:
        ax.cla()
        ax.imshow(image)

    for ann in COCO_GT.loadAnns(COCO_GT.getAnnIds(imgIds=frame_id)):
        mask = COCO_GT.annToMask(ann)
        show_mask(mask, ax=axs[1], category_id=ann["category_id"])
    for ann in COCO_PREDICT.loadAnns(COCO_PREDICT.getAnnIds(imgIds=frame_id)):
        mask = COCO_PREDICT.annToMask(ann)
        show_mask(mask, ax=axs[2], category_id=ann["category_id"])
    axs[0].set_title(f"Frame {frame['order_in_video']}")
    axs[1].set_title("Ground Truth")
    axs[2].set_title("Prediction")


def get_corresponding_frames(prompt_info: PromptInfo):
    frames = get_dicts_by_field_value(FRAMES, "video_id", prompt_info.video_id)
    frames = [frame for frame in frames if frame["is_det_keyframe"]]
    frames = sort_dicts_by_field(frames, "order_in_video")
    frame_ids = []
    for frame in frames:
        if frame["order_in_video"] < prompt_info.clip_range.start_idx:
            continue
        if frame["order_in_video"] > prompt_info.clip_range.end_idx:
            continue
        frame_ids.append(frame["id"])
    return frame_ids


def create_gif(image_paths, output_path):
    image_paths = natsorted(image_paths)
    images = [imageio.v3.imread(path) for path in image_paths]
    imageio.mimsave(
        output_path,
        images,
        fps=1.5,
    )


def visualize_based_on_prompt_info(prompt_info: PromptInfo):
    frame_ids = get_corresponding_frames(prompt_info)
    os.makedirs(
        os.path.join(OUTPUT_PATH, f"video_{prompt_info.video_id}"), exist_ok=True
    )
    if prompt_info.video_id not in IMAGE_PATH_FOR_GIF:
        IMAGE_PATH_FOR_GIF[prompt_info.video_id] = []

    fig, axs = plt.subplots(1, 4, figsize=(18, 3))
    plt.tight_layout()
    visualize_prompt_frame(prompt_info, axs[0])
    for frame_id in frame_ids:
        visualize_current_frame(frame_id, axs[1:])
        image_path = os.path.join(
            OUTPUT_PATH,
            f"video_{prompt_info.video_id}",
            f"{COCO_GT.loadImgs(frame_id)[0]['file_name']}",
        )
        for ax in axs:
            ax.axis("off")

        plt.savefig(image_path)
        IMAGE_PATH_FOR_GIF[prompt_info.video_id].append(image_path)
    plt.close()


def visualize(gt_path, predict_path, prompt_path):
    global COCO_GT, COCO_PREDICT, MOD, FRAMES, OUTPUT_PATH, IMAGE_PATH_FOR_GIF

    if COCO_GT is None:
        COCO_GT = COCO(gt_path)
        MOD = max(COCO_GT.getCatIds()) + 1
        FRAMES = COCO_GT.loadImgs(COCO_GT.getImgIds())
    COCO_PREDICT = COCO_GT.loadRes(predict_path)
    OUTPUT_PATH = os.path.dirname(os.path.join(os.getcwd(), predict_path))
    IMAGE_PATH_FOR_GIF = {}
    #
    logger.info(f"visualize {predict_path}")
    with open(
        prompt_path,
        "rb",
    ) as f:
        prompt_data: List[PromptInfo] = pickle.load(f)

    for prompt_info in tqdm(prompt_data):
        visualize_based_on_prompt_info(prompt_info)

    os.makedirs(os.path.join(OUTPUT_PATH, "gif"), exist_ok=True)

    logger.info("create gif")
    for video_id, image_paths in IMAGE_PATH_FOR_GIF.items():
        gif_path = os.path.join(OUTPUT_PATH, "gif", f"video_{video_id}.gif")
        create_gif(image_paths, gif_path)


if __name__ == "__main__":
    gt_path = "coco_annotations.json"
    predict_path = "/bd_byta6000i0/users/sam2/kyyang/sam2_predict/output/bbox/bbox_60_frames/predict.json"
    prompt_path = "/bd_byta6000i0/users/sam2/kyyang/sam2_predict/output/bbox/bbox_60_frames/prompt.pkl"
    visualize(gt_path, predict_path, prompt_path)
