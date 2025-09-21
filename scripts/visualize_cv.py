import os
import pickle
from typing import List

import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from natsort import natsorted
from pycocotools.coco import COCO
from tqdm import tqdm
import colorsys
import random


from utils import PromptInfo, get_dicts_by_field_value, sort_dicts_by_field

COCO_GT = None
COCO_PREDICT = None
MOD = None
FRAMES = None
OUTPUT_PATH = None
# CMAP = plt.get_cmap("tab20")


def generate_random_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.8 + random.random() * 0.2
        value = 0.8 + random.random() * 0.2
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))
    return colors


CMAP = generate_random_colors(20)
IMAGE_PATH_FOR_GIF = None

logger.add("logs/visualize.log")


def alpha_composition(overlay, image, alpha):
    """
    将前景图像、背景图像和 alpha 掩码进行混合。

    参数:
    foreground (numpy.ndarray): 前景图像，形状为 (H, W, 3)，数据类型为 np.uint8。
    background (numpy.ndarray): 背景图像，形状为 (H, W, 3)，数据类型为 np.uint8。
    alpha (numpy.ndarray): alpha 掩码，形状为 (H, W, 1)，数据类型为 np.uint8。

    返回:
    numpy.ndarray: 混合后的图像，形状为 (H, W, 3)，数据类型为 np.uint8。
    """
    # 将 uint8 转换为 float

    # 归一化 alpha 掩码，使其强度在 0 和 1 之间
    alpha = alpha.astype(float)

    outImage = overlay * alpha + image * (1 - alpha)

    return outImage.astype(np.uint8)


def show_box(box, image, category_id):
    x0, y0 = int(box[0]), int(box[1])
    x1, y1 = int(box[2]), int(box[3])

    # 从 color_map 中获取指定类别的颜色
    color = CMAP[category_id % len(CMAP)]
    color = tuple(reversed(color))

    overlay = np.zeros_like(image).astype(np.uint8)
    overlay[y0:y1, x0:x1] = color

    alpha_mask = np.zeros((image.shape[0], image.shape[1], 1), dtype=np.float32)
    alpha_mask[y0:y1, x0:x1] = 0.5  # 在矩形区域内设置透明度

    image = alpha_composition(overlay, image, alpha_mask)

    cv2.rectangle(image, (x0, y0), (x1, y1), color, 3)

    return image


def show_points(points, labels, image, category_id):
    for point, label in zip(points, labels):
        marker_color = CMAP[category_id % len(CMAP)]
        marker_color = tuple(reversed(marker_color))  # BGR 格式
        if label == 0:
            marker_size = 15
            marker_type = cv2.MARKER_TILTED_CROSS
            cv2.drawMarker(
                image,
                (int(point[0]), int(point[1])),
                (255, 255, 255),
                marker_type,
                marker_size,
                thickness=6,
            )
            cv2.drawMarker(
                image,
                (int(point[0]), int(point[1])),
                marker_color,
                marker_type,
                marker_size,
                thickness=3,
            )

        else:
            cv2.circle(
                image,
                (int(point[0]), int(point[1])),
                radius=8,
                color=marker_color,
                thickness=-1,
            )
            cv2.circle(
                image,
                (int(point[0]), int(point[1])),
                radius=9,
                color=(255, 255, 255),
                thickness=2,
            )
    return image


def show_mask(mask, image, category_id):
    # 获取颜色
    color = CMAP[category_id % len(CMAP)]
    color = tuple(reversed(color))  # BGR 格式

    mask = mask.astype(np.uint8)

    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = color

    alpha_mask = mask * 0.5
    alpha_mask = alpha_mask.reshape(image.shape[0], image.shape[1], 1)
    image = alpha_composition(colored_mask, image, alpha_mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color=color, thickness=2)

    return image


def visualize_prompt_frame(prompt_info: PromptInfo):
    image_path = prompt_info.path
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    for obj in prompt_info.prompt_objs:
        obj_id = obj.obj_id % MOD
        match prompt_info.prompt_type:
            case "points":
                image = show_points(obj.points, obj.pos_or_neg_label, image, obj_id)
            case "bbox":
                image = show_box(obj.bbox, image, obj_id)
            case "mask":
                image = show_mask(obj.mask, image, obj_id)
    return image


def visualize_current_frame(frame_id):
    frame = COCO_GT.loadImgs(frame_id)[0]
    raw_image = cv2.imread(frame["path"], cv2.IMREAD_UNCHANGED)

    images = [raw_image, raw_image.copy(), raw_image.copy()]
    # 用于存储合并后的掩码
    merged_masks_gt = {}
    merged_masks_predict = {}

    # 合并 COCO_GT 的掩码
    for ann in COCO_GT.loadAnns(COCO_GT.getAnnIds(imgIds=frame_id)):
        mask = COCO_GT.annToMask(ann)
        category_id = ann["category_id"] % MOD
        if category_id not in merged_masks_gt:
            merged_masks_gt[category_id] = mask
        else:
            merged_masks_gt[category_id] = np.logical_or(
                merged_masks_gt[category_id], mask
            )

    # 合并 COCO_PREDICT 的掩码
    for ann in COCO_PREDICT.loadAnns(COCO_PREDICT.getAnnIds(imgIds=frame_id)):
        mask = COCO_PREDICT.annToMask(ann)
        category_id = ann["category_id"] % MOD
        if category_id not in merged_masks_predict:
            merged_masks_predict[category_id] = mask
        else:
            merged_masks_predict[category_id] = np.logical_or(
                merged_masks_predict[category_id], mask
            )

    # 应用合并后的掩码到图像
    for category_id, mask in merged_masks_gt.items():
        images[1] = show_mask(mask, images[1], category_id)

    for category_id, mask in merged_masks_predict.items():
        images[2] = show_mask(mask, images[2], category_id)

    for i in range(3):
        images[i] = cv2.copyMakeBorder(
            images[i],
            20,
            20,
            10,
            10,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )

    return images


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
    if "10_" not in str(prompt_info.video_id).lower():
        return
    frame_ids = get_corresponding_frames(prompt_info)
    os.makedirs(
        os.path.join(OUTPUT_PATH, f"video_{prompt_info.video_id}"), exist_ok=True
    )
    if prompt_info.video_id not in IMAGE_PATH_FOR_GIF:
        IMAGE_PATH_FOR_GIF[prompt_info.video_id] = []

    prompt_image = visualize_prompt_frame(prompt_info)
    prompt_image = cv2.copyMakeBorder(
        prompt_image,
        20,
        20,
        10,
        10,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )

    for frame_id in frame_ids:
        current_images = visualize_current_frame(frame_id)
        image_path = os.path.join(
            OUTPUT_PATH,
            f"video_{prompt_info.video_id}",
            f"{COCO_GT.loadImgs(frame_id)[0]['file_name']}",
        )

        # Skip if the image already exists
        # if os.path.exists(image_path):
        #     logger.info(f"Skipping existing image: {image_path}")
        #     continue

        current_images.insert(0, prompt_image)
        combined_image = np.hstack(current_images)
        print(image_path)
        cv2.imwrite(image_path, combined_image)
        IMAGE_PATH_FOR_GIF[prompt_info.video_id].append(image_path)


def visualize(gt_path, predict_path, prompt_path):
    global COCO_GT, COCO_PREDICT, MOD, FRAMES, OUTPUT_PATH, IMAGE_PATH_FOR_GIF

    if COCO_GT is None:
        COCO_GT = COCO(gt_path)
        MOD = max(COCO_GT.getCatIds()) + 1
        FRAMES = COCO_GT.loadImgs(COCO_GT.getImgIds())
    try:
        COCO_PREDICT = COCO_GT.loadRes(predict_path)
    except:
        COCO_PREDICT = COCO_GT
    OUTPUT_PATH = "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2-video-training/baseline_results/endovis17/3_mem/wl"
    print(OUTPUT_PATH)
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

        # Skip if GIF already exists
        if os.path.exists(gif_path):
            logger.info(f"Skipping existing GIF: {gif_path}")
            continue

        create_gif(image_paths, gif_path)
        logger.info(f"Created GIF: {gif_path}")


if __name__ == "__main__":
    print(1)
    gt_path = "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2-video-training/data/endovis17_coco_annotations_val_closed.json"
    predict_path = "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2-video-training/baseline_results/endovis17/3_mem/eval/predict.json"
    prompt_path = "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2-video-training/baseline_results/endovis17/3_mem/eval/prompt.pkl"
    visualize(gt_path, predict_path, prompt_path)
