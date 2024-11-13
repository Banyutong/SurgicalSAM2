import json
import os
import pickle
import random
import tempfile
from pprint import pformat
from typing import List, Set

import numpy as np
import torch
from icecream import ic
from loguru import logger
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
from sam2.build_sam import build_sam2_video_predictor

from PromptObjNoiseAdder import PromptObjNoiseAdder
from utils import (
    ClipRange,
    PromptInfo,
    PromptObj,
    mask_to_bbox,
    mask_to_masks,
    mask_to_points,
    init_grid,
)

logger.add("dense_points/log.log")

# ic.disable()
# Enable autocast for mixed precision on CUDA devices
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# Enable TensorFloat32 (tf32) for Ampere GPUs
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


# Path to the SAM2 checkpoint and model configuration
sam2_checkpoint = (
    "/bd_byta6000i0/users/sam2/kyyang/SurgicalSAM2/checkpoints/sam2_hiera_tiny.pt"
)
model_cfg = "sam2_hiera_t.yaml"

######################
#
# torch initialize
#
######################


PROMPT_INFO = []
OUTPUT_PATH = None
VIDEO_ID_SET = set()
COCO_INFO = None
OBJ_COUNT = 0
MOD = None
NOISED_PROMPT = False
NUM_POINTS = 0
NOISE_ADDER = None
NUM_NEG_POINTS = 0
INCLUDE_CENTER = True


########################
#
# initialize the global variables
#
########################
################################################################################
#
# type definitions
#
################################################################################
def get_dicts_by_field_value(data, field_name, target_value):
    """
    Get dictionaries from a list where a specified field has a specific value.

    Args:
        data (list): List of dictionaries.
        field_name (str): Field name to check.
        target_value: Value to match.

    Returns:
        list: List of dictionaries that match the criteria.
    """
    return [item for item in data if item.get(field_name) == target_value]


def sort_dicts_by_field(data, field_name, reverse=False):
    """
    Sort a list of dictionaries by a specified field.

    Args:
        data (list): List of dictionaries.
        field_name (str): Field name to sort by.
        reverse (bool): Whether to sort in descending order.

    Returns:
        list: Sorted list of dictionaries.
    """
    return sorted(data, key=lambda item: item.get(field_name), reverse=reverse)


def get_imgs(coco_info):
    """
    Get images from COCO dataset.

    Args:
        coco_info (COCO): COCO dataset object.

    Returns:
        list: List of image information.
    """
    img_ids = coco_info.getImgIds()
    imgs = coco_info.loadImgs(img_ids)
    return imgs


################################################################################
#
# helper functions
#
################################################################################


def find_prompt_frame(frames, clip_range: ClipRange):
    """
    Find the first frame within the clip range that has annotations.

    Args:
        frames (list): List of frame information.
        clip_range (ClipRange): Range of the clip.

    Returns:
        dict: Information about the prompt frame.
    """
    clip_start = clip_range.start_idx
    clip_end = clip_range.end_idx

    for frame in frames:
        if frame["is_det_keyframe"] is False:
            continue

        if frame["order_in_video"] < clip_start or frame["order_in_video"] > clip_end:
            continue

        ann_ids = COCO_INFO.getAnnIds(imgIds=frame["id"])

        if ann_ids == []:
            continue

        return frame

    return None


def create_symbol_link_for_video(frames_info):
    """
    Create symbolic links for video frames in a temporary directory.

    Args:
        frames_info (list): List of frame information.

    Returns:
        str: Path to the temporary directory.
    """
    video_dir = tempfile.mkdtemp()
    current_dir = os.getcwd()

    for idx, frame in enumerate(frames_info):
        frame_name = str(idx).zfill(8)  # 填充到5位宽度
        dst_path = os.path.join(video_dir, f"{frame_name}.jpg")
        src_path = os.path.join(current_dir, frame["path"])
        os.symlink(src_path, dst_path)

    return video_dir


def fluctuate_point(point, beta, width, height):
    x, y = point
    dx = random.uniform(-beta, beta)
    dy = random.uniform(-beta, beta)

    new_x = max(0, min(width - 1, x + dx))
    new_y = max(0, min(height - 1, y + dy))

    return [int(new_x), int(new_y)]


def generate_negative_samples(
    sampled_points, sampled_point_classes, n, height, width, beta
):
    sampled_points = flatten_outer_list(sampled_points)
    sampled_point_classes = flatten_outer_list(sampled_point_classes)

    class_to_points = {}
    for cls, point in zip(sampled_point_classes, sampled_points):
        if cls not in class_to_points:
            class_to_points[cls] = []
        class_to_points[cls].append(point)

    negative_sampled_points = []
    negative_sampled_point_classes = []

    for cls in set(sampled_point_classes):
        other_points = [
            p for c, p in zip(sampled_point_classes, sampled_points) if c != cls
        ]

        class_negative_samples = []
        class_negative_classes = []

        for _ in range(n):
            if len(other_points) == 0:
                continue
            sampled_point = random.choice(other_points)
            fluctuated_point = fluctuate_point(sampled_point, beta, width, height)
            class_negative_samples.append(fluctuated_point)

            # Instead of sampling another class, we record the current class
            class_negative_classes.append(cls)

        negative_sampled_points.append(class_negative_samples)
        negative_sampled_point_classes.append(class_negative_classes)

    return negative_sampled_points, negative_sampled_point_classes


def flatten_outer_list(nested_list):
    return [item for sublist in nested_list for item in sublist]


def merge_point_lists(all_pos_points, negative_points):
    if len(all_pos_points) != len(negative_points):
        raise ValueError("Input lists must have the same length")

    merged_points = []
    pos_or_neg_labels = []

    for pos_points, neg_points in zip(all_pos_points, negative_points):
        # Correctly combine points by extending a new list
        all_points = []
        all_points.extend(pos_points)
        all_points.extend(neg_points)
        merged_points.append(all_points)

        # Create labels list
        labels = [1] * len(pos_points) + [0] * len(neg_points)
        pos_or_neg_labels.append(labels)

    return merged_points, pos_or_neg_labels


def get_each_obj(prompt_frame, cats: Set[int] = None, beta=10):
    """
    Extract objects from the prompt frame.

    Args:
        prompt_frame (dict): Information about the prompt frame.
        cats (Set[int]): Set of category IDs to filter by.

    Returns:
        list: List of objects.
    """
    global OBJ_COUNT
    ann_ids = COCO_INFO.getAnnIds(imgIds=prompt_frame["id"])
    anns = COCO_INFO.loadAnns(ann_ids)

    objs = []

    for ann in anns:
        if cats is not None and ann["category_id"] not in cats:
            continue
        rle = ann["segmentation"]
        raw_mask = maskUtils.decode(rle)  # 将RLE解码为二进制掩码
        masks = mask_to_masks(raw_mask)

        for mask in masks:
            obj_id = OBJ_COUNT * MOD + ann["category_id"]
            pos_points = mask_to_points(
                mask,
                num_points=NUM_POINTS,
                include_center=INCLUDE_CENTER,
            )
            negative_points = mask_to_points(
                np.logical_not(mask),
                num_points=NUM_NEG_POINTS,
                include_center=False,
            )

            pos_labels = np.ones(len(pos_points))
            neg_labels = np.zeros(len(negative_points))

            obj = PromptObj(
                mask=mask,
                bbox=mask_to_bbox(mask),
                points=np.concatenate([pos_points, negative_points]),
                obj_id=obj_id,
                pos_or_neg_label=np.concatenate([pos_labels, neg_labels]),
            )
            objs.append(obj)

            OBJ_COUNT += 1

    return objs


def get_obj_from_masks(video_segment):
    """
    Extract objects from video segments.

    Args:
        video_segment (dict): Dictionary containing video segments.

    Returns:
        list: List of objects.
    """
    objs = []
    for obj_id, obj_seg in video_segment.items():
        if obj_seg["mask"].sum() == 0:
            continue

        masks = mask_to_masks(np.squeeze(obj_seg["mask"], axis=0))
        for mask in masks:
            bbox = mask_to_bbox(mask)
            points = mask_to_points(mask)
            obj = PromptObj(
                mask=mask,
                bbox=bbox,
                points=points,
                obj_id=obj_id,
                pos_or_neg_label=np.ones(len(points)),
            )
            objs.append(obj)

    return objs


def add_prompt(
    prompt_objs: List[PromptObj],
    predictor,
    inference_state,
    prompt_frame_order_in_video,
    prompt_type,
):
    """
    Add prompts to the predictor.

    Args:
        prompt_objs (list): List of prompt objects.
        predictor (object): Predictor object.
        inference_state (object): Inference state object.
        prompt_frame_order_in_video (int): Order of the prompt frame in the video.
        prompt_type (str): Type of prompt (points, bbox, mask).

    Returns:
        tuple: Updated predictor, inference state, object IDs, and mask logits.
    """
    logger.info(f"lenght of prompt_objs: {len(prompt_objs)}")
    for obj in prompt_objs:
        if NOISED_PROMPT:
            obj = NOISE_ADDER.add_noise_to_obj(obj, prompt_type)
            if obj is None:
                continue
        match prompt_type:
            case "points":
                logger.debug(f"number of points:{len(obj.points)}")
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=prompt_frame_order_in_video,
                    obj_id=obj.obj_id,
                    points=obj.points,
                    labels=obj.pos_or_neg_label,
                )
            case "bbox":
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=prompt_frame_order_in_video,
                    obj_id=obj.obj_id,
                    box=obj.bbox,
                )
            case "mask":
                mask_tensor = torch.from_numpy(obj.mask).to(torch.bool)
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=prompt_frame_order_in_video,
                    obj_id=obj.obj_id,
                    mask=mask_tensor,
                )

    return predictor, inference_state, out_obj_ids, out_mask_logits


def predict_on_video(predictor, inference_state, start_idx):
    """
    Predict segmentation masks for the video.

    Args:
        predictor (object): Predictor object.
        inference_state (object): Inference state object.
        start_idx (int): Start index of the video.

    Returns:
        dict: Dictionary containing video segments.
    """
    video_segments = {}
    # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state, reverse=True
    ):
        video_segments[out_frame_idx + start_idx] = {
            out_obj_id: {
                "mask": (out_mask_logits[i] > 0.0).cpu().numpy(),
                "score": torch.sigmoid(out_mask_logits[i])
                .max()
                .item(),  # 使用 sigmoid 转换为概率，取最大值作为 score
            }
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        inference_state
    ):
        video_segments[out_frame_idx + start_idx] = {
            out_obj_id: {
                "mask": (out_mask_logits[i] > 0.0).cpu().numpy(),
                "score": torch.sigmoid(out_mask_logits[i])
                .max()
                .item(),  # 使用 sigmoid 转换为概率，取最大值作为 score
            }
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    return video_segments


def save_prompt_frame(
    clip_prompts: List[PromptInfo],
):
    """
    Save prompt frames.

    Args:
        clip_prompts (List[PromptInfo]): List of prompt information.
    """
    global PROMPT_INFO
    PROMPT_INFO.extend(clip_prompts)


def process_video_clip(frames, clip_prompts: List[PromptInfo], clip_range: ClipRange):
    """
    Process a video clip.

    Args:
        frames (list): List of frame information.
        clip_prompts (List[PromptInfo]): List of prompt information.
        clip_range (ClipRange): Range of the clip.

    Returns:
        dict: Dictionary containing video segments.
    """
    start_idx = clip_range.start_idx
    end_idx = clip_range.end_idx

    video_dir = create_symbol_link_for_video(frames[start_idx : end_idx + 1])

    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(video_path=video_dir)

    for prompt_info in clip_prompts:
        prompt_objs = prompt_info.prompt_objs
        prompt_frame_idx = prompt_info.frame_idx - start_idx
        prompt_type = prompt_info.prompt_type
        predictor, inference_state, out_obj_ids, out_mask_logits = add_prompt(
            prompt_objs,
            predictor,
            inference_state,
            prompt_frame_idx,
            prompt_type,
        )

    video_segments = predict_on_video(predictor, inference_state, start_idx)

    del predictor
    torch.cuda.empty_cache()

    return video_segments


def get_num_categories(frame):
    """
    Get the number of categories in a frame.

    Args:
        frame (dict): Information about the frame.

    Returns:
        set: Set of category IDs.
    """
    ann_ids = COCO_INFO.getAnnIds(imgIds=frame["id"])
    anns = COCO_INFO.loadAnns(ann_ids)
    cat_set = set()
    for ann in anns:
        cat_set.add(ann["category_id"])
    return cat_set


def generate_prompts_by_categories(frames, prompt_type: str):
    existing_cats = set()
    previous_start_idx = None
    previous_prompt_info = None
    prompt_and_ranges = []

    for frame in frames:
        if frame["is_det_keyframe"] is False:
            continue
        # ic(get_num_categories(frame))
        if get_num_categories(frame).issubset(existing_cats):
            continue

        diff_cats = get_num_categories(frame).difference(existing_cats)
        existing_cats = existing_cats.union(diff_cats)

        prompt_objs = get_each_obj(frame)

        if prompt_objs is None:
            continue

        prompt_frame_idx = frame["order_in_video"]

        prompt_info = PromptInfo(
            prompt_objs=prompt_objs,
            frame_idx=prompt_frame_idx,
            prompt_type=prompt_type,
            video_id=str(frame["video_id"]),
            path=frame["path"],
            clip_range=None,
        )

        if previous_prompt_info is None:
            previous_prompt_info = prompt_info
            previous_start_idx = prompt_frame_idx
            continue

        previous_prompt_info.clip_range = ClipRange(
            previous_start_idx, prompt_frame_idx - 1
        )
        prompt_and_ranges.append(
            (
                [previous_prompt_info],
                ClipRange(previous_start_idx, prompt_frame_idx - 1),
            ),
        )
        previous_prompt_info = prompt_info
        previous_start_idx = prompt_frame_idx

    if previous_start_idx != len(frames) - 1:
        previous_prompt_info.clip_range = ClipRange(previous_start_idx, len(frames) - 1)
        prompt_and_ranges.append(
            ([previous_prompt_info], ClipRange(previous_start_idx, len(frames) - 1))
        )

    for prompt_and_range in prompt_and_ranges:
        yield prompt_and_range


def generate_prompts_by_clip_length(frames, prompt_type, clip_length: int):
    if clip_length is None:
        clip_length = len(frames)

    current_clip_start = 0
    current_clip_end = -1
    current_clip_prompts = []

    for start_idx in range(0, len(frames), clip_length):
        end_idx = min(start_idx + clip_length - 1, len(frames) - 1)
        clip_range = ClipRange(start_idx, end_idx)

        prompt_frame = find_prompt_frame(frames, clip_range)

        if prompt_frame is None:
            logger.warning(
                f"No prompt frame found for clip {clip_range} for video {frames[0]['video_id']}"
            )
            current_clip_end = end_idx  # Extend the current clip range
            continue

        # If we found a prompt frame and we have a current clip that needs to be yielded
        if current_clip_start <= current_clip_end:
            for prompt_info in current_clip_prompts:
                prompt_info.clip_range = ClipRange(current_clip_start, current_clip_end)
            yield current_clip_prompts, ClipRange(current_clip_start, current_clip_end)
            current_clip_prompts = []  # Reset current clip prompts

        prompt_objs = get_each_obj(prompt_frame)
        current_clip_prompts.append(
            PromptInfo(
                prompt_objs=prompt_objs,
                frame_idx=prompt_frame["order_in_video"],
                prompt_type=prompt_type,
                video_id=str(prompt_frame["video_id"]),
                path=prompt_frame["path"],
                clip_range=None,
            )
        )
        current_clip_start = start_idx  # Start a new clip
        current_clip_end = end_idx

    # Yield the last clip if it exists
    if current_clip_start <= current_clip_end:
        for prompt_info in current_clip_prompts:
            prompt_info.clip_range = ClipRange(current_clip_start, current_clip_end)
        yield current_clip_prompts, ClipRange(current_clip_start, current_clip_end)


def merge_prompts(prompts_by_categories, prompts_by_clip_length):
    merged_prompts = []
    category_ranges = [
        (prompt_info, clip_range) for prompt_info, clip_range in prompts_by_categories
    ]
    clip_length_ranges = [
        (prompt_info, clip_range) for prompt_info, clip_range in prompts_by_clip_length
    ]

    ic("category_ranges")
    for _, clip_range in category_ranges:
        ic(clip_range)
    ic("clip_length_ranges")
    for _, clip_range in clip_length_ranges:
        ic(clip_range)

    range_dict = {}
    for prompt_info, clip_range in category_ranges + clip_length_ranges:
        range_dict[clip_range.start_idx] = (prompt_info, clip_range)

    # 将字典中的元素转换为列表并按 start_idx 排序
    all_ranges = sorted(range_dict.values(), key=lambda x: x[1].start_idx)

    current_start = None
    current_end = None
    current_prompts = []

    for i, (prompt_info, clip_range) in enumerate(all_ranges):
        if current_start is None:
            current_start = clip_range.start_idx
            current_end = clip_range.end_idx
            current_prompts = prompt_info

        elif clip_range.start_idx < current_end:
            for p in current_prompts:
                p.clip_range = ClipRange(current_start, clip_range.start_idx - 1)
            merged_prompts.append(
                (current_prompts, ClipRange(current_start, clip_range.start_idx - 1))
            )

            current_start = clip_range.start_idx
            current_end = clip_range.end_idx
            current_prompts = prompt_info

        else:
            for p in current_prompts:
                p.clip_range = ClipRange(current_start, current_end)
            merged_prompts.append(
                (current_prompts, ClipRange(current_start, current_end))
            )
            current_start = clip_range.start_idx
            current_end = clip_range.end_idx
            current_prompts = prompt_info

    if current_start is not None:
        for p in current_prompts:
            p.clip_range = ClipRange(current_start, current_end)
        merged_prompts.append((current_prompts, ClipRange(current_start, current_end)))
    ic("merged_prompts")
    for _, clip_range in merged_prompts:
        ic(clip_range)
    return merged_prompts


def process_singel_video(
    frames, prompt_type, clip_length: int = None, variable_cats: bool = False
):
    """
    Process a single video.

    Args:
        frames (list): List of frame information.
        prompt_type (str): Type of prompt (points, bbox, mask).
        clip_length (int): Length of the clip.
        variable_cats (bool): Whether to use variable categories for prompts.

    Returns:
        dict: Dictionary containing video segments.
    """
    global OBJ_COUNT
    OBJ_COUNT = 0

    video_segments = {}

    if variable_cats:
        prompts_by_categories = generate_prompts_by_categories(frames, prompt_type)
        prompts_by_clip_length = generate_prompts_by_clip_length(
            frames, prompt_type, clip_length
        )
        gen_clip_prompts = merge_prompts(prompts_by_categories, prompts_by_clip_length)
    else:
        gen_clip_prompts = generate_prompts_by_clip_length(
            frames, prompt_type, clip_length
        )

    for clip_prompts, clip_range in gen_clip_prompts:
        save_prompt_frame(clip_prompts)
        logger.info(clip_range)
        video_segments.update(process_video_clip(frames, clip_prompts, clip_range))

    torch.cuda.empty_cache()
    return video_segments


def process_all_videos(prompt_type, clip_length, variable_cats):
    """
    Process all videos.

    Args:
        prompt_type (str): Type of prompt (points, bbox, mask).
        clip_length (int): Length of the clip.

    Returns:
        dict: Dictionary containing all video segments.
    """
    all_video_segments = {}
    for video_id in VIDEO_ID_SET:
        logger.info(f"video_id: {video_id}")
        frames = get_dicts_by_field_value(get_imgs(COCO_INFO), "video_id", video_id)
        video_segments = process_singel_video(
            frames, prompt_type, clip_length, variable_cats
        )
        # print(video_segments)
        all_video_segments[video_id] = video_segments
        torch.cuda.empty_cache()
        free_memory, total_memory = torch.cuda.mem_get_info()
        logger.info(f"free memory: {free_memory/1024**3:.2f} GB\n")

    return all_video_segments


def save_as_coco_format(all_video_segments, save_video_list):
    """
    Save the results in COCO format.

    Args:
        all_video_segments (dict): Dictionary containing all video segments.
        save_video_list (list): The videos to save.

    Returns:
        tuple: Paths to the saved prediction and prompt files.
    """
    coco_annotations: list = []

    if save_video_list is None:
        save_video_list = VIDEO_ID_SET

    for video_id in save_video_list:
        video_segments = all_video_segments[video_id]

        frames = get_dicts_by_field_value(get_imgs(COCO_INFO), "video_id", video_id)
        frames = sort_dicts_by_field(frames, "order_in_video")

        for frame in frames:
            # if frame["is_det_keyframe"] is False:
            #     continue
            merged_mask = {}

            ## merge the mask
            for key, mask_info in video_segments[frame["order_in_video"]].items():
                remainder = key % MOD
                m_mask = np.logical_or.reduce(
                    mask_info["mask"], axis=0
                )  # 使用 mask_info['mask']
                score = mask_info["score"]  # 获取 score

                if remainder not in merged_mask:
                    merged_mask[remainder] = m_mask
                else:
                    merged_mask[remainder] = np.logical_or(
                        merged_mask[remainder], m_mask
                    )

            for key, mask in merged_mask.items():
                if mask.sum() == 0:
                    continue

                rle = maskUtils.encode(np.asfortranarray(mask))
                rle["counts"] = rle["counts"].decode("utf-8")
                annotation = {
                    "image_id": frame["id"],
                    "category_id": key,
                    "segmentation": rle,
                    "bbox": mask_to_bbox(mask),  # 添加 bbox 字段
                    "iscrowd": 0,
                    "score": score,  # 添加 score 字段
                }
                coco_annotations.append(annotation)

    predict_data = coco_annotations

    predict_path = os.path.join(OUTPUT_PATH, "predict.json")
    prompt_path = os.path.join(OUTPUT_PATH, "prompt.pkl")

    with open(predict_path, "w") as f:
        json.dump(predict_data, f, indent=4)

    with open(prompt_path, "wb") as f:
        pickle.dump(PROMPT_INFO, f)

    return predict_path, prompt_path


def inference(
    coco_path,
    output_path,
    prompt_type,
    save_video_list=None,
    clip_length=None,
    variable_cats=False,
    num_points=1,
    include_center=True,
    noised_prompt=False,
    noise_intensity=0.1,
    bbox_noise_type="shift_scale",
    num_neg_points=0,
    grid_spaceing=None,
):
    """
    Perform inference on COCO dataset.

    Args:
        coco_path (str): Path to COCO annotations file.
        output_path (str): Path to output directory.
        prompt_type (str): Type of prompt (points, bbox, mask).
        clip_length (int): Length of the clip.
        save_video_list (list): The videos to save.

    Returns:
        tuple: Paths to the saved prediction and prompt files.
    """
    global \
        OUTPUT_PATH, \
        VIDEO_ID_SET, \
        COCO_INFO, \
        MOD, \
        NOISED_PROMPT, \
        NUM_POINTS, \
        NOISE_ADDER, \
        NUM_NEG_POINTS, \
        INCLUDE_CENTER

    logger.info("Starting inference with parameters:")
    logger.info("\n" + pformat(locals(), indent=4, sort_dicts=False))

    NUM_NEG_POINTS = num_neg_points
    NUM_POINTS = num_points
    INCLUDE_CENTER = include_center
    NOISED_PROMPT = noised_prompt
    if NOISED_PROMPT:
        NOISE_ADDER = PromptObjNoiseAdder(bbox_noise_type, noise_intensity)

    OUTPUT_PATH = os.path.join("dense_points", prompt_type, output_path)
    # OUTPUT_PATH = os.path.join(output_path, prompt_type)
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    COCO_INFO = COCO(coco_path)
    MOD = max(COCO_INFO.getCatIds()) + 1

    img_ids = COCO_INFO.getImgIds()
    imgs = COCO_INFO.loadImgs(img_ids)

    if grid_spaceing is not None:
        init_grid((imgs[0]["height"], imgs[0]["width"]), grid_spaceing)

    for img in imgs:
        VIDEO_ID_SET.add(img["video_id"])

    all_videos_segments = process_all_videos(prompt_type, clip_length, variable_cats)
    predict_path, prompt_path = save_as_coco_format(
        all_videos_segments, save_video_list
    )

    return predict_path, prompt_path


if __name__ == "__main__":
    # global OUTPUT_PATH

    inference(
        coco_path="/bd_byta6000i0/users/sam2/kyyang/sam2_predict/coco_annotations.json",
        output_path="./test",
        prompt_type="points",  # bbox, mask
        clip_length=None,
        variable_cats=False,
        save_video_list=None,
        noised_prompt=False,
        grid_spaceing=None,
        num_points=int(1),
        num_neg_points=0,
    )
