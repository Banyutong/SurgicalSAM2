import tempfile
import json
import os
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
import matplotlib.pyplot as plt
from icecream import ic
import cv2
import numpy as np
from copy import deepcopy
import uuid
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import (
    show_mask,
    show_points,
    show_box,
    mask_to_masks,
    mask_to_bbox,
    mask_to_points,
)
from visualization import visualize_all_frames, visualize_first_frame_comprehensive

# Enable autocast for mixed precision on CUDA devices
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# Enable TensorFloat32 (tf32) for Ampere GPUs
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor

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


def find_prompt_frame(video_info, video_order, coco_info, clip_start, clip_end):
    """
    Find the first frame within the clip range that has annotations.

    Args:
        video_info (dict): Information about the video.
        video_order (int): Order of the video.
        coco_info (COCO): COCO dataset object.
        clip_start (int): Start index of the clip.
        clip_end (int): End index of the clip.

    Returns:
        dict: Information about the prompt frame.
    """
    img_ids = coco_info.getImgIds()
    imgs = coco_info.loadImgs(img_ids)

    prompt_frame = None  # Initialize prompt_frame to None

    for img in imgs:
        if img["order_in_video"] < clip_start or img["order_in_video"] > clip_end:
            continue

        ann_ids = coco_info.getAnnIds(imgIds=img["id"])
        if ann_ids == []:
            continue

        if img["video_id"] == video_info[video_order]["video_id"]:
            prompt_frame = img
            break

    return prompt_frame


def create_symbol_link_for_video(frames_info):
    """
    Create symbolic links for video frames in a temporary directory.

    Args:
        frames_info (list): List of frame information.

    Returns:
        str: Path to the temporary directory.
    """
    video_dir = tempfile.mkdtemp()

    for idx, frame in enumerate(frames_info):
        frame_name = formatted_number = str(idx).zfill(8)  # 填充到5位宽度
        dst_path = os.path.join(video_dir, f"{frame_name}.jpg")
        src_path = frame["path"]
        os.symlink(src_path, dst_path)

    return video_dir


def get_each_obj(prompt_frame, coco_info, num_points=1):
    """
    Extract objects from the prompt frame.

    Args:
        prompt_frame (dict): Information about the prompt frame.
        coco_info (COCO): COCO dataset object.
        num_points (int): Number of points to extract.

    Returns:
        list: List of objects.
    """
    ann_ids = coco_info.getAnnIds(imgIds=prompt_frame["id"])
    anns = coco_info.loadAnns(ann_ids)
    num_cat = len(coco_info.cats)

    obj_count = 0
    objs = []

    for ann in anns:
        rle = ann["segmentation"]
        mask = maskUtils.decode(rle)  # 将RLE解码为二进制掩码
        masks = mask_to_masks(mask)

        for mask in masks:
            obj = {
                "mask": mask,
                "bbox": mask_to_bbox(mask),
                "points": mask_to_points(mask, num_points),
                "obj_id": obj_count * (num_cat + 1) + ann["category_id"],
            }
            objs.append(obj)
            obj_count += 1

    return objs


def add_prompt(
    prompt_objs,
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
    for obj in prompt_objs:
        match prompt_type:
            case "points":
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=prompt_frame_order_in_video,
                    obj_id=obj["obj_id"],
                    points=obj["points"],
                    labels=np.ones(len(obj["points"])),
                )
            case "bbox":
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=prompt_frame_order_in_video,
                    obj_id=obj["obj_id"],
                    box=obj["bbox"],
                )
            case "mask":
                mask_tensor = torch.from_numpy(obj["mask"]).to(torch.bool)
                _, out_obj_ids, out_mask_logits = predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=prompt_frame_order_in_video,
                    obj_id=obj["obj_id"],
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
    frame_info,
    prompt_objects,
    prompt_type,
    out_obj_ids,
    out_mask_logits,
    num_cat,
    output_path,
):
    """
    Save the prompt frame and prediction results.

    Args:
        frame_info (dict): Information about the frame.
        prompt_objects (list): List of prompt objects.
        prompt_type (str): Type of prompt (points, bbox, mask).
        out_obj_ids (list): List of object IDs.
        out_mask_logits (list): List of mask logits.
        num_cat (int): Number of categories.
        output_path (str): Path to save the output.
    """
    # Load the image
    image = cv2.imread(frame_info["path"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a figure and axis for the prompt frame and prediction results
    fig, axs = plt.subplots(1, 2, figsize=(24, 8))

    # Display the image in the first subplot (prompt frame)
    axs[0].imshow(image)
    axs[0].axis("off")

    # Show masks in the first subplot
    for obj in prompt_objects:
        show_mask(obj["mask"], axs[0], obj_id=obj["obj_id"], random_color=True)

    # Show points or box based on prompt_type in the first subplot
    for obj in prompt_objects:
        if prompt_type == "points":
            show_points(obj["points"], np.ones(len(obj["points"])), axs[0])
        elif prompt_type == "bbox":
            show_box(obj["bbox"], axs[0])

    # Display the image in the second subplot (prediction results)
    axs[1].imshow(image)
    axs[1].axis("off")

    # Show masks from prediction results in the second subplot
    for i, obj_id in enumerate(out_obj_ids):
        mask = (out_mask_logits[i] > 0.0).cpu().numpy()
        show_mask(mask, axs[1], obj_id=obj_id, random_color=True)

    # Save the figure for the prompt frame and prediction results
    output_file = os.path.join(output_path, f"combined_frame_{frame_info['file_name']}")
    plt.savefig(output_file, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def process_video_clip(
    video_info, video_order, coco_info, prompt_type, start_idx, end_idx, output_path
):
    """
    Process a video clip.

    Args:
        video_info (dict): Information about the video.
        video_order (int): Order of the video.
        coco_info (COCO): COCO dataset object.
        prompt_type (str): Type of prompt (points, bbox, mask).
        start_idx (int): Start index of the clip.
        end_idx (int): End index of the clip.
        output_path (str): Path to save the output.

    Returns:
        dict: Dictionary containing video segments.
    """
    video_dir = create_symbol_link_for_video(
        video_info[video_order]["frames"][start_idx : end_idx + 1]
    )
    prompt_frame = find_prompt_frame(
        video_info, video_order, coco_info, start_idx, end_idx
    )

    if prompt_frame is None:
        return {}

    prompt_objs = get_each_obj(prompt_frame, coco_info)
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(video_path=video_dir)

    predictor, inference_state, out_obj_ids, out_mask_logits = add_prompt(
        prompt_objs,
        predictor,
        inference_state,
        prompt_frame["order_in_video"] - start_idx,
        prompt_type,
    )

    video_output_path = os.path.join(
        output_path, f"video_{video_info[video_order]["video_id"]}"
    )
    os.makedirs(video_output_path, exist_ok=True)
    save_prompt_frame(
        video_info[video_order]["frames"][prompt_frame["order_in_video"]],
        prompt_objs,
        prompt_type,
        out_obj_ids,
        out_mask_logits,
        len(coco_info.cats),
        video_output_path,
    )

    video_segments = predict_on_video(predictor, inference_state, start_idx)

    del predictor
    torch.cuda.empty_cache()

    return video_segments


def process_singel_video(
    video_info, video_order, coco_info, prompt_type, clip_length, output_path
):
    """
    Process a single video.

    Args:
        video_info (dict): Information about the video.
        video_order (int): Order of the video.
        coco_info (COCO): COCO dataset object.
        prompt_type (str): Type of prompt (points, bbox, mask).
        clip_length (int): Length of the clip.
        output_path (str): Path to save the output.

    Returns:
        dict: Dictionary containing video segments.
    """
    video_segments = {}

    if clip_length is None:
        clip_length = len(video_info[video_order]["frames"])

    for start_idx in range(0, len(video_info[video_order]["frames"]), clip_length):
        end_idx = min(
            start_idx + clip_length - 1, len(video_info[video_order]["frames"]) - 1
        )

        ic(start_idx, end_idx)

        video_segments.update(
            process_video_clip(
                video_info,
                video_order,
                coco_info,
                prompt_type,
                start_idx,
                end_idx,
                output_path,
            )
        )

    return video_segments


def process_all_videos(video_info, coco_info, prompt_type, output_path):
    """
    Process all videos.

    Args:
        video_info (dict): Information about the videos.
        coco_info (COCO): COCO dataset object.
        prompt_type (str): Type of prompt (points, bbox, mask).
        output_path (str): Path to save the output.

    Returns:
        dict: Dictionary containing all video segments.
    """
    all_video_segments = {}
    for video_order in range(len(video_info)):
        video_segments = process_singel_video(
            video_info, video_order, coco_info, prompt_type, 30, output_path
        )
        all_video_segments[video_order] = video_segments
    return all_video_segments


def save_as_coco_format(all_video_segments, video_info, save_video_list ,coco_info, output_path):
    """
    Save the results in COCO format.

    Args:
        all_video_segments (dict): Dictionary containing all video segments.
        video_info (dict): Information about the videos.
        save_video_list(list): The videos to save.
        coco_info (COCO): COCO dataset object.
        output_path (str): Path to save the output.
    """
    ann_total = 0
    coco_annotations: list = []
    num_cat: int = len(coco_info.cats)

    for video_order in save_video_list:
        video_segments = all_video_segments[video_order]

        for frame_id in range(len(video_info[video_order]["frames"])):
            current_frame = video_info[video_order]["frames"]
            if current_frame[frame_id]["id"] == None:
                continue

            merged_mask = {}

            ## merge the mask
            for key, mask_info in video_segments[frame_id].items():
                remainder = key % (num_cat + 1)
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
                    rle = maskUtils.encode(np.asfortranarray(mask))
                    rle["counts"] = rle["counts"].decode("utf-8")
                    annotation = {
                        "id": ann_total,
                        "image_id": current_frame[frame_id]["id"],
                        "category_id": key,
                        "segmentation": rle,
                        "bbox": mask_to_bbox(mask),
                        "area": int(np.sum(mask)),
                        "iscrowd": 0,
                        "score": score,  # 添加 score 字段
                    }
                    coco_annotations.append(annotation)
                    ann_total += 1

    img_ids = coco_info.getImgIds()
    cat_ids = coco_info.getCatIds()

    coco_images = coco_info.loadImgs(img_ids)
    coco_cats = coco_info.loadCats(cat_ids)
    predict_data = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_cats,
    }

    with open(os.path.join(output_path, "predict.json"), "w") as f:
        json.dump(predict_data, f, indent=4)


if __name__ == "__main__":
    with open("endoscapes_video.json", "r") as f:
        video_info = json.load(f)
    annotation_file = "coco_annotations.json"

    coco_info = COCO(annotation_file)

    output_path = "./"
    prompt_type = "points"

    output_path = os.path.join(output_path, "output", prompt_type)
    os.makedirs(output_path, exist_ok=True)

    all_videos_segments = process_all_videos(
        [video_info[0]], coco_info, prompt_type, output_path
    )

    save_as_coco_format(all_videos_segments, video_info,[0] ,coco_info, output_path)
