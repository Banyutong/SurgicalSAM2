import argparse
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from pycocotools.coco import COCO
from sam2.build_sam import build_sam2_video_predictor
from utils.mask_helpers import get_model_cfg
from utils.visualization_bbox import  visualize_all_frames, get_color_map
from utils.utils import find_frames
from utils.output_utils import save_pixel_masks, create_coco_annotations, save_visualizations, save_coco_json


def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 Video Segmentation with Bounding Box")
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing video frames')
    parser.add_argument('--sam2_checkpoint', type=str, required=True, help='Path to SAM2 checkpoint')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    parser.add_argument('--vis_frame_stride', type=int, default=15, help='Stride for visualization frames')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth COCO JSON')
    return parser.parse_args()


def setup_environment():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def extract_image_info(gt_data, image_filename):
    video_id, frame_id = image_filename.split('.')[0].split('_')
    image_id = int(f"{video_id}0{frame_id}")

    for img in gt_data.get('images', []):
        if img['id'] == image_id:
            return img['id']
    return None

def process_ground_truth(args, frame_names):
    with open(args.gt_path, 'r') as f:
        gt_data = json.load(f)
    coco = COCO(args.gt_path)
    gt_data_filtered = []
    first_valid_frame_index = None

    for i, frame_name in enumerate(frame_names):
        image_id = extract_image_info(gt_data, frame_name)
        if image_id is None or not coco.getAnnIds(imgIds=image_id):
            print(f"Warning: No corresponding gt found for {frame_name}")
            gt_data_filtered.append([])  # Empty list for bbox
            continue
        print(f"Ground truth found for {frame_name}")
        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)
        frame_gt_data = [ann['bbox'] for ann in anns]
        gt_data_filtered.append(frame_gt_data)
        if first_valid_frame_index is None:
            first_valid_frame_index = i
    return gt_data_filtered, first_valid_frame_index


def initialize_predictor(args, frame_names, gt_data, first_valid_frame_index):
    model_cfg = get_model_cfg(os.path.basename(args.sam2_checkpoint))
    predictor = build_sam2_video_predictor(model_cfg, args.sam2_checkpoint)
    inference_state = predictor.init_state(video_path=args.video_dir)

    ann_frame_idx = first_valid_frame_index
    first_frame_bboxes = gt_data[first_valid_frame_index]

    all_video_res_masks = []
    for obj_id, bbox in enumerate(first_frame_bboxes, start=1):
        x, y, w, h = bbox
        box = np.array([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max] format

        try:
            frame_idx, obj_ids, video_res_masks = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=obj_id,
                box=box,
            )
            all_video_res_masks.extend(video_res_masks)
        except Exception as e:
            print(f"Error in add_new_points_or_box for object {obj_id}: {str(e)}")
            raise

    return predictor, inference_state, frame_idx, len(first_frame_bboxes), all_video_res_masks


def main(args):
    setup_environment()

    frame_names = find_frames(args.video_dir)
    gt_data, first_valid_frame_index = process_ground_truth(args, frame_names)
    predictor, inference_state, init_frame_idx, num_objects, init_video_res_masks = initialize_predictor(args,
                                                                                                         frame_names,
                                                                                                         gt_data,
                                                                                                         first_valid_frame_index)

    video_segments = {init_frame_idx: {obj_id: mask.cpu().numpy().squeeze() for obj_id, mask in
                                       zip(range(1, num_objects + 1), init_video_res_masks)}}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    os.makedirs(args.output_dir, exist_ok=True)
    # Visualize all frames
    visualize_all_frames(video_segments, frame_names, args.video_dir, args.output_dir, gt_data, first_valid_frame_index,
                          gt_type="bbox", show_first_frame=True, show_points=False)

    # Save outputs
    save_pixel_masks(video_segments, args.output_dir)
    coco_annotations, coco_images = create_coco_annotations(video_segments, frame_names)

    # Determine the maximum object ID in video_segments
    max_obj_id = max(max(segment.keys()) for segment in video_segments.values())
    object_colors = get_color_map(max_obj_id)
    save_visualizations(video_segments, frame_names, args.video_dir, args.output_dir, object_colors,
                        args.vis_frame_stride)
    save_coco_json(coco_annotations, coco_images, max_obj_id, args.output_dir)

    print(f"Results saved in {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)