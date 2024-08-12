import argparse
import os
import torch
import numpy as np
from PIL import Image
import json
from sam2.build_sam import build_sam2_video_predictor
from utils.mask_helpers import rle_to_binary_mask, get_model_cfg
from utils.visualization import visualize_first_frame_comprehensive, get_color_map
from utils.utils import find_frames
from utils.groundtruth2point import sample_points_from_bboxes, sample_points_from_masks, sample_points_from_pixel_mask
from utils.output_utils import save_pixel_masks, create_coco_annotations, save_visualizations, save_coco_json

def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 Video Segmentation")
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing video frames')
    parser.add_argument('--sam2_checkpoint', type=str, required=True, help='Path to SAM2 checkpoint')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    parser.add_argument('--vis_frame_stride', type=int, default=15, help='Stride for visualization frames')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth data')
    parser.add_argument('--gt_type', type=str, choices=['bbox', 'mask', 'pixel_mask'], required=True,
                        help='Type of ground truth (bbox, mask, or pixel_mask)')
    parser.add_argument('--sample_points', type=int, default=1, help='Number of points to sample for each object')
    return parser.parse_args()

def setup_environment():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def process_ground_truth(args):
    if args.gt_type == 'pixel_mask':
        gt_mask = np.array(Image.open(args.gt_path))
        sampled_points = sample_points_from_pixel_mask(gt_mask, num_points=args.sample_points, include_center=True)
        gt_data = gt_mask
    else:
        with open(args.gt_path, 'r') as f:
            gt_data = json.load(f)

        if args.gt_type == 'bbox':
            gt_data = [ann['bbox'] for ann in gt_data['annotations']]
            sampled_points = sample_points_from_bboxes(gt_data, num_points=args.sample_points, include_center=True)
        elif args.gt_type == 'mask':
            masks = [ann['segmentation'] for ann in gt_data['annotations']]
            gt_data = [rle_to_binary_mask(mask) for mask in masks]
            sampled_points = sample_points_from_masks(gt_data, num_points=args.sample_points, include_center=True)

    return gt_data, sampled_points

def initialize_predictor(args, frame_names, sampled_points):
    model_cfg = get_model_cfg(os.path.basename(args.sam2_checkpoint))
    predictor = build_sam2_video_predictor(model_cfg, args.sam2_checkpoint)
    inference_state = predictor.init_state(video_path=args.video_dir)

    prompts = {}
    ann_frame_idx = 0  # Assuming we're using the first frame for annotation

    for obj_id, points in enumerate(sampled_points, start=1):
        labels = np.ones(args.sample_points, dtype=np.int32)  # All points are positive
        prompts[obj_id] = points, labels
        predictor.add_new_points(
            inference_state=inference_state,
            frame_idx=ann_frame_idx,
            obj_id=obj_id,
            points=np.array(points, dtype=np.float32),
            labels=labels,
        )

    return predictor, inference_state

def main(args):
    setup_environment()

    frame_names = find_frames(args.video_dir)
    gt_data, sampled_points = process_ground_truth(args)
    predictor, inference_state = initialize_predictor(args, frame_names, sampled_points)

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    os.makedirs(args.output_dir, exist_ok=True)

    # Visualize first frame
    first_frame_path = os.path.join(args.video_dir, frame_names[0])
    first_frame = np.array(Image.open(first_frame_path))
    first_frame_predictions = np.zeros_like(first_frame[:,:,0])
    for obj_id, mask in video_segments[0].items():
        first_frame_predictions[mask] = obj_id

    combined_output_path = os.path.join(args.output_dir, 'first_frame_visualization.png')
    visualize_first_frame_comprehensive(
        first_frame,
        gt_data,
        sampled_points,
        first_frame_predictions,
        combined_output_path,
        args.gt_type
    )

    # Save outputs
    save_pixel_masks(video_segments, args.output_dir)
    coco_annotations, coco_images = create_coco_annotations(video_segments, frame_names)
    object_colors = get_color_map(len(sampled_points))
    save_visualizations(video_segments, frame_names, args.video_dir, args.output_dir, object_colors, args.vis_frame_stride)
    save_coco_json(coco_annotations, coco_images, len(sampled_points), args.output_dir)

    print(f"Results saved in {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)