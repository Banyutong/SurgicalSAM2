import argparse
import os
import torch
import numpy as np
from PIL import Image
import json
from sam2.build_sam import build_sam2_video_predictor
from utils.mask_helpers import rle_to_binary_mask, get_model_cfg, mask_to_rle, mask_to_bbox
from utils.visualization import visualize_first_frame_mask, get_color_map, visualize_first_frame_bbx, \
	visualize_pixel_mask
from utils.utils import find_frames
from utils.groundtruth2point import sample_points_from_bboxes, sample_points_from_masks, sample_points_from_pixel_mask
import cv2
import matplotlib.pyplot as plt
import re


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


def main(args):
	torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

	if torch.cuda.get_device_properties(0).major >= 8:
		torch.backends.cuda.matmul.allow_tf32 = True
		torch.backends.cudnn.allow_tf32 = True

	# IF images have pixel masks in the same dir, then may need to rewrite find_frames functions and misc.py to filter masks.
	frame_names = find_frames(args.video_dir)
	model_cfg = get_model_cfg(os.path.basename(args.sam2_checkpoint))
	predictor = build_sam2_video_predictor(model_cfg, args.sam2_checkpoint)
	inference_state = predictor.init_state(video_path=args.video_dir)

	prompts = {}
	ann_frame_idx = 0  # Assuming we're using the first frame for annotation

	# first frame path
	first_frame_path = os.path.join(args.video_dir, frame_names[0])
	first_frame = Image.open(first_frame_path)
	os.makedirs(args.output_dir, exist_ok=True)
	combined_output_path = os.path.join(args.output_dir, 'first_frame_visualization.png')

	if args.gt_type == 'pixel_mask':
		gt_mask = np.array(Image.open(args.gt_path))
		sampled_points = sample_points_from_pixel_mask(gt_mask, num_points=args.sample_points, include_center=True)
		visualize_pixel_mask(first_frame, gt_mask, sampled_points, combined_output_path)
	else:
		with open(args.gt_path, 'r') as f:
			gt_data = json.load(f)

	# Process ground truth data
	if args.gt_type == 'bbox':
		bboxes = [ann['bbox'] for ann in gt_data['annotations']]
		sampled_points = sample_points_from_bboxes(bboxes, num_points=args.sample_points, include_center=True)
		visualize_first_frame_bbx(first_frame, bboxes, sampled_points, combined_output_path)

	elif args.gt_type == 'mask':
		masks = [ann['segmentation'] for ann in gt_data['annotations']]
		binary_masks = [rle_to_binary_mask(mask) for mask in masks]
		sampled_points = sample_points_from_masks(binary_masks, num_points=args.sample_points, include_center=True)
		visualize_first_frame_mask(first_frame, binary_masks, sampled_points, combined_output_path)

	# Add points for each object
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

	video_segments = {}
	for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
		video_segments[out_frame_idx] = {
			out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
			for i, out_obj_id in enumerate(out_obj_ids)
		}

	# Prepare output directories
	os.makedirs(os.path.join(args.output_dir, 'pixel_masks'), exist_ok=True)
	os.makedirs(os.path.join(args.output_dir, 'visualizations'), exist_ok=True)

	# Prepare COCO format data
	coco_annotations = []
	coco_images = []
	annotation_id = 1

	# Generate color map for objects
	object_colors = get_color_map(len(sampled_points))

	gif_frames = []
	for frame_idx, frame_name in enumerate(frame_names):
		original_img = Image.open(os.path.join(args.video_dir, frame_name))
		contour_image = np.array(original_img)

		# Save pixel masks and create COCO annotations
		for out_obj_id, out_mask in video_segments[frame_idx].items():
			# Ensure mask is 2D
			if out_mask.ndim == 3:
				out_mask = out_mask.squeeze()
			if out_mask.ndim != 2:
				print(f"Unexpected mask shape for frame {frame_idx}, object {out_obj_id}: {out_mask.shape}")
				continue

			# Save pixel mask
			mask_img = Image.fromarray((out_mask * 255).astype(np.uint8))
			mask_img.save(os.path.join(args.output_dir, 'pixel_masks', f'frame_{frame_idx:04d}_obj_{out_obj_id}.png'))

			# Create COCO annotation
			rle = mask_to_rle(out_mask)
			bbox = mask_to_bbox(out_mask)
			if bbox is not None:
				coco_ann = {
					"id": annotation_id,
					"image_id": frame_idx,
					"category_id": out_obj_id,
					"segmentation": rle,
					"area": int(np.sum(out_mask)),
					"bbox": bbox,
					"iscrowd": 0,
				}
				coco_annotations.append(coco_ann)
				annotation_id += 1

			# Create visualization
			color = [int(c * 255) for c in object_colors[out_obj_id - 1]]
			contours, _ = cv2.findContours(out_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
			cv2.drawContours(contour_image, contours, -1, color, 2)

		# Save visualization
		result = Image.fromarray(contour_image)

		if frame_idx % args.vis_frame_stride == 0:
			gif_frames.append(result)

		# Add image info to COCO format
		coco_images.append({
			"id": frame_idx,
			"file_name": frame_name,
			"height": original_img.height,
			"width": original_img.width
		})
	# Save GIF
	gif_frames[0].save(os.path.join(args.output_dir, 'visualization.gif'), save_all=True, append_images=gif_frames[1:],
	                   duration=500, loop=0)
	# Save COCO format JSON
	coco_data = {
		"images": coco_images,
		"annotations": coco_annotations,
		"categories": [{"id": i, "name": f"Object {i}"} for i in range(1, len(sampled_points) + 1)]
	}
	with open(os.path.join(args.output_dir, 'coco_annotations.json'), 'w') as f:
		json.dump(coco_data, f)

	print(f"Results saved in {args.output_dir}")


if __name__ == "__main__":
	args = parse_args()
	main(args)