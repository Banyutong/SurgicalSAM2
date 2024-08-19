import argparse
import os
import torch
import numpy as np
from PIL import Image
import json
from sam2.build_sam import build_sam2_video_predictor
from utils.mask_helpers import rle_to_binary_mask, get_model_cfg
from utils.visualization import visualize_first_frame_comprehensive, get_color_map, visualize_all_frames
from utils.utils import find_frames, process_gt_pixel_mask, get_class_to_color_mapping
from utils.groundtruth2point import sample_points_from_bboxes, sample_points_from_masks, sample_points_from_pixel_mask
from utils.output_utils import save_pixel_masks, create_coco_annotations, save_visualizations, save_coco_json

from pycocotools.coco import COCO

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
	parser.add_argument('--negative_sample_points', type=int, default=2, help='Number of negative points to sample for each object')
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
	if args.gt_type == 'pixel_mask':
		gt_data = process_gt_pixel_mask(frame_names, args.gt_path)
		gt_mask = np.array(Image.open(args.gt_path))
		sampled_points = sample_points_from_pixel_mask(gt_mask, num_points=args.sample_points, include_center=True)

		return gt_data, sampled_points, 0  # Assuming the first frame is always valid for pixel_mask

	with open(args.gt_path, 'r') as f:
		gt_data = json.load(f)

	coco = COCO(args.gt_path)

	gt_data_filtered = []
	sampled_points = None
	first_valid_frame_index = None

	for i, frame_name in enumerate(frame_names):
		image_id = extract_image_info(gt_data, frame_name)

		if image_id is None or not coco.getAnnIds(imgIds=image_id):
			print(f"Warning: No corresponding gt found for {frame_name}")
			gt_data_filtered.append([])  # Empty list for both bbox and mask
			continue

		print(f"Ground truth found for {frame_name}")

		ann_ids = coco.getAnnIds(imgIds=image_id)
		anns = coco.loadAnns(ann_ids)

		if args.gt_type == 'bbox':
			frame_gt_data = [ann['bbox'] for ann in anns]
			sample_func = sample_points_from_bboxes
		elif args.gt_type == 'mask':
			masks = [ann['segmentation'] for ann in anns]
			frame_gt_data = [rle_to_binary_mask(mask) for mask in masks]
			sample_func = sample_points_from_masks
		else:
			raise ValueError(f"Unsupported gt_type: {args.gt_type}")

		gt_data_filtered.append(frame_gt_data)  # Append as a list for each frame

		if sampled_points is None:
			sampled_points = sample_func(
				frame_gt_data,
				num_points=args.sample_points,
				include_center=True
			)
			first_valid_frame_index = i

	return gt_data_filtered, sampled_points, first_valid_frame_index


def add_positive_points_(predictor, inference_state, prompt_frame_index,  sampled_points):
	prompts = {}
	for i, points in enumerate(sampled_points):
		labels = np.array([1 for _ in range(len(points))], dtype=np.int32) # 1 for positive
		# obj_id = sampled_points_classes[i]
		# labels = np.array([1, 1], np.int32)
		obj_id = i
		prompts[obj_id] = points, labels
		points = np.array(points, dtype=np.float32)
		predictor.add_new_points_or_box(
			inference_state=inference_state,
			frame_idx=prompt_frame_index,
			obj_id=obj_id,
			points= points,
			labels= labels,
		)
		k=1
		# if i == 1: #todo
		# 	continue
		# if i == 2:
		# 	break




def add_negative_points_(predictor, inference_state, prompt_frame_index,  sampled_points, n_each_class):
	prompts = {}
	for i, points in enumerate(sampled_points):
		labels = np.array([0 for _ in range(len(points))], dtype=np.int32)
		# obj_id = sampled_points_classes[i]
		# labels = np.array([1, 1], np.int32)
		obj_id = i
		prompts[obj_id] = points, labels
		points = np.array(points, dtype=np.float32)
		predictor.add_new_points_or_box(
			inference_state=inference_state,
			frame_idx=prompt_frame_index,
			obj_id=obj_id,
			points= points,
			labels= labels,
		)


import random

import random


def generate_negative_samples(sampled_point_classes, sampled_points, n):
	class_to_points = {}
	for cls, point in zip(sampled_point_classes, sampled_points):
		if cls not in class_to_points:
			class_to_points[cls] = []
		class_to_points[cls].append(point)

	negative_sampled_points = []
	negative_sampled_point_classes = []

	for cls in set(sampled_point_classes):
		other_points = [p for c, p in zip(sampled_point_classes, sampled_points) if c != cls]

		# Sample n points for this class
		class_negative_samples = random.sample(other_points, k=n)
		negative_sampled_points.append(class_negative_samples)

		# Sample classes for these points
		other_classes = [c for c in set(sampled_point_classes) if c != cls]
		class_negative_classes = [random.choice(other_classes) for _ in range(n)]
		negative_sampled_point_classes.append(class_negative_classes)

	return negative_sampled_points, negative_sampled_point_classes


def main(args):
	setup_environment()

	frame_names = find_frames(args.video_dir)
	model_cfg = get_model_cfg(os.path.basename(args.sam2_checkpoint))
	predictor = build_sam2_video_predictor(model_cfg, args.sam2_checkpoint)

	inference_state = predictor.init_state(video_path=args.video_dir)
	try:
		gt_data, sampled_points, first_valid_frame_index = process_ground_truth(args, frame_names)
		print(f"Using ground truth from frame index: {first_valid_frame_index}")
	except ValueError as e:
		print(f"Error: {e}")
		return

	prompt_frame_index = first_valid_frame_index

	if args.gt_type =="pixel_mask":
		info = sampled_points
		prompt_points = info["sampled_points"]
		sampled_point_classes = info["classes_points"]
		class_to_color_mapper = info["class_to_color_mapper"]
		# Create a dictionary to store points by class
		points_by_class = {}
		classes_by_class = {}

		for point, class_label in zip(prompt_points, sampled_point_classes):
			if class_label not in points_by_class:
				points_by_class[class_label] = []
				classes_by_class[class_label] = []
			points_by_class[class_label].append(point)
			classes_by_class[class_label].append(class_label)

		# Convert the dictionaries to the desired format
		merged_prompt_points = [[points_by_class[class_label]] for class_label in sorted(points_by_class.keys())]
		merged_point_classes = [classes_by_class[class_label] for class_label in sorted(classes_by_class.keys())]

		# Flatten the merged_prompt_points list by one level
		merged_prompt_points = [point for sublist in merged_prompt_points for point in sublist]

		add_positive_points_(predictor, inference_state, prompt_frame_index, merged_prompt_points)

		# negative_points, negative_classes = generate_negative_samples(sampled_point_classes, prompt_points, args.negative_sample_points)
		# add_negative_points_(predictor, inference_state, prompt_frame_index, negative_points)
	else:
		add_positive_points_(predictor, inference_state, prompt_frame_index, sampled_points)
		class_to_color_mapper=None

	# official way
	video_segments = {}
	for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
		video_segments[out_frame_idx] = {
			out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
			for i, out_obj_id in enumerate(out_obj_ids)
		}

	os.makedirs(args.output_dir, exist_ok=True)

	# Visualize prompt frame
	prompt_frame_path = os.path.join(args.video_dir, frame_names[prompt_frame_index])
	prompt_frame_img = np.array(Image.open(prompt_frame_path))
	prompt_frame_predictions = np.zeros_like(prompt_frame_img[:,:,0])
	for obj_id, mask in video_segments[prompt_frame_index].items():
		# if class_labels is not None:
		#     obj_id = class_labels[obj_id]
		prompt_frame_predictions[mask] = obj_id



	visualize_all_frames(video_segments, frame_names, args.video_dir, args.output_dir, gt_data, prompt_frame_index, prompt_points, args.gt_type, class_to_color_mapper)
	# Save outputs
	save_pixel_masks(video_segments, args.output_dir)
	coco_annotations, coco_images = create_coco_annotations(video_segments, frame_names)
	object_colors = get_color_map(len(sampled_points))
	#save_visualizations(video_segments, frame_names, args.video_dir, args.output_dir, object_colors, args.vis_frame_stride)
	save_coco_json(coco_annotations, coco_images, len(sampled_points), args.output_dir)

	print(f"Results saved in {args.output_dir}")

if __name__ == "__main__":
	args = parse_args()
	main(args)