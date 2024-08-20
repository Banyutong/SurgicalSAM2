import argparse
import os
import torch
import numpy as np
from PIL import Image
import json
import random

# add_all_points_(predictor, inference_state, prompt_frame_index, merged_points, class_lists)
def add_all_points_(predictor, inference_state, prompt_frame_index,  merged_points, class_lists):
	prompts = {}
	for i, (points, labels )in enumerate(zip(merged_points, class_lists)):
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




def fluctuate_point(point, beta, width, height):
    x, y = point
    dx = random.uniform(-beta, beta)
    dy = random.uniform(-beta, beta)

    new_x = max(0, min(width - 1, x + dx))
    new_y = max(0, min(height - 1, y + dy))

    return [int(new_x), int(new_y)]

def generate_negative_samples(sampled_point_classes, sampled_points, n, height, width, beta):



    class_to_points = {}
    for cls, point in zip(sampled_point_classes, sampled_points):
        if cls not in class_to_points:
            class_to_points[cls] = []
        class_to_points[cls].append(point)

    negative_sampled_points = []
    negative_sampled_point_classes = []

    for cls in set(sampled_point_classes):
        other_points = [p for c, p in zip(sampled_point_classes, sampled_points) if c != cls]

        class_negative_samples = []
        class_negative_classes = []

        for _ in range(n):
            sampled_point = random.choice(other_points)
            fluctuated_point = fluctuate_point(sampled_point, beta, width, height)
            class_negative_samples.append(fluctuated_point)

            # Instead of sampling another class, we record the current class
            class_negative_classes.append(cls)

        negative_sampled_points.append(class_negative_samples)
        negative_sampled_point_classes.append(class_negative_classes)

    return negative_sampled_points, negative_sampled_point_classes

def merge_point_lists(positive_prompt_points, negative_points):
    if len(positive_prompt_points) != len(negative_points):
        raise ValueError("Input lists must have the same length")

    merged_points = []
    class_lists = []

    for prompt_points, neg_points in zip(positive_prompt_points, negative_points):
        # Combine points
        all_points = prompt_points + neg_points
        merged_points.append(all_points)

        # Create class list
        class_list = [1] * len(prompt_points) + [0] * len(neg_points)
        class_lists.append(class_list)

    return merged_points, class_lists

def flatten_outer_list(nested_list):
    return [item for sublist in nested_list for item in sublist]