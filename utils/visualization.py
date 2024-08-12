import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import colorsys
from itertools import groupby
from skimage import measure
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pycocotools.mask as mask_util
import random


def show_mask(mask, ax, obj_id=None, random_color=False):
	if random_color:
		color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
	else:
		cmap = plt.get_cmap("tab10")
		cmap_idx = 0 if obj_id is None else obj_id
		color = np.array([*cmap(cmap_idx)[:3], 0.6])
	h, w = mask.shape[-2:]
	mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
	ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
	pos_points = coords[labels == 1]
	neg_points = coords[labels == 0]
	ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
	           linewidth=1.25)
	ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
	           linewidth=1.25)


def get_color_map(num_classes):
	"""Generate a color map for visualizing different objects."""
	colors = []
	for i in range(num_classes):
		hue = i / num_classes
		rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
		colors.append(rgb)  # Keep as float values in range 0-1
	return colors


def ensure_color_range(color):
	"""Ensure color values are within 0-1 range."""
	return tuple(max(0, min(c, 1)) for c in color)


def visualize_first_frame_bbx(image, bboxes, sampled_points, output_path):
	# print(f"Number of bboxes: {len(bboxes)}")
	# print(f"Number of sampled points: {len(sampled_points)}")

	fig, ax = plt.subplots(figsize=(10, 10))
	ax.imshow(image)

	colors = get_color_map(len(bboxes))
	# print(f"Generated colors: {colors}")

	for i, (bbox, points) in enumerate(zip(bboxes, sampled_points)):
		color = ensure_color_range(colors[i])
		# print(f"Color for bbox {i}: {color}")
		# print(f"Color type: {type(color)}")
		# print(f"Color values: R={color[0]}, G={color[1]}, B={color[2]}")

		# Draw bounding box
		x, y, w, h = bbox
		# print(f"Bbox {i}: x={x}, y={y}, w={w}, h={h}")
		rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=2)
		ax.add_patch(rect)

		# Plot sampled points
		points = np.array(points)
		# print(f"Points for bbox {i}: {points}")
		ax.scatter(points[:, 0], points[:, 1], c=[color], s=50)

	ax.axis('off')
	plt.tight_layout()
	plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
	plt.close()
	print(f"First frame visualization (bboxes and points) saved to {output_path}")


# If you're using this color map elsewhere in your code, make sure to use ensure_color_range there as well
def get_color_map_255(num_classes):
	"""Generate a color map for visualizing different objects, with values in 0-255 range."""
	colors = get_color_map(num_classes)
	return [(int(r * 255), int(g * 255), int(b * 255)) for r, g, b in colors]


def visualize_first_frame_mask(image, masks, sampled_points, output_path):
	# Convert image to RGB if it's not already
	if isinstance(image, str):
		image = cv2.imread(image)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	elif isinstance(image, Image.Image):
		image = np.array(image)

	# Create a copy of the image for drawing filled contours
	filled_image = image.copy()

	# Generate a color map for the masks
	colors = get_color_map(len(masks))

	# Create an overlay for filled contours
	overlay = np.zeros_like(filled_image)

	# Loop over each mask and fill the contour
	for i, (mask, points) in enumerate(zip(masks, sampled_points)):
		color = [int(c * 255) for c in colors[i]]

		# Find contours of the mask
		contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# Fill contours on the overlay
		cv2.fillPoly(overlay, contours, color)

		# Plot sampled points
		for point in points:
			cv2.circle(filled_image, tuple(map(int, point)), 5, color, -1)

	# Blend the filled contours with the original image
	alpha = 0.3  # Adjust this value to change the transparency of the filled areas
	filled_image = cv2.addWeighted(filled_image, 1 - alpha, overlay, alpha, 0)

	# Visualize the image with the filled contours
	plt.figure(figsize=(10, 10))
	plt.imshow(filled_image)
	plt.title("First Frame with Filled Mask Contours and Sampled Points")
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
	plt.close()

	print(f"First frame visualization saved to {output_path}")


def get_color_map(n):
	def hsv2rgb(h, s, v):
		return tuple(round(i * 255) for i in colorsys.hsv_to_rgb(h, s, v))

	return [hsv2rgb(i / n, 1, 1) for i in range(n)]


def rle_to_binary_mask(rle):
	if isinstance(rle, dict):
		return mask_util.decode(rle)
	elif isinstance(rle, list):
		# If RLE is in COCO format (list of counts)
		h, w = rle['size'] if 'size' in rle else (0, 0)
		rle_dict = {'counts': rle, 'size': [h, w]}
		return mask_util.decode(rle_dict)
	else:
		raise ValueError("Unsupported RLE format")


def mask_to_rle(binary_mask):
	rle = mask_util.encode(np.asfortranarray(binary_mask))
	rle['counts'] = rle['counts'].decode('utf-8')
	return rle


def mask_to_bbox(mask):
	pos = np.where(mask)
	if len(pos[0]) == 0:
		return None
	xmin, ymin = np.min(pos[1]), np.min(pos[0])
	xmax, ymax = np.max(pos[1]), np.max(pos[0])
	return [float(xmin), float(ymin), float(xmax - xmin + 1), float(ymax - ymin + 1)]


def visualize_pixel_mask(image, pixel_mask, points, output_path):
	"""
    Visualize a pixel mask (color or grayscale) with sampled points.

    Args:
    image (PIL.Image.Image): The original image.
    pixel_mask (np.ndarray): A 2D or 3D numpy array representing the pixel mask.
    points (list): List of lists containing sampled points for each object.
    output_path (str): Path to save the visualization.
    """
	plt.figure(figsize=(12, 6))

	# Plot original image
	plt.subplot(1, 2, 1)
	plt.imshow(image)
	plt.title("Original Image")
	plt.axis('off')

	# Plot mask
	plt.subplot(1, 2, 2)
	if pixel_mask.ndim == 2:
		plt.imshow(pixel_mask, cmap='tab20')
	else:
		plt.imshow(pixel_mask)
	plt.title("Pixel Mask with Sampled Points")
	plt.axis('off')

	# Plot points
	for obj_points in points:
		obj_points = np.array(obj_points)
		plt.scatter(obj_points[:, 1], obj_points[:, 0], c='red', s=30, marker='x')

	plt.tight_layout()
	plt.savefig(output_path)
	plt.close()


def create_mask_overlay(image, pixel_mask):
	"""
    Create an overlay of the pixel mask on the original image.

    Args:
    image (PIL.Image.Image): The original image.
    pixel_mask (np.ndarray): A 2D or 3D numpy array representing the pixel mask.

    Returns:
    PIL.Image.Image: The image with mask overlay.
    """
	image_array = np.array(image)

	if pixel_mask.ndim == 2:
		# For grayscale masks, create a color map
		color_map = plt.get_cmap('tab20')
		unique_labels = np.unique(pixel_mask)
		colored_mask = np.zeros((*pixel_mask.shape, 4))
		for label in unique_labels[1:]:  # Skip background
			colored_mask[pixel_mask == label] = color_map(label / len(unique_labels))
	else:
		# For color masks, use the mask as is
		colored_mask = pixel_mask / 255.0  # Normalize to [0, 1]
		if colored_mask.shape[2] == 3:
			colored_mask = np.concatenate([colored_mask, np.ones((*colored_mask.shape[:2], 1))], axis=2)

	# Create overlay
	overlay = image_array * (1 - colored_mask[:, :, 3:]) + (colored_mask[:, :, :3] * 255 * colored_mask[:, :, 3:])
	return Image.fromarray(overlay.astype(np.uint8))