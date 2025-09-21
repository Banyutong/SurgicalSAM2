"""
Visualization utilities for creating composite frames and animated GIFs.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import cv2
import imageio
from loguru import logger
import sys

@logger.catch(onerror=lambda _: sys.exit(1))
def create_composite_visualization(
    frame_0: torch.Tensor,  # [C, H, W]
    image: torch.Tensor,  # [C, H, W]
    gt_mask: torch.Tensor,  # [C, H, W]
    pred_mask: torch.Tensor,  # [C, H, W]
    prompts: Dict[str, Any],
    obj_to_cat: List[int],
    num_categories: int = 13,
    title: str = "Visualization",
) -> np.ndarray:
    """Create composite visualization: Image | GT Mask | Prompts | Prediction."""
    import matplotlib.pyplot as plt
    import cv2
    import random
    import colorsys

    # Generate random colors for different categories
    def generate_random_colors(n):
        colors = []
        for i in range(n):
            # Generate vibrant colors using HSV color space
            hue = i / n
            saturation = 0.8 + random.random() * 0.2  # 0.8-1.0
            value = 0.8 + random.random() * 0.2  # 0.8-1.0
            r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors

    category_colors = generate_random_colors(num_categories)

    # Convert tensors to numpy with proper denormalization
    if isinstance(image, torch.Tensor):
        image_np = image.permute(1, 2, 0).cpu().detach().numpy()
        # Denormalize if using ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = image_np * std + mean
        image_np = np.clip(image_np, 0, 1)  # Ensure valid range
        # Convert to uint8 for display
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
    else:
        image_np = image

    if isinstance(frame_0, torch.Tensor):
        frame_0_np = frame_0.permute(1, 2, 0).cpu().detach().numpy()
        # Denormalize if using ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        frame_0_np = frame_0_np * std + mean
        frame_0_np = np.clip(frame_0_np, 0, 1)  # Ensure valid range
        # Convert to uint8 for display
        if frame_0_np.max() <= 1.0:
            frame_0_np = (frame_0_np * 255).astype(np.uint8)
    else:
        frame_0_np = frame_0

    # Handle multi-channel masks - process each category separately
    if isinstance(gt_mask, torch.Tensor):
        if gt_mask.dim() == 3 and gt_mask.shape[0] > 1:
            gt_mask_np = gt_mask.cpu().detach().numpy()  # [C, H, W]
        else:
            gt_mask_np = gt_mask.squeeze().cpu().detach().numpy()
    else:
        gt_mask_np = gt_mask

    if isinstance(pred_mask, torch.Tensor):
        if pred_mask.dim() == 3 and pred_mask.shape[0] > 1:
            pred_mask_np = pred_mask.cpu().detach().numpy()  # [C, H, W]
        else:
            pred_mask_np = pred_mask.squeeze().cpu().detach().numpy()
    else:
        pred_mask_np = pred_mask

    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)

    # 1. Original Image
    axes[0, 0].imshow(image_np)
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis("off")

    # Helper function to create alpha composite with contours
    def create_alpha_composite_with_contours(base_image, mask_array, colors, alpha=0.4):
        # Create a transparent overlay
        overlay = np.zeros_like(base_image, dtype=np.uint8)

        # Apply semi-transparent masks
        if mask_array.ndim == 3:  # Multi-category mask
            for category_idx in range(mask_array.shape[0]):
                mask = mask_array[category_idx] > 0.5
                if mask.any():
                    color = colors[category_idx % len(colors)]
                    for c in range(3):
                        overlay[:, :, c][mask] = color[c]
        else:  # Single category mask
            mask = mask_array > 0.5
            if mask.any():
                color = colors[0]
                for c in range(3):
                    overlay[:, :, c][mask] = color[c]

        # Create alpha composite
        composite = base_image.copy()
        # Alpha blending formula: composite = base * (1 - alpha) + overlay * alpha
        composite = cv2.addWeighted(base_image, 1 - alpha, overlay, alpha, 0)

        # Add contours
        if mask_array.ndim == 3:  # Multi-category mask
            for category_idx in range(mask_array.shape[0]):
                mask = (mask_array[category_idx] > 0.5).astype(np.uint8)
                if mask.any():
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    color = colors[category_idx % len(colors)]
                    cv2.drawContours(composite, contours, -1, color, 2)
        else:  # Single category mask
            mask = (mask_array > 0.5).astype(np.uint8)
            if mask.any():
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                color = colors[0]
                cv2.drawContours(composite, contours, -1, color, 2)

        return composite

    # 2. Ground Truth Mask with alpha composite and contours
    gt_display = create_alpha_composite_with_contours(
        image_np, gt_mask_np, category_colors, alpha=0.2
    )
    axes[0, 1].imshow(gt_display)
    axes[0, 1].set_title("GT Mask (Semi-transparent)")
    axes[0, 1].axis("off")

    # 3. Prompts Overlay - visualize all prompts
    prompt_img = frame_0_np.copy()
    prompt_mask_np = np.zeros_like(gt_mask_np)

    if prompts["prompt_type"] == "point":
        for i in range(prompts["point_inputs"]["point_coords"].shape[0]):
            color = category_colors[obj_to_cat[i]]
            points = prompts["point_inputs"]["point_coords"][i].cpu().detach().numpy()
            labels = prompts["point_inputs"]["point_labels"][i].cpu().detach().numpy()
            for idx, point in enumerate(points):
                label = labels[idx]
                if label == 1:  # 画点 (圆形标记)
                    cv2.circle(
                        prompt_img,
                        (int(point[0]), int(point[1])),
                        8,
                        color,
                        -1,
                    )
                    # 添加白色边框
                    cv2.circle(
                        prompt_img,
                        (int(point[0]), int(point[1])),
                        8,
                        (255, 255, 255),
                        2,
                    )
                elif label == 0:  # 画叉 (十字标记)
                    cv2.drawMarker(
                        prompt_img,
                        (int(point[0]), int(point[1])),
                        color,
                        markerType=cv2.MARKER_CROSS,
                        markerSize=16,
                        thickness=3,
                        line_type=cv2.LINE_AA,
                    )
    elif prompts["prompt_type"] == "bbox":
        for i in range(prompts["point_inputs"]["point_labels"].shape[0]):
            color = category_colors[obj_to_cat[i]]
            x1, y1, x2, y2 = 0, 0, 0, 0

            color = category_colors[obj_to_cat[i]]
            points = prompts["point_inputs"]["point_coords"][i].cpu().detach().numpy()
            labels = prompts["point_inputs"]["point_labels"][i].cpu().detach().numpy()
            for idx, point in enumerate(points):
                label = labels[idx]
                if label == 2:  # 画矩形 (矩形标记)
                    x1, y1 = point[0], point[1]
                if label == 3:
                    x2, y2 = point[0], point[1]

            # 画矩形
            cv2.rectangle(
                prompt_img,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness=2,
            )

    elif prompts["prompt_type"] == "mask":
        for i in range(prompts["mask_inputs"].shape[0]):
            color = category_colors[obj_to_cat[i]]
            mask_prompt = prompts["mask_inputs"][i].cpu().detach().numpy()
            # Create alpha composite for this mask
            prompt_mask_np[obj_to_cat[i]] = np.maximum(
                prompt_mask_np[obj_to_cat[i]], mask_prompt
            )

    if prompts["prompt_type"] == "mask":
        prompt_img = create_alpha_composite_with_contours(
            frame_0_np, prompt_mask_np, category_colors, alpha=0.2
        )
    axes[1, 0].imshow(prompt_img)
    axes[1, 0].set_title("Prompts (All Objects)")
    axes[1, 0].axis("off")

    # 4. Prediction Overlay with alpha composite and contours
    pred_display = create_alpha_composite_with_contours(
        image_np, pred_mask_np, category_colors, alpha=0.2
    )
    axes[1, 1].imshow(pred_display)
    axes[1, 1].set_title("Prediction (Semi-transparent)")
    axes[1, 1].axis("off")

    # Convert matplotlib figure to numpy array
    fig.canvas.draw()
    # Use buffer_rgba() for newer matplotlib versions
    buf = fig.canvas.buffer_rgba()
    composite = np.asarray(buf)
    # Convert from RGBA to RGB by dropping alpha channel
    composite = composite[:, :, :3]
    plt.close(fig)
    return composite


@logger.catch(onerror=lambda _: sys.exit(1))
def create_visualization_gif(
    frames: torch.Tensor,
    gt_masks: torch.Tensor,
    outs_per_frame: List[Dict[str, torch.Tensor]],
    obj_to_cat: List[int],
    max_length: int = 4,
    stride: int = 1,
) -> np.ndarray:
    """Create visualization GIF and return path to GIF file in /tmp."""

    # Create list to store frames for GIF
    gif_frames = []

    # [T, C, H, W]
    predictions = torch.stack(
        [out["multistep_pred_multimasks_high_res"][0] for out in outs_per_frame]
    ).squeeze(2)
    num_categories = predictions.shape[1]
    point_prompts = outs_per_frame[0]["point_inputs"]
    mask_prompts = outs_per_frame[0]["mask_inputs"]

    if point_prompts is not None:
        unique_values = point_prompts["point_labels"].unique()
        if 2 in unique_values or 3 in unique_values:
            prompt_type = "bbox"
        else:
            prompt_type = "point"
    else:
        prompt_type = "mask"

    # Process frames with stride
    length = min(max_length, frames.shape[0], predictions.shape[0])
    indices = range(0, length, stride)

    prompts = {
        "point_inputs": point_prompts if point_prompts is not None else None,
        "mask_inputs": mask_prompts if mask_prompts is not None else None,
        "prompt_type": prompt_type,
    }

    for i in indices:
        frame = frames[i]  # [C, H, W]
        gt_mask = gt_masks[i]
        pred_mask = predictions[i]

        # Create composite visualization
        composite = create_composite_visualization(
            frame_0=frames[0],
            image=frame,
            gt_mask=gt_mask,
            pred_mask=pred_mask,
            prompts=prompts,
            title=f"Frame {i}",
            obj_to_cat=obj_to_cat,
            num_categories=num_categories,
        )

        # Add frame to GIF list
        gif_frames.append(composite)

    gif_np = np.stack(gif_frames, axis=0).transpose(0, 3, 1, 2)
    return gif_np

    # Create GIF from frames
    if gif_frames:
        # Create temporary file in /tmp directory
        import tempfile

        gif_filename = tempfile.mktemp(
            suffix=".gif", prefix="visualization_", dir="/tmp"
        )

        # Save as GIF
        with imageio.get_writer(gif_filename, mode="I", duration=0.5, loop=0) as writer:
            for frame in gif_frames:
                writer.append_data(frame)

        return gif_filename

    return ""
