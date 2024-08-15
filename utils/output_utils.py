import os
import cv2
import numpy as np
from PIL import Image
import json
from utils.mask_helpers import mask_to_rle, mask_to_bbox

def save_pixel_masks(video_segments, output_dir):
    os.makedirs(os.path.join(output_dir, 'pixel_masks'), exist_ok=True)
    for frame_idx, frame_segments in video_segments.items():
        for out_obj_id, out_mask in frame_segments.items():
            if out_mask.ndim == 3:
                out_mask = out_mask.squeeze()
            if out_mask.ndim != 2:
                print(f"Unexpected mask shape for frame {frame_idx}, object {out_obj_id}: {out_mask.shape}")
                continue
            mask_img = Image.fromarray((out_mask * 255).astype(np.uint8))
            mask_img.save(os.path.join(output_dir, 'pixel_masks', f'frame_{frame_idx:04d}_obj_{out_obj_id}.png'))

def create_coco_annotations(video_segments, frame_names):
    coco_annotations = []
    coco_images = []
    annotation_id = 1

    for frame_idx, frame_name in enumerate(frame_names):
        frame_segments = video_segments[frame_idx]
        for out_obj_id, out_mask in frame_segments.items():
            if out_mask.ndim == 3:
                out_mask = out_mask.squeeze()
            if out_mask.ndim != 2:
                continue

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

        coco_images.append({
            "id": frame_idx,
            "file_name": frame_name,
            "height": out_mask.shape[0],
            "width": out_mask.shape[1]
        })

    return coco_annotations, coco_images

def save_visualizations(video_segments, frame_names, video_dir, output_dir, object_colors, vis_frame_stride):
    os.makedirs(os.path.join(output_dir, 'visualizations_for_gif'), exist_ok=True)
    gif_frames = []
    for frame_idx, frame_name in enumerate(frame_names):
        original_img = Image.open(os.path.join(video_dir, frame_name))
        original_array = np.array(original_img)
        overlay = np.zeros_like(original_array)
        
        if frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[frame_idx].items():
                if out_mask.ndim == 3:
                    out_mask = out_mask.squeeze()
                if out_mask.ndim != 2:
                    continue
                
                color = np.array([int(c * 255) for c in object_colors[out_obj_id - 1]])
                color = np.clip(color, 0, 255).astype(np.uint8)  # Ensure color values are within 0-255 range
                
                contours, _ = cv2.findContours(out_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.fillPoly(overlay, contours, color.tolist())
        
        alpha = 0.5
        result_array = cv2.addWeighted(original_array, 1 - alpha, overlay, alpha, 0)
        result = Image.fromarray(result_array)
        
        if frame_idx % vis_frame_stride == 0:
            gif_frames.append(result)
            # Also save individual frame
            result.save(os.path.join(output_dir, 'visualizations_for_gif', f'vis_{frame_name}'))
    
    # Save GIF
    if gif_frames:
        gif_frames[0].save(os.path.join(output_dir, 'visualization.gif'), 
                           save_all=True, 
                           append_images=gif_frames[1:],
                           duration=500, 
                           loop=0)

def save_coco_json(coco_annotations, coco_images, num_objects, output_dir):
    coco_data = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [{"id": i, "name": f"Object {i}"} for i in range(1, num_objects + 1)]
    }
    with open(os.path.join(output_dir, 'coco_annotations.json'), 'w') as f:
        json.dump(coco_data, f)