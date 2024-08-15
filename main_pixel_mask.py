import argparse
import os
import torch
import numpy as np
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from utils.mask_helpers import get_model_cfg
from utils.visualization import visualize_first_frame_comprehensive, get_color_map
from utils.utils import find_frames
from utils.output_utils import save_pixel_masks, create_coco_annotations, save_visualizations, save_coco_json

def parse_args():
    parser = argparse.ArgumentParser(description="SAM2 Video Segmentation with Pixel Mask")
    parser.add_argument('--video_dir', type=str, required=True, help='Directory containing video frames')
    parser.add_argument('--sam2_checkpoint', type=str, required=True, help='Path to SAM2 checkpoint')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    parser.add_argument('--vis_frame_stride', type=int, default=15, help='Stride for visualization frames')
    parser.add_argument('--gt_path', type=str, required=True, help='Path to ground truth pixel mask')
    return parser.parse_args()

def setup_environment():
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

def process_ground_truth(gt_path):
    pixel_mask = np.array(Image.open(gt_path))
    if pixel_mask.ndim == 3:
        pixel_mask = pixel_mask[:, :, 0]  # Take the first channel if it's a 3D array
    return pixel_mask

def initialize_predictor(args, frame_names, pixel_mask):
    model_cfg = get_model_cfg(os.path.basename(args.sam2_checkpoint))
    predictor = build_sam2_video_predictor(model_cfg, args.sam2_checkpoint)
    inference_state = predictor.init_state(video_path=args.video_dir)

    ann_frame_idx = 0  # Assuming we're using the first frame for annotation

    unique_objects = np.unique(pixel_mask)
    unique_objects = unique_objects[unique_objects != 0]  # Exclude background (0)

    all_video_res_masks = []
    for obj_id in unique_objects:
        binary_mask = (pixel_mask == obj_id).astype(np.uint8)
        binary_mask_tensor = torch.from_numpy(binary_mask).to(torch.bool)

        try:
            frame_idx, obj_ids, video_res_masks = predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=int(obj_id),
                mask=binary_mask_tensor,
            )
            all_video_res_masks.extend(video_res_masks)
        except Exception as e:
            print(f"Error in add_new_mask for object {obj_id}: {str(e)}")
            raise

    return predictor, inference_state, frame_idx, unique_objects, all_video_res_masks

def resize_mask(mask, target_shape):
    return np.array(Image.fromarray(mask).resize(target_shape[::-1], resample=Image.NEAREST))

def main(args):
    setup_environment()

    frame_names = find_frames(args.video_dir)
    pixel_mask = process_ground_truth(args.gt_path)
    predictor, inference_state, init_frame_idx, unique_objects, init_video_res_masks = initialize_predictor(args, frame_names, pixel_mask)

    video_segments = {init_frame_idx: {obj_id: mask.cpu().numpy().squeeze() for obj_id, mask in zip(unique_objects, init_video_res_masks)}}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    os.makedirs(args.output_dir, exist_ok=True)

    # Visualize first frame
    first_frame_path = os.path.join(args.video_dir, frame_names[0])
    first_frame = np.array(Image.open(first_frame_path))
    first_frame_predictions = np.zeros_like(first_frame[:, :, 0])

    for obj_id, mask in video_segments[0].items():
        if mask.shape != first_frame_predictions.shape:
            mask = resize_mask(mask, first_frame_predictions.shape)
        first_frame_predictions[mask] = obj_id

    combined_output_path = os.path.join(args.output_dir, 'first_frame_visualization.png')
    visualize_first_frame_comprehensive(
        first_frame,
        pixel_mask,
        None,  # No sampled points in this version
        first_frame_predictions,
        combined_output_path,
        'pixel_mask'
    )

    # Save outputs
    save_pixel_masks(video_segments, args.output_dir)
    coco_annotations, coco_images = create_coco_annotations(video_segments, frame_names)

    # Determine the maximum object ID in video_segments
    max_obj_id = max(max(segment.keys()) for segment in video_segments.values())
    object_colors = get_color_map(max_obj_id)

    save_visualizations(video_segments, frame_names, args.video_dir, args.output_dir, object_colors, args.vis_frame_stride)
    save_coco_json(coco_annotations, coco_images, max_obj_id, args.output_dir)

    print(f"Results saved in {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)