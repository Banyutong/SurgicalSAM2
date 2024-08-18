import argparse
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sam2.build_sam import build_sam2_video_predictor
from utils.mask_helpers import get_model_cfg
from utils.visualization import visualize_first_frame_comprehensive, get_color_map, visualize_all_frames
# from utils.visualization_mask import visualize_all_frames_masks
from utils.utils import find_frames, process_gt_pixel_mask, get_class_to_color_mapping
from utils.output_utils import save_pixel_masks, create_coco_annotations, save_visualizations, save_coco_json
from utils.groundtruth2point import  sample_points_from_pixel_mask

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

def process_ground_truth(args, frame_names):

    gt_data = process_gt_pixel_mask(frame_names, args.gt_path)
    return gt_data # Assuming the first frame is always valid for pixel_mask



def initialize_predictor(args, frame_names, pixel_mask):
    model_cfg = get_model_cfg(os.path.basename(args.sam2_checkpoint))
    predictor = build_sam2_video_predictor(model_cfg, args.sam2_checkpoint)
    inference_state = predictor.init_state(video_path=args.video_dir)

    ann_frame_idx = 0  # Assuming we're using the first frame for annotation

    # Check if pixel_mask is 3D, and if so, convert it to 2D
    if pixel_mask.ndim == 3:
        # Assuming the pixel_mask uses color to represent different objects,
        # we can convert it to a 2D mask where each unique color becomes a unique integer
        pixel_mask_2d = np.zeros((pixel_mask.shape[0], pixel_mask.shape[1]), dtype=np.int32)
        unique_colors = np.unique(pixel_mask.reshape(-1, pixel_mask.shape[2]), axis=0)
        for i, color in enumerate(unique_colors):
            if (not np.all(color == 0)) and (not np.all(color == 255)):  # Skip background (assuming it's black)
                mask = np.all(pixel_mask == color, axis=2)
                pixel_mask_2d[mask] = i + 1  # +1 to reserve 0 for background
    else:
        pixel_mask_2d = pixel_mask

    unique_objects = np.unique(pixel_mask_2d)
    unique_objects = unique_objects[unique_objects != 0]  # Exclude background (0)

    all_video_res_masks = []
    for obj_id in unique_objects:
        binary_mask = (pixel_mask_2d == obj_id).astype(np.uint8)
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



def main(args):
    setup_environment()

    frame_names = find_frames(args.video_dir)
    gt_data = process_ground_truth(args, frame_names)
    gt_mask = np.array(Image.open(args.gt_path))
    predictor, inference_state, init_frame_idx, unique_objects, init_video_res_masks = initialize_predictor(args, frame_names, gt_mask)

    video_segments = {init_frame_idx: {obj_id: mask.cpu().numpy().squeeze() for obj_id, mask in zip(unique_objects, init_video_res_masks)}}

    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    os.makedirs(args.output_dir, exist_ok=True)
    class_to_color_mapper = get_class_to_color_mapping(gt_mask)
    # _, _, class_to_color_mapper = sample_points_from_pixel_mask(gt_mask,  num_points=1, include_center=True)#, remove_small=False)
    # Visualize all frames
    visualize_all_frames(video_segments, frame_names, args.video_dir, args.output_dir, gt_data,  class_to_color_mapper=class_to_color_mapper, show_first_frame=True, show_points=False)
    
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