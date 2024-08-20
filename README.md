



## Installation

Refer to the official repository: [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2).

### Required Dependencies
To visualize the masks as contours:
```
pip install scikit-image
pip install pycocotools
```

To convert a video into frames, install `ffmpeg`:

```bash
sudo apt update
sudo apt install ffmpeg
pip install ffmpeg-python
```

### Update Log
- **8.18:** Added support for using bounding boxes directly as ground truth in `main_bbox.py`.
- **8.19:** Implemented negative points sampling for pixel masks in `main_point.py` (for pixel mask only).
- **8.20:** Added support for negative points visualization for pixel masks in `main_point.py` (for pixel mask only).

### Updated Examples
Refer [https://sjtu.feishu.cn/docx/Q8YBd1VAOo0e8YxUe4Zcs5f9nYf](https://sjtu.feishu.cn/docx/Q8YBd1VAOo0e8YxUe4Zcs5f9nYf) for the results.

#### For Pixel Mask Sampling

```bash
python main_point.py --sampled_points 2 --negative_sample_points 1 --video_dir examples/video_pixel2/frames --sam2_checkpoint checkpoints/sam2_hiera_large.pt --output_dir test_negative_output --gt_path examples/video_pixel2/0_mask.png --gt_type pixel_mask
```

- `--sampled_points`: Determines how many positive points to sample.
- `--negative_sample_points`: Determines how many negative points to sample (a class's sampled negative points are near the other-class sampled points).
  - You can change the `beta` to decide how near it is in `utils/negative_helpers.py` in `generate_negative_samples(sampled_point_classes, sampled_points, n, height, width, beta)`.

#### For Bounding Box as Ground Truth

```bash
python main_bbox.py --video_dir examples/video_mask/frames --sam2_checkpoint checkpoints/sam2_hiera_large.pt --output_dir bbox_output --gt_path examples/video_mask/annotation_coco_vid.json
```



# Usage: Mask Prompts


Currently, the script only supports processing a single video, provided as individual frames. To convert a video to frames:



```bash
ffmpeg -i /path/to/video.mp4 -q:v 2 -start_number 0 /path/to/output/frames/'%05d.jpg'
```
## Example
Only pixel-mask GT are tested.
```bash
python main_pixel_mask.py --video_dir examples/video_pixel  --sam2_checkpoint checkpoints/sam2_hiera_tiny.pt --output_dir pm_color_pixel_output  --gt_path examples/video_pixel/frame_561_endo_color_mask.png
```


# Usage: Point Prompts

This case, we convert any other GT format to point prompts, e.g., a pixel mask's center can be one point prompt.
Currently, the script only supports processing a single video, provided as individual frames. To convert a video to frames:

```bash
ffmpeg -i /path/to/video.mp4 -q:v 2 -start_number 0 /path/to/output/frames/'%05d.jpg'
```

**Note:** Ensure frame filenames are indexed numerically in time order.

### Running the Script

Run `main_point.py` with the following command:

```bash
python main_point.py --video_dir <path_to_video_frames> --sam2_checkpoint <path_to_sam2_checkpoint> --output_dir <output_directory> --gt_path <path_to_ground_truth_json> --gt_type <bbox_or_mask>
```

### Arguments:

- `--video_dir`: Directory containing video frames (required).
- `--sam2_checkpoint`: Path to the SAM2 checkpoint file (required).
- `--output_dir`: Output directory for results (default: current directory).
- `--vis_frame_stride`: Stride for visualization frames (default: 15).
- `--gt_path`: Path to ground truth data in JSON format or the path to a pixel mask image.
- `--gt_type`: Type of ground truth data, either 'bbox' or 'mask' or 'pixel-mask' (required).
- `--sample_points`: Number of points to sample for each object (default: 1).

### Script Workflow 

1. Uses ground truth (GT) data from the first frame only.
2. Generates point prompts from bounding boxes or masks.
3. Initializes SAM2 with these prompts.
4. Segments the first frame and propagates to subsequent frames.
5. Efficiently segments video using only initial frame's GT data.

## Output

The script generates in the specified output directory:

1. Pixel masks for each frame and object.
2. Visualizations of segmentation results.
3. GIF animation of the segmentation process.
4. COCO format annotations in JSON.
5. Visualization of the first frame with bounding boxes and sampled points.


## Examples- Convert to Point Prompts

### Bounding Boxes (bbx) as ground truths

Convert video to frames:

```bash
mkdir -p examples/video3/frames/
ffmpeg -i examples/video3/video3.mp4 -q:v 2 -start_number 0 examples/video3/frames/'%05d.jpg'
```

Generate masks:

```bash
python main_point.py --video_dir examples/video_mask/frames --sam2_checkpoint checkpoints/sam2_hiera_tiny.pt --output_dir test_bbx_output --gt_path examples/video_mask/annotation_coco_vid.json --gt_type bbox
```

### Masks as ground truths

Assuming frames are in `examples/video_mask/frames`:

```bash
python main_point.py --video_dir examples/video_mask/frames  --sam2_checkpoint checkpoints/sam2_hiera_tiny.pt --output_dir test_mask_output --gt_path examples/video_mask/annotation_coco_vid.json --gt_type mask
```

### Pixel Masks as ground truths

Assuming frames and pixel-mask gt are in `examples/video_pixel`:

This is for watershed pixel mask
```bash
python main_point.py --video_dir examples/video_pixel  --sam2_checkpoint checkpoints/sam2_hiera_tiny.pt --output_dir test_watermask_pixel_output --gt_path examples/video_pixel/frame_561_endo_watershed_mask.png --gt_type pixel_mask
```
This is for colored pixel mask
```bash
python main_point.py --video_dir examples/video_pixel  --sam2_checkpoint checkpoints/sam2_hiera_tiny.pt --output_dir test_color_pixel_output --gt_path examples/video_pixel/frame_561_endo_color_mask.png --gt_type pixel_mask
```


# Completed Pipeline on Endoscapes2023 Dataset
See `SurgicalSAM2/Endoscapes2023_Pipeline` for details.

-  `Endoscapes2023_Pipeline/point_prompt.ipynb` provides a step-by-step guide on how to use point prompts for segmentation within the Endoscapes2023 dataset. Additionally, it showcases the process of generating point prompts directly from ground truth masks.
- `Endoscapes2023_Pipeline/main.py` demonstrates how to batch process videos.
-  `Endoscapes2023_Pipeline/endoscapes_video.json` contains the information for the batch processing.
## Evaluation
The folder `Evaluation` contains different metric for different datasets. See `utils.py` for details.

|    Dataset    |          Metric           |
| :-----------: | :-----------------------: |
|  EndoVis'18   | Dice score, IOU per class |
|  CholecSeg8k  | Dice score, IOU per class |
| Endoscape2023 |      Ap50, MAP 50-95      |
|     Cadis     |                           |
|   EndoNeRF    |      Dice, IOU, MAE       |
|  EndoVis’17   |      Dice, IOU, MAE       |
|  SurgToolLoc  |      Dice, IOU, MAE,      |


## Reinitialization for every N frames
One version of reinitialization is implemented in `Eodoscapes2023/main.py`. You can focus on the following pieces of code to have an overview of the reinitialization method.

```python
def process_video_clip(
    video_info, video_order, coco_info, prompt_type, start_idx, end_idx, output_path
):
    video_dir = create_symbol_link_for_video(
        video_info[video_order]["frames"][start_idx : end_idx + 1]
    )
    prompt_frame = find_prompt_frame(
        video_info, video_order, coco_info, start_idx, end_idx
    )
    if prompt_frame is None:
        return {}
    prompt_objs = get_each_obj(prompt_frame, coco_info)
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    inference_state = predictor.init_state(video_path=video_dir)
    predictor, inference_state, out_obj_ids, out_mask_logits = add_prompt(
        prompt_objs,
        predictor,
        inference_state,
        prompt_frame["order_in_video"] - start_idx,
        prompt_type,
    )
    video_output_path = os.path.join(
        output_path, f"video_{video_info[video_order]["video_id"]}"
    )
    os.makedirs(video_output_path, exist_ok=True)
    save_prompt_frame(
        video_info[video_order]["frames"][prompt_frame["order_in_video"]],
        prompt_objs,
        prompt_type,
        out_obj_ids,
        out_mask_logits,
        len(coco_info.cats),
        video_output_path,
    )
    video_segments = predict_on_video(predictor, inference_state, start_idx)
    del predictor
    torch.cuda.empty_cache()
    return video_segments

```
```python
def process_singel_video(
    video_info, video_order, coco_info, prompt_type, clip_length, output_path
):
    video_segments = {}
    if clip_length is None:
        clip_length = len(video_info[video_order]["frames"])
    for start_idx in range(0, len(video_info[video_order]["frames"]), clip_length):
        end_idx = min(
            start_idx + clip_length - 1, len(video_info[video_order]["frames"]) - 1
        )
        video_segments.update(
            process_video_clip(
                video_info,
                video_order,
                coco_info,
                prompt_type,
                start_idx,
                end_idx,
                output_path,
            )
        )
    return video_segments
```
For other details, please have a look at the `Eodoscapes2023_Pipeline/main.py` file.
### A few things to notice
1. The code now is **NOT** based on the `main_point.py` or `main_pixel_mask.py`. As COCO data format is used in Endoscapes2023 instead of pixel masks. And there are extra configuration you need to make it work on segments.
2. The `process_video_clip` function is designed to process a segment of a video, which is defined by `start_idx` and `end_idx`. This function is called repeatedly by `process_singel_video` to process the entire video in segments.
3. The `clip_length` parameter in `process_singel_video` determines the length of each video segment to be processed. If `clip_length` is set to `None`, the entire video will be processed as a single segment.


## Add Todo
- [x] support to process a single video with a GT json file that contains complete annotations
- [x] Implement search for first frame with valid GT
- [ ] Disentangle contents in main_point.py
- [x] Develop main_mask.py for direct mask input processing
- [ ] Develop main_bbx.py for direct bounding box input handling
- [ ] Interactive point prompting in a .ipynb file
- [ ] Implement multi-video batch processing