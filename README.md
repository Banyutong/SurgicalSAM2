# SAM2 Surgical Video Segmentation



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

## Usage

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
- `--gt_path`: Path to ground truth data in JSON format (required).
- `--gt_type`: Type of ground truth data, either 'bbox' or 'mask' (required).
- `--sample_points`: Number of points to sample for each object (default: 1).

## Script Workflow 

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

[//]: # (## Additional Files)

[//]: # ()
[//]: # (- `utils.py`: Utility functions for visualization, color mapping, and COCO annotation creation.)

[//]: # (- `groundtruth2point.py`: Functions for sampling points from bounding boxes and masks.)

## Examples

### Bounding Boxes (bbx) as ground truths

Convert video to frames:

```bash
mkdir -p examples/video3/frames/
ffmpeg -i examples/video3/video3.mp4 -q:v 2 -start_number 0 examples/video3/frames/'%05d.jpg'
```

Generate masks:

```bash
python main_point.py --video_dir examples/video3/frames --sam2_checkpoint checkpoints/sam2_hiera_tiny.pt --output_dir test_bbx_output --gt_path examples/video3/bbox_video3.json --gt_type bbox
```

### Masks as ground truths

Assuming frames are in `examples/video_mask/frames`:

```bash
python main_point.py --video_dir examples/video_mask/frames  --sam2_checkpoint checkpoints/sam2_hiera_tiny.pt --output_dir test_mask_output --gt_path examples/video_mask/mask_gt.json --gt_type mask
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

### Complete Pipeline on Endoscapes2023 Dataset
See `SurgicalSAM2/Endoscapes2023_Pipeline` for details.

The `point_prompt.ipynb` provides a step-by-step guide on how to use point prompts for segmentation within the Endoscapes2023 dataset. Additionally, it showcases the process of generating point prompts directly from ground truth masks.
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


## Potential Issues 

- First frame may lack ground truths
- Ground truth format inconsistencies
- Dependency conflicts or version incompatibilities

## Current limitations

- As many frames lack of ground truths, the scripts only support to process a single video with a GT json file that contains exactly the annotation for the first frame. That means we have to manually find a frame that has GT in a complete json file.
- Due to various GT formats of different dataset, we may have to design custom GT preprocessing. 


## Add Todo
- [ ] support to process a single video with a GT json file that contains complete annotations
- [ ] Implement search for first frame with valid GT
- [ ] Disentangle contents in main_point.py
- [ ] Develop main_mask.py for direct mask input processing
- [ ] Develop main_bbx.py for direct bounding box input handling
- [ ] Interactive point prompting in a .ipynb file
- [ ] Implement multi-video batch processing