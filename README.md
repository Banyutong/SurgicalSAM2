# SAM2 Surgical Video Segmentation

## Installation

Refer to the official repository: [Segment Anything 2 (SAM2)](https://github.com/facebookresearch/segment-anything-2).

### Required Dependencies

To convert a video into frames, you will need to install `ffmpeg`. You can install it using the following commands:

```bash
sudo apt update
sudo apt install ffmpeg
pip install ffmpeg-python
```

## Usage

Currently, the script only supports processing a single video, which should be provided as individual frames. If your video is not already saved as frames, you can convert it using the following `ffmpeg` command:

```bash
ffmpeg -i /path/to/video.mp4 -q:v 2 -start_number 0 /path/to/output/frames/'%05d.jpg'
```

This command will convert the video into a sequence of images, saved in the specified output directory, with filenames **indexed numerically in time order**.

**Note:** If you already have the frames extracted, ensure that the filenames are indexed numerically in time order as well.

### Running the Script

You can run the main script `main_point.py` with the following command and arguments:

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

1. The script takes ground truth (GT) data from the first frame only, typically in the form of bounding boxes.
2. For each bounding box, it generates a point prompt by selecting the center of the box.
3. These point prompts are then used to initialize SAM2 for segmentation.
4. SAM2 uses these prompts to segment the first frame and then propagates the segmentation to all subsequent frames without additional prompts.
5. This approach allows for efficient video segmentation using only the initial frame's ground truth data.

## Output

The script generates the following outputs in the specified output directory:

1. Pixel masks for each frame and object.
2. Visualizations of segmentation results.
3. A GIF animation of the segmentation process.
4. COCO format annotations in JSON.
5. A visualization of the first frame with bounding boxes and sampled points.

## Additional Files

- `utils.py`: Contains utility functions for visualization, color mapping, and COCO annotation creation.
- `groundtruth2point.py`: Provides functions for sampling points from bounding boxes and masks. This script is crucial for converting ground truth data into point prompts for SAM2. It includes:
  - `sample_points_from_bbox`: Samples points from a bounding box. It can generate points from the center, edges, or randomly within the box. In the current workflow, it's used to select the center point of each bounding box.
  - `sample_points_from_mask`: Samples points from a binary mask. It can generate points from the contour or randomly within the mask. Note: This function is currently untested.
  - These functions are used to create prompts for SAM2 from different types of ground truth data, allowing for flexible input handling.

## Example

An example is provided. First, convert video3.mp4 to frames:

```bash
mkdir -p examples/video3/frames/
ffmpeg -i examples/video3/video3.mp4 -q:v 2 -start_number 0 examples/video3/frames/'%05d.jpg'
```

Then, generate masks for this video given the COCO ground truth bounding box file bbox_video3.json of the first frame:

```bash
python main_point.py --video_dir examples/video3/frames --sam2_checkpoint checkpoints/sam2_hiera_tiny.pt --output_dir test_output --gt_path examples/video3/bbox_video3.json --gt_type bbox
```

This example demonstrates how to use the script with bounding box ground truth data, but it can also work with mask data by changing the `--gt_type` argument to `mask`.