# SAM2 Video Training

This repository provides the reference training code used in our SAM2 memory module study on video instance segmentation. The focus is on a transparent, reproducible pipeline that researchers can easily adapt for their own datasets.

## Key Features

- Minimal PyTorch Lightning training loop with fail-fast logging via Loguru.
- Single Hydra configuration (`configs/config.yaml`) controlling data, model, trainer, callbacks, and evaluation.
- **Multi-Dataset Support**: Pre-configured for medical datasets (CholecSeg8k, EndoVis17, EndoVis18) with COCO-style annotations.
- **Modular Loss System**: Configurable loss combinations (focal, dice, BCE) with flexible weighting schemes.
- Dataset helpers for converting COCO-style annotations into fixed-length video clips with multi-object masks.
- **Multi-GPU Training**: Distributed training support with automatic GPU allocation.
- **Advanced Evaluation**: Grid search, threshold tuning, and batch evaluation capabilities.
- Evaluation scripts for reporting instance-level metrics on validation or hold-out sets.

## Repository Layout

### Core Training
- `train.py`: Hydra-powered entry point wrapping the Lightning module and data module.
- `multi_gpu_train.sh`: Script for distributed multi-GPU training.

### Configuration System
- `configs/config.yaml`: Main configuration file with sensible defaults.
- `configs/data/`: Dataset-specific configurations (cholecseg8k, endovis17, endovis18).
- `configs/combo/`: Pre-generated experiment combinations for different datasets and settings.
- `configs/losses/`: Modular loss configurations (dice_main, focal_main, equal).
- `configs/sam2/`: SAM2 model variant configurations.

### Core Package
- `sam2_video/`: Package containing data loaders, model wrapper, training module, and utilities.
  - `sam2_video/data/`: Dataset classes and data preprocessing utilities.
  - `sam2_video/training/`: Lightning modules and training logic.
  - `sam2_video/eval/`: Inference, evaluation, and threshold tuning utilities.

### Evaluation & Analysis
- `baseline_eval.py`: Single-model evaluation script with configurable prompting.
- `multi_baseline_eval.py`: Batch evaluation across multiple configurations.
- `grid_search_threshold.py`: Automated threshold optimization for post-processing.
- `baseline_utils.py`: Shared utilities for baseline comparisons.

### Data Processing
- `data/`: Dataset conversion and preprocessing scripts.
  - `convert_endovis_to_coco.py`: Convert EndoVis annotations to COCO format.
  - `apply_morphological_opening.py`: Post-processing for mask refinement.

### Utilities & Automation
- `generate_combo_yamls.py`: Generate experiment configuration combinations.
- `examples/`, `scripts/`, `notebooks/`: Supplementary material and analysis tools.

## Supported Datasets

The repository includes pre-configured support for medical video segmentation datasets:

### CholecSeg8k
- **Task**: Cholecystectomy tool and anatomy segmentation
- **Config**: `configs/data/cholecseg8k.yaml`
- **Image Size**: 512px, 8-frame clips
- **Categories**: 13 classes (tools + anatomical structures)

### EndoVis17 (MICCAI 2017)
- **Task**: Robotic instrument segmentation
- **Config**: `configs/data/endovis17.yaml`  
- **Image Size**: 384px, 10-frame clips
- **Categories**: 8 instrument classes

### EndoVis18 (MICCAI 2018)
- **Task**: Robotic scene segmentation with semantic labels
- **Config**: `configs/data/endovis18.yaml`
- **Image Size**: 384px, 10-frame clips  
- **Categories**: 7 semantic classes

Each dataset configuration includes optimized hyperparameters for image resolution, clip length, and batch sizing based on typical GPU memory constraints.

## Getting Started

### 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

Make sure the `sam2` checkpoint (`.pt` / `.pth`) referenced in your config is accessible locally.

### 2. Prepare Data

The dataloader expects COCO-style JSON with the following required fields:

- `images[*].video_id`: groups frames by video
- `images[*].order_in_video`: frame index within each video
- `annotations[*].segmentation`: RLE mask (per-object)

Frame ordering and category IDs must be consistent; empty categories are rejected to enforce fail-fast behaviour. See `sam2_video/data/dataset.py` for additional details.

### 3. Configure an Experiment

All knobs live in `configs/config.yaml`. You can use pre-configured datasets or specify custom paths:

#### Using Pre-configured Datasets
```bash
# Train on EndoVis18 with point prompts
python train.py --config-name combo/endovis18/1

# Train on CholecSeg8k with default settings  
python train.py data=cholecseg8k

# Train on EndoVis17 with custom loss weights
python train.py data=endovis17 loss.weight_dict.loss_mask=30
```

#### Custom Dataset Configuration
```bash
python train.py \
  model.checkpoint_path=/path/to/sam2.pt \
  model.config_path=/path/to/sam2.yaml \
  data.train_path=/data/train.json \
  data.val_path=/data/val.json \
  trainer.accelerator=gpu trainer.devices=1 trainer.precision=16-mixed
```

#### Loss Configuration Options
```bash
# Use dice-only loss
python train.py --config-name dice_loss_only

# Use focal loss with custom gamma
python train.py loss=focal_main loss.focal_gamma=2.5

# Use equal loss weighting
python train.py loss=equal
```

Logging defaults to the standard console output. Remote loggers (e.g., Weights & Biases) can be disabled by removing or overriding the `wandb` node in the config:

```bash
python train.py +wandb=null
```

### 4. Train

#### Single-GPU Training
```bash
python train.py
```

#### Multi-GPU Training
For distributed training across multiple GPUs:
```bash
# Use the provided multi-GPU script
bash multi_gpu_train.sh

# Or configure manually with Lightning
python train.py trainer.devices=4 trainer.strategy=ddp trainer.accelerator=gpu
```

Resolved configs and checkpoints are stored under `outputs/<date>/<time>/`.

### 5. Evaluate

#### Single Model Evaluation
To run evaluation on a trained checkpoint:

```bash
python baseline_eval.py \
  data.val_path=/data/val.json \
  model.checkpoint_path=/path/to/sam2.ckpt \
  output_dir=outputs/eval_run
```

#### Batch Evaluation
Evaluate multiple configurations in parallel:

```bash
# Evaluate all EndoVis18 combo configurations
python multi_baseline_eval.py

# Evaluate specific dataset configurations
python multi_baseline_eval.py --filter endovis17
```

#### Threshold Optimization
Automatically find optimal prediction thresholds:

```bash
# Grid search for optimal thresholds
python grid_search_threshold.py \
  --config-path configs/combo/endovis18/1.yaml \
  --output-dir outputs/threshold_search

# Apply optimized thresholds
python baseline_eval.py \
  --config-name combo/endovis18/1 \
  eval.threshold=0.35
```

Pass `--help` to any evaluation script for the full list of arguments.

## Data Preprocessing

### Converting Datasets to COCO Format

The repository includes utilities for converting medical datasets to the required COCO-style format:

```bash
# Convert EndoVis dataset annotations
python data/convert_endovis_to_coco.py \
  --input-dir /path/to/endovis/annotations \
  --output-file /path/to/endovis_coco.json \
  --image-root /path/to/endovis/images

# Apply morphological opening to masks (optional post-processing)
python data/apply_morphological_opening.py \
  --input-json /path/to/annotations.json \
  --output-json /path/to/annotations_processed.json \
  --kernel-size 3
```

### Generating Configuration Combinations

Automatically generate experiment configurations for systematic evaluation:

```bash
# Generate all dataset/prompt/loss combinations
python generate_combo_yamls.py \
  --datasets endovis17 endovis18 cholecseg8k \
  --prompt-types point box mask \
  --output-dir configs/combo/generated
```

## Reproducing Paper Results

Each experiment sweep is defined by a YAML file in `configs/`. The main paper setting corresponds to `configs/config.yaml` with prompt type `mask`. For ablations, we provide additional configurations:

### Main Experiments
```bash
# Default SAM2 configuration with balanced loss
python train.py --config-name=config

# Best performing configuration
python train.py --config-name=best

# Dice loss only ablation
python train.py --config-name=dice_loss_only
```

### Dataset-Specific Experiments
```bash
# Run pre-configured EndoVis18 experiments
python train.py --config-name=combo/endovis18/1  # Point prompts
python train.py --config-name=combo/endovis18/5  # Box prompts
python train.py --config-name=combo/endovis18/10 # Mask prompts

# Batch evaluation of all configurations
python multi_baseline_eval.py --filter endovis18
```

The `configs/combo/` directory contains systematically generated configurations covering all dataset/prompt/loss combinations used in the paper.

