# SAM2 Video Training

This repository provides the reference training code used in our SAM2 memory module study on video instance segmentation. The focus is on a transparent, reproducible pipeline that researchers can easily adapt for their own datasets.

## Key Features

- Minimal PyTorch Lightning training loop with fail-fast logging via Loguru.
- Single Hydra configuration (`configs/config.yaml`) controlling data, model, trainer, callbacks, and evaluation.
- Dataset helpers for converting COCO-style annotations into fixed-length video clips with multi-object masks.
- Evaluation scripts for reporting instance-level metrics on validation or hold-out sets.

## Repository Layout

- `train.py`: Hydra-powered entry point wrapping the Lightning module and data module.
- `configs/`: Default experiment configuration plus model variants under `configs/sam2/`.
- `sam2_video/`: Package containing data loaders, model wrapper, training module, and utilities.
- `baseline_utils.py`, `baseline_eval.py`: Utilities for reproducing baseline comparisons reported in the paper.
- `examples/`, `scripts/`, `notebooks/`: Supplementary material used during experimentation.

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

All knobs live in `configs/config.yaml`. Common overrides:

```bash
python train.py \
  model.checkpoint_path=/path/to/sam2.pt \
  model.config_path=/path/to/sam2.yaml \
  data.train_path=/data/train.json \
  data.val_path=/data/val.json \
  trainer.accelerator=gpu trainer.devices=1 trainer.precision=16-mixed
```

Logging defaults to the standard console output. Remote loggers (e.g., Weights & Biases) can be disabled by removing or overriding the `wandb` node in the config:

```bash
python train.py +wandb=null
```

### 4. Train

```bash
python train.py
```

Resolved configs and checkpoints are stored under `outputs/<date>/<time>/`.

### 5. Evaluate

To run evaluation on a trained checkpoint:

```bash
python baseline_eval.py \
  data.val_path=/data/val.json \
  model.checkpoint_path=/path/to/sam2.ckpt \
  output_dir=outputs/eval_run
```

Pass `--help` to `train.py` or `baseline_eval.py` for the full list of Hydra arguments.

## Reproducing Paper Results

Each experiment sweep is defined by a YAML file in `configs/`. The main paper setting corresponds to `configs/config.yaml` with prompt type `mask`. For ablations, we provide additional configs (e.g., `configs/dice_loss_only.yaml`).

To restore an experiment precisely:

```bash
python train.py --config-name=dice_loss_only
```

## Citation

If you use this codebase in your research, please cite our paper:

```
@inproceedings{<lead_author>2025sam2video,
  title     = {SAM2 Video Training for Memory-Augmented Video Segmentation},
  author    = {<Author List>},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}
```

Replace the placeholder fields with the final publication details once available.

## License

The license will be added upon camera-ready submission. Until then, all rights reserved.
