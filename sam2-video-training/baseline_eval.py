"""
Baseline evaluation script for all combo configurations.
Performs inference and evaluation on all configs in configs/combo/ directory.
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
from loguru import logger
from omegaconf import OmegaConf, DictConfig
import pandas as pd
import wandb

from sam2_video.eval import inference as eval_infer
from sam2_video.eval import eval as eval_eval
from concurrent.futures import ProcessPoolExecutor


@logger.catch(onerror=lambda _: sys.exit(1))
def discover_combo_configs(
    combo_dir: str = "configs/combo", specific_file: Optional[str] = None
) -> List[Path]:
    """Discover combo configuration files.

    Args:
        combo_dir: Directory containing combo configurations
        specific_file: Path to specific combo file (overrides combo_dir discovery)

    Returns:
        List of combo configuration file paths
    """
    if specific_file:
        combo_path = Path(specific_file)
        if not combo_path.exists():
            raise FileNotFoundError(f"Combo file not found: {specific_file}")
        if combo_path.suffix != ".yaml":
            raise ValueError(f"Combo file must be a .yaml file: {specific_file}")
        logger.info(f"Using specific combo file: {combo_path}")
        return [combo_path]

    combo_path = Path(combo_dir)
    combo_files = list(combo_path.rglob("*_mem.yaml"))
    logger.info(f"Found {len(combo_files)} combo configurations")
    return combo_files


@logger.catch(onerror=lambda _: sys.exit(1))
def parse_combo_config(config_path: Path) -> DictConfig:
    """Parse a combo configuration file and resolve references."""
    # Load the combo config
    combo_cfg = OmegaConf.load(config_path)

    # Load data config referenced by combo
    data_config_name = combo_cfg.defaults[0].split("@")[0].replace("/data/", "")
    data_cfg = OmegaConf.load(f"configs/data/{data_config_name}.yaml")

    # Merge only data config with combo config
    merged_cfg = OmegaConf.merge({"data": data_cfg}, combo_cfg)

    return merged_cfg


@logger.catch(onerror=lambda _: sys.exit(1))
def load_finetuned_state_dict(finetuned_model_path: str) -> Dict:
    """Load finetuned state dict following sam2model.py logic."""
    if finetuned_model_path is None:
        return None
    if "all" in finetuned_model_path:
        state_dict = torch.load(finetuned_model_path, weights_only=False)
        logger.info(f"Loaded full model state dict from {finetuned_model_path}")
        if hasattr(state_dict, "state_dict"):
            return state_dict.state_dict()
        else:
            return state_dict
    else:
        # Load only mask decoder for partial loading
        mask_decoder_state = torch.load(finetuned_model_path, weights_only=True)
        prompt_encoder_state = None

        logger.info(f"Loaded mask decoder state dict from {finetuned_model_path}")

        pe_path = finetuned_model_path.replace(".torch", "_prompt_encoder.torch")

        if os.path.exists(pe_path):
            prompt_encoder_state = torch.load(pe_path, weights_only=True)
            logger.info(f"Loaded prompt encoder state dict from {pe_path}")

        return mask_decoder_state, prompt_encoder_state


@logger.catch(onerror=lambda _: sys.exit(1))
def run_inference_and_eval(cfg: DictConfig, output_dir: Path) -> Dict[str, float]:
    """Run inference and evaluation for a single combo configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load finetuned state dict
    finetuned_state_dict = load_finetuned_state_dict(cfg.model.fintuned_model_path)

    # Run inference
    logger.info(f"Running inference with prompt type: {cfg.model.prompt_type}")
    _, _ = eval_infer.inference(
        run_dir=str(output_dir),
        coco_path=cfg.data.val_path,
        output_path=".",  # Files will be saved in output_dir directly
        prompt_type=cfg.model.prompt_type,
        clip_length=cfg.get("clip_length", None),
        variable_cats=cfg.get("variable_cats", False),
        num_points=cfg.model.get("num_pos_points", 1),
        include_center=cfg.model.get("include_center", True),
        noised_prompt=cfg.get("noised_prompt", False),
        noise_intensity=cfg.get("noise_intensity", 0.1),
        bbox_noise_type=cfg.get("bbox_noise_type", "shift_scale"),
        num_neg_points=cfg.model.get("num_neg_points", 0),
        grid_spaceing=cfg.get("grid_spacing", None),
        save_video_list=cfg.get("save_video_list", None),
        model_cfg_override="configs/sam2.1/sam2.1_hiera_t.yaml",  # Use default SAM2 config
        sam2_checkpoint_override="/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2/checkpoints/sam2.1_hiera_tiny.pt",  # Use default checkpoint
        finetuned_state_dict=finetuned_state_dict,
    )

    # Run evaluation
    logger.info("Running evaluation")
    eval_eval.eval(
        predict_path=str((output_dir / "eval" / "predict.json").resolve()),
        coco_path=cfg.data.val_path,
        output_path=str(output_dir.resolve()),
    )

    # Load and extract metrics
    with open(output_dir / "eval.pkl", "rb") as f:
        result = pickle.load(f)

    summary = {
        "mIoU": float(result["avg_scores"]["iou"]),
        "Dice": float(result["avg_scores"]["dice"]),
        "MAE": float(result["avg_scores"]["mae"]),
    }

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Results: mIoU={summary['mIoU']:.4f}, Dice={summary['Dice']:.4f}, MAE={summary['MAE']:.4f}"
    )
    return summary


@logger.catch(onerror=lambda _: sys.exit(1))
def save_summary_results(all_results: List[Dict], output_dir: Path):
    """Save aggregated results to CSV and JSON."""
    df = pd.DataFrame(all_results)
    df.to_csv(output_dir / "summary_results.csv", index=False)

    with open(output_dir / "summary_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Summary results saved to {output_dir}")


@logger.catch(onerror=lambda _: sys.exit(1))
def init_wandb_logging(dataset_name: str, combo_name: str) -> Optional[wandb.run]:
    """Initialize WandB logging for a combo evaluation."""
    try:
        run = wandb.init(
            project="sam2-baseline-eval",
            name=f"{dataset_name}-{combo_name}",
            tags=[dataset_name, combo_name],
            reinit=True,
        )
        return run
    except Exception as e:
        logger.warning(
            f"WandB initialization failed: {e}. Continuing without WandB logging."
        )
        return None


@logger.catch(onerror=lambda _: sys.exit(1))
def log_to_wandb(
    wandb_run: Optional[wandb.run], metrics: Dict[str, float], config_info: Dict
):
    """Log metrics and config to WandB if available."""
    if wandb_run is None:
        return

    try:
        # Log configuration
        wandb_run.config.update(config_info)

        # Log metrics
        wandb_run.summary.update(
            {
                "eval/mIoU": metrics["mIoU"],
                "eval/Dice": metrics["Dice"],
                "eval/MAE": metrics["MAE"],
            }
        )

        wandb_run.finish()
        logger.info("Results logged to WandB")
    except Exception as e:
        logger.warning(f"WandB logging failed: {e}")


@logger.catch(onerror=lambda _: sys.exit(1))
def main(combo_file: Optional[str] = None):
    """Main baseline evaluation function.

    Args:
        combo_file: Optional path to specific combo file to process
    """
    # Setup logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    output_base_dir = Path("baseline_results")
    output_base_dir.mkdir(exist_ok=True)

    # Add file logging
    logger.add(output_base_dir / "evaluation_log.log", rotation="10 MB")

    if combo_file:
        logger.info(
            f"Starting baseline evaluation for specific combo file: {combo_file}"
        )
    else:
        logger.info("Starting baseline evaluation for all combo configurations")

    # Discover combo configurations (single file or all files)
    combo_configs = discover_combo_configs(specific_file=combo_file)

    all_results = []

    for config_path in combo_configs:
        logger.info(f"Processing {config_path}")

        # Parse configuration
        cfg = parse_combo_config(config_path)

        # Extract dataset and combo name from path
        path_parts = config_path.parts
        dataset_name = path_parts[-2]  # e.g., 'cholecseg8k'
        combo_name = config_path.stem  # e.g., '1_mem'

        # Create output directory
        combo_output_dir = output_base_dir / dataset_name / combo_name

        # Initialize WandB logging
        wandb_run = init_wandb_logging(dataset_name, combo_name)

        # Save configuration
        combo_output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, combo_output_dir / "config.yaml")

        # Run inference and evaluation
        metrics = run_inference_and_eval(cfg, combo_output_dir)

        # Prepare config info for logging
        config_info = {
            "dataset": dataset_name,
            "combo": combo_name,
            "finetuned_model_path": cfg.model.fintuned_model_path,
            "prompt_type": cfg.model.prompt_type,
            "trainable_modules": cfg.model.trainable_modules,
        }

        # Log to WandB
        log_to_wandb(wandb_run, metrics, config_info)

        # Record results
        result_entry = {
            "dataset": dataset_name,
            "combo": combo_name,
            "config_path": str(config_path),
            "finetuned_model_path": cfg.model.fintuned_model_path,
            "prompt_type": cfg.model.prompt_type,
            "trainable_modules": cfg.model.trainable_modules,
            **metrics,
        }
        all_results.append(result_entry)

        logger.info(f"✅ Completed {dataset_name}/{combo_name}")

    # Save summary results
    # save_summary_results(all_results, output_base_dir)

    # logger.info(
    #     f"🎉 Baseline evaluation completed! Processed {len(all_results)} configurations"
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline evaluation for SAM2 combo configurations"
    )
    parser.add_argument(
        "--combo-file",
        type=str,
        help="Path to specific combo configuration file to process (e.g., configs/combo/cholecseg8k/1_mem.yaml)",
    )

    args = parser.parse_args()
    main(combo_file=args.combo_file)
