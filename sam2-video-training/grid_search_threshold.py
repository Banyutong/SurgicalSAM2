"""
Simplified SAM2 video training script.
Clean, direct implementation focused on core training functionality.
"""

import os
import sys
from pathlib import Path
import json
import pickle
import hydra
import lightning as L
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning.pytorch.callbacks import EarlyStopping
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import torch

from sam2_video.eval import inference as eval_infer
from sam2_video.eval import eval as eval_eval
from baseline_utils import extract_baseline_metrics, calculate_metrics_delta

from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()

@hydra.main(config_path="configs", config_name="best", version_base=None)
@logger.catch(onerror=lambda _: sys.exit(1))
def main(cfg: DictConfig) -> None:
    """Main entry point with consolidated training logic and Hydra configuration management."""

    # Resolve best checkpoint and export state_dict
    ckpt_path = "/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2-video-training/outputs/2025-09-13/09-54-33/checkpoints/sam2-epochepoch=03-val_lossval/total_loss=6.6774.ckpt"
    OUTPUT_DIR = Path("/bd_byta6000i0/users/surgicaldinov2/kyyang/sam2-video-training/outputs/2025-09-13/09-54-33/")
    if not ckpt_path:
        raise FileNotFoundError(
            "No best checkpoint found; ensure ModelCheckpoint is enabled and monitoring a metric"
        )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state_dict = state["state_dict"]
    if any(k.startswith("model.") for k in raw_state_dict.keys()):
        best_state_dict = {k[len("model.") :]: v for k, v in raw_state_dict.items()}
    else:
        best_state_dict = raw_state_dict

    # Inference → predict.json (under ${hydra:run.dir}/eval)
    _, _ = eval_infer.inference(
        run_dir=str(OUTPUT_DIR),
        coco_path=cfg.eval.coco_path,
        output_path=cfg.eval.output_subdir,
        prompt_type=cfg.eval.prompt_type,
        clip_length=cfg.eval.clip_length,
        variable_cats=cfg.eval.variable_cats,
        num_points=cfg.eval.num_points,
        include_center=cfg.eval.include_center,
        noised_prompt=cfg.eval.noised_prompt,
        noise_intensity=cfg.eval.noise_intensity,
        bbox_noise_type=cfg.eval.bbox_noise_type,
        num_neg_points=cfg.eval.num_neg_points,
        grid_spaceing=cfg.eval.grid_spacing,
        save_video_list=cfg.eval.save_video_list,
        model_cfg_override=cfg.model.config_path,
        sam2_checkpoint_override=cfg.model.checkpoint_path,
        finetuned_state_dict=best_state_dict,
        probs_out_dir = str(OUTPUT_DIR / "probs")
    )

    # Eval → eval.pkl + metrics.json (under ${hydra:run.dir}/eval)
    eval_dir = OUTPUT_DIR / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_eval.eval(
        predict_path=str(eval_dir / "predict.json"),
        coco_path=cfg.eval.coco_path,
        output_path=str(eval_dir),
    )

    with open(eval_dir / "eval.pkl", "rb") as f:
        result = pickle.load(f)
    summary = {
        "eval/mIoU": float(result["avg_scores"]["iou"]),
        "eval/Dice": float(result["avg_scores"]["dice"]),
        "eval/MAE": float(result["avg_scores"]["mae"]),
    }

    # Calculate and add baseline deltas
    baseline_metrics = extract_baseline_metrics(combo_name)
    if baseline_metrics:
        # Convert summary keys to match baseline format
        current_metrics = {
            "mIoU": summary["eval/mIoU"],
            "Dice": summary["eval/Dice"],
            "MAE": summary["eval/MAE"],
        }

        delta_metrics = calculate_metrics_delta(current_metrics, baseline_metrics)

        # Add deltas to summary with eval/ prefix
        for metric_key, delta_value in delta_metrics.items():
            summary[f"eval/{metric_key}"] = delta_value

        logger.info(f"Added baseline deltas for {combo_name}: {delta_metrics}")
    else:
        logger.warning(
            f"No baseline found for {combo_name}, skipping delta calculation"
        )

    wandb_logger.experiment.summary.update(summary)
    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    if cfg.eval.log_per_category:
        for cat_id, scores in result["cat_scores"].items():
            wandb_logger.experiment.summary.update(
                {
                    f"eval/cat/{cat_id}/IoU": float(scores["iou"]),
                    f"eval/cat/{cat_id}/Dice": float(scores["dice"]),
                    f"eval/cat/{cat_id}/MAE": float(scores["mae"]),
                }
            )


if __name__ == "__main__":
    # pass
    main()
