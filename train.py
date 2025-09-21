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
import os

from sam2_video.eval import inference as eval_infer
from sam2_video.eval import eval as eval_eval
from baseline_utils import extract_baseline_metrics, calculate_metrics_delta

from hydra.core.global_hydra import GlobalHydra

GlobalHydra.instance().clear()


@hydra.main(config_path="configs", config_name="best", version_base=None)
@logger.catch(onerror=lambda _: sys.exit(1))
def main(cfg: DictConfig) -> None:
    """Main entry point with consolidated training logic and Hydra configuration management."""

    OUTPUT_DIR = Path(HydraConfig.get().run.dir)
    OAR_JOB_ID = os.getenv("OAR_JOB_ID", "000000")  # collision insurance
    # =====================================
    # SECTION 1: LOGGING SETUP
    # =====================================
    log_level = cfg.get("log_level", "INFO")
    logger.remove()
    logger.add(
        sys.stderr,
        level=log_level,
    )
    logger.add(OUTPUT_DIR / "training.log", rotation="10 MB", retention="10 days")
    # Extract combo name and number from config
    # Create tags for wandb
    if hasattr(cfg, "combo"):
        combo_name = getattr(cfg.combo, "name", "unknown")
    else:
        combo_name = "unknown"

    tags = [
        combo_name,
        cfg.model.prompt_type,
        cfg.data.name,
        f"stride{cfg.loss.gt_stride}",
    ].extend(cfg.model.trainable_modules)

    exp_name = f"{combo_name}__{OAR_JOB_ID}"

    # Weights & Biases
    logger.info(f"Initializing W&B logger for project: {cfg.wandb.project}")
    wandb_logger = instantiate(
        cfg.wandb,
        name=exp_name,
        id=exp_name,  # = W&B run id → immutable
        tags=tags,
        save_dir=OUTPUT_DIR,
    )

    wandb_logger.experiment.config.update(
        OmegaConf.to_container(cfg, resolve=True), allow_val_change=True
    )

    # =====================================
    # SECTION 2: CONFIGURATION & SETUP
    # =====================================
    # Convert to structured config

    # Set random seed
    L.seed_everything(cfg.seed)

    logger.info("Starting SAM2 training...")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg, resolve=True)}")

    # =====================================
    # SECTION 3: DIRECTORY & CONFIG SETUP
    # =====================================
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = OUTPUT_DIR / "config.yaml"
    OmegaConf.save(cfg, config_path)
    logger.info(f"Configuration saved to {config_path}")

    # =====================================
    # SECTION 4: LIGHTNING COMPONENTS
    # =====================================
    # Instantiate module and data module via Hydra targets, passing unpacked sections
    lightning_module = instantiate(cfg.module, _recursive_=False)
    data_module = instantiate(cfg.data_module, _recursive_=False)

    # =====================================
    # SECTION 5: CALLBACK CONFIGURATION
    # =====================================
    # Instantiate callbacks list defined in config
    callbacks = [instantiate(cb) for cb in cfg.callbacks]

    # =====================================
    # SECTION 7: TRAINER CREATION & EXECUTION
    # =====================================
    # Decide logger argument for Trainer

    trainer = instantiate(
        cfg.trainer,
        logger=wandb_logger,
        callbacks=callbacks,
        default_root_dir=(OUTPUT_DIR / "lightning_logs"),
        # default_root_dir="test",
    )

    # Start training
    logger.info("Starting training...")
    trainer.fit(lightning_module, data_module)

    logger.info("Training completed!")
    logger.info(f"Outputs saved to: {OUTPUT_DIR}")

    # =====================================
    # SECTION 8: POST-TRAINING INFERENCE + EVAL (fast-fail)
    # =====================================
    if not hasattr(cfg, "eval"):
        raise AttributeError(
            "Missing cfg.eval block; add eval settings to configs/config.yaml"
        )
    if not cfg.eval.enabled:
        logger.info("cfg.eval.enabled is False — enable it to run post-training eval")
        return
    if not os.path.exists(cfg.eval.coco_path):
        raise FileNotFoundError(f"Eval coco_path not found: {cfg.eval.coco_path}")

    # Resolve best checkpoint and export state_dict
    ckpt_path = trainer.checkpoint_callback.best_model_path
    print(ckpt_path)
    if not ckpt_path:
        best_state_dict = None
        logger.info("No best checkpoint found, skipping loading state_dict")
    else:
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
    summary["best_checkpoint_path"] = ckpt_path
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
