"""
Unified SAM2 Lightning trainer module.
This module provides PyTorch Lightning implementation for SAM2 training.
"""

import lightning as L
import torch
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from typing import Dict, Any, List
from loguru import logger
import sys
from hydra.utils import instantiate


from sam2_video.data.dataset import sam2_collate_fn
from sam2_video.model.losses import (
    MultiStepMultiMasksAndIous,
    BCECategoryLoss,
    CORE_LOSS_KEY,
)
from sam2_video.data.data_utils import BatchedVideoDatapoint

from torch.utils.data import DataLoader
import wandb


class SAM2LightningModule(L.LightningModule):
    """Lightning module for SAM2 video training.

    Simplifications:
    - Rely on Lightning for epoch-level metric aggregation via self.log(on_epoch=True)
    - Remove custom best-checkpoint and metrics JSON (ModelCheckpoint handles persistence)
    - Keep GIF logging behind config gate
    """

    def __init__(
        self,
        model: Any,
        loss: Any,
        optimizer: Any,
        scheduler: Any,
        visualization: Any,
    ):
        """
        Initialize Lightning module with unpacked configuration sections.

        Args:
            model: Model config node
            loss: Loss config node
            optimizer: Optimizer config node
            scheduler: Scheduler config node
            visualization: Visualization config node
        """
        super().__init__()

        # MODIFIED: 直接调用 save_hyperparameters()，它会自动捕获所有 __init__ 的参数
        # 这会自动创建 self.hparams.model, self.hparams.loss, self.hparams.optimizer 等
        self.save_hyperparameters()
        


        # Initialize model
        self.model = None

        # MODIFIED: 使用 self.hparams.loss 访问配置
        loss_type = getattr(self.hparams.loss, "type", None)
        if loss_type is not None and str(loss_type).lower() in {
            "bce",
            "bce_only",
            "ce_only",
        }:
            pos_weight = getattr(self.hparams.loss, "bce_pos_weight", None)
            reduction = getattr(self.hparams.loss, "bce_reduction", "mean")
            logit_temperature = getattr(
                self.hparams.loss, "bce_logit_temperature", 1.0
            )
            self.criterion = BCECategoryLoss(
                pos_weight=pos_weight,
                reduction=reduction,
                logit_temperature=logit_temperature,
            )
            logger.info("Using BCECategoryLoss (BCEWithLogits over [C,H,W])")
        else:
            self.criterion = MultiStepMultiMasksAndIous(
                weight_dict=self.hparams.loss.weight_dict,
                supervise_all_iou=self.hparams.loss.supervise_all_iou,
                iou_use_l1_loss=self.hparams.loss.iou_use_l1_loss,
                pred_obj_scores=self.hparams.loss.pred_obj_scores,
                focal_gamma_obj_score=self.hparams.loss.focal_gamma_obj_score,
                focal_alpha_obj_score=self.hparams.loss.focal_alpha_obj_score,
                logit_temperature=getattr(self.hparams.loss, "multistep_logit_temperature", 1.0),
            )
            logger.info("Using MultiStepMultiMasksAndIous loss")

        # MODIFIED: 使用 self.hparams.loss 访问配置
        self.loss_gt_stride: int = max(
            int(getattr(self.hparams.loss, "gt_stride", 1)), 1
        )

    @logger.catch(onerror=lambda _: sys.exit(1))
    def setup(self, stage: str):
        """Setup model when stage starts."""
        if stage == "fit":
            if self.model is None:
                logger.info("Loading SAM2 model via Hydra instantiate...")
                # MODIFIED: 使用 self.hparams.model 访问配置
                self.model = instantiate(self.hparams.model)
            self.model.load(str(self.device))
            self.model.train()
            model_info = self.model.get_info()
            logger.info(
                f"Model loaded: {model_info['total_parameters']:,} total params, "
                f"{model_info['trainable_parameters']:,} trainable params"
            )

    @logger.catch(onerror=lambda _: sys.exit(1))
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        # MODIFIED: 使用 self.hparams.optimizer 访问所有优化器相关的配置
        # 尤其是 lr=self.hparams.optimizer.lr，这是让 lr_finder 生效的关键
        if self.hparams.optimizer.type.lower() == "adamw":
            optimizer = optim.AdamW(
                trainable_params,
                lr=self.hparams.optimizer.lr,
                weight_decay=self.hparams.optimizer.weight_decay,
                betas=self.hparams.optimizer.betas,
            )
        else:
            optimizer = optim.Adam(
                trainable_params,
                lr=self.hparams.optimizer.lr,
                weight_decay=self.hparams.optimizer.weight_decay,
            )

        # MODIFIED: 使用 self.hparams.scheduler 访问调度器配置
        scheduler = None
        if getattr(self.hparams.scheduler, "enabled", True):
            total_steps = int(
                getattr(self.trainer, "estimated_stepping_batches", 0) or 0
            )
            total_steps = max(1, total_steps)
            warmup_steps = total_steps * self.hparams.optimizer.warmup_factor
            if warmup_steps >= total_steps:
                warmup_steps = max(0, total_steps - 1)

            num_cycles = float(getattr(self.hparams.scheduler, "num_cycles", 0.5))
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
                num_cycles=num_cycles,
            )

        logger.info(
            f"Configured optimizer: {self.hparams.optimizer.type}, "
            f"LR: {self.hparams.optimizer.lr}, Trainable params: {len(trainable_params):,}"
        )

        if scheduler is not None:
            logger.info(
                f"LR scheduler: cosine_with_warmup | warmup_steps={warmup_steps} | total_steps={total_steps} | num_cycles={num_cycles}"
            )

        if scheduler:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return {"optimizer": optimizer}

    # ... forward, _apply_gt_stride, training_step, validation_step ...
    # 这些方法内部没有直接使用 config，所以不需要修改
    # 比如 training_step 内部调用了 self._should_log_gif，我们需要去修改那个帮助函数
    @logger.catch(onerror=lambda _: sys.exit(1))
    def forward(self, batch: BatchedVideoDatapoint):
        """Forward pass returning per-frame outputs (list of dicts)."""
        out = self.model(batch)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()  # 把碎片还给 CUDA
        return out

    def _apply_gt_stride(
        self, outs_per_frame: List[Dict[str, Any]], target_masks: torch.Tensor
    ):
        """Subsample frames for loss computation according to self.loss_gt_stride.

        Keeps frame 0, then every k-th frame where k = gt_stride.
        """
        if self.loss_gt_stride <= 1:
            return outs_per_frame, target_masks
        total_frames = len(outs_per_frame)
        idxs = list(range(0, total_frames, self.loss_gt_stride))
        outs_sub = [outs_per_frame[i] for i in idxs]
        targets_sub = target_masks[idxs]
        return outs_sub, targets_sub

    @logger.catch(onerror=lambda _: sys.exit(1))
    def _should_log_gif(self, split: str, batch_idx: int) -> bool:
        """Check if GIF should be logged for current step/epoch."""
        # MODIFIED: 使用 self.hparams.visualization 访问配置
        if not self.hparams.visualization.enabled or not self.trainer.is_global_zero:
            return False

        if split == "train":
            total_steps = self.trainer.fit_loop.epoch_loop.batch_progress.total.ready
            steps = self.hparams.visualization.train_every_n_steps
            return steps > 0 and total_steps % steps == 0
        elif split == "val":
            epochs = self.hparams.visualization.val_first_batch_every_n_epochs
            if epochs == 0:
                return batch_idx == 0
            else:
                return batch_idx == 0 and self.current_epoch % epochs == 0

        return False

    @logger.catch(onerror=lambda _: sys.exit(1))
    def _log_gif(
        self,
        stage: str,
        frames: torch.Tensor,
        gt_masks: torch.Tensor,
        outs_per_frame: List[Dict],
        obj_to_cat: List[int],
        split: str,
    ) -> None:
        """Create and log GIF visualization."""
        from sam2_video.utils import create_visualization_gif

        # MODIFIED: 使用 self.hparams.visualization 访问配置
        gif_np = create_visualization_gif(
            frames=frames,
            gt_masks=gt_masks,
            outs_per_frame=outs_per_frame,
            obj_to_cat=obj_to_cat,
            max_length=self.hparams.visualization.max_length,
            stride=self.hparams.visualization.stride,
        )

        caption = f"{self.hparams.visualization.caption} | {split} e{self.current_epoch} s{self.global_step}"

        self.logger.experiment.log(
            {stage: wandb.Video(gif_np, caption=caption, format="gif", fps=4)}
        )

    # [NOTE] training_step 和 validation_step 内部逻辑不需要改变
    # 因为它们调用的是已经修改过的 _should_log_gif 方法
    def training_step(
        self, batch: BatchedVideoDatapoint, batch_idx: int
    ) -> torch.Tensor:
        # Strategic memory management before forward pass
        # Forward pass
        outs_per_frame, obj_to_cat = self.forward(batch)

        # Strategic memory management after forward pass, before loss computation
        target_masks = batch.masks
        outs_for_loss, targets_for_loss = self._apply_gt_stride(
            outs_per_frame, target_masks
        )
        losses = self.criterion(outs_for_loss, targets_for_loss)
        total_loss = losses[CORE_LOSS_KEY]
        self.log(
            "train/total_loss", total_loss, prog_bar=True, logger=True, batch_size=1
        )
        for k, v in losses.items():
            if k == CORE_LOSS_KEY or k == "logits":
                continue
            self.log(f"train/{k}", v, logger=True, batch_size=1)
        self.log(
            "train/learning_rate",
            self.optimizers().param_groups[0]["lr"],
            prog_bar=True,
            logger=True,
            batch_size=1,
        )
        if self._should_log_gif("train", batch_idx):
            frames = batch.img_batch.squeeze(1)
            self._log_gif(
                "train", frames, batch.masks, outs_per_frame, obj_to_cat, "train"
            )
        return total_loss

    def validation_step(
        self, batch: BatchedVideoDatapoint, batch_idx: int
    ) -> torch.Tensor:
        # Strategic memory management before forward pass
        # Forward pass
        outs_per_frame, obj_to_cat = self.forward(batch)

        # Strategic memory management after forward pass, before loss computation
        target_masks = batch.masks
        outs_for_loss, targets_for_loss = self._apply_gt_stride(
            outs_per_frame, target_masks
        )
        losses = self.criterion(outs_for_loss, targets_for_loss)
        total_loss = losses[CORE_LOSS_KEY]
        self.log(
            "val/total_loss",
            total_loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
        )
        for k, v in losses.items():
            if k == CORE_LOSS_KEY or k == "logits":
                continue
            self.log(f"val/{k}", v, on_step=False, on_epoch=True, batch_size=1)
        if self._should_log_gif("val", batch_idx):
            frames = batch.img_batch.squeeze(1)
            self._log_gif("val", frames, batch.masks, outs_per_frame, obj_to_cat, "val")

        # Strategic memory management after validation computation
        return total_loss


class SAM2LightningDataModule(L.LightningDataModule):
    """Lightning data module for SAM2 training."""

    def __init__(self, data: Any, train_shuffle: bool = True):
        """
        Initialize data module with unpacked data configuration section.

        Args:
            data: Data config node from Hydra/YAML.
        """
        super().__init__()

        # MODIFIED: 直接调用 save_hyperparameters()。
        # 它会自动将传入的 'data' 参数保存到 self.hparams.data。
        self.save_hyperparameters()

        # REMOVED: 不再需要手动创建 self.data_cfg。
        # self.data_cfg = data

        self.train_dataset = None
        self.val_dataset = None

    @logger.catch(onerror=lambda _: sys.exit(1))
    def setup(self, stage: str):
        """Setup datasets for training and validation."""
        # 将导入放在方法内部可以避免循环依赖，是一个好习惯
        from sam2_video.data.dataset import COCODataset

        if stage == "fit":
            # MODIFIED: 使用 self.hparams.data 访问配置
            # Training dataset
            self.train_dataset = COCODataset(
                config=self.hparams.data,
                coco_json_path=self.hparams.data.train_path,
            )

            # MODIFIED: 使用 self.hparams.data 访问配置
            # Validation dataset
            self.val_dataset = COCODataset(
                config=self.hparams.data,
                coco_json_path=self.hparams.data.val_path,
            )

    @logger.catch(onerror=lambda _: sys.exit(1))
    def train_dataloader(self):
        """Return training dataloader."""
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        # MODIFIED: 使用 self.hparams.data 访问配置
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.data.batch_size,
            num_workers=self.hparams.data.num_workers,
            shuffle=self.hparams.train_shuffle,  # 注意：在原始代码中为False，通常训练集会设为True
            pin_memory=True,
            collate_fn=sam2_collate_fn,
        )

    @logger.catch(onerror=lambda _: sys.exit(1))
    def val_dataloader(self):
        """Return validation dataloader."""
        if self.val_dataset is None:
            raise RuntimeError(
                "Validation dataset not initialized. Call setup() first."
            )

        # MODIFIED: 使用 self.hparams.data 访问配置
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.data.batch_size,
            num_workers=self.hparams.data.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=sam2_collate_fn,
        )
