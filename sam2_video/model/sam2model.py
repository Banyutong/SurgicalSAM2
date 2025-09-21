"""
Unified SAM2 model for video training that combines configuration management
with core training functionality. Eliminates the wrapper pattern for better
maintainability.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
import os
import torch
import torch.nn as nn
from loguru import logger
import sys
from sam2.build_sam import build_sam2

# SAM2 core imports
# Use a package-relative import to avoid relying on a top-level
# module named "modeling" when importing this file as part of
# the sam2_video package.
from .modeling.sam2_base import SAM2Base

from sam2_video import utils
from sam2_video.data.data_utils import BatchedVideoDatapoint
from sam2_video.utils import merge_object_results_to_category


class SAM2Model(SAM2Base):
    """
    Unified SAM2 model for video training that combines configuration management
    with core training functionality. Eliminates the wrapper pattern for better
    maintainability.

    This class provides:
    - Simplified SAM2 training without iterative correction complexity
    - Configuration management and utility methods
    - Direct inheritance from SAM2Base for better performance
    - Sequential frame processing for video object tracking
    - Only support batch size one.
    - Not use the multi-mask tracking feature.
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str,
        fintuned_model_path: Optional[str] = None,
        trainable_modules: Optional[List[str]] = None,
        device: str = "cuda",
        prompt_type: str = "point",
        forward_backbone_per_frame_for_eval: bool = False,
        num_pos_points: int = 1,
        num_neg_points: int = 0,
        include_center: bool = True,
        use_activation_checkpoint: bool = False,
        random_init_memory_modules: bool = False,
    ):
        """
        Initialize unified SAM2 model with configuration management and training capabilities.

        Args:
            model_config: ModelConfig containing all model configuration
        """

        # 1. 保存配置参数为成员变量
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.prompt_type = prompt_type
        assert self.prompt_type in ["point", "box", "mask"], (
            f"prompt_type must be one of ['point', 'box', 'mask'], got {self.prompt_type}"
        )
        self.forward_backbone_per_frame_for_eval = forward_backbone_per_frame_for_eval
        self.num_pos_points = num_pos_points
        self.num_neg_points = num_neg_points
        self.include_center = include_center
        self.fintuned_model_path = fintuned_model_path

        # 3. 先构建一个完整的 SAM2 模型
        # Resolve config path relative to repository if needed
        sam2_model = build_sam2(
            self.config_path, self.checkpoint_path, device=device, mode="train"
        )
        super().__init__(
            image_encoder=sam2_model.image_encoder,
            memory_attention=sam2_model.memory_attention,
            memory_encoder=sam2_model.memory_encoder,
            use_activation_checkpoint=use_activation_checkpoint,
        )

        for attr_name in dir(sam2_model):
            if attr_name.startswith("_"):  # 私有成员
                continue
            if attr_name in [
                "forward",
                "forward_image",
                "device",
                "load",
                "prepare_prompt_inputs",
                "track_step",
                "count_trainable_parameters",
                "count_total_parameters",
                "get_info",
            ]:
                continue
            setattr(self, attr_name, getattr(sam2_model, attr_name))

        # Optionally load fine-tuned weights before setting up trainable modules
        # Follows the prescribed loading behavior.
        if self.fintuned_model_path is not None:
            if self.fintuned_model_path.count("all") > 0:
                state_dict = torch.load(self.fintuned_model_path, weights_only=False)
                if not isinstance(state_dict, OrderedDict):
                    self.load_state_dict(state_dict.state_dict(), strict=False)
                else:
                    self.load_state_dict(state_dict, strict=False)
            else:
                self.sam_mask_decoder.load_state_dict(
                    torch.load(self.fintuned_model_path), strict=True
                )
                pe_path = self.fintuned_model_path.replace(
                    ".torch", "_prompt_encoder.torch"
                )
                if os.path.exists(pe_path):
                    self.sam_prompt_encoder.load_state_dict(
                        torch.load(pe_path), strict=True
                    )

        # Optionally reinitialize memory modules to random weights
        if random_init_memory_modules:
            self._randomly_initialize_memory_modules()

        # 6. 按需冻结权重
        trainable_modules = trainable_modules or [
            "memory_attention",
            "memory_encoder",
        ]
        self._setup_trainable_modules(trainable_modules)

        logger.info("Unified SAM2 model initialized successfully")

    def load(self, device: str = None) -> "SAM2Model":
        """
        Compatibility method - model is already loaded during initialization.

        Args:
            device: Device to move model to

        Returns:
            Self for method chaining
        """
        return self

    @logger.catch(onerror=lambda _: sys.exit(1))
    def forward(
        self, input: BatchedVideoDatapoint
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[int]]:
        """Forward pass through simplified SAM2 training.

        Returns a list of per-frame dictionaries aggregated at category level,
        where each dict contains the keys expected by the loss, e.g.,
        "multistep_pred_multimasks_high_res", "multistep_pred_ious",
        and "multistep_object_score_logits".
        """
        if self.training or not self.forward_backbone_per_frame_for_eval:
            # precompute image features on all frames before tracking
            backbone_out = self.forward_image(input.flat_img_batch)
        else:
            # defer image feature computation on a frame until it's being tracked
            backbone_out = {"backbone_fpn": None, "vision_pos_enc": None}

        backbone_out = self.prepare_prompt_inputs(backbone_out, input)
        previous_stages_out = self.forward_tracking(backbone_out, input)
        out = merge_object_results_to_category(
            previous_stages_out,
            backbone_out["obj_to_cat"],
            backbone_out["num_categories"],
        )
        # print(len(backbone_out["obj_to_cat"]))
        return out, backbone_out["obj_to_cat"]

    @logger.catch(onerror=lambda _: sys.exit(1))
    def prepare_prompt_inputs(self, backbone_out, input, start_frame_idx=0):
        """
        Simplified prompt preparation - only adds prompt to first frame.

        Args:
            backbone_out: Backbone features dictionary
            input: BatchedVideoDatapoint with video data
            start_frame_idx: Starting frame index (default 0)

        Returns:
            backbone_out: Updated with prompt inputs for first frame only
        """
        # Load ground-truth masks for all frames
        gt_masks_per_frame = {
            stage_id: masks.unsqueeze(1)  # [B, 1, H_im, W_im]
            for stage_id, masks in enumerate(input.masks)
        }
        backbone_out["num_frames"] = input.num_frames

        # Initialize prompt containers
        backbone_out["mask_inputs_per_frame"] = {}
        backbone_out["point_inputs_per_frame"] = {}

        # Generate prompt for first frame only
        first_frame_cat_mask = gt_masks_per_frame[start_frame_idx]
        first_frame_mask, obj_to_cat, num_categories = utils.cat_to_obj_mask(
            first_frame_cat_mask
        )
        # GT masks should not track gradients
        # first_frame_mask = first_frame_mask.requires_grad_(True)
        backbone_out["obj_to_cat"] = obj_to_cat
        backbone_out["num_categories"] = num_categories

        if self.prompt_type == "mask":
            # Use GT mask directly as prompt
            backbone_out["mask_inputs_per_frame"][start_frame_idx] = first_frame_mask

        elif self.prompt_type == "box":
            # Generate box prompt from GT mask
            points, labels = utils.generate_box_prompt(first_frame_mask)
            point_inputs = {"point_coords": points, "point_labels": labels}
            backbone_out["point_inputs_per_frame"][start_frame_idx] = point_inputs

        elif self.prompt_type == "point":
            # Generate point prompt from GT mask
            points, labels = utils.generate_point_prompt(
                mask=first_frame_mask,
                num_pos_points=self.num_pos_points,
                num_neg_points=self.num_neg_points,
                include_center=self.include_center,
            )
            point_inputs = {"point_coords": points, "point_labels": labels}
            backbone_out["point_inputs_per_frame"][start_frame_idx] = point_inputs

        return backbone_out

    @logger.catch(onerror=lambda _: sys.exit(1))
    def _prepare_backbone_features_per_frame(self, img_batch, img_ids):
        """Compute the image backbone features on the fly for the given img_ids."""
        # Only forward backbone on unique image ids to avoid repetitive computation
        # (if `img_ids` has only one element, it's already unique so we skip this step).
        if img_ids.numel() > 1:
            unique_img_ids, inv_ids = torch.unique(img_ids, return_inverse=True)
        else:
            unique_img_ids, inv_ids = img_ids, None

        # Compute the image features on those unique image ids
        image = img_batch[unique_img_ids]
        backbone_out = self.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self._prepare_backbone_features(backbone_out)
        # Inverse-map image features for `unique_img_ids` to the final image features
        # for the original input `img_ids`.
        if inv_ids is not None:
            image = image[inv_ids]
            vision_feats = [x[:, inv_ids] for x in vision_feats]
            vision_pos_embeds = [x[:, inv_ids] for x in vision_pos_embeds]

        return image, vision_feats, vision_pos_embeds, feat_sizes

    @logger.catch(onerror=lambda _: sys.exit(1))
    def forward_tracking(
        self, backbone_out, input: BatchedVideoDatapoint, return_dict=False
    ):
        """
        Simplified forward video tracking - sequential processing without iterative correction.

        Args:
            backbone_out: Backbone features and prompt setup
            input: BatchedVideoDatapoint with video data
            return_dict: Whether to return dict format (default False)

        Returns:
            List of frame outputs for loss calculation
        """
        img_feats_already_computed = backbone_out["backbone_fpn"] is not None
        if img_feats_already_computed:
            # Prepare the backbone features
            # - vision_feats and vision_pos_embeds are in (HW)BC format
            (
                _,
                vision_feats,
                vision_pos_embeds,
                feat_sizes,
            ) = self._prepare_backbone_features(backbone_out)

        num_frames = backbone_out["num_frames"]
        all_frame_outputs = []
        output_dict = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <lightweight_mem_entry>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <lightweight_mem_entry>}
        }

        # Sequential processing: [0, 1, 2, ..., num_frames-1]
        number_of_objects = len(backbone_out["obj_to_cat"])
        for frame_idx in range(num_frames):
            # First frame is conditioning frame (has prompt), others are not
            is_init_cond_frame = frame_idx == 0

            # Get the image features for current frame
            # img_ids = input.flat_obj_to_img_idx[frame_idx]
            img_ids = torch.tensor([frame_idx]).repeat(number_of_objects)
            if img_feats_already_computed:
                # Retrieve image features according to img_ids (if already computed)
                current_vision_feats = [x[:, img_ids] for x in vision_feats]
                current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds]
            else:
                # Otherwise, compute image features on the fly for the given img_ids
                (
                    _,
                    current_vision_feats,
                    current_vision_pos_embeds,
                    feat_sizes,
                ) = self._prepare_backbone_features_per_frame(
                    input.flat_img_batch, img_ids
                )

            # Get output masks based on this frame's prompts and previous memory
            current_out = self.track_step(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                point_inputs=backbone_out["point_inputs_per_frame"].get(
                    frame_idx, None
                ),
                mask_inputs=backbone_out["mask_inputs_per_frame"].get(frame_idx, None),
                num_frames=num_frames,
                output_dict=output_dict,
            )

            # Create a lightweight memory entry for the memory bank to avoid retaining
            # unnecessary tensors and graphs across frames.
            mem_entry = {
                "maskmem_features": current_out.get("maskmem_features"),
                "maskmem_pos_enc": current_out.get("maskmem_pos_enc"),
                "obj_ptr": current_out.get("obj_ptr"),
            }
            # Detach tensors to prevent cross-frame graph accumulation
            mmf = mem_entry.get("maskmem_features")
            if isinstance(mmf, torch.Tensor):
                mem_entry["maskmem_features"] = mmf.detach()
            mmpe = mem_entry.get("maskmem_pos_enc")
            if isinstance(mmpe, torch.Tensor):
                mem_entry["maskmem_pos_enc"] = mmpe.detach()
            elif isinstance(mmpe, (list, tuple)):
                mem_entry["maskmem_pos_enc"] = [
                    t.detach() if torch.is_tensor(t) else t for t in mmpe
                ]
            optr = mem_entry.get("obj_ptr")
            if isinstance(optr, torch.Tensor):
                mem_entry["obj_ptr"] = optr.detach()

            # Save lightweight memory to the appropriate bank and prune non-cond window
            if is_init_cond_frame:
                output_dict["cond_frame_outputs"][frame_idx] = mem_entry
            else:
                output_dict["non_cond_frame_outputs"][frame_idx] = mem_entry
                # Keep only the last (num_maskmem - 1) non-conditioning frames
                max_noncond = (
                    max(int(self.num_maskmem) - 1, 0)
                    if hasattr(self, "num_maskmem")
                    else 0
                )
                if max_noncond == 0:
                    output_dict["non_cond_frame_outputs"].clear()
                else:
                    keys = sorted(output_dict["non_cond_frame_outputs"].keys())
                    while len(keys) > max_noncond:
                        old_k = keys.pop(0)
                        del output_dict["non_cond_frame_outputs"][old_k]

            # Do not keep mask memory tensors in the per-frame outputs returned for loss
            if "maskmem_features" in current_out:
                current_out["maskmem_features"] = None
            if "maskmem_pos_enc" in current_out:
                current_out["maskmem_pos_enc"] = None

            all_frame_outputs.append(current_out)

            # Release references to per-frame feature holders early
            del current_vision_feats
            del current_vision_pos_embeds

        if return_dict:
            return output_dict

        # Make DDP happy with activation checkpointing by removing unused keys
        keys_to_drop = {"obj_ptr", "maskmem_features", "maskmem_pos_enc"}
        all_frame_outputs = [
            {k: v for k, v in d.items() if k not in keys_to_drop}
            for d in all_frame_outputs
        ]

        return all_frame_outputs

    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        run_mem_encoder=True,  # Whether to run the memory encoder on the predicted masks.
        prev_sam_mask_logits=None,  # The previously predicted SAM mask logits.
        output_dict=None,  # Kept for compatibility but not used in simple version
    ):
        """
        Simplified track step - single prediction without iterative correction.

        Args:
            frame_idx: Current frame index
            is_init_cond_frame: Whether this is the conditioning frame (first frame)
            current_vision_feats: Vision features for current frame
            current_vision_pos_embeds: Vision position embeddings
            feat_sizes: Feature sizes
            point_inputs: Point prompts (only for first frame)
            mask_inputs: Mask prompts (only for first frame)
            num_frames: Total number of frames
            track_in_reverse: Whether tracking in reverse order (default False)
            run_mem_encoder: Whether to run memory encoder (default True)
            prev_sam_mask_logits: Previous SAM mask logits (default None)
            gt_masks: Ground truth masks (for compatibility)
            output_dict: Output dictionary (for compatibility)

        Returns:
            current_out: Frame prediction output for loss calculation
        """
        if output_dict is None:
            output_dict = {}

        # Core tracking step - this calls the base class functionality
        current_out, sam_outputs, high_res_features, pix_feat = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
        )

        # Extract SAM outputs
        (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        # Set up outputs (simplified - no iterative correction)
        # For the simple version, we only have single-step predictions
        current_out["multistep_pred_masks"] = low_res_masks
        current_out["multistep_pred_masks_high_res"] = high_res_masks
        current_out["multistep_pred_multimasks"] = [low_res_multimasks]
        current_out["multistep_pred_multimasks_high_res"] = [high_res_multimasks]
        current_out["multistep_pred_ious"] = [ious]
        current_out["multistep_point_inputs"] = [point_inputs]
        current_out["multistep_object_score_logits"] = [object_score_logits]

        # Expose single-step prompts for downstream visualization/merging
        current_out["point_inputs"] = point_inputs
        current_out["mask_inputs"] = mask_inputs

        # Use final prediction for output (no correction needed)
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # Run memory encoder on the predicted mask to encode it into a new memory feature
        # (that can be used in future frames)
        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out

    def count_trainable_parameters(self) -> int:
        """Count number of trainable parameters."""
        return utils.count_trainable_parameters(self)

    def count_total_parameters(self) -> int:
        """Count total number of parameters."""
        return utils.count_total_parameters(self)

    def get_info(self) -> Dict[str, Any]:
        """
        Get model information including parameter counts.

        Returns:
            Dictionary with model information
        """
        return utils.get_model_info(
            self,
            self.checkpoint_path,
            self.config_path,
            self.device,
        )

    @logger.catch(onerror=lambda _: sys.exit(1))
    def _randomly_initialize_memory_modules(self) -> None:
        """
        Reinitialize the memory modules (memory_attention and memory_encoder)
        with their default random initialization.

        This intentionally limits scope to the two memory-related modules so that
        the rest of the pretrained weights remain intact. It uses each submodule's
        own `reset_parameters` when available.
        """
        modules_to_reset = []
        for name in ["memory_attention", "memory_encoder"]:
            mod = getattr(self, name, None)
            if isinstance(mod, nn.Module):
                modules_to_reset.append((name, mod))
            else:
                logger.warning(f"Memory module '{name}' not found; skipping reinit")

        for name, module in modules_to_reset:
            reset_count = 0
            for m in module.modules():  # includes the module itself
                if hasattr(m, "reset_parameters") and callable(getattr(m, "reset_parameters")):
                    m.reset_parameters()
                    reset_count += 1
            logger.info(
                f"Random-initialized memory module '{name}' via reset_parameters() on {reset_count} submodules"
            )

    def _get_module_mapping(self) -> Dict[str, nn.Module]:
        """
        Get dictionary mapping module names to actual PyTorch modules.

        Returns:
            Dictionary mapping module names to modules
        """
        return {
            "image_encoder": self.image_encoder,
            "memory_attention": self.memory_attention,
            "memory_encoder": self.memory_encoder,
            "prompt_encoder": self.sam_prompt_encoder,
            "mask_decoder": self.sam_mask_decoder,
            "obj_ptr_proj": self.obj_ptr_proj,
            "obj_ptr_tpos_proj": self.obj_ptr_tpos_proj,
        }

    def _setup_trainable_modules(self, trainable_modules: List[str]) -> None:
        """
        Freeze all modules except those in trainable_modules list.

        Args:
            trainable_modules: List of module names that should remain trainable
        """
        module_mapping = self._get_module_mapping()
        utils.setup_trainable_modules(self, module_mapping, trainable_modules)

    def freeze_module(self, module_name: str) -> None:
        """
        Dynamically freeze a specific module.

        Args:
            module_name: Name of the module to freeze
        """
        module_mapping = self._get_module_mapping()
        utils.freeze_module_by_name(module_mapping, module_name)

    def unfreeze_module(self, module_name: str) -> None:
        """
        Dynamically unfreeze a specific module.

        Args:
            module_name: Name of the module to unfreeze
        """
        module_mapping = self._get_module_mapping()
        utils.unfreeze_module_by_name(module_mapping, module_name)

    def get_trainable_modules(self) -> List[str]:
        """
        Return list of currently trainable module names.

        Returns:
            List of module names that have trainable parameters
        """
        module_mapping = self._get_module_mapping()
        return utils.get_trainable_module_names(module_mapping)

    def save_config(self, path: str) -> None:
        """Save model configuration to file."""
        config = {
            "checkpoint_path": self.checkpoint_path,
            "config_path": self.config_path,
            "device": self.device,
            "prompt_type": self.model_config.prompt_type,
            "trainable_modules": self.model_config.trainable_modules,
        }
        utils.save_model_config(config, path)
