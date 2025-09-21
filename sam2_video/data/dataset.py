"""
Dataset module for SAM2 video training.
Single-responsibility: COCO image dataset -> temporal clip dataset -> collate.
KISS: only what is needed for the current training path.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms import functional as F

from loguru import logger
import sys
# icecream import removed

# Import pycocotools for RLE decoding (required)
from pycocotools import mask as mask_utils

# SAM2 data structures for collate function
from sam2_video.data.data_utils import BatchedVideoDatapoint, BatchedVideoMetaData


class COCOImageDataset(Dataset):
    """Minimal COCO-format dataset that returns a single image and its per-category mask tensor."""

    @logger.catch(onerror=lambda _: sys.exit(1))
    def __init__(
        self,
        config: Any,
        json_path: Optional[str] = None,
    ):
        """
        Initialize COCO image dataset.

        Args:
            config: DataConfig containing dataset configuration
        """
        self.config = config
        # Resolve annotation path (explicit override wins; otherwise use train_path)
        self.coco_json_path = Path(json_path or config.train_path)
        self.image_size = config.image_size

        # Validate input
        if not self.coco_json_path.exists():
            raise FileNotFoundError(f"COCO JSON file not found: {self.coco_json_path}")

        # Load COCO annotations
        with open(self.coco_json_path, "r") as f:
            coco_data = json.load(f)

        self.images: List[Dict[str, Any]] = coco_data.get("images", [])
        self.images = [img for img in self.images if img["is_det_keyframe"]]

        self.annotations: List[Dict[str, Any]] = coco_data.get("annotations", [])
        self.categories: List[Dict[str, Any]] = coco_data.get("categories", [])

        # Build category-id -> contiguous index mapping (fail-fast if missing)
        if not self.categories:
            raise ValueError(
                "COCO JSON must include non-empty 'categories' list for fail-fast semantics"
            )
        sorted_cats = sorted(self.categories, key=lambda c: c.get("id", 0))
        self.catid_to_idx = {c["id"]: i for i, c in enumerate(sorted_cats)}
        inferred_num_cats = len(sorted_cats)

        self.num_categories = (
            config.num_categories
            if config.num_categories is not None
            else inferred_num_cats
        )
        # Create image ID to annotations mapping
        self.image_id_to_annotations = {}
        for ann in self.annotations:
            image_id = ann["image_id"]
            if image_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[image_id] = []
            self.image_id_to_annotations[image_id].append(ann)

        # Group images by video for clip generation and sort by frame order
        self.video_to_images = {}
        for img in self.images:
            video_id = img.get("video_id", 0)
            if video_id not in self.video_to_images:
                self.video_to_images[video_id] = []
            self.video_to_images[video_id].append(img)

        for video_id in self.video_to_images:
            self.video_to_images[video_id].sort(
                key=lambda x: x.get("order_in_video", 0)
            )

        # Index: image-id -> index in self.images for O(1) lookups
        self.image_id_to_idx: Dict[int, int] = {
            img["id"]: i for i, img in enumerate(self.images)
        }

        # Default transform
        self.transform = T.Compose(
            [
                T.Resize(config.image_size),
                T.CenterCrop(config.image_size),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Cache for decoded masks per image
        self.mask_cache: Dict[int, torch.Tensor] = {}

        logger.info(f"Loaded COCO image dataset with {len(self.images)} images")

    @logger.catch(onerror=lambda _: sys.exit(1))
    def _decode_rle_mask(self, rle_data: Dict) -> np.ndarray:
        """
        Decode RLE mask to binary mask.

        Args:
            rle_data: RLE encoded mask data
        Returns:
            np.ndarray: Decoded binary mask
        """
        mask = mask_utils.decode(rle_data)
        # Ensure 2D [H, W]
        if mask.ndim == 3:
            mask = mask[..., 0]
        return mask.astype(np.uint8)

    @logger.catch(onerror=lambda _: sys.exit(1))
    def _load_gt_masks_for_image(self, image_id: int) -> torch.Tensor:
        """
        Load ground truth masks for a given image.

        Args:
            image_id: ID of the image
            height: Height of the image
            width: Width of the image

        Returns:
            torch.Tensor: Masks for this image
        """
        cache_key = image_id
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]

        annotations = self.image_id_to_annotations.get(image_id, [])
        masks = torch.zeros(
            (self.num_categories, self.image_size, self.image_size), dtype=torch.bool
        )

        raw_sum = 0
        for ann in annotations:
            segmentation = ann.get("segmentation")
            cat_id = ann.get("category_id")
            if segmentation is None or cat_id is None:
                continue

            # Map raw category id to contiguous index when possible
            cat_idx = self.catid_to_idx.get(cat_id)
            if cat_idx is None or cat_idx >= self.num_categories:
                # Skip categories out of configured range
                continue

            # Decode and transform mask to align with image transforms
            m_np = self._decode_rle_mask(segmentation)  # [H, W]
            m = torch.from_numpy(m_np).unsqueeze(0).float()  # [1, H, W]
            m = F.resize(m, self.image_size, T.InterpolationMode.NEAREST)
            m = F.center_crop(m, [self.image_size, self.image_size])
            m = m.squeeze(0) > 0.5  # [H, W] -> bool

            # Merge multiple instances of same category with OR
            masks[cat_idx] |= m
            raw_sum += m_np.sum()
        # Cache the result and return
        self.mask_cache[cache_key] = masks
        return masks

    def __len__(self):
        return len(self.images)

    @logger.catch(onerror=lambda _: sys.exit(1))
    def __getitem__(self, idx):
        """
        Get a single image with its masks.

        Args:
            idx: Index of the image

        Returns:
            dict: Contains 'image', 'masks', 'image_info', 'video_id'
        """
        img_info = self.images[idx]

        # Load image (require COCO-standard 'file_name'; fail-fast if absent)
        image_path = img_info["path"]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)

        # Load ground truth masks
        image_id = img_info["id"]
        mask_tensor = self._load_gt_masks_for_image(image_id)

        if mask_tensor.sum() == 0:
            logger.warning(f"Skipping image {image_id} due to empty masks.")
            # Randomly pick another index
            new_idx = (idx + 1) % len(self.images)
            return self[new_idx]

        return {
            "image": image_tensor,  # [C, H, W]
            "masks": mask_tensor,  # [N, H, W] where N is number of categories
        }


class VideoDataset(Dataset):
    """Video dataset that generates clips from a COCOImageDataset with configurable stride."""

    @logger.catch(onerror=lambda _: sys.exit(1))
    def __init__(
        self,
        image_dataset: COCOImageDataset,
        config: Any,
    ):
        """
        Initialize video dataset.

        Args:
            image_dataset: COCOImageDataset instance for loading individual images
            config: DataConfig containing video dataset configuration
        """
        self.image_dataset = image_dataset
        self.config = config
        self.video_clip_length = config.video_clip_length
        self.stride = config.stride
        self.image_size = config.image_size

        # Pre-generate all valid video clips
        self.clip_indices = []
        self._generate_clip_indices()

        logger.info(
            f"Generated {len(self.clip_indices)} video clips with stride={self.stride}"
        )

    def _generate_clip_indices(self):
        """Generate all valid clip indices based on video sequences and stride."""
        for video_id, images in self.image_dataset.video_to_images.items():
            video_length = len(images)

            # Generate clips with the specified stride
            clip_start = 0
            while clip_start + self.video_clip_length <= video_length:
                # Store the image indices for this clip
                clip_image_indices = []
                for i in range(self.video_clip_length):
                    img_id = images[clip_start + i]["id"]
                    img_idx = self.image_dataset.image_id_to_idx[img_id]
                    clip_image_indices.append(img_idx)

                self.clip_indices.append(
                    {
                        "video_id": video_id,
                        "clip_start": clip_start,
                        "image_indices": clip_image_indices,
                    }
                )

                clip_start += self.stride

    def __len__(self):
        return len(self.clip_indices)

    @logger.catch(onerror=lambda _: sys.exit(1))
    def __getitem__(self, idx):
        """
        Get a video clip.

        Args:
            idx: Index of the clip

        Returns:
            dict: Contains 'images', 'masks', 'prompts', 'video_id', 'clip_info'
        """
        clip_info = self.clip_indices[idx]
        image_indices = clip_info["image_indices"]

        # Materialize tensors for the clip
        images = torch.stack(
            [self.image_dataset[i]["image"] for i in image_indices]
        )  # [T, C, H, W]
        masks = torch.stack(
            [self.image_dataset[i]["masks"] for i in image_indices]
        )  # [T, N, H, W]

        return {
            "images": images,
            "masks": masks,
        }


class COCODataset(Dataset):
    """COCO-format video dataset: builds temporal clips on top of single-image COCO dataset."""

    @logger.catch(onerror=lambda _: sys.exit(1))
    def __init__(
        self,
        config: Any,
        coco_json_path: Optional[str] = None,  # Override for train/val
    ):
        """
        Initialize COCO dataset using the new architecture.

        Args:
            config: DataConfig containing dataset configuration
            coco_json_path: Optional path override for COCO JSON annotation file
        """
        self.config = config

        # Create the image dataset with explicit annotation path
        image_dataset = COCOImageDataset(
            config=config,
            json_path=coco_json_path or config.train_path,
        )

        # Create the video dataset
        self.video_dataset = VideoDataset(
            image_dataset=image_dataset,
            config=config,
        )

        logger.info(f"Loaded COCO dataset with {len(self.video_dataset)} video clips")

    def __len__(self):
        return len(self.video_dataset)

    @logger.catch(onerror=lambda _: sys.exit(1))
    def __getitem__(self, idx):
        """Get a video clip - delegates to VideoDataset."""
        return self.video_dataset[idx]


def sam2_collate_fn(batch_list: List[Dict[str, torch.Tensor]]) -> BatchedVideoDatapoint:
    """
    Optimized collate function for memory efficiency with batch_size=1.
    Removes unnecessary tensor operations and simplifies indexing structures.
    """
    # Since batch_size=1 is enforced, we can optimize memory usage
    # by avoiding unnecessary tensor operations and simplifying indexing

    # 1. Stack images [T, B, C, H, W] - optimized for batch_size=1
    images = torch.stack([s["images"] for s in batch_list]).permute(1, 0, 2, 3, 4)
    T, B, C, H, W = images.shape
    # KISS: current pipeline assumes batch size of 1 for tracking simplicity
    assert B == 1, (
        f"Only batch_size=1 is supported in the simplified pipeline, got B={B}"
    )

    # 2. Stack masks [B, T, N, H, W] -> [T, B, N, H, W]
    masks = torch.stack([s["masks"] for s in batch_list]).permute(1, 0, 2, 3, 4)
    N = masks.shape[2]  # Number of categories per frame

    # 3. Optimized indexing structures for batch_size=1
    # Since B=1, we can simplify the indexing and avoid nested loops
    obj_to_frame_idx = torch.zeros(T, N, 2, dtype=torch.int)
    for t in range(T):
        for n in range(N):
            obj_to_frame_idx[t, n] = torch.tensor([t, 0], dtype=torch.int)

    # Optimized objects_identifier for batch_size=1
    objects_identifier = torch.zeros(T, N, 3, dtype=torch.long)
    for t in range(T):
        for n in range(N):
            objects_identifier[t, n] = torch.tensor([0, n, t], dtype=torch.long)

    # Optimized frame_orig_size creation
    frame_orig_size = torch.full((T, N, 2), fill_value=H, dtype=torch.long)
    frame_orig_size[..., 1] = W

    # 4. Optimized mask reshaping [T, B*N, H, W]
    masks = masks.reshape(T, B * N, H, W)  # More efficient than flatten(1, 2)

    metadata = BatchedVideoMetaData(
        unique_objects_identifier=objects_identifier,
        frame_orig_size=frame_orig_size,
    )

    return BatchedVideoDatapoint(
        img_batch=images,
        obj_to_frame_idx=obj_to_frame_idx,
        masks=masks.bool(),
        metadata=metadata,
        dict_key="video_batch",
        batch_size=[T],
    )
