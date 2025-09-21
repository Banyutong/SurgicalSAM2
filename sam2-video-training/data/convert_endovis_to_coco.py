#!/usr/bin/env python3
"""
Convert EndoVis 2018 Additional Annotation dataset to COCO format.

This script converts the EndoVis 2018 dataset structure to COCO format
compatible with SAM2 training pipeline.

Performance improvements:
- Uses joblib for parallel processing of images
- Parallelizes image dimension extraction and mask processing
- Configurable number of jobs (default: all CPUs)
"""

import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image
import cv2
from joblib import Parallel, delayed


class EndoVisToCOCOConverter:
    """Convert EndoVis dataset to COCO format."""
    
    def __init__(self, source_dir: str, output_dir: str, n_jobs: int = -1):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.n_jobs = n_jobs
        
        # Load class labels
        with open(self.source_dir / "labels.json", "r") as f:
            self.class_labels = json.load(f)
        
        # Create class ID mapping
        self.class_id_mapping = self.create_class_id_mapping()
        
        self.image_id = 0
        self.annotation_id = 0
        
    def create_coco_categories(self) -> List[Dict]:
        """Create COCO categories from EndoVis labels."""
        categories = []
        for idx, label in enumerate(self.class_labels):
            categories.append({
                "id": idx,
                "name": label["name"]
            })
        return categories

    
    def create_class_id_mapping(self) -> Dict[int, int]:
        """Create mapping from original class IDs to 0-based category IDs."""
        mapping = {}
        for idx, label in enumerate(self.class_labels):
            mapping[label["classid"]] = idx
        return mapping
    
    def get_image_dimensions(self, image_path: str) -> Tuple[int, int]:
        """Get image width and height."""
        with Image.open(image_path) as img:
            return img.size  # Returns (width, height)
    
    def extract_sequence_and_frame(self, filename: str) -> Tuple[str, int]:
        """Extract sequence name and frame number from filename."""
        # Example: seq_10_frame000.png -> ("seq_10_", 0)
        parts = filename.replace(".png", "").split("_")
        seq_name = f"{parts[0]}_{parts[1]}_"
        frame_num = int(parts[2].replace("frame", ""))
        return seq_name, frame_num
    
    def mask_to_rle(self, mask: np.ndarray) -> Dict:
        """Convert binary mask to RLE format."""
        # Ensure mask is binary
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Convert to COCO RLE format
        from pycocotools import mask as mask_utils
        rle = mask_utils.encode(np.asfortranarray(binary_mask))
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    
    def get_bbox_from_mask(self, mask: np.ndarray) -> List[float]:
        """Get bounding box from binary mask in COCO format [x, y, width, height]."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not rows.any() or not cols.any():
            return [0, 0, 0, 0]
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return [float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)]

    
    def process_single_image(self, image_path: str, annotations_dir: Path, base_image_id: int) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Process a single image and its annotations. Returns image entry and annotations."""
        image_filename = os.path.basename(image_path)
        annotation_path = annotations_dir / image_filename
        
        if not annotation_path.exists():
            print(f"Warning: No annotation found for {image_filename}")
            return None, []
        
        # Get image dimensions
        width, height = self.get_image_dimensions(image_path)
        
        # Extract sequence and frame info
        seq_name, frame_num = self.extract_sequence_and_frame(image_filename)
        
        # Create image entry
        image_entry = {
            "file_name": image_filename,
            "path": str(image_path),
            "height": height,
            "width": width,
            "id": base_image_id,
            "video_id": seq_name,
            "is_det_keyframe": True,
            "order_in_video": frame_num
        }
        
        # Process annotations
        annotations = self.process_annotation_mask(str(annotation_path), base_image_id)
        
        return image_entry, annotations
    
    def process_annotation_mask(self, mask_path: str, image_id: int) -> List[Dict]:
        """Process annotation mask and create COCO annotations."""
        annotations = []
        
        # Load mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return annotations
        
        # Get unique class IDs (excluding background=0)
        unique_classes = np.unique(mask)
        unique_classes = unique_classes[unique_classes > 0]
        
        for ann_idx, class_id in enumerate(unique_classes):
            # Create binary mask for this class
            binary_mask = (mask == class_id).astype(np.uint8)
            
            # Skip if mask is empty
            if not binary_mask.any():
                continue
            
            # Get bounding box
            bbox = self.get_bbox_from_mask(binary_mask)
            if bbox == [0, 0, 0, 0]:
                continue
            
            # Calculate area
            area = int(np.sum(binary_mask))
            
            # Convert to RLE
            rle = self.mask_to_rle(binary_mask)
            
            # Map original class ID to 0-based category ID
            category_id = self.class_id_mapping.get(int(class_id), int(class_id))
            
            annotation = {
                "id": ann_idx,  # Temporary ID, will be reassigned later
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0
            }
            
            annotations.append(annotation)
        
        return annotations
    
    def convert_split(self, split: str) -> Dict[str, Any]:
        """Convert train or val split to COCO format."""
        split_dir = self.source_dir / split
        images_dir = split_dir / "images"
        annotations_dir = split_dir / "annotations"
        
        if not images_dir.exists() or not annotations_dir.exists():
            raise FileNotFoundError(f"Split directory {split_dir} not found or incomplete")
        
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": self.create_coco_categories()
        }
        
        # Get all image files
        image_files = sorted(glob.glob(str(images_dir / "*.png")))
        
        print(f"Processing {len(image_files)} images with {self.n_jobs} jobs...")
        
        # Process images in parallel
        results = Parallel(n_jobs=self.n_jobs, verbose=1)(
            delayed(self.process_single_image)(
                image_path, annotations_dir, self.image_id + idx
            ) for idx, image_path in enumerate(image_files)
        )
        
        # Collect results and reassign annotation IDs
        for image_entry, annotations in results:
            if image_entry is not None:
                coco_data["images"].append(image_entry)
                
                # Reassign annotation IDs to avoid conflicts
                for ann in annotations:
                    ann["id"] = self.annotation_id
                    self.annotation_id += 1
                
                coco_data["annotations"].extend(annotations)
                self.image_id += 1
        
        return coco_data
    
    def convert_dataset(self):
        """Convert the entire dataset to COCO format."""
        print("Converting EndoVis dataset to COCO format...")
        
        # Convert train split
        if (self.source_dir / "train").exists():
            print("Converting train split...")
            train_coco = self.convert_split("train")
            
            output_file = self.output_dir / "endovis18_coco_annotations_train.json"
            with open(output_file, "w") as f:
                json.dump(train_coco, f, indent=2)
            print(f"Train annotations saved to {output_file}")
            print(f"Train: {len(train_coco['images'])} images, {len(train_coco['annotations'])} annotations")
        
        # Reset IDs for val split
        self.image_id = 0
        self.annotation_id = 0
        
        # Convert val split
        if (self.source_dir / "val").exists():
            print("Converting val split...")
            val_coco = self.convert_split("val")
            
            output_file = self.output_dir / "endovis18_coco_annotations_val.json"
            with open(output_file, "w") as f:
                json.dump(val_coco, f, indent=2)
            print(f"Val annotations saved to {output_file}")
            print(f"Val: {len(val_coco['images'])} images, {len(val_coco['annotations'])} annotations")
        
        print("Conversion completed!")


def main():
    """Main function to run the conversion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert EndoVis dataset to COCO format")
    parser.add_argument(
        "--source_dir", 
        type=str, 
        default="/bd_byta6000i0/users/dataset/EndoVis_2018_Additional_Annotation",
        help="Path to EndoVis dataset directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./",
        help="Output directory for COCO annotations"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel jobs (-1 for all CPUs)"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    try:
        import pycocotools.mask
    except ImportError:
        print("Error: pycocotools is required. Install with: pip install pycocotools")
        return
    
    converter = EndoVisToCOCOConverter(args.source_dir, args.output_dir, args.n_jobs)
    converter.convert_dataset()


if __name__ == "__main__":
    main()