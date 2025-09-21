#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from typing import Dict, Set
import logging
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
def load_coco_file(file_path: Path) -> Dict:
    """Load COCO annotation file."""
    logger.info(f"Loading COCO file: {file_path}")
    with open(file_path, 'r') as f:
        return json.load(f)

def get_images_with_annotations(annotations: list) -> Set[int]:
    """Get set of image IDs that have annotations."""
    return {ann['image_id'] for ann in annotations}

def update_is_det_keyframe(coco_data: Dict, dry_run: bool = False) -> Dict:
    """Update is_det_keyframe to false for images with no annotations."""
    images_with_annotations = get_images_with_annotations(coco_data['annotations'])
    
    updated_count = 0
    for image in coco_data['images']:
        image_id = image['id']
        if image_id not in images_with_annotations:
            if image.get('is_det_keyframe', True):  # Only update if currently true
                if dry_run:
                    logger.info(f"Would update image_id {image_id}: {image.get('file_name', 'unknown')} -> is_det_keyframe=false")
                else:
                    image['is_det_keyframe'] = False
                updated_count += 1
    
    logger.info(f"Updated {updated_count} images to is_det_keyframe=false")
    return coco_data

def save_coco_file(coco_data: Dict, file_path: Path) -> None:
    """Save updated COCO annotation file."""
    logger.info(f"Saving updated COCO file: {file_path}")
    with open(file_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

def process_coco_file(file_path: Path, backup: bool = True, dry_run: bool = False) -> None:
    """Process a single COCO file."""
    logger.info(f"Processing {file_path}")
    
    # Create backup if requested
    if backup:
        backup_path = file_path.with_suffix('.json.backup')
        backup_path.write_text(file_path.read_text())
        logger.info(f"Created backup: {backup_path}")
    
    # Load, update, and save
    coco_data = load_coco_file(file_path)
    updated_data = update_is_det_keyframe(coco_data, dry_run)
    
    if not dry_run:
        save_coco_file(updated_data, file_path)

def main():
    parser = argparse.ArgumentParser(
        description="Update is_det_keyframe to false for images with no annotations in COCO files"
    )
    parser.add_argument(
        "files", 
        nargs="*", 
        help="COCO JSON files to process (if empty, processes all *.json files in data/)"
    )
    parser.add_argument(
        "--no-backup", 
        action="store_true", 
        help="Don't create backup files"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Show what would be updated without making changes"
    )
    
    args = parser.parse_args()
    
    # Determine files to process
    if args.files:
        files_to_process = [Path(f) for f in args.files]
    else:
        data_dir = Path("data")
        files_to_process = list(data_dir.glob("*.json"))
    
    if not files_to_process:
        logger.error("No JSON files found to process")
        sys.exit(1)
    
    logger.info(f"Found {len(files_to_process)} files to process")
    
    # Process each file
    for file_path in files_to_process:
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            continue
            
        process_coco_file(file_path, backup=not args.no_backup, dry_run=args.dry_run)
    
    logger.info("All files processed successfully")

if __name__ == "__main__":
    main()