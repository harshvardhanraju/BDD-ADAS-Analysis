#!/usr/bin/env python3
"""
Debug dataset directly to see coordinate transformations.
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.data.detr_dataset import BDD100KDETRDataset
import numpy as np

def debug_dataset_direct():
    print("🔍 Debugging Dataset Coordinate Transformations")
    print("=" * 60)
    
    # Create dataset
    dataset = BDD100KDETRDataset(
        annotations_file="data/analysis/processed_10class_corrected/val_annotations_10class.csv",
        images_root="data/raw/bdd100k/bdd100k/images/100k",
        split='val',
        image_size=(416, 416),
        augment=False  # No augmentation for debugging
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Get first item
    image, target = dataset[0]
    
    print(f"\\nImage shape: {image.shape}")
    print(f"Target keys: {list(target.keys())}")
    
    if 'boxes' in target and len(target['boxes']) > 0:
        boxes = target['boxes']
        labels = target['class_labels']
        
        print(f"\\nNumber of objects: {len(boxes)}")
        print(f"Boxes tensor shape: {boxes.shape}")
        print(f"Boxes dtype: {boxes.dtype}")
        
        # Check coordinate ranges
        print(f"\\nBox coordinate ranges (should be 0-1 normalized):")
        print(f"  Center X: [{boxes[:, 0].min():.4f}, {boxes[:, 0].max():.4f}]")
        print(f"  Center Y: [{boxes[:, 1].min():.4f}, {boxes[:, 1].max():.4f}]")
        print(f"  Width:    [{boxes[:, 2].min():.4f}, {boxes[:, 2].max():.4f}]")
        print(f"  Height:   [{boxes[:, 3].min():.4f}, {boxes[:, 3].max():.4f}]")
        
        # Convert back to pixel coordinates for verification
        img_h, img_w = 416, 416  # target image size
        pixel_boxes = []
        
        print(f"\\nFirst 5 boxes converted to pixel coordinates:")
        for i in range(min(5, len(boxes))):
            cx, cy, w, h = boxes[i]
            # Convert back to pixel [x1, y1, x2, y2]
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            
            pixel_boxes.append([x1, y1, x2, y2])
            
            label = labels[i].item()
            class_name = dataset.id_to_class.get(label, f'class_{label}')
            
            print(f"  Box {i}: {class_name}")
            print(f"    Normalized: cx={cx:.4f}, cy={cy:.4f}, w={w:.4f}, h={h:.4f}")
            print(f"    Pixels: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]")
            print(f"    Size: {x2-x1:.1f}x{y2-y1:.1f}")
        
        # Check for invalid boxes
        invalid_boxes = 0
        for i, box in enumerate(boxes):
            cx, cy, w, h = box
            if cx < 0 or cx > 1 or cy < 0 or cy > 1 or w <= 0 or h <= 0:
                invalid_boxes += 1
                if invalid_boxes <= 3:  # Show first 3 invalid boxes
                    print(f"  ❌ Invalid box {i}: cx={cx:.4f}, cy={cy:.4f}, w={w:.4f}, h={h:.4f}")
        
        if invalid_boxes > 0:
            print(f"\\n❌ Found {invalid_boxes} invalid boxes (out of bounds or invalid dimensions)")
        else:
            print(f"\\n✅ All boxes appear to be properly normalized")
    else:
        print("\\n❌ No boxes found in target")
    
    # Check a few more samples
    print(f"\\n" + "=" * 40)
    print("Checking 5 more samples...")
    
    all_valid = True
    for idx in range(1, min(6, len(dataset))):
        image, target = dataset[idx]
        if 'boxes' in target and len(target['boxes']) > 0:
            boxes = target['boxes']
            max_coord = max(boxes[:, 0].max(), boxes[:, 1].max())
            max_dim = max(boxes[:, 2].max(), boxes[:, 3].max())
            
            if max_coord > 1.0 or max_dim > 1.0:
                print(f"  Sample {idx}: ❌ Max coord: {max_coord:.4f}, Max dim: {max_dim:.4f}")
                all_valid = False
            else:
                print(f"  Sample {idx}: ✅ Max coord: {max_coord:.4f}, Max dim: {max_dim:.4f}")
        else:
            print(f"  Sample {idx}: No boxes")
    
    print(f"\\n{'✅ All samples valid' if all_valid else '❌ Some samples have invalid coordinates'}")

if __name__ == "__main__":
    debug_dataset_direct()