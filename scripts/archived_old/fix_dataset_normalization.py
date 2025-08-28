#!/usr/bin/env python3
"""
Fix the dataset bbox normalization issue by correcting the coordinate transformation.
"""

import torch
import sys
import cv2
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.data.detr_dataset import BDD100KDETRDataset
import pandas as pd
import numpy as np

def debug_image_loading():
    print("🔍 Debugging Image Loading and Bbox Transform")
    print("=" * 60)
    
    # Load annotations to check original coordinates
    csv_path = "data/analysis/processed_10class_corrected/val_annotations_10class.csv"
    df = pd.read_csv(csv_path)
    df = df[df['split'] == 'val'].copy()
    
    # Get first image
    first_image = df.iloc[0]['image_name']
    image_annotations = df[df['image_name'] == first_image]
    
    print(f"Image: {first_image}")
    print(f"Annotations: {len(image_annotations)}")
    
    # Check original bbox coordinates
    print(f"\\nOriginal bbox coordinates (from CSV):")
    for i, row in image_annotations.head(3).iterrows():
        print(f"  Bbox {i}: [{row['bbox_x1']:.1f}, {row['bbox_y1']:.1f}, {row['bbox_x2']:.1f}, {row['bbox_y2']:.1f}]")
    
    # Load actual image file to check dimensions
    images_root = Path("data/raw/bdd100k/bdd100k/images/100k")
    image_path = images_root / "val" / first_image
    
    if image_path.exists():
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        print(f"\\nLoaded image dimensions: {orig_w}x{orig_h}")
        
        # Check if coordinates match image dimensions
        max_x = image_annotations['bbox_x2'].max()
        max_y = image_annotations['bbox_y2'].max()
        print(f"Max bbox coordinates: x={max_x:.1f}, y={max_y:.1f}")
        
        if max_x <= orig_w and max_y <= orig_h:
            print("✅ Bbox coordinates match image dimensions")
        else:
            print("❌ Bbox coordinates exceed image dimensions!")
            
        # Test albumentations transform manually
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        
        # Test transform
        transform = A.Compose([
            A.Resize(height=416, width=416),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['class_labels'],
            min_visibility=0.3
        ))
        
        # Prepare bboxes
        bboxes = []
        labels = []
        for _, row in image_annotations.head(3).iterrows():
            bbox = [row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']]
            bboxes.append(bbox)
            labels.append(0)  # dummy label
            
        print(f"\\nTesting albumentations transform:")
        print(f"Input image size: {orig_w}x{orig_h}")
        print(f"Target image size: 416x416")
        print(f"Input bboxes (first 3):")
        for i, bbox in enumerate(bboxes):
            print(f"  {i}: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        
        # Apply transform
        transformed = transform(
            image=image,
            bboxes=bboxes,
            class_labels=labels
        )
        
        transformed_bboxes = transformed['bboxes']
        print(f"\\nTransformed bboxes (should be in 416x416 space):")
        for i, bbox in enumerate(transformed_bboxes):
            x1, y1, x2, y2 = bbox
            print(f"  {i}: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}] -> size: {x2-x1:.1f}x{y2-y1:.1f}")
            
        # Convert to normalized format manually (like dataset does)
        print(f"\\nManual normalization to DETR format:")
        for i, bbox in enumerate(transformed_bboxes):
            x1, y1, x2, y2 = bbox
            # Normalize by target image size
            x_center = (x1 + x2) / 2.0 / 416
            y_center = (y1 + y2) / 2.0 / 416
            width = (x2 - x1) / 416
            height = (y2 - y1) / 416
            
            print(f"  {i}: cx={x_center:.4f}, cy={y_center:.4f}, w={width:.4f}, h={height:.4f}")
            
            if x_center > 1.0 or y_center > 1.0 or width > 1.0 or height > 1.0:
                print(f"    ❌ Invalid normalized coordinates!")
            else:
                print(f"    ✅ Valid normalized coordinates")
        
    else:
        print(f"❌ Image file not found: {image_path}")

if __name__ == "__main__":
    debug_image_loading()