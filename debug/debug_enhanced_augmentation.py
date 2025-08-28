#!/usr/bin/env python3
"""
Test if enhanced augmentation is causing the bbox normalization issue.
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.data.detr_dataset import BDD100KDETRDataset
import numpy as np

def debug_augmentation_impact():
    print("🔍 Testing Enhanced vs Basic Augmentation")
    print("=" * 60)
    
    # Test with enhanced augmentation (current default)
    print("Testing with enhanced augmentation...")
    dataset_enhanced = BDD100KDETRDataset(
        annotations_file="data/analysis/processed_10class_corrected/val_annotations_10class.csv",
        images_root="data/raw/bdd100k/bdd100k/images/100k",
        split='val',
        image_size=(416, 416),
        augment=False,  # No augmentation for val
        use_enhanced_augmentation=True  # This is probably the issue
    )
    
    # Get first item
    image_enh, target_enh = dataset_enhanced[0]
    
    if len(target_enh['boxes']) > 0:
        boxes_enh = target_enh['boxes']
        max_coord_enh = max(boxes_enh[:, 0].max(), boxes_enh[:, 1].max())
        print(f"Enhanced aug - Max coordinate: {max_coord_enh:.4f}")
        print(f"Enhanced aug - First box: cx={boxes_enh[0,0]:.4f}, cy={boxes_enh[0,1]:.4f}")
        
        if max_coord_enh > 1.0:
            print("❌ Enhanced augmentation produces invalid coordinates")
        else:
            print("✅ Enhanced augmentation produces valid coordinates")
    
    print("\\n" + "-" * 40)
    
    # Test with basic augmentation
    print("Testing with basic augmentation...")
    dataset_basic = BDD100KDETRDataset(
        annotations_file="data/analysis/processed_10class_corrected/val_annotations_10class.csv",
        images_root="data/raw/bdd100k/bdd100k/images/100k",
        split='val',
        image_size=(416, 416),
        augment=False,  # No augmentation for val
        use_enhanced_augmentation=False  # Force basic transforms
    )
    
    # Get first item
    image_basic, target_basic = dataset_basic[0]
    
    if len(target_basic['boxes']) > 0:
        boxes_basic = target_basic['boxes']
        max_coord_basic = max(boxes_basic[:, 0].max(), boxes_basic[:, 1].max())
        print(f"Basic aug - Max coordinate: {max_coord_basic:.4f}")
        print(f"Basic aug - First box: cx={boxes_basic[0,0]:.4f}, cy={boxes_basic[0,1]:.4f}")
        
        if max_coord_basic > 1.0:
            print("❌ Basic augmentation produces invalid coordinates")
        else:
            print("✅ Basic augmentation produces valid coordinates")
    
    print("\\n" + "=" * 60)
    
    # Compare the two
    if len(target_enh['boxes']) > 0 and len(target_basic['boxes']) > 0:
        print(f"Coordinate difference between enhanced and basic:")
        diff = torch.abs(boxes_enh - boxes_basic).max()
        print(f"Max absolute difference: {diff:.6f}")
        
        if diff > 0.001:
            print("❌ Enhanced and basic augmentation produce different results!")
            print("This suggests enhanced augmentation is broken.")
        else:
            print("✅ Enhanced and basic augmentation produce similar results")

if __name__ == "__main__":
    debug_augmentation_impact()