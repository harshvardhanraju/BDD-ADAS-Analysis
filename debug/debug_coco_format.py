#!/usr/bin/env python3
"""
Debug COCO format conversion to see why IoU matching fails.
"""

import torch
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.models.detr_model import BDD100KDETR, BDD100KDetrConfig
from src.data.detr_dataset import BDD100KDETRDataset
from torch.utils.data import DataLoader
import numpy as np

def debug_coco_format():
    print("🔍 Debugging COCO Format Conversion")
    print("=" * 60)
    
    # Load model
    config = BDD100KDetrConfig()
    model = BDD100KDETR(config)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/complete_10class_demo/checkpoint_epoch_015.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create small validation dataset
    val_dataset = BDD100KDETRDataset(
        annotations_file="data/analysis/processed_10class_corrected/val_annotations_10class.csv",
        images_root="data/raw/bdd100k/bdd100k/images/100k",
        split='val',
        image_size=(416, 416),
        augment=False
    )
    
    # Create dataloader for just 1 image
    val_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, range(1)),
        batch_size=1,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Import our fixed extraction functions
    from scripts.run_comprehensive_evaluation import extract_predictions_from_output, extract_ground_truth
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            images, targets = batch
            images = images.to(device)
            
            print(f"Image shape: {images[0].shape}")
            
            # Forward pass
            outputs = model(images)
            
            image_id = "debug_img_0"
            target = targets[0]
            image_size = (images.shape[-2], images.shape[-1])  # (416, 416)
            
            print(f"Image size for conversion: {image_size}")
            
            # Extract predictions and GT using our functions
            predictions = extract_predictions_from_output(
                outputs, 0, image_id, confidence_threshold=0.05, image_size=image_size
            )
            ground_truth = extract_ground_truth(target, image_id, image_size=image_size)
            
            print(f"\\nFound {len(predictions)} predictions")
            print(f"Found {len(ground_truth)} ground truth annotations")
            
            # Show first few predictions
            print("\\nFirst 5 predictions:")
            for i, pred in enumerate(predictions[:5]):
                bbox = pred['bbox']
                print(f"  Pred {i}: class={pred['category_id']}, score={pred['score']:.4f}")
                print(f"    bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
                
            # Show first few ground truth
            print("\\nFirst 5 ground truth:")
            for i, gt in enumerate(ground_truth[:5]):
                bbox = gt['bbox']
                print(f"  GT {i}: class={gt['category_id']}, area={gt['area']:.1f}")
                print(f"    bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
            
            # Check for obvious issues
            pred_bboxes = np.array([p['bbox'] for p in predictions])
            gt_bboxes = np.array([g['bbox'] for g in ground_truth])
            
            if len(pred_bboxes) > 0:
                print(f"\\nPrediction bbox stats:")
                print(f"  X range: [{pred_bboxes[:, 0].min():.1f}, {pred_bboxes[:, 0].max():.1f}]")
                print(f"  Y range: [{pred_bboxes[:, 1].min():.1f}, {pred_bboxes[:, 1].max():.1f}]")
                print(f"  W range: [{pred_bboxes[:, 2].min():.1f}, {pred_bboxes[:, 2].max():.1f}]")
                print(f"  H range: [{pred_bboxes[:, 3].min():.1f}, {pred_bboxes[:, 3].max():.1f}]")
                
                # Check for negative values
                print(f"  Negative X: {(pred_bboxes[:, 0] < 0).sum()}")
                print(f"  Negative Y: {(pred_bboxes[:, 1] < 0).sum()}")
                print(f"  Zero/negative W: {(pred_bboxes[:, 2] <= 0).sum()}")
                print(f"  Zero/negative H: {(pred_bboxes[:, 3] <= 0).sum()}")
            
            if len(gt_bboxes) > 0:
                print(f"\\nGround truth bbox stats:")
                print(f"  X range: [{gt_bboxes[:, 0].min():.1f}, {gt_bboxes[:, 0].max():.1f}]")
                print(f"  Y range: [{gt_bboxes[:, 1].min():.1f}, {gt_bboxes[:, 1].max():.1f}]") 
                print(f"  W range: [{gt_bboxes[:, 2].min():.1f}, {gt_bboxes[:, 2].max():.1f}]")
                print(f"  H range: [{gt_bboxes[:, 3].min():.1f}, {gt_bboxes[:, 3].max():.1f}]")
                
                # Check for negative values
                print(f"  Negative X: {(gt_bboxes[:, 0] < 0).sum()}")
                print(f"  Negative Y: {(gt_bboxes[:, 1] < 0).sum()}")
                print(f"  Zero/negative W: {(gt_bboxes[:, 2] <= 0).sum()}")
                print(f"  Zero/negative H: {(gt_bboxes[:, 3] <= 0).sum()}")
            
            # Check raw model outputs
            print(f"\\nRaw model outputs for debugging:")
            logits = outputs['logits'][0]  # [100, 11]
            boxes = outputs['pred_boxes'][0]  # [100, 4]
            
            print(f"Raw boxes range:")
            print(f"  CX: [{boxes[:, 0].min():.4f}, {boxes[:, 0].max():.4f}]")
            print(f"  CY: [{boxes[:, 1].min():.4f}, {boxes[:, 1].max():.4f}]")
            print(f"  W:  [{boxes[:, 2].min():.4f}, {boxes[:, 2].max():.4f}]")
            print(f"  H:  [{boxes[:, 3].min():.4f}, {boxes[:, 3].max():.4f}]")
            
            # Check target format
            print(f"\\nTarget format:")
            print(f"  Keys: {list(target.keys())}")
            if 'boxes' in target:
                raw_gt_boxes = target['boxes']
                print(f"  GT boxes shape: {raw_gt_boxes.shape}")
                print(f"  Raw GT boxes range:")
                print(f"    CX: [{raw_gt_boxes[:, 0].min():.4f}, {raw_gt_boxes[:, 0].max():.4f}]")
                print(f"    CY: [{raw_gt_boxes[:, 1].min():.4f}, {raw_gt_boxes[:, 1].max():.4f}]")
                print(f"    W:  [{raw_gt_boxes[:, 2].min():.4f}, {raw_gt_boxes[:, 2].max():.4f}]")
                print(f"    H:  [{raw_gt_boxes[:, 3].min():.4f}, {raw_gt_boxes[:, 3].max():.4f}]")
            
            break

if __name__ == "__main__":
    debug_coco_format()