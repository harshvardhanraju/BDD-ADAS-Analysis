#!/usr/bin/env python3
"""
Debug script to check model predictions and find evaluation issues.
"""

import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from src.models.detr_model import BDD100KDETR, BDD100KDetrConfig
from src.data.detr_dataset import BDD100KDETRDataset
from torch.utils.data import DataLoader
import numpy as np

def debug_model_predictions():
    print("🔍 Debugging BDD100K Model Predictions")
    print("=" * 60)
    
    # Load model
    config = BDD100KDetrConfig()
    model = BDD100KDETR(config)
    
    # Load checkpoint
    checkpoint_path = "checkpoints/complete_10class_demo/checkpoint_epoch_015.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Loaded model from epoch {checkpoint['epoch']}")
    print(f"✅ Best validation loss: {checkpoint['best_val_loss']:.4f}")
    
    # Create small validation dataset
    val_dataset = BDD100KDETRDataset(
        annotations_file="data/analysis/processed_10class_corrected/val_annotations_10class.csv",
        images_root="data/raw/bdd100k/bdd100k/images/100k",
        split='val',
        image_size=(416, 416),
        augment=False
    )
    
    # Create dataloader for just 5 images
    val_loader = DataLoader(
        torch.utils.data.Subset(val_dataset, range(5)),
        batch_size=2,
        shuffle=False,
        collate_fn=val_dataset.collate_fn
    )
    
    print(f"✅ Created dataset with {len(val_dataset)} images")
    print("Class names:", config.class_names)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"✅ Using device: {device}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            print(f"\n--- Batch {batch_idx + 1} ---")
            images, targets = batch
            images = images.to(device)
            
            print(f"Batch size: {len(images)}")
            print(f"Image shape: {images[0].shape}")
            
            # Forward pass
            outputs = model(images)
            
            print(f"Output logits shape: {outputs.logits.shape}")
            print(f"Output pred_boxes shape: {outputs.pred_boxes.shape}")
            
            # Check predictions for first image
            img_logits = outputs.logits[0]  # [100, 11]
            img_boxes = outputs.pred_boxes[0]  # [100, 4]
            
            # Convert to probabilities
            probs = torch.softmax(img_logits, dim=-1)
            
            # Check confidence levels
            class_probs = probs[:, :-1]  # Exclude background
            max_probs, pred_classes = torch.max(class_probs, dim=1)
            
            print(f"\nPrediction Analysis for Image 0:")
            print(f"Max confidence: {max_probs.max().item():.4f}")
            print(f"Min confidence: {max_probs.min().item():.4f}")
            print(f"Mean confidence: {max_probs.mean().item():.4f}")
            
            # Count predictions above different thresholds
            for threshold in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]:
                count = (max_probs > threshold).sum().item()
                print(f"Predictions > {threshold}: {count}")
            
            # Check ground truth for first image
            target = targets[0]
            print(f"\nGround Truth for Image 0:")
            print(f"GT classes: {target.get('class_labels', [])}")
            print(f"GT boxes: {target.get('boxes', []).shape if 'boxes' in target else 'None'}")
            
            # Show top predictions with very low threshold
            threshold = 0.01
            valid_preds = max_probs > threshold
            if valid_preds.any():
                top_indices = valid_preds.nonzero().squeeze()
                if top_indices.numel() > 0:
                    if top_indices.dim() == 0:
                        top_indices = [top_indices.item()]
                    else:
                        top_indices = top_indices.tolist()[:10]  # Top 10
                        
                    print(f"\nTop predictions (threshold={threshold}):")
                    for idx in top_indices:
                        conf = max_probs[idx].item()
                        cls = pred_classes[idx].item()
                        box = img_boxes[idx]
                        class_name = config.class_names[cls] if cls < len(config.class_names) else f"class_{cls}"
                        print(f"  Query {idx}: {class_name} ({conf:.4f}) box: [{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}, {box[3]:.3f}]")
            else:
                print(f"❌ No predictions above threshold {threshold}")
            
            if batch_idx == 0:  # Only analyze first batch
                break
    
    print("\n" + "=" * 60)
    print("Debug analysis complete!")

if __name__ == "__main__":
    debug_model_predictions()