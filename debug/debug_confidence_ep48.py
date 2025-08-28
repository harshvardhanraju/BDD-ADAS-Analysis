#!/usr/bin/env python3

import torch
import json
import numpy as np
from pathlib import Path
from transformers import DetrForObjectDetection, DetrImageProcessor
from src.data.detr_dataset import BDD100KDETRDataset
from src.models.detr_model import BDD100KDETR, BDD100KDetrConfig
# from src.evaluation.coco_evaluator import extract_predictions_from_output
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_model_confidence():
    """Debug confidence scores from ep48 model"""
    
    # Load model
    model_path = "checkpoints/complete_10class_demo/checkpoint_epoch_048.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading model from: {model_path}")
    config = BDD100KDetrConfig()
    model = BDD100KDETR(config=config)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create dataloader for just a few samples
    dataset = BDD100KDETRDataset(
        data_dir="data/analysis/processed_10class_corrected",
        images_root="data/raw/bdd100k/bdd100k/images/100k",
        split='val',
        use_enhanced_augmentation=False
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=dataset.collate_fn
    )
    
    # Process just 5 samples
    all_confidences = []
    all_predictions = []
    
    with torch.no_grad():
        for i, (images, targets, batch_info) in enumerate(dataloader):
            if i >= 5:  # Only process first 5 batches
                break
                
            images = images.to(device)
            outputs = model(images)
            
            # Extract raw predictions
            logits = outputs['logits'][0]  # First image in batch
            boxes = outputs['pred_boxes'][0]
            
            # Get confidence scores (apply softmax to logits)
            probs = torch.softmax(logits, dim=-1)
            
            # Get max probability for each detection (excluding background class)
            max_probs = probs[:, :-1].max(dim=-1)[0]  # Exclude background (last class)
            
            print(f"\nSample {i+1}:")
            print(f"  Number of predictions: {len(max_probs)}")
            print(f"  Max confidence: {max_probs.max().item():.6f}")
            print(f"  Min confidence: {max_probs.min().item():.6f}")
            print(f"  Mean confidence: {max_probs.mean().item():.6f}")
            print(f"  Predictions above 0.1: {(max_probs > 0.1).sum().item()}")
            print(f"  Predictions above 0.05: {(max_probs > 0.05).sum().item()}")
            print(f"  Predictions above 0.01: {(max_probs > 0.01).sum().item()}")
            print(f"  Predictions above 0.001: {(max_probs > 0.001).sum().item()}")
            
            all_confidences.extend(max_probs.cpu().numpy())
    
    # Overall statistics
    all_confidences = np.array(all_confidences)
    print(f"\n=== OVERALL STATISTICS ===")
    print(f"Total predictions analyzed: {len(all_confidences)}")
    print(f"Overall max confidence: {all_confidences.max():.6f}")
    print(f"Overall mean confidence: {all_confidences.mean():.6f}")
    print(f"Overall std confidence: {all_confidences.std():.6f}")
    print(f"Predictions above 0.1: {(all_confidences > 0.1).sum()} ({(all_confidences > 0.1).sum()/len(all_confidences)*100:.1f}%)")
    print(f"Predictions above 0.05: {(all_confidences > 0.05).sum()} ({(all_confidences > 0.05).sum()/len(all_confidences)*100:.1f}%)")
    print(f"Predictions above 0.01: {(all_confidences > 0.01).sum()} ({(all_confidences > 0.01).sum()/len(all_confidences)*100:.1f}%)")
    print(f"Predictions above 0.001: {(all_confidences > 0.001).sum()} ({(all_confidences > 0.001).sum()/len(all_confidences)*100:.1f}%)")

if __name__ == "__main__":
    debug_model_confidence()