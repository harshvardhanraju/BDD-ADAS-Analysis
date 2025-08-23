#!/usr/bin/env python3
"""
DETR Evaluation Script for BDD100K Dataset

This script evaluates a trained DETR model on BDD100K validation dataset
and computes mAP metrics.
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import cv2

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.detr_model import create_bdd_detr_model
from scripts.train_detr_demo import BDDDemoDataset, collate_fn


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x1, y1, x2, y2] format."""
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def compute_ap(pred_scores, pred_correct, num_gt):
    """Compute Average Precision for a single class."""
    if num_gt == 0:
        return 0.0
    
    if len(pred_scores) == 0:
        return 0.0
    
    # Sort by confidence
    indices = np.argsort(pred_scores)[::-1]
    pred_correct = pred_correct[indices]
    
    # Compute precision and recall
    tp = np.cumsum(pred_correct)
    fp = np.cumsum(1 - pred_correct)
    
    recall = tp / num_gt
    precision = tp / (tp + fp)
    
    # Compute AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11
    
    return ap


def evaluate_model(model, dataloader, device, iou_threshold=0.5, conf_threshold=0.1):
    """Evaluate model on validation dataset."""
    model.eval()
    
    # Class names
    class_names = ['car', 'truck', 'bus', 'train', 'rider', 'traffic sign', 'traffic light']
    num_classes = len(class_names)
    
    # Collect all predictions and ground truth
    all_predictions = {i: {'scores': [], 'boxes': [], 'image_ids': []} for i in range(num_classes)}
    all_ground_truth = {i: {'boxes': [], 'image_ids': []} for i in range(num_classes)}
    
    print("🔍 Collecting predictions...")
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device)
            batch_size = images.shape[0]
            
            # Get model outputs
            outputs = model.model(pixel_values=images)
            
            # Process each image in batch
            for i in range(batch_size):
                image_id = batch_idx * dataloader.batch_size + i
                
                # Get predictions
                logits = outputs.logits[i]  # [num_queries, num_classes + 1]
                pred_boxes = outputs.pred_boxes[i]  # [num_queries, 4]
                
                # Convert to probabilities and get predictions
                probs = torch.softmax(logits, dim=-1)[:, :-1]  # Remove background
                scores, labels = probs.max(dim=-1)
                
                # Filter by confidence
                keep = scores > conf_threshold
                if keep.sum() > 0:
                    filtered_scores = scores[keep]
                    filtered_labels = labels[keep]
                    filtered_boxes = pred_boxes[keep]
                    
                    # Convert normalized boxes to pixel coordinates
                    target = targets[i]
                    orig_h, orig_w = target['orig_size']
                    
                    # Convert from center format to corner format
                    center_x = filtered_boxes[:, 0] * orig_w
                    center_y = filtered_boxes[:, 1] * orig_h
                    width = filtered_boxes[:, 2] * orig_w
                    height = filtered_boxes[:, 3] * orig_h
                    
                    x1 = center_x - width / 2
                    y1 = center_y - height / 2
                    x2 = center_x + width / 2
                    y2 = center_y + height / 2
                    
                    pixel_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                    
                    # Store predictions by class
                    for j in range(len(filtered_scores)):
                        class_id = filtered_labels[j].item()
                        all_predictions[class_id]['scores'].append(filtered_scores[j].item())
                        all_predictions[class_id]['boxes'].append(pixel_boxes[j].cpu().numpy())
                        all_predictions[class_id]['image_ids'].append(image_id)
                
                # Store ground truth
                target = targets[i]
                if len(target['class_labels']) > 0:
                    gt_labels = target['class_labels']
                    gt_boxes = target['boxes']
                    
                    # Convert normalized GT boxes to pixel coordinates
                    orig_h, orig_w = target['orig_size']
                    
                    for j in range(len(gt_labels)):
                        class_id = gt_labels[j].item()
                        
                        # Convert from center format to corner format
                        center_x = gt_boxes[j, 0] * orig_w
                        center_y = gt_boxes[j, 1] * orig_h
                        width = gt_boxes[j, 2] * orig_w
                        height = gt_boxes[j, 3] * orig_h
                        
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2
                        
                        gt_box = np.array([x1, y1, x2, y2])
                        all_ground_truth[class_id]['boxes'].append(gt_box)
                        all_ground_truth[class_id]['image_ids'].append(image_id)
    
    print("📊 Computing mAP metrics...")
    
    # Compute AP for each class
    class_aps = []
    results = {}
    
    for class_id in range(num_classes):
        class_name = class_names[class_id]
        
        # Get predictions and ground truth for this class
        pred_scores = np.array(all_predictions[class_id]['scores'])
        pred_boxes = np.array(all_predictions[class_id]['boxes'])
        pred_image_ids = np.array(all_predictions[class_id]['image_ids'])
        
        gt_boxes = all_ground_truth[class_id]['boxes']
        gt_image_ids = np.array(all_ground_truth[class_id]['image_ids'])
        
        num_gt = len(gt_boxes)
        
        if num_gt == 0:
            print(f"  {class_name:15}: No ground truth, skipping")
            class_aps.append(0.0)
            results[class_name] = {'ap': 0.0, 'num_gt': 0, 'num_pred': len(pred_scores)}
            continue
        
        if len(pred_scores) == 0:
            print(f"  {class_name:15}: No predictions, AP = 0.0")
            class_aps.append(0.0)
            results[class_name] = {'ap': 0.0, 'num_gt': num_gt, 'num_pred': 0}
            continue
        
        # Create GT lookup by image
        gt_by_image = {}
        for i, image_id in enumerate(gt_image_ids):
            if image_id not in gt_by_image:
                gt_by_image[image_id] = []
            gt_by_image[image_id].append((gt_boxes[i], False))  # (box, used)
        
        # Evaluate each prediction
        pred_correct = np.zeros(len(pred_scores))
        
        # Sort predictions by score
        sorted_indices = np.argsort(pred_scores)[::-1]
        
        for i, pred_idx in enumerate(sorted_indices):
            pred_box = pred_boxes[pred_idx]
            pred_image_id = pred_image_ids[pred_idx]
            
            if pred_image_id in gt_by_image:
                # Find best matching GT box
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, (gt_box, used) in enumerate(gt_by_image[pred_image_id]):
                    if not used:
                        iou = compute_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                
                # Check if match is good enough
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    pred_correct[i] = 1
                    # Mark GT as used
                    gt_by_image[pred_image_id][best_gt_idx] = (
                        gt_by_image[pred_image_id][best_gt_idx][0], True
                    )
        
        # Compute AP
        ap = compute_ap(pred_scores, pred_correct, num_gt)
        class_aps.append(ap)
        
        results[class_name] = {
            'ap': ap,
            'num_gt': num_gt,
            'num_pred': len(pred_scores)
        }
        
        print(f"  {class_name:15}: AP = {ap:.3f} (GT: {num_gt}, Pred: {len(pred_scores)})")
    
    # Compute mAP
    mean_ap = np.mean(class_aps)
    results['mAP'] = mean_ap
    
    print(f"\n📈 Overall mAP@{iou_threshold}: {mean_ap:.3f}")
    
    return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate DETR on BDD100K dataset')
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/detr_demo_checkpoint.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--val-annotations',
        type=str,
        default='data/analysis/processed/val_annotations.csv',
        help='Path to validation annotations'
    )
    parser.add_argument(
        '--images-root',
        type=str,
        default='data/raw/bdd100k/bdd100k/images/100k',
        help='Root directory containing images'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--conf-threshold',
        type=float,
        default=0.1,
        help='Confidence threshold for predictions'
    )
    parser.add_argument(
        '--iou-threshold',
        type=float,
        default=0.5,
        help='IoU threshold for mAP computation'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        default=200,
        help='Maximum number of validation images to evaluate'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    checkpoint_path = Path(args.checkpoint)
    val_ann_path = Path(args.val_annotations)
    images_path = Path(args.images_root)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    if not val_ann_path.exists():
        print(f"❌ Validation annotations not found: {val_ann_path}")
        return
    
    if not images_path.exists():
        print(f"❌ Images directory not found: {images_path}")
        return
    
    print("🚀 Starting DETR evaluation...")
    print(f"📁 Checkpoint: {checkpoint_path}")
    print(f"📁 Validation annotations: {val_ann_path}")
    print(f"📁 Images root: {images_path}")
    print(f"🔧 Confidence threshold: {args.conf_threshold}")
    print(f"🔧 IoU threshold: {args.iou_threshold}")
    print("-" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and checkpoint
    print("Loading model...")
    model = create_bdd_detr_model(pretrained=False)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Create validation dataset
    print("Creating validation dataset...")
    val_dataset = BDDDemoDataset(
        annotations_file=str(val_ann_path),
        images_root=str(images_path),
        split='val',
        max_images=args.max_images
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        dataloader=val_dataloader,
        device=device,
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold
    )
    
    # Save results
    results_path = Path('evaluation_results.json')
    import json
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    print("🎉 Evaluation completed!")


if __name__ == "__main__":
    main()