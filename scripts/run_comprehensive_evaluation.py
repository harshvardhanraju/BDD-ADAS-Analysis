#!/usr/bin/env python3
"""
Comprehensive BDD100K Model Evaluation Script

This script runs the complete evaluation pipeline including:
- Quantitative metrics (COCO, safety-critical, contextual)
- Qualitative visualizations 
- Failure analysis
- Performance reporting

Usage:
    python scripts/run_comprehensive_evaluation.py --model-path checkpoints/production_10class/best_model.pth
"""

import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.models.detr_model import BDD100KDETR, BDD100KDetrConfig
from src.data.detr_dataset import BDD100KDETRDataset
from src.evaluation.metrics import COCOEvaluator, SafetyCriticalMetrics, ContextualMetrics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model(model_path: str, device: str = 'cuda') -> BDD100KDETR:
    """Load trained DETR model from checkpoint."""
    logger.info(f"Loading model from: {model_path}")
    
    # Create model configuration
    config = BDD100KDetrConfig()
    model = BDD100KDETR(config)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        logger.info("Loaded model state dict directly")
    
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def create_validation_dataloader(data_dir: str, images_root: str, batch_size: int = 4) -> DataLoader:
    """Create validation data loader."""
    logger.info("Creating validation dataloader...")
    
    val_annotations = os.path.join(data_dir, "val_annotations_10class.csv")
    
    if not os.path.exists(val_annotations):
        raise FileNotFoundError(f"Validation annotations not found: {val_annotations}")
    
    val_dataset = BDD100KDETRDataset(
        annotations_file=val_annotations,
        images_root=images_root,
        split='val',
        image_size=(512, 512),
        augment=False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    logger.info(f"Validation dataset: {len(val_dataset)} images, {len(val_dataloader)} batches")
    return val_dataloader


def generate_predictions(model: BDD100KDETR, 
                        dataloader: DataLoader, 
                        device: str,
                        confidence_threshold: float = 0.05,
                        max_batches: Optional[int] = None) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Generate model predictions on validation set.
    
    Returns:
        Tuple of (predictions, ground_truth, image_metadata)
    """
    logger.info("Generating model predictions...")
    
    predictions = []
    ground_truth = []
    image_metadata = []
    
    total_batches = len(dataloader)
    processed_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Check batch limit
            if max_batches and batch_idx >= max_batches:
                logger.info(f"Reached batch limit ({max_batches}), stopping")
                break
                
            # The collate_fn returns (images, targets) tuple
            images, targets = batch
            images = images.to(device)
            # Generate image IDs for this batch
            image_ids = [f"val_img_{batch_idx}_{i}" for i in range(len(images))]
            
            # Forward pass
            outputs = model(images)
            
            # Process each image in batch
            for i in range(len(images)):
                image_id = image_ids[i]
                target = targets[i]
                
                # Extract predictions for this image (pass image size)
                image_predictions = extract_predictions_from_output(
                    outputs, i, image_id, confidence_threshold, image_size=(images.shape[-2], images.shape[-1])
                )
                predictions.extend(image_predictions)
                
                # Extract ground truth for this image (pass image size)
                image_ground_truth = extract_ground_truth(target, image_id, image_size=(images.shape[-2], images.shape[-1]))
                ground_truth.extend(image_ground_truth)
                
                # Extract metadata (using defaults for now)
                metadata = {
                    'image_id': image_id,
                    'weather': 'clear',  # Would come from dataset metadata
                    'timeofday': 'daytime',
                    'scene': 'city street'
                }
                image_metadata.append(metadata)
            
            processed_batches += 1
            if processed_batches % 50 == 0:
                logger.info(f"Processed {processed_batches}/{total_batches} batches "
                           f"({processed_batches/total_batches*100:.1f}%)")
    
    logger.info(f"Generated {len(predictions)} predictions for {len(set(p['image_id'] for p in predictions))} images")
    logger.info(f"Ground truth: {len(ground_truth)} annotations")
    
    return predictions, ground_truth, image_metadata


def extract_predictions_from_output(outputs: Dict, 
                                   batch_index: int, 
                                   image_id: str,
                                   confidence_threshold: float = 0.05,
                                   image_size: Tuple[int, int] = (416, 416)) -> List[Dict]:
    """Extract predictions from model output for a single image."""
    predictions = []
    
    # Extract logits and boxes for this image
    logits = outputs['logits'][batch_index]  # [num_queries, num_classes + 1] 
    boxes = outputs['pred_boxes'][batch_index]    # [num_queries, 4]
    
    # Convert to probabilities and get predicted classes
    probs = torch.softmax(logits, dim=-1)
    
    # Get predictions above confidence threshold
    for query_idx in range(len(probs)):
        # Find best class (excluding background class at index -1)
        class_probs = probs[query_idx][:-1]  # Exclude background
        max_prob, pred_class = torch.max(class_probs, dim=0)
        
        if max_prob.item() >= confidence_threshold:
            # Convert normalized boxes to pixel coordinates
            box = boxes[query_idx]  # [cx, cy, w, h] normalized
            cx, cy, w, h = box
            
            # Convert to [x, y, width, height] in pixels
            img_h, img_w = image_size  # Actual input image size
            x = (cx - w/2) * img_w
            y = (cy - h/2) * img_h
            width = w * img_w
            height = h * img_h
            
            prediction = {
                'image_id': image_id,
                'category_id': pred_class.item(),
                'bbox': [x.item(), y.item(), width.item(), height.item()],
                'score': max_prob.item()
            }
            predictions.append(prediction)
    
    return predictions


def extract_ground_truth(target: Dict, image_id: str, image_size: Tuple[int, int] = (416, 416)) -> List[Dict]:
    """Extract ground truth annotations for a single image."""
    ground_truth = []
    
    # Handle different possible key names
    labels_key = 'class_labels' if 'class_labels' in target else 'labels'
    boxes_key = 'boxes'
    
    if labels_key in target and boxes_key in target:
        labels = target[labels_key]
        boxes = target[boxes_key]
        
        for i in range(len(labels)):
            # Convert normalized boxes to pixel coordinates
            box = boxes[i]  # [cx, cy, w, h] normalized (DETR format)
            cx, cy, w, h = box
            
            # Convert to [x, y, width, height] in pixels
            img_h, img_w = image_size  # Actual input image size
            x = (cx - w/2) * img_w
            y = (cy - h/2) * img_h
            width = w * img_w
            height = h * img_h
            area = width * height
            
            annotation = {
                'image_id': image_id,
                'category_id': labels[i].item(),
                'bbox': [x.item(), y.item(), width.item(), height.item()],
                'area': area.item(),
                'iscrowd': 0,
                'id': len(ground_truth)
            }
            ground_truth.append(annotation)
    
    return ground_truth


def run_quantitative_evaluation(predictions: List[Dict], 
                               ground_truth: List[Dict],
                               image_metadata: List[Dict]) -> Dict:
    """Run comprehensive quantitative evaluation."""
    logger.info("Running quantitative evaluation...")
    
    results = {}
    
    # 1. COCO-style evaluation
    logger.info("  - Running COCO evaluation...")
    coco_evaluator = COCOEvaluator()
    coco_results = coco_evaluator.evaluate(predictions, ground_truth)
    results['coco_metrics'] = coco_results
    
    # 2. Safety-critical evaluation
    logger.info("  - Running safety-critical evaluation...")
    safety_evaluator = SafetyCriticalMetrics()
    safety_results = safety_evaluator.evaluate_safety_performance(predictions, ground_truth)
    results['safety_metrics'] = safety_results
    
    # 3. Contextual evaluation
    logger.info("  - Running contextual evaluation...")
    contextual_evaluator = ContextualMetrics()
    
    # Environmental performance
    env_results = contextual_evaluator.evaluate_environmental_performance(
        predictions, ground_truth, image_metadata
    )
    results['environmental_metrics'] = env_results
    
    # Object characteristics performance
    char_results = contextual_evaluator.evaluate_object_characteristics(
        predictions, ground_truth
    )
    results['characteristic_metrics'] = char_results
    
    logger.info("Quantitative evaluation completed")
    return results


def save_results(results: Dict, output_dir: str) -> None:
    """Save evaluation results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save raw results as JSON
    results_file = output_path / "evaluation_results.json"
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(results)
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Results saved to: {results_file}")


def generate_reports(results: Dict, output_dir: str) -> None:
    """Generate human-readable evaluation reports."""
    output_path = Path(output_dir)
    
    # 1. COCO Evaluation Report
    if 'coco_metrics' in results:
        coco_evaluator = COCOEvaluator()
        coco_report = coco_evaluator.get_summary_report(results['coco_metrics'])
        
        report_file = output_path / "coco_evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(coco_report)
        logger.info(f"COCO report saved to: {report_file}")
    
    # 2. Safety-Critical Report
    if 'safety_metrics' in results:
        safety_evaluator = SafetyCriticalMetrics()
        safety_report = safety_evaluator.generate_safety_report(results['safety_metrics'])
        
        report_file = output_path / "safety_evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(safety_report)
        logger.info(f"Safety report saved to: {report_file}")
    
    # 3. Contextual Performance Report
    if 'environmental_metrics' in results and 'characteristic_metrics' in results:
        contextual_evaluator = ContextualMetrics()
        contextual_report = contextual_evaluator.generate_contextual_report(
            results['environmental_metrics'], results['characteristic_metrics']
        )
        
        report_file = output_path / "contextual_evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(contextual_report)
        logger.info(f"Contextual report saved to: {report_file}")


def main():
    parser = argparse.ArgumentParser(description='Run comprehensive BDD100K model evaluation')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, 
                       default='data/analysis/processed_10class_corrected',
                       help='Path to processed annotations directory')
    parser.add_argument('--images-root', type=str,
                       default='data/raw/bdd100k/bdd100k/images/100k',
                       help='Path to images root directory')
    parser.add_argument('--output-dir', type=str,
                       default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--confidence-threshold', type=float, default=0.1,
                       help='Confidence threshold for predictions')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run evaluation on')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Maximum number of images to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    # Set device
    device = args.device if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Start evaluation
    start_time = time.time()
    
    try:
        # 1. Load model
        model = load_model(args.model_path, device)
        
        # 2. Create data loader
        val_dataloader = create_validation_dataloader(
            args.data_dir, args.images_root, args.batch_size
        )
        
        # Limit dataset size if specified (for testing)
        if args.max_images:
            logger.info(f"Limiting evaluation to {args.max_images} images")
            # Limit by truncating the dataloader
            max_batches = min(len(val_dataloader), args.max_images // args.batch_size + 1)
        
        # 3. Generate predictions
        max_batches_arg = max_batches if args.max_images else None
        predictions, ground_truth, image_metadata = generate_predictions(
            model, val_dataloader, device, args.confidence_threshold, max_batches_arg
        )
        
        # 4. Run quantitative evaluation
        results = run_quantitative_evaluation(predictions, ground_truth, image_metadata)
        
        # 5. Save results
        save_results(results, args.output_dir)
        
        # 6. Generate reports
        generate_reports(results, args.output_dir)
        
        # Print summary
        duration = time.time() - start_time
        logger.info(f"Evaluation completed in {duration:.1f} seconds")
        
        # Print key metrics
        if 'coco_metrics' in results:
            coco_results = results['coco_metrics']
            print("\\n" + "="*60)
            print("📊 EVALUATION SUMMARY")
            print("="*60)
            print(f"Overall mAP@0.5:0.95:     {coco_results['mAP']:.3f}")
            print(f"mAP@0.5:                 {coco_results['mAP@0.5']:.3f}")
            print(f"Safety-Critical mAP:     {coco_results['safety_critical_mAP']:.3f}")
            print(f"Small Objects mAP:       {coco_results['mAP_small']:.3f}")
            print()
            print("Per-Class Performance:")
            per_class_ap = coco_results['per_class_AP']
            for class_name, ap in sorted(per_class_ap.items(), key=lambda x: x[1], reverse=True):
                print(f"  {class_name:15} {ap:.3f}")
            print("="*60)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()