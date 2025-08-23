"""
mAP Evaluation for DETR on BDD100K Dataset

This module implements comprehensive evaluation metrics including mAP, per-class AP,
and detailed analysis for object detection performance on BDD100K dataset.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import pycocotools for mAP computation
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    import pycocotools.mask as mask_util
except ImportError:
    print("Warning: pycocotools not found. Please install: pip install pycocotools")

from ..models.detr_model import BDD100KDETR, create_bdd_detr_model
from ..data.detr_dataset import BDD100KDETRDataset


class BDD100KmAPEvaluator:
    """
    Comprehensive mAP evaluator for BDD100K DETR model.
    """
    
    def __init__(
        self,
        model: BDD100KDETR,
        val_dataloader: DataLoader,
        device: str = 'cuda',
        confidence_threshold: float = 0.1,
        iou_thresholds: Optional[List[float]] = None
    ):
        """
        Initialize mAP evaluator.
        
        Args:
            model: Trained DETR model
            val_dataloader: Validation data loader
            device: Device for evaluation
            confidence_threshold: Minimum confidence for predictions
            iou_thresholds: IoU thresholds for mAP calculation
        """
        self.model = model.to(device)
        self.val_dataloader = val_dataloader
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        if iou_thresholds is None:
            # Standard COCO mAP IoU thresholds
            self.iou_thresholds = np.arange(0.5, 1.0, 0.05)
        else:
            self.iou_thresholds = np.array(iou_thresholds)
        
        # Class information
        self.class_names = model.config.class_names
        self.num_classes = len(self.class_names)
        
        # Results storage
        self.predictions = []
        self.ground_truth = []
        self.evaluation_results = {}
    
    def _postprocess_predictions(
        self, 
        outputs: Dict, 
        target_sizes: torch.Tensor
    ) -> List[Dict]:
        """
        Postprocess DETR outputs to get final predictions.
        
        Args:
            outputs: Raw DETR outputs
            target_sizes: Original image sizes [batch_size, 2] (H, W)
            
        Returns:
            List of prediction dictionaries for each image
        """
        logits = outputs.logits  # [batch_size, num_queries, num_classes + 1]
        pred_boxes = outputs.pred_boxes  # [batch_size, num_queries, 4]
        
        batch_size = logits.shape[0]
        predictions = []
        
        for i in range(batch_size):
            # Get probabilities and filter background class
            probs = F.softmax(logits[i], dim=-1)[:, :-1]  # Remove background
            scores, labels = probs.max(dim=-1)
            
            # Filter by confidence threshold
            keep = scores > self.confidence_threshold
            
            if keep.sum() == 0:
                # No predictions above threshold
                predictions.append({
                    'boxes': torch.empty((0, 4)),
                    'scores': torch.empty((0,)),
                    'labels': torch.empty((0,), dtype=torch.long)
                })
                continue
            
            # Get filtered predictions
            filtered_boxes = pred_boxes[i][keep]
            filtered_scores = scores[keep]
            filtered_labels = labels[keep]
            
            # Convert normalized boxes to pixel coordinates
            target_h, target_w = target_sizes[i]
            
            # DETR boxes are in format [center_x, center_y, width, height] normalized
            # Convert to [x1, y1, x2, y2] format
            center_x = filtered_boxes[:, 0] * target_w
            center_y = filtered_boxes[:, 1] * target_h
            width = filtered_boxes[:, 2] * target_w
            height = filtered_boxes[:, 3] * target_h
            
            x1 = center_x - width / 2
            y1 = center_y - height / 2
            x2 = center_x + width / 2
            y2 = center_y + height / 2
            
            pixel_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            
            predictions.append({
                'boxes': pixel_boxes,
                'scores': filtered_scores,
                'labels': filtered_labels
            })
        
        return predictions
    
    def _compute_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Compute IoU between two sets of boxes.
        
        Args:
            box1: First set of boxes [N, 4] in (x1, y1, x2, y2) format
            box2: Second set of boxes [M, 4] in (x1, y1, x2, y2) format
            
        Returns:
            IoU matrix [N, M]
        """
        # Expand dimensions for broadcasting
        box1 = box1.unsqueeze(1)  # [N, 1, 4]
        box2 = box2.unsqueeze(0)  # [1, M, 4]
        
        # Calculate intersection
        inter_x1 = torch.max(box1[:, :, 0], box2[:, :, 0])
        inter_y1 = torch.max(box1[:, :, 1], box2[:, :, 1])
        inter_x2 = torch.min(box1[:, :, 2], box2[:, :, 2])
        inter_y2 = torch.min(box1[:, :, 3], box2[:, :, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union
        area1 = (box1[:, :, 2] - box1[:, :, 0]) * (box1[:, :, 3] - box1[:, :, 1])
        area2 = (box2[:, :, 2] - box2[:, :, 0]) * (box2[:, :, 3] - box2[:, :, 1])
        union_area = area1 + area2 - inter_area
        
        # Calculate IoU
        iou = inter_area / (union_area + 1e-7)
        return iou
    
    def _compute_ap(
        self, 
        pred_scores: np.ndarray, 
        pred_correct: np.ndarray, 
        num_gt: int
    ) -> float:
        """
        Compute Average Precision for a single class.
        
        Args:
            pred_scores: Prediction scores sorted in descending order
            pred_correct: Binary array indicating correct predictions
            num_gt: Number of ground truth boxes
            
        Returns:
            Average precision value
        """
        if num_gt == 0:
            return 0.0
        
        if len(pred_scores) == 0:
            return 0.0
        
        # Compute precision and recall
        tp = np.cumsum(pred_correct)
        fp = np.cumsum(1 - pred_correct)
        
        recall = tp / num_gt
        precision = tp / (tp + fp)
        
        # Compute AP using 101-point interpolation
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap += p / 11
        
        return ap
    
    def collect_predictions(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Collect predictions and ground truth from validation set.
        
        Returns:
            Tuple of (predictions, ground_truth)
        """
        print("🔍 Collecting predictions from validation set...")
        
        self.model.eval()
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(self.val_dataloader, desc="Evaluating")):
                # Move to device
                images = images.to(self.device)
                batch_size = images.shape[0]
                
                # Get model outputs
                outputs = self.model.model(pixel_values=images)
                
                # Get target sizes for postprocessing
                target_sizes = torch.stack([t['orig_size'] for t in targets])
                
                # Postprocess predictions
                batch_predictions = self._postprocess_predictions(outputs, target_sizes)
                
                # Process each image in batch
                for i in range(batch_size):
                    image_id = batch_idx * self.val_dataloader.batch_size + i
                    
                    # Store predictions
                    pred_dict = {
                        'image_id': image_id,
                        'boxes': batch_predictions[i]['boxes'].cpu(),
                        'scores': batch_predictions[i]['scores'].cpu(),
                        'labels': batch_predictions[i]['labels'].cpu()
                    }
                    predictions.append(pred_dict)
                    
                    # Store ground truth
                    target = targets[i]
                    
                    # Convert normalized boxes back to pixel coordinates
                    if len(target['boxes']) > 0:
                        orig_h, orig_w = target['orig_size']
                        norm_boxes = target['boxes']
                        
                        # Convert from [center_x, center_y, width, height] to [x1, y1, x2, y2]
                        center_x = norm_boxes[:, 0] * orig_w
                        center_y = norm_boxes[:, 1] * orig_h
                        width = norm_boxes[:, 2] * orig_w
                        height = norm_boxes[:, 3] * orig_h
                        
                        x1 = center_x - width / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width / 2
                        y2 = center_y + height / 2
                        
                        gt_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
                    else:
                        gt_boxes = torch.empty((0, 4))
                    
                    gt_dict = {
                        'image_id': image_id,
                        'boxes': gt_boxes,
                        'labels': target['class_labels']
                    }
                    ground_truth.append(gt_dict)
        
        return predictions, ground_truth
    
    def evaluate_map(self) -> Dict[str, float]:
        """
        Evaluate mAP across different IoU thresholds and classes.
        
        Returns:
            Dictionary containing mAP results
        """
        print("📊 Computing mAP metrics...")
        
        # Collect predictions if not already done
        if not hasattr(self, 'predictions') or not self.predictions:
            self.predictions, self.ground_truth = self.collect_predictions()
        
        results = {}
        class_aps = {iou_thresh: [] for iou_thresh in self.iou_thresholds}
        
        # Evaluate each class
        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            
            # Collect all predictions and ground truth for this class
            all_pred_scores = []
            all_pred_boxes = []
            all_gt_boxes = []
            all_image_ids = []
            
            # Ground truth count for this class
            total_gt = 0
            
            for pred, gt in zip(self.predictions, self.ground_truth):
                image_id = pred['image_id']
                
                # Get predictions for this class
                class_mask = pred['labels'] == class_idx
                if class_mask.sum() > 0:
                    pred_boxes = pred['boxes'][class_mask]
                    pred_scores = pred['scores'][class_mask]
                    
                    all_pred_boxes.extend(pred_boxes)
                    all_pred_scores.extend(pred_scores.tolist())
                    all_image_ids.extend([image_id] * len(pred_boxes))
                
                # Get ground truth for this class
                gt_class_mask = gt['labels'] == class_idx
                if gt_class_mask.sum() > 0:
                    gt_boxes = gt['boxes'][gt_class_mask]
                    all_gt_boxes.append((image_id, gt_boxes))
                    total_gt += len(gt_boxes)
            
            if total_gt == 0:
                print(f"  No ground truth for class {class_name}, skipping")
                continue
            
            if len(all_pred_scores) == 0:
                print(f"  No predictions for class {class_name}, AP = 0.0")
                for iou_thresh in self.iou_thresholds:
                    class_aps[iou_thresh].append(0.0)
                continue
            
            # Convert to tensors and sort by score
            all_pred_scores = torch.tensor(all_pred_scores)
            all_pred_boxes = torch.stack(all_pred_boxes)
            all_image_ids = torch.tensor(all_image_ids)
            
            # Sort by confidence score
            sorted_indices = torch.argsort(all_pred_scores, descending=True)
            all_pred_scores = all_pred_scores[sorted_indices]
            all_pred_boxes = all_pred_boxes[sorted_indices]
            all_image_ids = all_image_ids[sorted_indices]
            
            # Evaluate at different IoU thresholds
            for iou_thresh in self.iou_thresholds:
                pred_correct = torch.zeros(len(all_pred_scores), dtype=torch.bool)
                
                # Create dict for easy GT lookup
                gt_dict = {image_id.item(): boxes for image_id, boxes in all_gt_boxes}
                used_gt = {image_id.item(): torch.zeros(len(boxes), dtype=torch.bool) 
                          for image_id, boxes in all_gt_boxes}
                
                # Check each prediction
                for pred_idx in range(len(all_pred_scores)):
                    image_id = all_image_ids[pred_idx].item()
                    pred_box = all_pred_boxes[pred_idx].unsqueeze(0)
                    
                    if image_id in gt_dict:
                        gt_boxes = gt_dict[image_id]
                        if len(gt_boxes) > 0:
                            # Compute IoU with all GT boxes in this image
                            ious = self._compute_iou(pred_box, gt_boxes).squeeze(0)
                            
                            # Find best matching GT box
                            max_iou, max_idx = ious.max(0)
                            
                            if max_iou >= iou_thresh and not used_gt[image_id][max_idx]:
                                pred_correct[pred_idx] = True
                                used_gt[image_id][max_idx] = True
                
                # Compute AP for this IoU threshold
                ap = self._compute_ap(
                    all_pred_scores.numpy(),
                    pred_correct.numpy(),
                    total_gt
                )
                class_aps[iou_thresh].append(ap)
            
            print(f"  {class_name}: AP@0.5 = {class_aps[0.5][-1]:.3f}")
        
        # Compute mAP across classes
        results['per_class_ap'] = {}
        for iou_thresh in self.iou_thresholds:
            mean_ap = np.mean(class_aps[iou_thresh]) if class_aps[iou_thresh] else 0.0
            results[f'mAP@{iou_thresh:.2f}'] = mean_ap
            
            # Store per-class results
            results['per_class_ap'][f'iou_{iou_thresh:.2f}'] = {
                self.class_names[i]: ap for i, ap in enumerate(class_aps[iou_thresh])
            }
        
        # Compute mAP@0.5:0.95 (average across all IoU thresholds)
        results['mAP@0.5:0.95'] = np.mean([results[f'mAP@{iou:.2f}'] for iou in self.iou_thresholds])
        
        # Store detailed results
        self.evaluation_results = results
        
        return results
    
    def generate_evaluation_report(
        self, 
        results: Dict[str, float], 
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate detailed evaluation report.
        
        Args:
            results: Evaluation results from evaluate_map()
            save_path: Path to save the report
            
        Returns:
            Report as string
        """
        report_lines = []
        
        report_lines.extend([
            "# BDD100K DETR Model Evaluation Report",
            "=" * 60,
            f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Confidence Threshold: {self.confidence_threshold}",
            f"IoU Thresholds: {self.iou_thresholds.tolist()}",
            "",
            "## Overall Performance",
            f"- mAP@0.5:0.95: {results['mAP@0.5:0.95']:.3f}",
            f"- mAP@0.5: {results['mAP@0.50']:.3f}",
            f"- mAP@0.75: {results['mAP@0.75']:.3f}",
            "",
            "## Per-Class Performance (mAP@0.5)",
        ])
        
        # Add per-class results
        if 'per_class_ap' in results and 'iou_0.50' in results['per_class_ap']:
            class_aps = results['per_class_ap']['iou_0.50']
            for class_name, ap in class_aps.items():
                report_lines.append(f"- {class_name:15}: {ap:.3f}")
        
        report_lines.extend([
            "",
            "## IoU Threshold Analysis",
        ])
        
        # Add IoU threshold breakdown
        for iou_thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
            if f'mAP@{iou_thresh:.2f}' in results:
                report_lines.append(f"- mAP@{iou_thresh}: {results[f'mAP@{iou_thresh:.2f}']:.3f}")
        
        report_content = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            print(f"📄 Evaluation report saved to: {save_path}")
        
        return report_content
    
    def create_evaluation_plots(self, save_dir: Optional[str] = None) -> List[str]:
        """
        Create visualization plots for evaluation results.
        
        Args:
            save_dir: Directory to save plots
            
        Returns:
            List of saved plot paths
        """
        if not self.evaluation_results:
            print("No evaluation results found. Run evaluate_map() first.")
            return []
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = []
        results = self.evaluation_results
        
        # 1. Per-class mAP@0.5 bar plot
        if 'per_class_ap' in results and 'iou_0.50' in results['per_class_ap']:
            plt.figure(figsize=(12, 6))
            class_aps = results['per_class_ap']['iou_0.50']
            
            classes = list(class_aps.keys())
            aps = list(class_aps.values())
            
            bars = plt.bar(classes, aps, color='skyblue', alpha=0.8)
            plt.title('Per-Class Average Precision (mAP@0.5)', fontweight='bold', fontsize=14)
            plt.xlabel('Object Classes')
            plt.ylabel('Average Precision')
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, ap in zip(bars, aps):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{ap:.3f}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            if save_dir:
                plot_path = save_dir / 'per_class_map.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths.append(str(plot_path))
            
            plt.close()
        
        # 2. mAP vs IoU threshold plot
        iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        map_values = [results.get(f'mAP@{iou:.2f}', 0) for iou in iou_thresholds]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iou_thresholds, map_values, marker='o', linewidth=2, markersize=8)
        plt.title('mAP vs IoU Threshold', fontweight='bold', fontsize=14)
        plt.xlabel('IoU Threshold')
        plt.ylabel('mAP')
        plt.grid(True, alpha=0.3)
        plt.xlim(0.45, 1.0)
        plt.ylim(0, max(map_values) * 1.1 if max(map_values) > 0 else 1)
        
        # Add value labels
        for iou, map_val in zip(iou_thresholds, map_values):
            plt.annotate(f'{map_val:.3f}', (iou, map_val), 
                        textcoords="offset points", xytext=(0,10), ha='center')
        
        plt.tight_layout()
        
        if save_dir:
            plot_path = save_dir / 'map_vs_iou.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths.append(str(plot_path))
        
        plt.close()
        
        return plot_paths


def evaluate_detr_checkpoint(
    checkpoint_path: str,
    val_annotations: str,
    images_root: str,
    confidence_threshold: float = 0.1,
    batch_size: int = 8,
    save_dir: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a trained DETR model checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        val_annotations: Path to validation annotations
        images_root: Root directory for images
        confidence_threshold: Minimum confidence for predictions
        batch_size: Batch size for evaluation
        save_dir: Directory to save evaluation results
        
    Returns:
        Evaluation results dictionary
    """
    print(f"🔍 Evaluating DETR checkpoint: {checkpoint_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model and checkpoint
    model = create_bdd_detr_model(pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create validation dataloader
    from ..data.detr_dataset import BDD100KDETRDataset
    
    val_dataset = BDD100KDETRDataset(
        annotations_file=val_annotations,
        images_root=images_root,
        split='val',
        image_size=(512, 512),
        augment=False
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=val_dataset.collate_fn
    )
    
    # Create evaluator
    evaluator = BDD100KmAPEvaluator(
        model=model,
        val_dataloader=val_dataloader,
        device=str(device),
        confidence_threshold=confidence_threshold
    )
    
    # Run evaluation
    results = evaluator.evaluate_map()
    
    # Generate report and plots
    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save evaluation report
        report_path = save_path / 'evaluation_report.md'
        evaluator.generate_evaluation_report(results, str(report_path))
        
        # Create plots
        evaluator.create_evaluation_plots(str(save_path))
        
        # Save results as JSON
        results_path = save_path / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    # Test evaluation setup
    print("Testing DETR evaluation setup...")
    
    checkpoint_path = "checkpoints/best_model.pth"
    val_annotations = "data/analysis/processed/val_annotations.csv"
    images_root = "data/raw/bdd100k/bdd100k/images/100k"
    
    if Path(checkpoint_path).exists() and Path(val_annotations).exists():
        results = evaluate_detr_checkpoint(
            checkpoint_path=checkpoint_path,
            val_annotations=val_annotations,
            images_root=images_root,
            save_dir="evaluation_results"
        )
        print(f"Evaluation results: {results}")
    else:
        print("Required files not found for evaluation.")