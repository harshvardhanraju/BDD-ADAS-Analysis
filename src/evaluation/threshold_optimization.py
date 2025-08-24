"""
Confidence Threshold Optimization for DETR Model

This module implements automated confidence threshold tuning based on
precision/recall trade-offs to optimize model performance for deployment.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.detr_model import BDD100KDETR


class ConfidenceThresholdOptimizer:
    """
    Automated confidence threshold optimization system.
    """
    
    def __init__(
        self,
        model: BDD100KDETR,
        val_dataloader: DataLoader,
        device: str = 'cuda',
        output_dir: str = 'threshold_optimization_results'
    ):
        """
        Initialize threshold optimizer.
        
        Args:
            model: Trained DETR model
            val_dataloader: Validation data loader
            device: Device for inference
            output_dir: Output directory for results
        """
        self.model = model.to(device)
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.class_names = model.config.class_names
        self.num_classes = len(self.class_names)
        
        # Threshold range to test
        self.threshold_range = np.arange(0.05, 0.95, 0.05)
        
        # Results storage
        self.threshold_results = {}
        self.optimal_thresholds = {}
        
        print(f"Threshold optimizer initialized")
        print(f"Testing {len(self.threshold_range)} thresholds: {self.threshold_range[0]:.2f} to {self.threshold_range[-1]:.2f}")
    
    def collect_predictions_and_scores(self, max_images: int = 500) -> Dict:
        """
        Collect model predictions with confidence scores.
        
        Args:
            max_images: Maximum images to evaluate
            
        Returns:
            Dictionary with predictions and ground truth data
        """
        print(f"🔍 Collecting predictions from {max_images} validation images...")
        
        self.model.eval()
        all_predictions = []
        all_ground_truth = []
        
        processed_images = 0
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(self.val_dataloader, desc="Collecting predictions")):
                if processed_images >= max_images:
                    break
                
                images = images.to(self.device)
                batch_size = images.shape[0]
                
                # Get model outputs
                outputs = self.model.model(pixel_values=images)
                
                # Process each image in batch
                for i in range(batch_size):
                    if processed_images >= max_images:
                        break
                    
                    # Get predictions
                    logits = outputs.logits[i]  # [num_queries, num_classes + 1]
                    pred_boxes = outputs.pred_boxes[i]  # [num_queries, 4]
                    
                    # Convert to probabilities
                    probs = torch.softmax(logits, dim=-1)[:, :-1]  # Remove background
                    scores, labels = probs.max(dim=-1)
                    
                    # Store all predictions with scores
                    predictions = []
                    for j in range(len(scores)):
                        if scores[j] > 0.01:  # Very low threshold to capture all
                            predictions.append({
                                'class_id': labels[j].item(),
                                'class_name': self.class_names[labels[j].item()],
                                'confidence': scores[j].item(),
                                'bbox': pred_boxes[j].cpu().numpy()
                            })
                    
                    # Process ground truth
                    target = targets[i]
                    ground_truth = []
                    
                    if len(target['class_labels']) > 0:
                        for j in range(len(target['class_labels'])):
                            ground_truth.append({
                                'class_id': target['class_labels'][j].item(),
                                'class_name': self.class_names[target['class_labels'][j].item()],
                                'bbox': target['boxes'][j].cpu().numpy()
                            })
                    
                    all_predictions.append({
                        'image_id': processed_images,
                        'predictions': predictions,
                        'ground_truth': ground_truth
                    })
                    
                    processed_images += 1
        
        print(f"✅ Collected predictions from {processed_images} images")
        return {
            'predictions': all_predictions,
            'num_images': processed_images
        }
    
    def compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes."""
        # Convert from center format to corner format if needed
        if len(box1) == 4 and len(box2) == 4:
            # Assume center format: [cx, cy, w, h]
            x1_min = box1[0] - box1[2] / 2
            y1_min = box1[1] - box1[3] / 2
            x1_max = box1[0] + box1[2] / 2
            y1_max = box1[1] + box1[3] / 2
            
            x2_min = box2[0] - box2[2] / 2
            y2_min = box2[1] - box2[3] / 2
            x2_max = box2[0] + box2[2] / 2
            y2_max = box2[1] + box2[3] / 2
            
            # Compute intersection
            inter_xmin = max(x1_min, x2_min)
            inter_ymin = max(y1_min, y2_min)
            inter_xmax = min(x1_max, x2_max)
            inter_ymax = min(y1_max, y2_max)
            
            if inter_xmax <= inter_xmin or inter_ymax <= inter_ymin:
                return 0.0
            
            inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            
            union_area = box1_area + box2_area - inter_area
            return inter_area / union_area if union_area > 0 else 0.0
        
        return 0.0
    
    def evaluate_threshold(self, data: Dict, threshold: float, iou_threshold: float = 0.5) -> Dict:
        """
        Evaluate performance at a specific confidence threshold.
        
        Args:
            data: Prediction data
            threshold: Confidence threshold to test
            iou_threshold: IoU threshold for matching
            
        Returns:
            Performance metrics
        """
        class_metrics = {name: {'tp': 0, 'fp': 0, 'fn': 0} for name in self.class_names}
        
        for image_data in data['predictions']:
            predictions = image_data['predictions']
            ground_truth = image_data['ground_truth']
            
            # Filter predictions by threshold
            filtered_preds = [p for p in predictions if p['confidence'] >= threshold]
            
            # Track matched ground truth
            gt_matched = [False] * len(ground_truth)
            
            # Evaluate each prediction
            for pred in filtered_preds:
                pred_class = pred['class_name']
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching ground truth
                for gt_idx, gt in enumerate(ground_truth):
                    if gt['class_name'] == pred_class and not gt_matched[gt_idx]:
                        iou = self.compute_iou(pred['bbox'], gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    # True positive
                    class_metrics[pred_class]['tp'] += 1
                    gt_matched[best_gt_idx] = True
                else:
                    # False positive
                    class_metrics[pred_class]['fp'] += 1
            
            # Count false negatives (unmatched ground truth)
            for gt_idx, gt in enumerate(ground_truth):
                if not gt_matched[gt_idx]:
                    class_metrics[gt['class_name']]['fn'] += 1
        
        # Calculate metrics
        results = {}
        overall_tp = overall_fp = overall_fn = 0
        
        for class_name, metrics in class_metrics.items():
            tp, fp, fn = metrics['tp'], metrics['fp'], metrics['fn']
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }
            
            overall_tp += tp
            overall_fp += fp
            overall_fn += fn
        
        # Overall metrics
        overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
        overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
        overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
        
        results['overall'] = {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'tp': overall_tp,
            'fp': overall_fp,
            'fn': overall_fn
        }
        
        return results
    
    def optimize_thresholds(
        self,
        data: Dict,
        optimization_metric: str = 'f1_score',
        class_specific: bool = True
    ) -> Dict:
        """
        Optimize confidence thresholds for best performance.
        
        Args:
            data: Prediction data
            optimization_metric: Metric to optimize ('f1_score', 'precision', 'recall')
            class_specific: Whether to optimize per-class thresholds
            
        Returns:
            Optimal thresholds and performance metrics
        """
        print(f"🎯 Optimizing thresholds based on {optimization_metric}...")
        
        # Test all thresholds
        threshold_results = {}
        
        for threshold in tqdm(self.threshold_range, desc="Testing thresholds"):
            results = self.evaluate_threshold(data, threshold)
            threshold_results[threshold] = results
        
        self.threshold_results = threshold_results
        
        # Find optimal thresholds
        optimal_thresholds = {}
        
        if class_specific:
            # Optimize per-class thresholds
            for class_name in self.class_names:
                best_threshold = 0.5
                best_score = 0
                
                for threshold, results in threshold_results.items():
                    if class_name in results:
                        score = results[class_name][optimization_metric]
                        if score > best_score:
                            best_score = score
                            best_threshold = threshold
                
                optimal_thresholds[class_name] = {
                    'threshold': best_threshold,
                    'score': best_score
                }
            
            # Overall optimal threshold
            best_overall_threshold = 0.5
            best_overall_score = 0
            
            for threshold, results in threshold_results.items():
                score = results['overall'][optimization_metric]
                if score > best_overall_score:
                    best_overall_score = score
                    best_overall_threshold = threshold
            
            optimal_thresholds['overall'] = {
                'threshold': best_overall_threshold,
                'score': best_overall_score
            }
        
        else:
            # Single global threshold
            best_threshold = 0.5
            best_score = 0
            
            for threshold, results in threshold_results.items():
                score = results['overall'][optimization_metric]
                if score > best_score:
                    best_score = score
                    best_threshold = threshold
            
            optimal_thresholds = {
                'global': {
                    'threshold': best_threshold,
                    'score': best_score
                }
            }
        
        self.optimal_thresholds = optimal_thresholds
        return optimal_thresholds
    
    def plot_threshold_curves(self):
        """Plot precision-recall curves for different thresholds."""
        if not self.threshold_results:
            print("❌ No threshold results available. Run optimization first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Confidence Threshold Optimization Results', fontsize=16)
        
        # Overall metrics vs threshold
        thresholds = list(self.threshold_results.keys())
        overall_precision = [self.threshold_results[t]['overall']['precision'] for t in thresholds]
        overall_recall = [self.threshold_results[t]['overall']['recall'] for t in thresholds]
        overall_f1 = [self.threshold_results[t]['overall']['f1_score'] for t in thresholds]
        
        axes[0, 0].plot(thresholds, overall_precision, 'b-', label='Precision', marker='o')
        axes[0, 0].plot(thresholds, overall_recall, 'r-', label='Recall', marker='s')
        axes[0, 0].plot(thresholds, overall_f1, 'g-', label='F1 Score', marker='^')
        
        # Mark optimal threshold
        if 'overall' in self.optimal_thresholds:
            opt_thresh = self.optimal_thresholds['overall']['threshold']
            axes[0, 0].axvline(x=opt_thresh, color='black', linestyle='--', 
                              label=f'Optimal: {opt_thresh:.2f}')
        
        axes[0, 0].set_xlabel('Confidence Threshold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Overall Performance vs Threshold')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Precision-Recall curve
        axes[0, 1].plot(overall_recall, overall_precision, 'b-', marker='o')
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].grid(True)
        
        # Class-specific F1 scores
        class_f1_data = []
        for class_name in self.class_names:
            class_f1 = [self.threshold_results[t][class_name]['f1_score'] for t in thresholds]
            axes[1, 0].plot(thresholds, class_f1, label=class_name, marker='o')
            class_f1_data.append(class_f1)
        
        axes[1, 0].set_xlabel('Confidence Threshold')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_title('Class-Specific F1 Scores')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Optimal thresholds heatmap
        if len(self.optimal_thresholds) > 1:
            class_names = [name for name in self.class_names if name in self.optimal_thresholds]
            opt_thresholds = [self.optimal_thresholds[name]['threshold'] for name in class_names]
            opt_scores = [self.optimal_thresholds[name]['score'] for name in class_names]
            
            scatter = axes[1, 1].scatter(opt_thresholds, range(len(class_names)), 
                                       c=opt_scores, cmap='viridis', s=100)
            axes[1, 1].set_yticks(range(len(class_names)))
            axes[1, 1].set_yticklabels(class_names)
            axes[1, 1].set_xlabel('Optimal Threshold')
            axes[1, 1].set_title('Class-Specific Optimal Thresholds')
            plt.colorbar(scatter, ax=axes[1, 1], label='F1 Score')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'threshold_optimization_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Threshold optimization curves saved to {plot_path}")
    
    def generate_threshold_report(self) -> str:
        """Generate comprehensive threshold optimization report."""
        if not self.optimal_thresholds:
            return "No optimization results available."
        
        report_lines = [
            "# Confidence Threshold Optimization Report",
            "=" * 60,
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Threshold range tested: {self.threshold_range[0]:.2f} to {self.threshold_range[-1]:.2f}",
            f"Step size: {self.threshold_range[1] - self.threshold_range[0]:.2f}",
            "",
            "## Optimal Thresholds Summary",
            ""
        ]
        
        # Overall optimal threshold
        if 'overall' in self.optimal_thresholds:
            overall = self.optimal_thresholds['overall']
            report_lines.extend([
                f"**Overall Optimal Threshold**: {overall['threshold']:.3f}",
                f"**Best F1 Score**: {overall['score']:.3f}",
                ""
            ])
        
        # Class-specific thresholds
        if any(name in self.optimal_thresholds for name in self.class_names):
            report_lines.extend([
                "## Class-Specific Optimal Thresholds",
                "",
                "| Class | Threshold | F1 Score | Recommendation |",
                "|-------|-----------|----------|----------------|"
            ])
            
            for class_name in self.class_names:
                if class_name in self.optimal_thresholds:
                    opt = self.optimal_thresholds[class_name]
                    threshold = opt['threshold']
                    score = opt['score']
                    
                    # Generate recommendation
                    if threshold < 0.3:
                        recommendation = "Low threshold - high recall focus"
                    elif threshold > 0.7:
                        recommendation = "High threshold - high precision focus"
                    else:
                        recommendation = "Balanced threshold"
                    
                    report_lines.append(
                        f"| {class_name} | {threshold:.3f} | {score:.3f} | {recommendation} |"
                    )
            
            report_lines.append("")
        
        # Performance analysis at optimal thresholds
        if 'overall' in self.optimal_thresholds and self.threshold_results:
            opt_threshold = self.optimal_thresholds['overall']['threshold']
            opt_results = self.threshold_results[opt_threshold]
            
            report_lines.extend([
                "## Performance at Optimal Threshold",
                f"*Threshold: {opt_threshold:.3f}*",
                "",
                "| Class | Precision | Recall | F1 Score | TP | FP | FN |",
                "|-------|-----------|--------|----------|----|----|----| "
            ])
            
            for class_name in self.class_names:
                if class_name in opt_results:
                    metrics = opt_results[class_name]
                    report_lines.append(
                        f"| {class_name} | {metrics['precision']:.3f} | "
                        f"{metrics['recall']:.3f} | {metrics['f1_score']:.3f} | "
                        f"{metrics['tp']} | {metrics['fp']} | {metrics['fn']} |"
                    )
            
            # Overall performance
            overall_metrics = opt_results['overall']
            report_lines.extend([
                "",
                f"**Overall Performance:**",
                f"- Precision: {overall_metrics['precision']:.3f}",
                f"- Recall: {overall_metrics['recall']:.3f}",
                f"- F1 Score: {overall_metrics['f1_score']:.3f}",
                f"- Total TP: {overall_metrics['tp']}",
                f"- Total FP: {overall_metrics['fp']}",
                f"- Total FN: {overall_metrics['fn']}",
                ""
            ])
        
        # Recommendations
        report_lines.extend([
            "## Deployment Recommendations",
            "",
            "### Production Deployment:",
            "1. **Safety-Critical Applications**: Use higher thresholds (0.6-0.8) to minimize false positives",
            "2. **General Detection**: Use optimal overall threshold for balanced performance",
            "3. **High Recall Applications**: Use lower thresholds (0.2-0.4) to capture more objects",
            "",
            "### Class-Specific Recommendations:",
        ])
        
        for class_name in self.class_names:
            if class_name in self.optimal_thresholds:
                threshold = self.optimal_thresholds[class_name]['threshold']
                
                if class_name in ['train', 'rider']:  # Safety critical
                    report_lines.append(
                        f"- **{class_name}**: Consider lower threshold ({threshold-0.1:.2f}) for safety"
                    )
                elif class_name in ['car', 'truck', 'bus']:  # Common objects
                    report_lines.append(
                        f"- **{class_name}**: Use optimal threshold ({threshold:.2f}) for balanced performance"
                    )
                else:  # Traffic signs/lights
                    report_lines.append(
                        f"- **{class_name}**: Consider context-based thresholds around {threshold:.2f}"
                    )
        
        # Save report
        report_file = self.output_dir / "threshold_optimization_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Threshold optimization report saved to {report_file}")
        return str(report_file)
    
    def run_complete_optimization(
        self,
        max_images: int = 500,
        optimization_metric: str = 'f1_score'
    ) -> Dict:
        """
        Run complete threshold optimization pipeline.
        
        Args:
            max_images: Maximum images to evaluate
            optimization_metric: Metric to optimize
            
        Returns:
            Complete optimization results
        """
        print("🚀 Starting complete confidence threshold optimization...")
        print("=" * 60)
        
        # Step 1: Collect predictions
        data = self.collect_predictions_and_scores(max_images)
        
        # Step 2: Optimize thresholds
        optimal_thresholds = self.optimize_thresholds(data, optimization_metric)
        
        # Step 3: Generate visualizations
        self.plot_threshold_curves()
        
        # Step 4: Generate report
        report_path = self.generate_threshold_report()
        
        # Step 5: Save results
        results = {
            'optimal_thresholds': optimal_thresholds,
            'threshold_results': self.threshold_results,
            'evaluation_summary': {
                'images_evaluated': data['num_images'],
                'thresholds_tested': len(self.threshold_range),
                'optimization_metric': optimization_metric,
                'best_overall_threshold': optimal_thresholds.get('overall', {}).get('threshold', 0.5),
                'best_overall_score': optimal_thresholds.get('overall', {}).get('score', 0.0)
            }
        }
        
        results_file = self.output_dir / 'optimization_results.json'
        with open(results_file, 'w') as f:
            # Convert numpy types for JSON serialization
            results_json = json.loads(json.dumps(results, default=float))
            json.dump(results_json, f, indent=2)
        
        print("🎉 Confidence threshold optimization complete!")
        print(f"📊 Results summary:")
        print(f"   • Images evaluated: {data['num_images']}")
        print(f"   • Optimal overall threshold: {optimal_thresholds.get('overall', {}).get('threshold', 'N/A'):.3f}")
        print(f"   • Best F1 score: {optimal_thresholds.get('overall', {}).get('score', 'N/A'):.3f}")
        print(f"📄 Complete results: {report_path}")
        print(f"💾 Data saved to: {self.output_dir}")
        
        return results


if __name__ == "__main__":
    print("Confidence Threshold Optimization Module")
    print("Use this module with a trained DETR model and validation data")
    print("Example usage:")
    print("  optimizer = ConfidenceThresholdOptimizer(model, val_dataloader)")
    print("  results = optimizer.run_complete_optimization()")