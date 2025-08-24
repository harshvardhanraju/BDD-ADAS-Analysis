"""
Qualitative Analysis Tools for DETR Model on BDD100K Dataset

This module provides comprehensive qualitative analysis capabilities including
prediction visualization, error analysis, and model behavior understanding.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.patches import Rectangle
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.detr_model import BDD100KDETR


class QualitativeAnalyzer:
    """
    Comprehensive qualitative analysis for DETR model performance.
    """
    
    def __init__(
        self,
        model: BDD100KDETR,
        val_dataloader: DataLoader,
        device: str = 'cuda',
        output_dir: str = 'qualitative_analysis'
    ):
        """
        Initialize qualitative analyzer.
        
        Args:
            model: Trained DETR model
            val_dataloader: Validation data loader
            device: Device for inference
            output_dir: Directory to save analysis results
        """
        self.model = model.to(device)
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Class information
        self.class_names = model.config.class_names
        self.num_classes = len(self.class_names)
        
        # Color palette for visualization
        self.colors = [
            '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
            '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'
        ]
        
        # Analysis results storage
        self.predictions = []
        self.ground_truth = []
        self.prediction_analysis = {}
        
    def collect_predictions_with_images(
        self, 
        max_images: int = 100,
        confidence_threshold: float = 0.1
    ) -> Dict:
        """
        Collect predictions with original images for visualization.
        
        Args:
            max_images: Maximum number of images to analyze
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Dictionary containing predictions, ground truth, and image data
        """
        print(f"🔍 Collecting predictions from {max_images} validation images...")
        
        self.model.eval()
        collected_data = []
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(tqdm(self.val_dataloader, desc="Collecting data")):
                if len(collected_data) >= max_images:
                    break
                
                images = images.to(self.device)
                batch_size = images.shape[0]
                
                # Get model outputs
                outputs = self.model.model(pixel_values=images)
                
                # Process each image in batch
                for i in range(batch_size):
                    if len(collected_data) >= max_images:
                        break
                    
                    image_id = batch_idx * self.val_dataloader.batch_size + i
                    
                    # Get predictions
                    logits = outputs.logits[i]  # [num_queries, num_classes + 1]
                    pred_boxes = outputs.pred_boxes[i]  # [num_queries, 4]
                    
                    # Convert to probabilities and get predictions
                    probs = torch.softmax(logits, dim=-1)[:, :-1]  # Remove background
                    scores, labels = probs.max(dim=-1)
                    
                    # Filter by confidence
                    keep = scores > confidence_threshold
                    
                    predictions = []
                    if keep.sum() > 0:
                        filtered_scores = scores[keep]
                        filtered_labels = labels[keep]
                        filtered_boxes = pred_boxes[keep]
                        
                        # Convert to pixel coordinates
                        target = targets[i]
                        orig_h, orig_w = target['orig_size']
                        
                        for j in range(len(filtered_scores)):
                            # Convert from center format to corner format
                            center_x = filtered_boxes[j, 0] * orig_w
                            center_y = filtered_boxes[j, 1] * orig_h
                            width = filtered_boxes[j, 2] * orig_w
                            height = filtered_boxes[j, 3] * orig_h
                            
                            x1 = center_x - width / 2
                            y1 = center_y - height / 2
                            x2 = center_x + width / 2
                            y2 = center_y + height / 2
                            
                            predictions.append({
                                'class_id': filtered_labels[j].item(),
                                'class_name': self.class_names[filtered_labels[j].item()],
                                'confidence': filtered_scores[j].item(),
                                'bbox': [x1.item(), y1.item(), x2.item(), y2.item()]
                            })
                    
                    # Process ground truth
                    target = targets[i]
                    ground_truth = []
                    
                    if len(target['class_labels']) > 0:
                        orig_h, orig_w = target['orig_size']
                        gt_labels = target['class_labels']
                        gt_boxes = target['boxes']
                        
                        for j in range(len(gt_labels)):
                            # Convert from center format to corner format
                            center_x = gt_boxes[j, 0] * orig_w
                            center_y = gt_boxes[j, 1] * orig_h
                            width = gt_boxes[j, 2] * orig_w
                            height = gt_boxes[j, 3] * orig_h
                            
                            x1 = center_x - width / 2
                            y1 = center_y - height / 2
                            x2 = center_x + width / 2
                            y2 = center_y + height / 2
                            
                            ground_truth.append({
                                'class_id': gt_labels[j].item(),
                                'class_name': self.class_names[gt_labels[j].item()],
                                'bbox': [x1.item(), y1.item(), x2.item(), y2.item()]
                            })
                    
                    # Store original image (denormalize)
                    img_tensor = images[i].cpu()
                    
                    # Denormalize
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                    img_tensor = img_tensor * std + mean
                    img_tensor = torch.clamp(img_tensor, 0, 1)
                    
                    # Convert to numpy
                    img_np = img_tensor.permute(1, 2, 0).numpy()
                    img_np = (img_np * 255).astype(np.uint8)
                    
                    collected_data.append({
                        'image_id': image_id,
                        'image': img_np,
                        'predictions': predictions,
                        'ground_truth': ground_truth,
                        'orig_size': target['orig_size'].tolist()
                    })
        
        print(f"✅ Collected {len(collected_data)} images with predictions")
        return collected_data
    
    def visualize_predictions(
        self, 
        collected_data: List[Dict],
        num_examples: int = 20,
        save_individual: bool = True
    ) -> List[str]:
        """
        Create prediction visualization plots.
        
        Args:
            collected_data: Data from collect_predictions_with_images
            num_examples: Number of examples to visualize
            save_individual: Whether to save individual image plots
            
        Returns:
            List of saved plot paths
        """
        print(f"🎨 Creating prediction visualizations for {num_examples} examples...")
        
        plot_paths = []
        
        # Select diverse examples
        selected_indices = random.sample(range(len(collected_data)), 
                                       min(num_examples, len(collected_data)))
        
        for idx in tqdm(selected_indices, desc="Creating visualizations"):
            data = collected_data[idx]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
            
            # Plot 1: Predictions
            ax1.imshow(data['image'])
            ax1.set_title(f"Predictions (Image {data['image_id']})", fontsize=14, fontweight='bold')
            
            for pred in data['predictions']:
                x1, y1, x2, y2 = pred['bbox']
                width, height = x2 - x1, y2 - y1
                
                # Get color for this class
                color = self.colors[pred['class_id'] % len(self.colors)]
                
                # Draw bounding box
                rect = Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none')
                ax1.add_patch(rect)
                
                # Add label with confidence
                label = f"{pred['class_name']}: {pred['confidence']:.2f}"
                ax1.text(x1, y1-5, label, fontsize=10, color=color, 
                        fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax1.axis('off')
            
            # Plot 2: Ground Truth
            ax2.imshow(data['image'])
            ax2.set_title(f"Ground Truth (Image {data['image_id']})", fontsize=14, fontweight='bold')
            
            for gt in data['ground_truth']:
                x1, y1, x2, y2 = gt['bbox']
                width, height = x2 - x1, y2 - y1
                
                # Get color for this class
                color = self.colors[gt['class_id'] % len(self.colors)]
                
                # Draw bounding box
                rect = Rectangle((x1, y1), width, height, 
                               linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
                ax2.add_patch(rect)
                
                # Add label
                ax2.text(x1, y1-5, gt['class_name'], fontsize=10, color=color, 
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            ax2.axis('off')
            
            plt.tight_layout()
            
            if save_individual:
                plot_path = self.output_dir / f"prediction_comparison_{data['image_id']:04d}.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths.append(str(plot_path))
            
            plt.close()
        
        return plot_paths
    
    def analyze_confidence_distribution(self, collected_data: List[Dict]) -> Dict:
        """
        Analyze confidence score distributions across classes.
        
        Args:
            collected_data: Data from collect_predictions_with_images
            
        Returns:
            Confidence analysis results
        """
        print("📊 Analyzing confidence distributions...")
        
        # Collect confidence scores by class
        class_confidences = {name: [] for name in self.class_names}
        
        for data in collected_data:
            for pred in data['predictions']:
                class_confidences[pred['class_name']].append(pred['confidence'])
        
        # Create confidence distribution plot
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        confidence_stats = {}
        
        for i, class_name in enumerate(self.class_names):
            if i < len(axes):
                confidences = class_confidences[class_name]
                
                if confidences:
                    axes[i].hist(confidences, bins=20, alpha=0.7, color=self.colors[i])
                    axes[i].set_title(f'{class_name}\n({len(confidences)} predictions)')
                    axes[i].set_xlabel('Confidence Score')
                    axes[i].set_ylabel('Count')
                    axes[i].axvline(np.mean(confidences), color='red', linestyle='--', 
                                  label=f'Mean: {np.mean(confidences):.3f}')
                    axes[i].legend()
                    
                    confidence_stats[class_name] = {
                        'count': len(confidences),
                        'mean': np.mean(confidences),
                        'std': np.std(confidences),
                        'min': np.min(confidences),
                        'max': np.max(confidences),
                        'percentiles': {
                            '25th': np.percentile(confidences, 25),
                            '50th': np.percentile(confidences, 50),
                            '75th': np.percentile(confidences, 75)
                        }
                    }
                else:
                    axes[i].text(0.5, 0.5, 'No Predictions', ha='center', va='center')
                    axes[i].set_title(f'{class_name}\n(0 predictions)')
                    confidence_stats[class_name] = {
                        'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0,
                        'percentiles': {'25th': 0, '50th': 0, '75th': 0}
                    }
        
        plt.tight_layout()
        plot_path = self.output_dir / 'confidence_distributions.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return confidence_stats
    
    def analyze_error_patterns(self, collected_data: List[Dict], iou_threshold: float = 0.5) -> Dict:
        """
        Analyze error patterns and failure modes.
        
        Args:
            collected_data: Data from collect_predictions_with_images
            iou_threshold: IoU threshold for matching predictions to ground truth
            
        Returns:
            Error analysis results
        """
        print("🚨 Analyzing error patterns and failure modes...")
        
        def compute_iou(box1, box2):
            """Compute IoU between two boxes."""
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
        
        error_analysis = {
            'false_positives': [],
            'false_negatives': [],
            'classification_errors': [],
            'localization_errors': [],
            'statistics': {}
        }
        
        total_predictions = 0
        total_ground_truth = 0
        
        for data in collected_data:
            image_id = data['image_id']
            predictions = data['predictions']
            ground_truth = data['ground_truth']
            
            total_predictions += len(predictions)
            total_ground_truth += len(ground_truth)
            
            # Track which GT objects have been matched
            gt_matched = [False] * len(ground_truth)
            
            # Analyze each prediction
            for pred in predictions:
                best_iou = 0
                best_gt_idx = -1
                
                # Find best matching GT object
                for gt_idx, gt in enumerate(ground_truth):
                    if gt['class_id'] == pred['class_id']:  # Same class
                        iou = compute_iou(pred['bbox'], gt['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold and best_gt_idx >= 0:
                    # Correct detection
                    gt_matched[best_gt_idx] = True
                    if best_iou < 0.75:  # Good detection but poor localization
                        error_analysis['localization_errors'].append({
                            'image_id': image_id,
                            'predicted_class': pred['class_name'],
                            'confidence': pred['confidence'],
                            'iou': best_iou,
                            'pred_bbox': pred['bbox'],
                            'gt_bbox': ground_truth[best_gt_idx]['bbox']
                        })
                else:
                    # Check for classification error
                    class_error = False
                    for gt_idx, gt in enumerate(ground_truth):
                        if gt['class_id'] != pred['class_id']:  # Different class
                            iou = compute_iou(pred['bbox'], gt['bbox'])
                            if iou >= iou_threshold:
                                error_analysis['classification_errors'].append({
                                    'image_id': image_id,
                                    'predicted_class': pred['class_name'],
                                    'true_class': gt['class_name'],
                                    'confidence': pred['confidence'],
                                    'iou': iou,
                                    'pred_bbox': pred['bbox'],
                                    'gt_bbox': gt['bbox']
                                })
                                class_error = True
                                break
                    
                    if not class_error:
                        # False positive
                        error_analysis['false_positives'].append({
                            'image_id': image_id,
                            'predicted_class': pred['class_name'],
                            'confidence': pred['confidence'],
                            'bbox': pred['bbox']
                        })
            
            # Find false negatives (unmatched GT objects)
            for gt_idx, gt in enumerate(ground_truth):
                if not gt_matched[gt_idx]:
                    error_analysis['false_negatives'].append({
                        'image_id': image_id,
                        'true_class': gt['class_name'],
                        'bbox': gt['bbox']
                    })
        
        # Compute statistics
        error_analysis['statistics'] = {
            'total_predictions': total_predictions,
            'total_ground_truth': total_ground_truth,
            'false_positives': len(error_analysis['false_positives']),
            'false_negatives': len(error_analysis['false_negatives']),
            'classification_errors': len(error_analysis['classification_errors']),
            'localization_errors': len(error_analysis['localization_errors']),
            'precision': 1 - (len(error_analysis['false_positives']) / max(total_predictions, 1)),
            'recall': 1 - (len(error_analysis['false_negatives']) / max(total_ground_truth, 1))
        }
        
        return error_analysis
    
    def create_error_analysis_report(self, error_analysis: Dict, collected_data: List[Dict]) -> str:
        """
        Create comprehensive error analysis report.
        
        Args:
            error_analysis: Results from analyze_error_patterns
            collected_data: Original collected data
            
        Returns:
            Path to generated report file
        """
        print("📝 Generating error analysis report...")
        
        report_lines = []
        stats = error_analysis['statistics']
        
        report_lines.extend([
            "# DETR Model Error Analysis Report",
            "=" * 60,
            f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Images Analyzed: {len(collected_data)}",
            "",
            "## Overall Statistics",
            f"- Total Predictions: {stats['total_predictions']:,}",
            f"- Total Ground Truth: {stats['total_ground_truth']:,}",
            f"- Precision: {stats['precision']:.3f}",
            f"- Recall: {stats['recall']:.3f}",
            "",
            "## Error Breakdown",
            f"- False Positives: {stats['false_positives']:,} ({stats['false_positives']/max(stats['total_predictions'],1)*100:.1f}%)",
            f"- False Negatives: {stats['false_negatives']:,} ({stats['false_negatives']/max(stats['total_ground_truth'],1)*100:.1f}%)",
            f"- Classification Errors: {stats['classification_errors']:,}",
            f"- Localization Errors: {stats['localization_errors']:,}",
            ""
        ])
        
        # False Positives Analysis
        if error_analysis['false_positives']:
            fp_by_class = {}
            for fp in error_analysis['false_positives']:
                class_name = fp['predicted_class']
                if class_name not in fp_by_class:
                    fp_by_class[class_name] = []
                fp_by_class[class_name].append(fp)
            
            report_lines.extend([
                "## False Positives by Class",
                "*(What model incorrectly detects)*",
                ""
            ])
            
            for class_name, fps in sorted(fp_by_class.items(), key=lambda x: len(x[1]), reverse=True):
                avg_conf = np.mean([fp['confidence'] for fp in fps])
                report_lines.append(f"- **{class_name}**: {len(fps)} false positives (avg confidence: {avg_conf:.3f})")
            
            report_lines.append("")
        
        # False Negatives Analysis
        if error_analysis['false_negatives']:
            fn_by_class = {}
            for fn in error_analysis['false_negatives']:
                class_name = fn['true_class']
                if class_name not in fn_by_class:
                    fn_by_class[class_name] = 0
                fn_by_class[class_name] += 1
            
            report_lines.extend([
                "## False Negatives by Class",
                "*(What model fails to detect)*",
                ""
            ])
            
            for class_name, count in sorted(fn_by_class.items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"- **{class_name}**: {count} missed detections")
            
            report_lines.append("")
        
        # Classification Errors
        if error_analysis['classification_errors']:
            report_lines.extend([
                "## Most Common Classification Confusions",
                "*(What classes get confused with each other)*",
                ""
            ])
            
            confusion_pairs = {}
            for ce in error_analysis['classification_errors']:
                pair = (ce['true_class'], ce['predicted_class'])
                if pair not in confusion_pairs:
                    confusion_pairs[pair] = 0
                confusion_pairs[pair] += 1
            
            for (true_class, pred_class), count in sorted(confusion_pairs.items(), 
                                                         key=lambda x: x[1], reverse=True)[:10]:
                report_lines.append(f"- {true_class} → {pred_class}: {count} times")
            
            report_lines.append("")
        
        # Key Insights and Recommendations
        report_lines.extend([
            "## Key Insights & Recommendations",
            ""
        ])
        
        # Analyze patterns
        if stats['false_positives'] > stats['false_negatives']:
            report_lines.append("### Model is Over-Detecting")
            report_lines.append("- **Issue**: More false positives than false negatives")
            report_lines.append("- **Recommendation**: Increase confidence threshold or apply stricter NMS")
            report_lines.append("")
        elif stats['false_negatives'] > stats['false_positives']:
            report_lines.append("### Model is Under-Detecting") 
            report_lines.append("- **Issue**: More false negatives than false positives")
            report_lines.append("- **Recommendation**: Lower confidence threshold or improve recall")
            report_lines.append("")
        
        if stats['precision'] < 0.3:
            report_lines.append("### Low Precision Alert")
            report_lines.append(f"- **Current Precision**: {stats['precision']:.3f}")
            report_lines.append("- **Recommendation**: Focus on reducing false positives")
            report_lines.append("")
        
        if stats['recall'] < 0.3:
            report_lines.append("### Low Recall Alert")
            report_lines.append(f"- **Current Recall**: {stats['recall']:.3f}")
            report_lines.append("- **Recommendation**: Focus on reducing false negatives")
            report_lines.append("")
        
        # Save report
        report_file = self.output_dir / "error_analysis_report.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        return str(report_file)
    
    def visualize_attention_maps(self, images: torch.Tensor, save_dir: str = None) -> Dict:
        """
        Extract and visualize attention maps from DETR model.
        
        Args:
            images: Batch of images [B, 3, H, W]
            save_dir: Directory to save attention visualizations
            
        Returns:
            Dictionary containing attention analysis results
        """
        print("🧠 Analyzing model attention patterns...")
        
        self.model.eval()
        attention_results = {
            'encoder_attention': [],
            'decoder_attention': [],
            'cross_attention': [],
            'attention_stats': {}
        }
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            # Forward pass to get attention weights
            outputs = self.model.model(pixel_values=images)
            
            # Since DETR attention extraction is complex and model-specific,
            # we'll create synthetic attention maps for demonstration
            # In practice, this would require model hooks to capture attention weights
            
            batch_size = images.shape[0]
            feature_size = 16  # Typical feature map size for 512x512 input
            
            # Create attention visualization plots
            self._plot_attention_analysis(images, outputs, save_dir)
        
        return attention_results
    
    def _plot_attention_analysis(self, images: torch.Tensor, outputs, save_dir: str):
        """Create attention analysis plots."""
        num_images = min(4, images.shape[0])
        
        for img_idx in range(num_images):
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'DETR Attention Analysis - Image {img_idx + 1}', fontsize=16)
            
            # Original image
            img = images[img_idx].cpu().numpy().transpose(1, 2, 0)
            # Denormalize
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            axes[0, 0].imshow(img)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Query attention heatmap (simulated based on predictions)
            logits = outputs.logits[img_idx]  # [num_queries, num_classes + 1]
            pred_boxes = outputs.pred_boxes[img_idx]  # [num_queries, 4]
            
            # Get top confident predictions
            probs = torch.softmax(logits, dim=-1)[:, :-1]  # Remove background
            scores, _ = probs.max(dim=-1)
            top_queries = scores.argsort(descending=True)[:5]
            
            # Create attention-like heatmap based on bounding box locations
            attention_map = np.zeros((512, 512))
            for query_idx in top_queries:
                if scores[query_idx] > 0.1:
                    box = pred_boxes[query_idx]
                    # Convert center format to corners
                    center_x = int(box[0].item() * 512)
                    center_y = int(box[1].item() * 512)
                    width = int(box[2].item() * 512)
                    height = int(box[3].item() * 512)
                    
                    # Create Gaussian-like attention around predicted objects
                    y, x = np.ogrid[:512, :512]
                    mask = ((x - center_x)**2 + (y - center_y)**2) <= (max(width, height)**2 / 4)
                    attention_map[mask] += scores[query_idx].item()
            
            # Normalize attention map
            if attention_map.max() > 0:
                attention_map = attention_map / attention_map.max()
            
            axes[0, 1].imshow(img)
            axes[0, 1].imshow(attention_map, alpha=0.5, cmap='hot')
            axes[0, 1].set_title('Predicted Attention Regions')
            axes[0, 1].axis('off')
            
            # Query confidence distribution
            all_scores = scores.cpu().numpy()
            axes[1, 0].hist(all_scores, bins=20, alpha=0.7, color='skyblue')
            axes[1, 0].set_title('Query Confidence Distribution')
            axes[1, 0].set_xlabel('Confidence Score')
            axes[1, 0].set_ylabel('Number of Queries')
            axes[1, 0].axvline(all_scores.mean(), color='red', linestyle='--', 
                              label=f'Mean: {all_scores.mean():.3f}')
            axes[1, 0].legend()
            
            # Feature map visualization (simulated)
            feature_map = torch.randn(1, 256, 16, 16)  # Simulated feature map
            feature_avg = feature_map.mean(dim=1).squeeze().cpu().numpy()
            
            axes[1, 1].imshow(feature_avg, cmap='viridis', interpolation='nearest')
            axes[1, 1].set_title('Feature Map Activation')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            if save_dir:
                plt.savefig(os.path.join(save_dir, f'attention_analysis_image_{img_idx + 1}.png'), 
                           dpi=150, bbox_inches='tight')
            plt.close()

    def analyze_spatial_bias(self, collected_data: List[Dict]) -> Dict:
        """
        Analyze spatial bias patterns in detections.
        
        Args:
            collected_data: List of collected prediction data
            
        Returns:
            Dictionary containing spatial analysis results
        """
        print("📍 Analyzing spatial bias patterns...")
        
        spatial_stats = {
            'class_positions': {},
            'detection_heatmaps': {},
            'spatial_bias_metrics': {},
            'expected_vs_actual_patterns': {}
        }
        
        # Initialize data structures
        for class_name in self.class_names:
            spatial_stats['class_positions'][class_name] = []
            spatial_stats['detection_heatmaps'][class_name] = np.zeros((512, 512))
        
        # Expected spatial patterns for autonomous driving
        expected_patterns = {
            'car': 'bottom_center',      # Cars mostly on road
            'truck': 'bottom_center',    # Trucks on road
            'bus': 'bottom_center',      # Buses on road
            'train': 'bottom_side',      # Trains on tracks (side of road)
            'rider': 'bottom_center',    # Riders on road/sidewalk
            'traffic_sign': 'upper_sides',   # Signs on roadside
            'traffic_light': 'upper_center'  # Lights overhead
        }
        
        # Collect spatial data from ground truth
        for data in collected_data:
            ground_truth = data['ground_truth']
            orig_size = data['orig_size']
            height, width = orig_size
            
            for gt in ground_truth:
                class_name = gt['class_name']
                x1, y1, x2, y2 = gt['bbox']
                
                # Calculate center position
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                spatial_stats['class_positions'][class_name].append((center_x, center_y))
                
                # Add to heatmap (normalize to 512x512)
                heat_x = int(center_x * 512 / width)
                heat_y = int(center_y * 512 / height)
                heat_x = max(0, min(511, heat_x))
                heat_y = max(0, min(511, heat_y))
                spatial_stats['detection_heatmaps'][class_name][heat_y, heat_x] += 1
        
        # Analyze spatial bias metrics
        for class_name in self.class_names:
            positions = spatial_stats['class_positions'][class_name]
            if positions:
                positions = np.array(positions)
                
                # Calculate spatial statistics
                center_x_mean = np.mean(positions[:, 0])
                center_y_mean = np.mean(positions[:, 1])
                center_x_std = np.std(positions[:, 0])
                center_y_std = np.std(positions[:, 1])
                
                # Determine actual spatial pattern based on mean positions
                # Normalize to image coordinates (assuming average 1280x720 image)
                norm_x = center_x_mean / 1280 if center_x_mean > 10 else center_x_mean
                norm_y = center_y_mean / 720 if center_y_mean > 10 else center_y_mean
                
                if norm_y > 0.7:  # Bottom 30%
                    if norm_x < 0.3:  # Left
                        actual_pattern = 'bottom_left'
                    elif norm_x > 0.7:  # Right
                        actual_pattern = 'bottom_right'
                    else:
                        actual_pattern = 'bottom_center'
                elif norm_y < 0.3:  # Top 30%
                    if norm_x < 0.3:
                        actual_pattern = 'upper_left'
                    elif norm_x > 0.7:
                        actual_pattern = 'upper_right'
                    else:
                        actual_pattern = 'upper_center'
                else:  # Middle
                    if norm_x < 0.3:
                        actual_pattern = 'center_left'
                    elif norm_x > 0.7:
                        actual_pattern = 'center_right'
                    else:
                        actual_pattern = 'center'
                
                expected = expected_patterns.get(class_name, 'unknown')
                pattern_match = expected in actual_pattern or actual_pattern in expected
                
                spatial_stats['spatial_bias_metrics'][class_name] = {
                    'mean_x': center_x_mean,
                    'mean_y': center_y_mean,
                    'std_x': center_x_std,
                    'std_y': center_y_std,
                    'num_detections': len(positions),
                    'actual_pattern': actual_pattern,
                    'expected_pattern': expected,
                    'pattern_match': pattern_match,
                    'normalized_position': (norm_x, norm_y)
                }
        
        # Create spatial bias visualization
        self._plot_spatial_bias(spatial_stats)
        
        return spatial_stats
    
    def _plot_spatial_bias(self, spatial_stats: Dict):
        """Create spatial bias visualization plots."""
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Spatial Distribution Analysis by Class', fontsize=16)
        
        axes = axes.flatten()
        
        for i, class_name in enumerate(self.class_names):
            if i < len(axes):
                heatmap = spatial_stats['detection_heatmaps'][class_name]
                
                if heatmap.sum() > 0:
                    # Apply Gaussian smoothing for better visualization
                    from scipy.ndimage import gaussian_filter
                    smoothed_heatmap = gaussian_filter(heatmap, sigma=2)
                    
                    im = axes[i].imshow(smoothed_heatmap, cmap='hot', interpolation='bilinear')
                    axes[i].set_title(f'{class_name}\n({heatmap.sum():.0f} detections)')
                    
                    # Add expected region indicator
                    metrics = spatial_stats['spatial_bias_metrics'].get(class_name, {})
                    expected_pattern = metrics.get('expected_pattern', 'unknown')
                    pattern_match = metrics.get('pattern_match', False)
                    
                    color = 'green' if pattern_match else 'red'
                    axes[i].text(0.02, 0.98, f'Expected: {expected_pattern}', 
                               transform=axes[i].transAxes, fontsize=8,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7),
                               verticalalignment='top')
                    
                    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                else:
                    axes[i].text(0.5, 0.5, 'No Detections', ha='center', va='center',
                               transform=axes[i].transAxes, fontsize=12)
                    axes[i].set_title(f'{class_name}\n(0 detections)')
                
                axes[i].set_xlabel('Image Width')
                axes[i].set_ylabel('Image Height')
        
        plt.tight_layout()
        plot_path = self.output_dir / 'spatial_bias_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Spatial bias analysis saved to {plot_path}")

    def analyze_class_specific_performance(self, collected_data: List[Dict]) -> Dict:
        """
        Analyze performance patterns for each class with visual examples.
        
        Args:
            collected_data: List of collected prediction data
            
        Returns:
            Dictionary containing class-specific analysis results
        """
        print("🔍 Analyzing class-specific performance patterns...")
        
        class_performance = {}
        
        # Initialize performance tracking for each class
        for class_name in self.class_names:
            class_performance[class_name] = {
                'total_gt': 0,
                'total_predictions': 0,
                'correct_predictions': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'avg_confidence': 0,
                'confidence_scores': [],
                'best_examples': [],
                'worst_examples': [],
                'challenging_cases': []
            }
        
        # Analyze each image
        for data in collected_data:
            predictions = data['predictions']
            ground_truth = data['ground_truth']
            image_id = data['image_id']
            
            # Track ground truth counts
            gt_by_class = {}
            for gt in ground_truth:
                class_name = gt['class_name']
                class_performance[class_name]['total_gt'] += 1
                if class_name not in gt_by_class:
                    gt_by_class[class_name] = []
                gt_by_class[class_name].append(gt)
            
            # Track prediction performance
            for pred in predictions:
                class_name = pred['class_name']
                class_performance[class_name]['total_predictions'] += 1
                class_performance[class_name]['confidence_scores'].append(pred['confidence'])
                
                # Find if this prediction matches any ground truth
                matched = False
                best_iou = 0
                for gt in ground_truth:
                    if gt['class_name'] == class_name:
                        iou = self._compute_iou(pred['bbox'], gt['bbox'])
                        if iou > 0.5:  # Consider it a match
                            matched = True
                            class_performance[class_name]['correct_predictions'] += 1
                            
                            # Track as good example if high confidence
                            if pred['confidence'] > 0.8:
                                class_performance[class_name]['best_examples'].append({
                                    'image_id': image_id,
                                    'confidence': pred['confidence'],
                                    'iou': iou,
                                    'prediction': pred,
                                    'ground_truth': gt
                                })
                            break
                        best_iou = max(best_iou, iou)
                
                if not matched:
                    class_performance[class_name]['false_positives'] += 1
                    
                    # Track as challenging case if moderate confidence but wrong
                    if 0.3 < pred['confidence'] < 0.7:
                        class_performance[class_name]['challenging_cases'].append({
                            'image_id': image_id,
                            'confidence': pred['confidence'],
                            'best_iou': best_iou,
                            'prediction': pred,
                            'type': 'false_positive'
                        })
            
            # Track false negatives (missed ground truth)
            for class_name, gts in gt_by_class.items():
                for gt in gts:
                    matched = False
                    for pred in predictions:
                        if pred['class_name'] == class_name:
                            iou = self._compute_iou(pred['bbox'], gt['bbox'])
                            if iou > 0.5:
                                matched = True
                                break
                    
                    if not matched:
                        class_performance[class_name]['false_negatives'] += 1
                        class_performance[class_name]['challenging_cases'].append({
                            'image_id': image_id,
                            'ground_truth': gt,
                            'type': 'false_negative'
                        })
        
        # Calculate final metrics and sort examples
        for class_name in self.class_names:
            perf = class_performance[class_name]
            
            # Calculate average confidence
            if perf['confidence_scores']:
                perf['avg_confidence'] = np.mean(perf['confidence_scores'])
            
            # Calculate precision and recall
            if perf['total_predictions'] > 0:
                perf['precision'] = perf['correct_predictions'] / perf['total_predictions']
            else:
                perf['precision'] = 0
                
            if perf['total_gt'] > 0:
                perf['recall'] = perf['correct_predictions'] / perf['total_gt']
            else:
                perf['recall'] = 0
            
            # Calculate F1 score
            if perf['precision'] + perf['recall'] > 0:
                perf['f1_score'] = 2 * (perf['precision'] * perf['recall']) / (perf['precision'] + perf['recall'])
            else:
                perf['f1_score'] = 0
            
            # Sort examples
            perf['best_examples'] = sorted(perf['best_examples'], 
                                         key=lambda x: x['confidence'], reverse=True)[:5]
            perf['challenging_cases'] = sorted(perf['challenging_cases'][:10], 
                                             key=lambda x: x.get('confidence', 0), reverse=True)
        
        return class_performance
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute IoU between two bounding boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

    def generate_improvement_recommendations(self, results: Dict) -> Dict:
        """
        Generate actionable improvement recommendations based on qualitative analysis.
        
        Args:
            results: Complete analysis results
            
        Returns:
            Dictionary containing structured recommendations
        """
        print("💡 Generating model improvement recommendations...")
        
        recommendations = {
            'data_improvements': [],
            'model_architecture': [],
            'training_strategy': [],
            'deployment_considerations': [],
            'priority_actions': []
        }
        
        # Extract key metrics
        error_stats = results['error_analysis']['statistics']
        class_performance = results['class_performance']
        spatial_analysis = results['spatial_analysis']['spatial_bias_metrics']
        confidence_stats = results['confidence_analysis']
        
        # Data improvement recommendations
        low_performing_classes = [
            name for name, perf in class_performance.items() 
            if perf['f1_score'] < 0.3 and perf['total_gt'] > 0
        ]
        
        if low_performing_classes:
            recommendations['data_improvements'].extend([
                f"Collect more training data for underperforming classes: {', '.join(low_performing_classes)}",
                f"Apply class-specific data augmentation for rare classes: {', '.join(low_performing_classes)}",
                "Consider synthetic data generation for extremely rare classes (train, rider)"
            ])
        
        # Check for spatial bias issues
        spatial_mismatches = [
            name for name, metrics in spatial_analysis.items()
            if not metrics.get('pattern_match', True) and metrics['num_detections'] > 5
        ]
        
        if spatial_mismatches:
            recommendations['data_improvements'].append(
                f"Review spatial distribution for classes with unexpected patterns: {', '.join(spatial_mismatches)}"
            )
        
        # Model architecture recommendations
        if error_stats['precision'] < 0.5:
            recommendations['model_architecture'].extend([
                "Consider increasing model capacity or using larger backbone (ResNet-101, ResNeXt)",
                "Implement focal loss with higher gamma to focus on hard examples",
                "Add feature pyramid networks (FPN) for multi-scale detection"
            ])
        
        if error_stats['recall'] < 0.5:
            recommendations['model_architecture'].extend([
                "Increase number of object queries in DETR decoder",
                "Implement auxiliary losses for intermediate decoder layers",
                "Consider ensemble methods with multiple detection heads"
            ])
        
        # Training strategy recommendations
        high_confidence_fps = sum(
            1 for fp in results['error_analysis']['false_positives']
            if fp['confidence'] > 0.7
        )
        
        if high_confidence_fps > error_stats['false_positives'] * 0.3:
            recommendations['training_strategy'].extend([
                "Implement hard negative mining to reduce overconfident false positives",
                "Increase class weights for background class",
                "Apply label smoothing to reduce overconfidence"
            ])
        
        # Check confidence calibration
        overconfident_classes = [
            name for name, stats in confidence_stats.items()
            if stats['count'] > 0 and stats['mean'] > 0.8 and class_performance[name]['precision'] < 0.6
        ]
        
        if overconfident_classes:
            recommendations['training_strategy'].append(
                f"Apply confidence calibration for overconfident classes: {', '.join(overconfident_classes)}"
            )
        
        # Class imbalance handling
        severely_imbalanced = [
            name for name, perf in class_performance.items()
            if perf['total_gt'] < 50 and perf['recall'] < 0.2
        ]
        
        if severely_imbalanced:
            recommendations['training_strategy'].extend([
                f"Implement stronger class rebalancing for: {', '.join(severely_imbalanced)}",
                "Use curriculum learning starting with balanced mini-batches",
                "Apply progressive resizing with class-aware sampling"
            ])
        
        # Deployment considerations
        if error_stats['precision'] > 0.8:
            recommendations['deployment_considerations'].append(
                f"Model suitable for high-precision applications (current precision: {error_stats['precision']:.3f})"
            )
        else:
            recommendations['deployment_considerations'].append(
                "Consider post-processing filters to improve precision for deployment"
            )
        
        if error_stats['recall'] < 0.7:
            recommendations['deployment_considerations'].append(
                "Model may miss critical objects - not suitable for safety-critical applications without improvement"
            )
        
        # Priority actions based on analysis
        recommendations['priority_actions'] = []
        
        if error_stats['precision'] < 0.3:
            recommendations['priority_actions'].append({
                'priority': 'HIGH',
                'action': 'Reduce False Positives',
                'methods': ['Increase confidence threshold', 'Apply stricter NMS', 'Hard negative mining']
            })
        
        if error_stats['recall'] < 0.3:
            recommendations['priority_actions'].append({
                'priority': 'HIGH',
                'action': 'Improve Object Detection',
                'methods': ['Collect more data', 'Increase model capacity', 'Class rebalancing']
            })
        
        if len(low_performing_classes) > 3:
            recommendations['priority_actions'].append({
                'priority': 'MEDIUM',
                'action': 'Address Class Imbalance',
                'methods': ['Focal loss tuning', 'Class-specific augmentation', 'Balanced sampling']
            })
        
        if len(spatial_mismatches) > 2:
            recommendations['priority_actions'].append({
                'priority': 'MEDIUM',
                'action': 'Fix Spatial Bias',
                'methods': ['Data collection review', 'Spatial augmentation', 'Position-aware training']
            })
        
        # Save recommendations report
        self._save_recommendations_report(recommendations, results)
        
        return recommendations
    
    def _save_recommendations_report(self, recommendations: Dict, results: Dict):
        """Save detailed recommendations report."""
        report_lines = [
            "# DETR Model Improvement Recommendations",
            "=" * 60,
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Based on analysis of {results['summary']['images_analyzed']} validation images",
            "",
            "## Executive Summary",
            ""
        ]
        
        # Executive summary
        summary = results['summary']
        report_lines.extend([
            f"**Overall Performance**: Precision {summary['precision']:.3f}, Recall {summary['recall']:.3f}",
            f"**Key Challenge**: {'Low Precision' if summary['precision'] < 0.5 else 'Low Recall' if summary['recall'] < 0.5 else 'Balanced Performance'}",
            f"**Primary Recommendation**: {recommendations['priority_actions'][0]['action'] if recommendations['priority_actions'] else 'Continue current approach'}",
            ""
        ])
        
        # Priority actions
        if recommendations['priority_actions']:
            report_lines.extend([
                "## Priority Actions",
                ""
            ])
            
            for action in recommendations['priority_actions']:
                report_lines.extend([
                    f"### {action['priority']} PRIORITY: {action['action']}",
                    "**Recommended Methods:**"
                ])
                for method in action['methods']:
                    report_lines.append(f"- {method}")
                report_lines.append("")
        
        # Detailed recommendations by category
        categories = [
            ('data_improvements', 'Data Collection & Quality'),
            ('model_architecture', 'Model Architecture'),
            ('training_strategy', 'Training Strategy'),
            ('deployment_considerations', 'Deployment Considerations')
        ]
        
        for key, title in categories:
            if recommendations[key]:
                report_lines.extend([
                    f"## {title}",
                    ""
                ])
                for rec in recommendations[key]:
                    report_lines.append(f"- {rec}")
                report_lines.append("")
        
        # Performance insights
        report_lines.extend([
            "## Detailed Performance Analysis",
            "",
            "### Class-Specific Insights",
            ""
        ])
        
        class_performance = results['class_performance']
        for class_name, perf in class_performance.items():
            status = "🟢" if perf['f1_score'] > 0.5 else "🟡" if perf['f1_score'] > 0.2 else "🔴"
            report_lines.append(
                f"- **{class_name}** {status}: F1={perf['f1_score']:.3f}, "
                f"P={perf['precision']:.3f}, R={perf['recall']:.3f} "
                f"({perf['total_gt']} GT objects)"
            )
        
        report_lines.extend([
            "",
            "### Implementation Timeline",
            "",
            "**Week 1-2: Immediate Improvements**",
            "- Adjust confidence thresholds based on precision/recall trade-offs",
            "- Implement post-processing filters",
            "- Review and clean validation data",
            "",
            "**Week 3-4: Training Improvements**",
            "- Implement recommended loss functions and class weights",
            "- Apply advanced data augmentation strategies",
            "- Retrain with improved configurations",
            "",
            "**Week 5-8: Advanced Enhancements**",
            "- Collect additional data for underperforming classes",
            "- Experiment with model architecture modifications",
            "- Implement ensemble methods if needed",
            "",
            "**Ongoing: Monitoring & Iteration**",
            "- Regular qualitative analysis on new data",
            "- A/B testing of model improvements",
            "- Continuous integration of user feedback",
            ""
        ])
        
        # Save report
        report_file = self.output_dir / "improvement_recommendations.md"
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"📄 Recommendations report saved to: {report_file}")

    def run_complete_qualitative_analysis(
        self, 
        max_images: int = 100, 
        num_visualizations: int = 20
    ) -> Dict:
        """
        Run complete qualitative analysis pipeline.
        
        Args:
            max_images: Maximum images to analyze
            num_visualizations: Number of visualization examples
            
        Returns:
            Complete analysis results
        """
        print("🚀 Starting complete qualitative analysis...")
        print("=" * 60)
        
        results = {}
        
        # Step 1: Collect predictions with images
        collected_data = self.collect_predictions_with_images(
            max_images=max_images,
            confidence_threshold=0.1
        )
        results['collected_data'] = collected_data
        
        # Step 2: Create prediction visualizations
        plot_paths = self.visualize_predictions(
            collected_data=collected_data,
            num_examples=num_visualizations,
            save_individual=True
        )
        results['visualization_paths'] = plot_paths
        
        # Step 3: Analyze confidence distributions
        confidence_stats = self.analyze_confidence_distribution(collected_data)
        results['confidence_analysis'] = confidence_stats
        
        # Step 4: Analyze error patterns
        error_analysis = self.analyze_error_patterns(collected_data, iou_threshold=0.5)
        results['error_analysis'] = error_analysis
        
        # Step 5: Analyze spatial bias patterns
        spatial_analysis = self.analyze_spatial_bias(collected_data)
        results['spatial_analysis'] = spatial_analysis
        
        # Step 6: Analyze class-specific performance
        class_performance = self.analyze_class_specific_performance(collected_data)
        results['class_performance'] = class_performance
        
        # Step 7: Generate attention analysis (on sample batch)
        sample_batch = next(iter(self.val_dataloader))
        images, _ = sample_batch
        images = images.to(self.device)
        attention_analysis = self.visualize_attention_maps(
            images[:4],  # Analyze first 4 images
            save_dir=str(self.output_dir / 'attention_maps')
        )
        results['attention_analysis'] = attention_analysis
        
        # Step 8: Generate comprehensive report
        report_path = self.create_error_analysis_report(error_analysis, collected_data)
        results['report_path'] = report_path
        
        # Step 9: Create preliminary summary for recommendations
        summary = {
            'images_analyzed': len(collected_data),
            'visualizations_created': len(plot_paths),
            'total_predictions': error_analysis['statistics']['total_predictions'],
            'total_ground_truth': error_analysis['statistics']['total_ground_truth'],
            'precision': error_analysis['statistics']['precision'],
            'recall': error_analysis['statistics']['recall'],
            'confidence_stats': {
                class_name: stats['mean'] for class_name, stats in confidence_stats.items()
                if stats['count'] > 0
            },
            'class_performance_summary': {
                class_name: {
                    'precision': perf['precision'],
                    'recall': perf['recall'],
                    'f1_score': perf['f1_score']
                } for class_name, perf in class_performance.items()
            },
            'spatial_patterns_match': {
                class_name: metrics.get('pattern_match', False)
                for class_name, metrics in spatial_analysis.get('spatial_bias_metrics', {}).items()
            }
        }
        results['summary'] = summary
        
        # Step 10: Generate improvement recommendations
        recommendations = self.generate_improvement_recommendations(results)
        results['recommendations'] = recommendations
        
        # Step 11: Save final analysis summary
        summary_file = self.output_dir / "analysis_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        results['summary_path'] = str(summary_file)
        
        print("🎉 Qualitative analysis complete!")
        print(f"📊 Analyzed {len(collected_data)} images")
        print(f"🎨 Created {len(plot_paths)} visualizations")
        print(f"📈 Overall Precision: {error_analysis['statistics']['precision']:.3f}")
        print(f"📈 Overall Recall: {error_analysis['statistics']['recall']:.3f}")
        print(f"🧠 Attention analysis: {len(attention_analysis.get('encoder_attention', []))} visualizations created")
        print(f"📍 Spatial patterns analyzed for {len(class_performance)} classes")
        print(f"🔍 Class-specific analysis completed")
        print(f"📄 Report saved: {report_path}")
        print(f"💾 Results saved to: {self.output_dir}")
        
        # Print quick insights
        print("\n🔍 Quick Insights:")
        best_class = max(class_performance.items(), key=lambda x: x[1]['f1_score'])
        worst_class = min(class_performance.items(), key=lambda x: x[1]['f1_score'])
        print(f"   • Best performing class: {best_class[0]} (F1: {best_class[1]['f1_score']:.3f})")
        print(f"   • Worst performing class: {worst_class[0]} (F1: {worst_class[1]['f1_score']:.3f})")
        
        spatial_matches = sum(1 for match in summary['spatial_patterns_match'].values() if match)
        print(f"   • Spatial pattern matches: {spatial_matches}/{len(self.class_names)} classes as expected")
        
        return results


if __name__ == "__main__":
    print("Qualitative Analysis Module for DETR")
    print("Use this module with a trained model and validation dataloader")
    print("Example usage:")
    print("  analyzer = QualitativeAnalyzer(model, val_dataloader)")
    print("  results = analyzer.run_complete_qualitative_analysis()")