"""
Failure Case Analysis for BDD100K Object Detection

This module provides systematic failure analysis including false positive/negative detection,
localization errors, and classification mistakes with safety-critical focus.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import json
from pathlib import Path
import cv2
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import seaborn as sns


class FailureAnalyzer:
    """
    Comprehensive failure analysis for object detection models.
    
    Analyzes different types of failures (FN, FP, localization errors) and
    identifies patterns in failure modes for targeted improvements.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None, iou_threshold: float = 0.5):
        """
        Initialize failure analyzer.
        
        Args:
            class_names: List of class names
            iou_threshold: IoU threshold for matching predictions to ground truth
        """
        if class_names is None:
            self.class_names = [
                'pedestrian', 'rider', 'car', 'truck', 'bus',
                'train', 'motorcycle', 'bicycle', 'traffic_light', 'traffic_sign'
            ]
        else:
            self.class_names = class_names
            
        self.iou_threshold = iou_threshold
        
        # Safety-critical classes for prioritized analysis
        self.safety_critical_classes = {'pedestrian', 'rider', 'bicycle', 'motorcycle'}
        
        # Define failure categories
        self.failure_types = {
            'false_negative': 'Ground truth object not detected',
            'false_positive': 'Predicted object with no corresponding ground truth',
            'classification_error': 'Correct detection but wrong class',
            'localization_error': 'Correct class but poor localization (IoU < threshold)',
            'duplicate_detection': 'Multiple predictions for same ground truth object'
        }
    
    def analyze_failures(self, 
                        predictions: List[Dict], 
                        ground_truth: List[Dict],
                        image_metadata: Optional[List[Dict]] = None) -> Dict:
        """
        Comprehensive failure analysis across all images.
        
        Args:
            predictions: List of prediction dictionaries
            ground_truth: List of ground truth annotations
            image_metadata: Optional metadata for contextual analysis
            
        Returns:
            Dictionary containing detailed failure analysis
        """
        print("Running comprehensive failure analysis...")
        
        # Organize data by image
        pred_by_image = self._group_by_image(predictions)
        gt_by_image = self._group_by_image(ground_truth)
        
        # Initialize analysis results
        results = {
            'failure_counts': defaultdict(int),
            'failure_by_class': defaultdict(lambda: defaultdict(int)),
            'failure_cases': {failure_type: [] for failure_type in self.failure_types.keys()},
            'safety_critical_failures': [],
            'failure_patterns': {},
            'localization_errors': [],
            'confidence_analysis': {},
            'contextual_failures': {}
        }
        
        if image_metadata:
            metadata_by_image = {meta['image_id']: meta for meta in image_metadata}
        else:
            metadata_by_image = {}
        
        # Analyze each image
        all_image_ids = set(pred_by_image.keys()) | set(gt_by_image.keys())
        
        for image_id in all_image_ids:
            image_preds = pred_by_image.get(image_id, [])
            image_gts = gt_by_image.get(image_id, [])
            image_meta = metadata_by_image.get(image_id, {})
            
            # Analyze this image's failures
            image_failures = self._analyze_image_failures(
                image_preds, image_gts, image_id, image_meta
            )
            
            # Accumulate results
            for failure_type, cases in image_failures.items():
                results['failure_cases'][failure_type].extend(cases)
                results['failure_counts'][failure_type] += len(cases)
                
                # Track by class
                for case in cases:
                    if 'class_name' in case:
                        results['failure_by_class'][failure_type][case['class_name']] += 1
                    
                    # Track safety-critical failures
                    if case.get('class_name') in self.safety_critical_classes:
                        case['failure_type'] = failure_type
                        results['safety_critical_failures'].append(case)
        
        # Post-process analysis
        results['failure_patterns'] = self._identify_failure_patterns(results)
        results['confidence_analysis'] = self._analyze_confidence_patterns(results)
        results['contextual_failures'] = self._analyze_contextual_patterns(results, metadata_by_image)
        
        # Add summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _group_by_image(self, annotations: List[Dict]) -> Dict[str, List[Dict]]:
        """Group annotations by image_id."""
        grouped = defaultdict(list)
        for ann in annotations:
            grouped[ann['image_id']].append(ann)
        return dict(grouped)
    
    def _analyze_image_failures(self, 
                               predictions: List[Dict], 
                               ground_truths: List[Dict],
                               image_id: str,
                               metadata: Dict) -> Dict[str, List[Dict]]:
        """Analyze failures for a single image."""
        failures = {failure_type: [] for failure_type in self.failure_types.keys()}
        
        if not ground_truths:
            # All predictions are false positives
            for pred in predictions:
                failure_case = {
                    'image_id': image_id,
                    'prediction': pred,
                    'class_name': self._get_class_name(pred['category_id']),
                    'confidence': pred.get('score', 0.0),
                    'metadata': metadata
                }
                failures['false_positive'].append(failure_case)
            return failures
        
        if not predictions:
            # All ground truths are false negatives
            for gt in ground_truths:
                failure_case = {
                    'image_id': image_id,
                    'ground_truth': gt,
                    'class_name': self._get_class_name(gt['category_id']),
                    'metadata': metadata
                }
                failures['false_negative'].append(failure_case)
            return failures
        
        # Calculate IoU matrix
        iou_matrix = self._calculate_iou_matrix(predictions, ground_truths)
        
        # Track matches
        matched_preds = set()
        matched_gts = set()
        
        # Find best matches for each prediction
        for pred_idx, pred in enumerate(predictions):
            best_gt_idx = -1
            best_iou = 0.0
            best_class_match = False
            
            for gt_idx, gt in enumerate(ground_truths):
                if gt_idx in matched_gts:
                    continue
                    
                iou = iou_matrix[pred_idx][gt_idx]
                class_match = pred['category_id'] == gt['category_id']
                
                # Prioritize class-matching high-IoU matches
                if class_match and iou >= self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_class_match = True
                elif not best_class_match and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
                    best_class_match = class_match
            
            # Analyze this prediction
            if best_gt_idx >= 0:
                gt = ground_truths[best_gt_idx]
                class_match = pred['category_id'] == gt['category_id']
                
                if class_match and best_iou >= self.iou_threshold:
                    # Good match - no failure
                    matched_preds.add(pred_idx)
                    matched_gts.add(best_gt_idx)
                    
                elif class_match and best_iou < self.iou_threshold:
                    # Localization error
                    failure_case = {
                        'image_id': image_id,
                        'prediction': pred,
                        'ground_truth': gt,
                        'class_name': self._get_class_name(pred['category_id']),
                        'iou': best_iou,
                        'confidence': pred.get('score', 0.0),
                        'metadata': metadata
                    }
                    failures['localization_error'].append(failure_case)
                    matched_preds.add(pred_idx)
                    matched_gts.add(best_gt_idx)
                    
                elif not class_match and best_iou >= self.iou_threshold:
                    # Classification error
                    failure_case = {
                        'image_id': image_id,
                        'prediction': pred,
                        'ground_truth': gt,
                        'predicted_class': self._get_class_name(pred['category_id']),
                        'true_class': self._get_class_name(gt['category_id']),
                        'iou': best_iou,
                        'confidence': pred.get('score', 0.0),
                        'metadata': metadata
                    }
                    failures['classification_error'].append(failure_case)
                    matched_preds.add(pred_idx)
                    matched_gts.add(best_gt_idx)
                else:
                    # No good match - false positive
                    failure_case = {
                        'image_id': image_id,
                        'prediction': pred,
                        'class_name': self._get_class_name(pred['category_id']),
                        'confidence': pred.get('score', 0.0),
                        'best_iou': best_iou,
                        'metadata': metadata
                    }
                    failures['false_positive'].append(failure_case)
            else:
                # No ground truth nearby - false positive
                failure_case = {
                    'image_id': image_id,
                    'prediction': pred,
                    'class_name': self._get_class_name(pred['category_id']),
                    'confidence': pred.get('score', 0.0),
                    'metadata': metadata
                }
                failures['false_positive'].append(failure_case)
        
        # Unmatched ground truths are false negatives
        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx not in matched_gts:
                failure_case = {
                    'image_id': image_id,
                    'ground_truth': gt,
                    'class_name': self._get_class_name(gt['category_id']),
                    'metadata': metadata
                }
                failures['false_negative'].append(failure_case)
        
        # Check for duplicate detections
        for gt_idx in matched_gts:
            matching_preds = [i for i, pred in enumerate(predictions) 
                            if i in matched_preds and 
                            iou_matrix[i][gt_idx] >= self.iou_threshold and
                            pred['category_id'] == ground_truths[gt_idx]['category_id']]
            
            if len(matching_preds) > 1:
                # Multiple predictions for same ground truth
                for pred_idx in matching_preds[1:]:  # Skip the first (best) match
                    failure_case = {
                        'image_id': image_id,
                        'prediction': predictions[pred_idx],
                        'ground_truth': ground_truths[gt_idx],
                        'class_name': self._get_class_name(predictions[pred_idx]['category_id']),
                        'confidence': predictions[pred_idx].get('score', 0.0),
                        'metadata': metadata
                    }
                    failures['duplicate_detection'].append(failure_case)
        
        return failures
    
    def _calculate_iou_matrix(self, predictions: List[Dict], ground_truths: List[Dict]) -> List[List[float]]:
        """Calculate IoU matrix between predictions and ground truths."""
        iou_matrix = []
        
        for pred in predictions:
            pred_ious = []
            for gt in ground_truths:
                iou = self._calculate_bbox_iou(pred['bbox'], gt['bbox'])
                pred_ious.append(iou)
            iou_matrix.append(pred_ious)
        
        return iou_matrix
    
    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes."""
        # Convert [x, y, width, height] to [x1, y1, x2, y2]
        x1_1, y1_1, w1, h1 = bbox1
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        
        x1_2, y1_2, w2, h2 = bbox2
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"unknown_{class_id}"
    
    def _identify_failure_patterns(self, results: Dict) -> Dict:
        """Identify patterns in failure modes."""
        patterns = {}
        
        # 1. Most problematic classes
        class_failure_rates = {}
        for failure_type in self.failure_types.keys():
            class_failures = results['failure_by_class'][failure_type]
            for class_name, count in class_failures.items():
                if class_name not in class_failure_rates:
                    class_failure_rates[class_name] = defaultdict(int)
                class_failure_rates[class_name][failure_type] = count
        
        patterns['class_failure_rates'] = dict(class_failure_rates)
        
        # 2. Common classification confusions
        classification_errors = results['failure_cases']['classification_error']
        confusion_matrix = defaultdict(lambda: defaultdict(int))
        
        for error in classification_errors:
            true_class = error['true_class']
            pred_class = error['predicted_class']
            confusion_matrix[true_class][pred_class] += 1
        
        patterns['classification_confusions'] = dict(confusion_matrix)
        
        # 3. Confidence distribution by failure type
        confidence_by_failure = {}
        for failure_type, cases in results['failure_cases'].items():
            confidences = [case.get('confidence', 0.0) for case in cases if 'confidence' in case]
            if confidences:
                confidence_by_failure[failure_type] = {
                    'mean': np.mean(confidences),
                    'std': np.std(confidences),
                    'median': np.median(confidences),
                    'min': np.min(confidences),
                    'max': np.max(confidences)
                }
        
        patterns['confidence_patterns'] = confidence_by_failure
        
        return patterns
    
    def _analyze_confidence_patterns(self, results: Dict) -> Dict:
        """Analyze confidence score patterns in failures."""
        confidence_analysis = {}
        
        # Analyze false positives by confidence
        fp_cases = results['failure_cases']['false_positive']
        fp_confidences = [case.get('confidence', 0.0) for case in fp_cases if 'confidence' in case]
        
        if fp_confidences:
            # Bin confidences
            bins = np.linspace(0, 1, 11)  # 10 bins
            bin_counts, _ = np.histogram(fp_confidences, bins=bins)
            
            confidence_analysis['false_positive_distribution'] = {
                'bins': bins.tolist(),
                'counts': bin_counts.tolist(),
                'high_confidence_fps': sum(1 for c in fp_confidences if c > 0.8),
                'total_fps': len(fp_confidences)
            }
        
        # Analyze localization errors by confidence
        loc_cases = results['failure_cases']['localization_error']
        loc_confidences = [case.get('confidence', 0.0) for case in loc_cases if 'confidence' in case]
        loc_ious = [case.get('iou', 0.0) for case in loc_cases if 'iou' in case]
        
        if loc_confidences and loc_ious:
            # Correlation between confidence and IoU
            correlation = np.corrcoef(loc_confidences, loc_ious)[0, 1] if len(loc_confidences) > 1 else 0.0
            
            confidence_analysis['localization_confidence_correlation'] = {
                'correlation': float(correlation),
                'mean_confidence': np.mean(loc_confidences),
                'mean_iou': np.mean(loc_ious)
            }
        
        return confidence_analysis
    
    def _analyze_contextual_patterns(self, results: Dict, metadata_by_image: Dict) -> Dict:
        """Analyze failure patterns by environmental context."""
        contextual_failures = {}
        
        if not metadata_by_image:
            return contextual_failures
        
        # Group failures by weather/lighting/scene
        context_keys = ['weather', 'timeofday', 'scene']
        
        for context_key in context_keys:
            context_failures = defaultdict(lambda: defaultdict(int))
            
            for failure_type, cases in results['failure_cases'].items():
                for case in cases:
                    metadata = case.get('metadata', {})
                    context_value = metadata.get(context_key, 'unknown')
                    context_failures[context_value][failure_type] += 1
            
            if context_failures:
                contextual_failures[context_key] = dict(context_failures)
        
        return contextual_failures
    
    def generate_failure_report(self, results: Dict, save_path: Optional[str] = None) -> str:
        """Generate comprehensive failure analysis report."""
        report = []
        report.append("=" * 80)
        report.append("BDD100K MODEL - COMPREHENSIVE FAILURE ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        # Overall failure statistics
        report.append("📊 OVERALL FAILURE STATISTICS")
        report.append("-" * 50)
        
        total_failures = sum(results['failure_counts'].values())
        for failure_type, count in results['failure_counts'].items():
            percentage = (count / total_failures * 100) if total_failures > 0 else 0
            description = self.failure_types[failure_type]
            report.append(f"{failure_type:20} {count:6} ({percentage:5.1f}%) - {description}")
        
        report.append("")
        
        # Safety-critical failures
        safety_failures = len(results['safety_critical_failures'])
        report.append("🚨 SAFETY-CRITICAL FAILURES")
        report.append("-" * 50)
        report.append(f"Total safety-critical failures: {safety_failures}")
        
        if safety_failures > 0:
            safety_by_type = defaultdict(int)
            safety_by_class = defaultdict(int)
            
            for failure in results['safety_critical_failures']:
                safety_by_type[failure['failure_type']] += 1
                safety_by_class[failure['class_name']] += 1
            
            report.append("\\nBy failure type:")
            for failure_type, count in safety_by_type.items():
                percentage = (count / safety_failures * 100)
                report.append(f"  {failure_type:20} {count:6} ({percentage:5.1f}%)")
            
            report.append("\\nBy class:")
            for class_name, count in safety_by_class.items():
                percentage = (count / safety_failures * 100)
                report.append(f"  {class_name:15} {count:6} ({percentage:5.1f}%)")
        
        report.append("")
        
        # Class-specific analysis
        report.append("🎯 PER-CLASS FAILURE ANALYSIS")
        report.append("-" * 50)
        
        patterns = results.get('failure_patterns', {})
        class_failure_rates = patterns.get('class_failure_rates', {})
        
        for class_name in self.class_names:
            if class_name in class_failure_rates:
                class_failures = class_failure_rates[class_name]
                total_class_failures = sum(class_failures.values())
                
                report.append(f"\\n{class_name.upper()}:")
                if class_name in self.safety_critical_classes:
                    report.append("  ⚠️ SAFETY-CRITICAL CLASS")
                
                for failure_type, count in class_failures.items():
                    if count > 0:
                        percentage = (count / total_class_failures * 100)
                        report.append(f"  {failure_type:20} {count:4} ({percentage:5.1f}%)")
        
        report.append("")
        
        # Classification confusion analysis
        if 'classification_confusions' in patterns:
            confusions = patterns['classification_confusions']
            if confusions:
                report.append("🔄 CLASSIFICATION CONFUSIONS")
                report.append("-" * 50)
                
                # Find most common confusions
                confusion_list = []
                for true_class, pred_classes in confusions.items():
                    for pred_class, count in pred_classes.items():
                        confusion_list.append((count, true_class, pred_class))
                
                confusion_list.sort(reverse=True)
                
                report.append("Most common misclassifications:")
                for count, true_class, pred_class in confusion_list[:10]:
                    report.append(f"  {true_class:15} → {pred_class:15} ({count:3} cases)")
                
                report.append("")
        
        # Confidence analysis
        if 'confidence_patterns' in patterns:
            conf_patterns = patterns['confidence_patterns']
            if conf_patterns:
                report.append("📈 CONFIDENCE ANALYSIS")
                report.append("-" * 50)
                
                for failure_type, stats in conf_patterns.items():
                    report.append(f"\\n{failure_type.upper()}:")
                    report.append(f"  Mean confidence: {stats['mean']:.3f}")
                    report.append(f"  Std deviation:   {stats['std']:.3f}")
                    report.append(f"  Median:          {stats['median']:.3f}")
                    report.append(f"  Range:           {stats['min']:.3f} - {stats['max']:.3f}")
        
        report.append("")
        
        # Contextual failure analysis
        contextual = results.get('contextual_failures', {})
        if contextual:
            report.append("🌤️ ENVIRONMENTAL CONTEXT FAILURES")
            report.append("-" * 50)
            
            for context_type, context_data in contextual.items():
                report.append(f"\\n{context_type.upper()} CONDITIONS:")
                
                # Sort by total failures
                context_totals = [(sum(failures.values()), condition, failures) 
                                for condition, failures in context_data.items()]
                context_totals.sort(reverse=True)
                
                for total_failures, condition, failures in context_totals:
                    report.append(f"  {condition:15} Total: {total_failures:4}")
                    for failure_type, count in failures.items():
                        if count > 0:
                            report.append(f"    {failure_type:20} {count:3}")
        
        report_text = "\\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
        
        return report_text
    
    def visualize_failure_patterns(self, results: Dict, output_dir: str) -> None:
        """Create visualizations of failure patterns."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Failure distribution pie chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        failure_counts = results['failure_counts']
        labels = []
        sizes = []
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
        
        for i, (failure_type, count) in enumerate(failure_counts.items()):
            if count > 0:
                labels.append(f"{failure_type.replace('_', ' ').title()}\\n({count})")
                sizes.append(count)
        
        if sizes:
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors[:len(sizes)])
            ax.set_title('Distribution of Failure Types', fontsize=16, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Failures Detected', ha='center', va='center', 
                   fontsize=16, transform=ax.transAxes)
        
        plt.savefig(output_path / 'failure_distribution.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # 2. Per-class failure heatmap
        patterns = results.get('failure_patterns', {})
        class_failure_rates = patterns.get('class_failure_rates', {})
        
        if class_failure_rates:
            failure_matrix = []
            class_labels = []
            failure_labels = list(self.failure_types.keys())
            
            for class_name in self.class_names:
                if class_name in class_failure_rates:
                    class_labels.append(class_name)
                    row = []
                    for failure_type in failure_labels:
                        count = class_failure_rates[class_name].get(failure_type, 0)
                        row.append(count)
                    failure_matrix.append(row)
            
            if failure_matrix:
                fig, ax = plt.subplots(1, 1, figsize=(12, 8))
                
                im = ax.imshow(failure_matrix, cmap='Reds', aspect='auto')
                
                # Set ticks and labels
                ax.set_xticks(range(len(failure_labels)))
                ax.set_yticks(range(len(class_labels)))
                ax.set_xticklabels([f.replace('_', '\\n') for f in failure_labels], rotation=45, ha='right')
                ax.set_yticklabels(class_labels)
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label='Number of Failures')
                
                # Add text annotations
                for i in range(len(class_labels)):
                    for j in range(len(failure_labels)):
                        value = failure_matrix[i][j]
                        if value > 0:
                            ax.text(j, i, str(value), ha='center', va='center',
                                  color='white' if value > np.max(failure_matrix) * 0.6 else 'black')
                
                ax.set_title('Failure Patterns by Class', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(output_path / 'class_failure_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
        
        # 3. Safety-critical failure analysis
        safety_failures = results.get('safety_critical_failures', [])
        if safety_failures:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Safety failures by type
            safety_by_type = defaultdict(int)
            safety_by_class = defaultdict(int)
            
            for failure in safety_failures:
                safety_by_type[failure['failure_type']] += 1
                safety_by_class[failure['class_name']] += 1
            
            # Plot by failure type
            types = list(safety_by_type.keys())
            counts = list(safety_by_type.values())
            
            bars1 = ax1.bar(types, counts, color='red', alpha=0.7)
            ax1.set_title('Safety-Critical Failures by Type', fontweight='bold')
            ax1.set_ylabel('Count')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars1, counts):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            # Plot by class
            classes = list(safety_by_class.keys())
            class_counts = list(safety_by_class.values())
            
            bars2 = ax2.bar(classes, class_counts, color='orange', alpha=0.7)
            ax2.set_title('Safety-Critical Failures by Class', fontweight='bold')
            ax2.set_ylabel('Count')
            
            for bar, count in zip(bars2, class_counts):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path / 'safety_critical_failures.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print(f"✅ Failure analysis visualizations saved to {output_dir}")
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate summary statistics for the failure analysis."""
        summary = {}
        
        # Basic counts
        failure_counts = results['failure_counts']
        summary['total_failures'] = sum(failure_counts.values())
        summary['false_negatives'] = failure_counts.get('false_negative', 0)
        summary['false_positives'] = failure_counts.get('false_positive', 0) 
        summary['classification_errors'] = failure_counts.get('classification_error', 0)
        summary['localization_errors'] = failure_counts.get('localization_error', 0)
        summary['duplicate_detections'] = failure_counts.get('duplicate_detection', 0)
        
        # Safety-critical summary
        safety_failures = results.get('safety_critical_failures', [])
        summary['safety_critical_failures'] = len(safety_failures)
        
        # Most problematic failure type
        if failure_counts:
            most_common_failure = max(failure_counts.items(), key=lambda x: x[1])
            summary['most_common_failure_type'] = most_common_failure[0]
            summary['most_common_failure_count'] = most_common_failure[1]
        else:
            summary['most_common_failure_type'] = 'none'
            summary['most_common_failure_count'] = 0
        
        # Class with most failures
        failure_by_class = results.get('failure_by_class', {})
        all_class_failures = defaultdict(int)
        for failure_type_dict in failure_by_class.values():
            for class_name, count in failure_type_dict.items():
                all_class_failures[class_name] += count
        
        if all_class_failures:
            most_problematic_class = max(all_class_failures.items(), key=lambda x: x[1])
            summary['most_problematic_class'] = most_problematic_class[0]
            summary['most_problematic_class_failures'] = most_problematic_class[1]
        else:
            summary['most_problematic_class'] = 'none'
            summary['most_problematic_class_failures'] = 0
        
        return summary