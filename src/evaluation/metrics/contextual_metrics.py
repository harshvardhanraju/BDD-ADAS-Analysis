"""
Contextual Performance Metrics for BDD100K Object Detection

This module evaluates model performance across different environmental contexts
(weather, lighting, scene type) and object characteristics (size, position, occlusion).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class ContextualMetrics:
    """
    Evaluates model performance across different contexts and conditions.
    
    Analyzes how environmental factors and object characteristics affect
    detection performance, crucial for autonomous driving applications.
    """
    
    def __init__(self):
        """Initialize contextual metrics evaluator."""
        self.class_names = [
            'pedestrian', 'rider', 'car', 'truck', 'bus',
            'train', 'motorcycle', 'bicycle', 'traffic_light', 'traffic_sign'
        ]
        
        # Environmental context categories
        self.weather_conditions = ['clear', 'rainy', 'snowy', 'foggy', 'overcast', 'partly cloudy']
        self.lighting_conditions = ['daytime', 'night', 'dawn', 'dusk']  
        self.scene_types = ['city street', 'highway', 'residential', 'parking lot', 'gas station', 'tunnel']
        
        # Object characteristic categories
        self.size_categories = {
            'tiny': (0, 16**2),       # < 256 px²
            'small': (16**2, 32**2),   # 256 - 1024 px²
            'medium': (32**2, 96**2),  # 1024 - 9216 px²
            'large': (96**2, float('inf'))  # > 9216 px²
        }
        
        # Image position categories (normalized coordinates)
        self.position_categories = {
            'top_left': (0.0, 0.0, 0.33, 0.33),
            'top_center': (0.33, 0.0, 0.67, 0.33),
            'top_right': (0.67, 0.0, 1.0, 0.33),
            'middle_left': (0.0, 0.33, 0.33, 0.67),
            'middle_center': (0.33, 0.33, 0.67, 0.67),
            'middle_right': (0.67, 0.33, 1.0, 0.67),
            'bottom_left': (0.0, 0.67, 0.33, 1.0),
            'bottom_center': (0.33, 0.67, 0.67, 1.0),
            'bottom_right': (0.67, 0.67, 1.0, 1.0)
        }
        
    def evaluate_environmental_performance(self, 
                                         predictions: List[Dict],
                                         ground_truth: List[Dict],
                                         image_metadata: List[Dict]) -> Dict:
        """
        Evaluate performance across environmental conditions.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth annotations
            image_metadata: List of image metadata with keys:
                - 'image_id': Image identifier
                - 'weather': Weather condition
                - 'timeofday': Lighting condition  
                - 'scene': Scene type
                
        Returns:
            Dictionary with environmental performance analysis
        """
        # Create metadata lookup
        metadata_lookup = {meta['image_id']: meta for meta in image_metadata}
        
        results = {
            'weather_performance': {},
            'lighting_performance': {},
            'scene_performance': {},
            'environmental_summary': {}
        }
        
        # Evaluate by weather condition
        results['weather_performance'] = self._evaluate_by_condition(
            predictions, ground_truth, metadata_lookup, 'weather', self.weather_conditions
        )
        
        # Evaluate by lighting condition
        results['lighting_performance'] = self._evaluate_by_condition(
            predictions, ground_truth, metadata_lookup, 'timeofday', self.lighting_conditions
        )
        
        # Evaluate by scene type
        results['scene_performance'] = self._evaluate_by_condition(
            predictions, ground_truth, metadata_lookup, 'scene', self.scene_types
        )
        
        # Generate environmental summary
        results['environmental_summary'] = self._generate_environmental_summary(results)
        
        return results
    
    def evaluate_object_characteristics(self,
                                      predictions: List[Dict],
                                      ground_truth: List[Dict]) -> Dict:
        """
        Evaluate performance by object characteristics (size, position).
        
        Returns:
            Dictionary with object characteristic performance analysis
        """
        results = {
            'size_performance': {},
            'position_performance': {},
            'characteristic_summary': {}
        }
        
        # Evaluate by object size
        results['size_performance'] = self._evaluate_by_size(predictions, ground_truth)
        
        # Evaluate by object position
        results['position_performance'] = self._evaluate_by_position(predictions, ground_truth)
        
        # Generate characteristic summary
        results['characteristic_summary'] = self._generate_characteristic_summary(results)
        
        return results
    
    def _evaluate_by_condition(self,
                              predictions: List[Dict],
                              ground_truth: List[Dict],
                              metadata_lookup: Dict,
                              condition_key: str,
                              conditions: List[str]) -> Dict:
        """Evaluate performance for each condition in a category."""
        condition_results = {}
        
        for condition in conditions:
            # Filter data for this condition
            condition_image_ids = set()
            for image_id, metadata in metadata_lookup.items():
                if metadata.get(condition_key) == condition:
                    condition_image_ids.add(image_id)
            
            if not condition_image_ids:
                continue
                
            # Filter predictions and ground truth
            condition_preds = [p for p in predictions if p['image_id'] in condition_image_ids]
            condition_gt = [gt for gt in ground_truth if gt['image_id'] in condition_image_ids]
            
            # Calculate metrics for this condition
            if condition_gt:
                metrics = self._calculate_detection_metrics(condition_preds, condition_gt)
                metrics['num_images'] = len(condition_image_ids)
                metrics['num_objects'] = len(condition_gt)
                condition_results[condition] = metrics
        
        return condition_results
    
    def _evaluate_by_size(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate performance by object size categories."""
        size_results = {}
        
        for size_category, (min_area, max_area) in self.size_categories.items():
            # Filter ground truth by size
            size_gt = []
            size_image_ids = set()
            
            for gt in ground_truth:
                area = gt.get('area', gt['bbox'][2] * gt['bbox'][3])
                if min_area <= area < max_area:
                    size_gt.append(gt)
                    size_image_ids.add(gt['image_id'])
            
            # Filter predictions to matching images
            size_preds = [p for p in predictions if p['image_id'] in size_image_ids]
            
            if size_gt:
                metrics = self._calculate_detection_metrics(size_preds, size_gt)
                metrics['num_objects'] = len(size_gt)
                metrics['avg_area'] = np.mean([gt.get('area', gt['bbox'][2] * gt['bbox'][3]) for gt in size_gt])
                size_results[size_category] = metrics
                
        return size_results
    
    def _evaluate_by_position(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Evaluate performance by object position in image."""
        position_results = {}
        
        for position_name, (x_min, y_min, x_max, y_max) in self.position_categories.items():
            # Filter ground truth by position
            position_gt = []
            position_image_ids = set()
            
            for gt in ground_truth:
                # Calculate object center (normalized coordinates)
                bbox = gt['bbox']  # [x, y, width, height]
                center_x = (bbox[0] + bbox[2] / 2) / 1280.0  # BDD100K width
                center_y = (bbox[1] + bbox[3] / 2) / 720.0   # BDD100K height
                
                # Check if center falls in this position category
                if x_min <= center_x < x_max and y_min <= center_y < y_max:
                    position_gt.append(gt)
                    position_image_ids.add(gt['image_id'])
            
            # Filter predictions to matching images  
            position_preds = [p for p in predictions if p['image_id'] in position_image_ids]
            
            if position_gt:
                metrics = self._calculate_detection_metrics(position_preds, position_gt)
                metrics['num_objects'] = len(position_gt)
                position_results[position_name] = metrics
                
        return position_results
    
    def _calculate_detection_metrics(self, predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
        """Calculate basic detection metrics for filtered data."""
        if not ground_truth:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'ap': 0.0}
        
        # Simple IoU-based matching for basic metrics
        tp, fp, fn = self._calculate_tp_fp_fn_simple(predictions, ground_truth)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Estimate AP (simplified)
        ap = recall * precision  # Rough approximation
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'ap': ap,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        }
    
    def _calculate_tp_fp_fn_simple(self, predictions: List[Dict], ground_truth: List[Dict]) -> Tuple[int, int, int]:
        """Simplified TP/FP/FN calculation using IoU threshold of 0.5."""
        if not ground_truth:
            return 0, len(predictions), 0
        if not predictions:
            return 0, 0, len(ground_truth)
            
        # Group by image for efficient matching
        pred_by_image = defaultdict(list)
        gt_by_image = defaultdict(list)
        
        for pred in predictions:
            pred_by_image[pred['image_id']].append(pred)
            
        for gt in ground_truth:
            gt_by_image[gt['image_id']].append(gt)
        
        total_tp = 0
        total_fp = 0
        total_fn = 0
        
        # Process each image
        all_image_ids = set(pred_by_image.keys()) | set(gt_by_image.keys())
        
        for image_id in all_image_ids:
            image_preds = pred_by_image.get(image_id, [])
            image_gts = gt_by_image.get(image_id, [])
            
            if not image_gts:
                total_fp += len(image_preds)
                continue
                
            if not image_preds:
                total_fn += len(image_gts)
                continue
            
            # Calculate IoU matrix for this image
            matched_gt = set()
            matched_pred = set()
            
            # Sort predictions by confidence
            sorted_preds = sorted(enumerate(image_preds), 
                                key=lambda x: x[1]['score'], reverse=True)
            
            for pred_idx, pred in sorted_preds:
                best_iou = 0.0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(image_gts):
                    if gt_idx in matched_gt:
                        continue
                        
                    # Only match same class
                    if pred['category_id'] != gt['category_id']:
                        continue
                        
                    iou = self._calculate_bbox_iou(pred['bbox'], gt['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                
                # Match if IoU > 0.5
                if best_iou >= 0.5 and best_gt_idx >= 0:
                    matched_pred.add(pred_idx)
                    matched_gt.add(best_gt_idx)
            
            # Count matches for this image
            total_tp += len(matched_pred)
            total_fp += len(image_preds) - len(matched_pred)
            total_fn += len(image_gts) - len(matched_gt)
        
        return total_tp, total_fp, total_fn
    
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
    
    def _generate_environmental_summary(self, results: Dict) -> Dict:
        """Generate summary of environmental performance patterns."""
        summary = {
            'best_conditions': {},
            'worst_conditions': {},
            'condition_impact': {},
            'robustness_analysis': {}
        }
        
        # Find best and worst conditions for each metric
        for condition_type in ['weather_performance', 'lighting_performance', 'scene_performance']:
            if condition_type in results and results[condition_type]:
                condition_results = results[condition_type]
                
                # Sort by F1 score
                sorted_conditions = sorted(condition_results.items(),
                                         key=lambda x: x[1]['f1_score'], reverse=True)
                
                if sorted_conditions:
                    best_condition = sorted_conditions[0]
                    worst_condition = sorted_conditions[-1]
                    
                    summary['best_conditions'][condition_type] = {
                        'condition': best_condition[0],
                        'f1_score': best_condition[1]['f1_score']
                    }
                    
                    summary['worst_conditions'][condition_type] = {
                        'condition': worst_condition[0],
                        'f1_score': worst_condition[1]['f1_score']
                    }
                    
                    # Calculate performance variance
                    f1_scores = [metrics['f1_score'] for metrics in condition_results.values()]
                    summary['condition_impact'][condition_type] = {
                        'mean_f1': np.mean(f1_scores),
                        'std_f1': np.std(f1_scores),
                        'variance_ratio': np.std(f1_scores) / np.mean(f1_scores) if np.mean(f1_scores) > 0 else 0
                    }
        
        # Overall robustness assessment
        variance_ratios = [impact['variance_ratio'] for impact in summary['condition_impact'].values()]
        if variance_ratios:
            avg_variance = np.mean(variance_ratios)
            if avg_variance < 0.15:
                robustness = "ROBUST - Consistent across conditions"
            elif avg_variance < 0.30:
                robustness = "MODERATE - Some condition sensitivity"  
            else:
                robustness = "SENSITIVE - High variance across conditions"
                
            summary['robustness_analysis'] = {
                'average_variance_ratio': avg_variance,
                'assessment': robustness
            }
        
        return summary
    
    def _generate_characteristic_summary(self, results: Dict) -> Dict:
        """Generate summary of object characteristic performance patterns."""
        summary = {
            'size_trends': {},
            'position_bias': {},
            'recommendations': []
        }
        
        # Analyze size performance trends
        if 'size_performance' in results and results['size_performance']:
            size_results = results['size_performance']
            size_order = ['tiny', 'small', 'medium', 'large']
            
            f1_by_size = {}
            for size in size_order:
                if size in size_results:
                    f1_by_size[size] = size_results[size]['f1_score']
            
            summary['size_trends'] = f1_by_size
            
            # Check if performance decreases with smaller objects
            if len(f1_by_size) >= 2:
                sizes_with_scores = [(size, score) for size, score in f1_by_size.items()]
                if sizes_with_scores[0][1] < sizes_with_scores[-1][1]:  # smaller objects perform worse
                    summary['recommendations'].append(
                        "Small object detection needs improvement - consider multi-scale training"
                    )
        
        # Analyze position bias
        if 'position_performance' in results and results['position_performance']:
            position_results = results['position_performance']
            position_f1 = {pos: metrics['f1_score'] for pos, metrics in position_results.items()}
            
            summary['position_bias'] = position_f1
            
            # Check for strong position bias
            if position_f1:
                min_f1 = min(position_f1.values())
                max_f1 = max(position_f1.values())
                bias_ratio = (max_f1 - min_f1) / max_f1 if max_f1 > 0 else 0
                
                if bias_ratio > 0.30:
                    summary['recommendations'].append(
                        "Strong position bias detected - model may be learning shortcuts"
                    )
        
        return summary
    
    def generate_contextual_report(self, 
                                 environmental_results: Dict,
                                 characteristic_results: Dict) -> str:
        """Generate comprehensive contextual performance report."""
        report = []
        report.append("=" * 80)
        report.append("BDD100K MODEL - CONTEXTUAL PERFORMANCE ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        # Environmental Performance
        report.append("🌤️  ENVIRONMENTAL PERFORMANCE")
        report.append("-" * 50)
        
        env_summary = environmental_results.get('environmental_summary', {})
        
        # Weather performance
        if 'weather_performance' in environmental_results:
            report.append("Weather Conditions:")
            weather_results = environmental_results['weather_performance']
            for weather, metrics in sorted(weather_results.items(), 
                                         key=lambda x: x[1]['f1_score'], reverse=True):
                report.append(f"  {weather:12} F1: {metrics['f1_score']:.3f} "
                            f"(Objects: {metrics['num_objects']:,})")
        
        report.append("")
        
        # Lighting performance  
        if 'lighting_performance' in environmental_results:
            report.append("Lighting Conditions:")
            lighting_results = environmental_results['lighting_performance']
            for lighting, metrics in sorted(lighting_results.items(),
                                          key=lambda x: x[1]['f1_score'], reverse=True):
                report.append(f"  {lighting:12} F1: {metrics['f1_score']:.3f} "
                            f"(Objects: {metrics['num_objects']:,})")
        
        report.append("")
        
        # Robustness assessment
        if 'robustness_analysis' in env_summary:
            robustness = env_summary['robustness_analysis']
            report.append("Environmental Robustness:")
            report.append(f"  Assessment: {robustness['assessment']}")
            report.append(f"  Variance Ratio: {robustness['average_variance_ratio']:.3f}")
        
        report.append("")
        
        # Object Characteristics Performance
        report.append("📏 OBJECT CHARACTERISTICS PERFORMANCE")
        report.append("-" * 50)
        
        char_summary = characteristic_results.get('characteristic_summary', {})
        
        # Size performance
        if 'size_performance' in characteristic_results:
            report.append("Object Size Performance:")
            size_results = characteristic_results['size_performance']
            for size, metrics in size_results.items():
                report.append(f"  {size:8} F1: {metrics['f1_score']:.3f} "
                            f"(Avg Area: {metrics['avg_area']:.0f} px²)")
        
        report.append("")
        
        # Position performance
        if 'position_performance' in characteristic_results:
            report.append("Position Performance (Top 5):")
            position_results = characteristic_results['position_performance']
            sorted_positions = sorted(position_results.items(),
                                    key=lambda x: x[1]['f1_score'], reverse=True)[:5]
            
            for position, metrics in sorted_positions:
                report.append(f"  {position:13} F1: {metrics['f1_score']:.3f} "
                            f"(Objects: {metrics['num_objects']:,})")
        
        report.append("")
        
        # Recommendations
        recommendations = char_summary.get('recommendations', [])
        if recommendations:
            report.append("🛠️  IMPROVEMENT RECOMMENDATIONS")
            report.append("-" * 50)
            for i, rec in enumerate(recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        return "\n".join(report)