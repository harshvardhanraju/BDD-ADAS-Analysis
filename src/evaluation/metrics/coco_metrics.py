"""
COCO-Style Evaluation Metrics for BDD100K Object Detection

This module implements standard COCO evaluation metrics (mAP) adapted for BDD100K dataset.
Provides comprehensive detection performance evaluation across all 10 object classes.
"""

import json
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import tempfile
import os
from collections import defaultdict


class COCOEvaluator:
    """
    COCO-style evaluator for BDD100K object detection models.
    
    Provides standard mAP metrics with BDD100K class mapping and safety-critical
    class highlighting for autonomous driving applications.
    """
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize COCO evaluator.
        
        Args:
            class_names: List of class names. If None, uses BDD100K 10-class names.
        """
        if class_names is None:
            self.class_names = [
                'pedestrian', 'rider', 'car', 'truck', 'bus',
                'train', 'motorcycle', 'bicycle', 'traffic_light', 'traffic_sign'
            ]
        else:
            self.class_names = class_names
            
        self.num_classes = len(self.class_names)
        
        # Safety-critical class indices for autonomous driving
        self.safety_critical_classes = {
            'pedestrian': 0,
            'rider': 1, 
            'bicycle': 7,
            'motorcycle': 6
        }
        
        # Object size thresholds (in pixels²) - aligned with COCO
        self.size_thresholds = {
            'small': (0, 32**2),      # < 1024 px² 
            'medium': (32**2, 96**2), # 1024 - 9216 px²
            'large': (96**2, float('inf'))  # > 9216 px²
        }
        
    def evaluate(self, 
                 predictions: List[Dict], 
                 ground_truth: List[Dict],
                 iou_thresholds: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Evaluate model predictions using COCO metrics.
        
        Args:
            predictions: List of prediction dicts with keys:
                - 'image_id': Image identifier
                - 'category_id': Class prediction (0-based)
                - 'bbox': [x, y, width, height] in pixels
                - 'score': Confidence score [0-1]
            ground_truth: List of ground truth dicts with keys:
                - 'image_id': Image identifier  
                - 'category_id': True class (0-based)
                - 'bbox': [x, y, width, height] in pixels
                - 'area': Bounding box area in pixels²
                - 'iscrowd': Crowd annotation flag
                - 'id': Annotation ID
            iou_thresholds: IoU thresholds for evaluation. 
                          If None, uses COCO standard [0.5:0.05:0.95]
                          
        Returns:
            Dict containing evaluation metrics:
                - 'mAP': Overall mAP@0.5:0.95
                - 'mAP@0.5': mAP at IoU=0.5
                - 'mAP@0.75': mAP at IoU=0.75  
                - 'mAP_small': mAP for small objects
                - 'mAP_medium': mAP for medium objects
                - 'mAP_large': mAP for large objects
                - 'per_class_AP': Per-class Average Precision
                - 'safety_critical_mAP': Weighted mAP for safety classes
        """
        # Convert to COCO format and create temporary files
        gt_coco_format = self._convert_gt_to_coco(ground_truth)
        pred_coco_format = self._convert_pred_to_coco(predictions)
        
        # Create temporary files for COCO evaluation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as gt_file:
            json.dump(gt_coco_format, gt_file, indent=2)
            gt_file_path = gt_file.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as pred_file:
            json.dump(pred_coco_format, pred_file, indent=2)  
            pred_file_path = pred_file.name
            
        try:
            # Initialize COCO objects
            coco_gt = COCO(gt_file_path)
            coco_dt = coco_gt.loadRes(pred_file_path)
            
            # Create evaluator
            coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
            
            # Set IoU thresholds
            if iou_thresholds is not None:
                coco_eval.params.iouThrs = np.array(iou_thresholds)
            
            # Run evaluation
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            
            # Extract standard metrics
            results = {
                'mAP': coco_eval.stats[0],          # mAP@0.5:0.95
                'mAP@0.5': coco_eval.stats[1],      # mAP@0.5
                'mAP@0.75': coco_eval.stats[2],     # mAP@0.75
                'mAP_small': coco_eval.stats[3],    # mAP for small objects
                'mAP_medium': coco_eval.stats[4],   # mAP for medium objects  
                'mAP_large': coco_eval.stats[5],    # mAP for large objects
                'mAR': coco_eval.stats[6],          # mAR@0.5:0.95
                'mAR_small': coco_eval.stats[9],    # mAR for small objects
                'mAR_medium': coco_eval.stats[10],  # mAR for medium objects
                'mAR_large': coco_eval.stats[11]    # mAR for large objects
            }
            
            # Calculate per-class Average Precision
            per_class_ap = self._calculate_per_class_ap(coco_eval)
            results['per_class_AP'] = per_class_ap
            
            # Calculate safety-critical weighted mAP
            safety_critical_map = self._calculate_safety_critical_map(per_class_ap)
            results['safety_critical_mAP'] = safety_critical_map
            
            return results
            
        finally:
            # Clean up temporary files
            os.unlink(gt_file_path)
            os.unlink(pred_file_path)
            
    def _convert_gt_to_coco(self, ground_truth: List[Dict]) -> Dict:
        """Convert ground truth annotations to COCO format."""
        # Create COCO dataset structure
        dataset = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories (classes)
        for idx, class_name in enumerate(self.class_names):
            dataset['categories'].append({
                'id': idx + 1,  # COCO uses 1-based class IDs
                'name': class_name,
                'supercategory': 'object'
            })
        
        # Track unique images
        image_ids = set()
        
        # Convert annotations
        for ann in ground_truth:
            image_id = ann['image_id']
            
            # Add image info if not already present
            if image_id not in image_ids:
                dataset['images'].append({
                    'id': image_id,
                    'width': 1280,  # BDD100K standard width
                    'height': 720,  # BDD100K standard height
                    'file_name': f"{image_id}.jpg"
                })
                image_ids.add(image_id)
            
            # Add annotation
            dataset['annotations'].append({
                'id': ann.get('id', len(dataset['annotations'])),
                'image_id': image_id,
                'category_id': ann['category_id'] + 1,  # Convert to 1-based
                'bbox': ann['bbox'],  # [x, y, width, height]
                'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                'iscrowd': ann.get('iscrowd', 0)
            })
            
        return dataset
    
    def _convert_pred_to_coco(self, predictions: List[Dict]) -> List[Dict]:
        """Convert predictions to COCO format."""
        coco_predictions = []
        
        for pred in predictions:
            coco_predictions.append({
                'image_id': pred['image_id'],
                'category_id': pred['category_id'] + 1,  # Convert to 1-based
                'bbox': pred['bbox'],  # [x, y, width, height]
                'score': pred['score']
            })
            
        return coco_predictions
    
    def _calculate_per_class_ap(self, coco_eval: COCOeval) -> Dict[str, float]:
        """Calculate per-class Average Precision."""
        per_class_ap = {}
        
        # Extract per-class AP from COCO evaluation
        # coco_eval.eval['precision'] shape: [T, R, K, A, M]
        # T: IoU thresholds, R: recall thresholds, K: categories, A: area ranges, M: max detections
        
        if coco_eval.eval is not None:
            precision = coco_eval.eval['precision']
            
            # Average over IoU thresholds and recall thresholds
            # Use all area ranges (A=0 means all areas)
            for idx, class_name in enumerate(self.class_names):
                # Get precision for this class (K=idx, A=0 for all areas, M=-1 for max detections)
                class_precision = precision[:, :, idx, 0, -1]
                
                # Calculate AP as mean precision where recall > 0
                valid_precisions = class_precision[class_precision > -1]
                if len(valid_precisions) > 0:
                    per_class_ap[class_name] = np.mean(valid_precisions)
                else:
                    per_class_ap[class_name] = 0.0
        else:
            # Fallback if evaluation failed
            per_class_ap = {class_name: 0.0 for class_name in self.class_names}
            
        return per_class_ap
    
    def _calculate_safety_critical_map(self, per_class_ap: Dict[str, float]) -> float:
        """
        Calculate weighted mAP for safety-critical classes.
        
        Safety-critical classes get higher weights based on their importance
        for autonomous driving safety.
        """
        safety_weights = {
            'pedestrian': 3.0,    # Highest priority - vulnerable road user
            'rider': 2.5,         # High priority - vulnerable road user  
            'bicycle': 2.0,       # High priority - vulnerable road user
            'motorcycle': 2.5     # High priority - vulnerable road user
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for class_name, weight in safety_weights.items():
            if class_name in per_class_ap:
                weighted_sum += per_class_ap[class_name] * weight
                total_weight += weight
                
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.0
    
    def evaluate_by_size(self, 
                        predictions: List[Dict], 
                        ground_truth: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance by object size categories.
        
        Returns:
            Dict with size categories as keys and metrics as values.
        """
        results = {}
        
        for size_category, (min_area, max_area) in self.size_thresholds.items():
            # Filter ground truth and predictions by size
            filtered_gt = []
            filtered_pred = []
            
            # Create mapping of valid image_ids for this size category
            valid_image_ids = set()
            
            for gt in ground_truth:
                area = gt.get('area', gt['bbox'][2] * gt['bbox'][3])
                if min_area <= area < max_area:
                    filtered_gt.append(gt)
                    valid_image_ids.add(gt['image_id'])
            
            # Only include predictions for images that have GT objects of this size
            for pred in predictions:
                if pred['image_id'] in valid_image_ids:
                    filtered_pred.append(pred)
            
            # Evaluate if we have sufficient data
            if len(filtered_gt) > 0:
                size_results = self.evaluate(filtered_pred, filtered_gt)
                results[size_category] = size_results
            else:
                results[size_category] = {'mAP': 0.0, 'mAP@0.5': 0.0, 'mAP@0.75': 0.0}
                
        return results
    
    def get_summary_report(self, results: Dict[str, float]) -> str:
        """
        Generate human-readable summary report.
        
        Args:
            results: Results dictionary from evaluate()
            
        Returns:
            Formatted string report suitable for both technical and business audiences.
        """
        report = []
        report.append("=" * 80)
        report.append("BDD100K OBJECT DETECTION MODEL - PERFORMANCE SUMMARY")
        report.append("=" * 80)
        report.append("")
        
        # Overall Performance
        report.append("📊 OVERALL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Overall mAP (0.5:0.95):     {results['mAP']:.3f}")
        report.append(f"mAP @ IoU=0.5:             {results['mAP@0.5']:.3f}")
        report.append(f"mAP @ IoU=0.75:            {results['mAP@0.75']:.3f}")
        report.append("")
        
        # Object Size Performance  
        report.append("📏 PERFORMANCE BY OBJECT SIZE")
        report.append("-" * 40)
        report.append(f"Small objects (traffic signs): {results['mAP_small']:.3f}")
        report.append(f"Medium objects (pedestrians):   {results['mAP_medium']:.3f}")
        report.append(f"Large objects (vehicles):       {results['mAP_large']:.3f}")
        report.append("")
        
        # Safety-Critical Performance
        report.append("🚨 SAFETY-CRITICAL PERFORMANCE")
        report.append("-" * 40)
        report.append(f"Safety-Critical mAP:           {results['safety_critical_mAP']:.3f}")
        report.append("")
        
        # Per-Class Performance
        report.append("🎯 PER-CLASS PERFORMANCE")
        report.append("-" * 40)
        per_class_ap = results['per_class_AP']
        
        # Sort classes by performance for better readability
        sorted_classes = sorted(per_class_ap.items(), key=lambda x: x[1], reverse=True)
        
        for class_name, ap in sorted_classes:
            # Highlight safety-critical classes
            if class_name in self.safety_critical_classes:
                report.append(f"{class_name:15} {ap:.3f} 🚨 (Safety Critical)")
            else:
                report.append(f"{class_name:15} {ap:.3f}")
        
        report.append("")
        
        # Business Interpretation
        report.append("💼 BUSINESS INTERPRETATION")
        report.append("-" * 40)
        
        # Deployment readiness assessment
        if results['mAP'] >= 0.45:
            deployment_status = "✅ READY for production deployment"
        elif results['mAP'] >= 0.35:
            deployment_status = "⚠️  NEEDS IMPROVEMENT before deployment"
        else:
            deployment_status = "❌ NOT READY for deployment"
            
        report.append(f"Deployment Status: {deployment_status}")
        
        # Safety assessment
        if results['safety_critical_mAP'] >= 0.35:
            safety_status = "✅ ACCEPTABLE safety performance"
        else:
            safety_status = "❌ INSUFFICIENT safety performance - HIGH RISK"
            
        report.append(f"Safety Assessment: {safety_status}")
        report.append("")
        
        return "\n".join(report)