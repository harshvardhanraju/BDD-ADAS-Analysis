"""
Safety-Critical Metrics for Autonomous Driving Object Detection

This module implements specialized metrics focused on safety-critical classes
(pedestrians, riders, bicycles, motorcycles) for autonomous driving applications.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class SafetyCriticalMetrics:
    """
    Specialized metrics evaluator for safety-critical object detection.
    
    Focuses on vulnerable road users (VRU) and provides metrics specifically
    designed for autonomous driving safety assessment.
    """
    
    def __init__(self):
        """Initialize safety-critical metrics evaluator."""
        self.class_names = [
            'pedestrian', 'rider', 'car', 'truck', 'bus',
            'train', 'motorcycle', 'bicycle', 'traffic_light', 'traffic_sign'
        ]
        
        # Define safety-critical classes (Vulnerable Road Users)
        self.safety_critical_classes = {
            'pedestrian': {'index': 0, 'weight': 3.0, 'priority': 'CRITICAL'},
            'rider': {'index': 1, 'weight': 2.5, 'priority': 'CRITICAL'},
            'bicycle': {'index': 7, 'weight': 2.0, 'priority': 'HIGH'},
            'motorcycle': {'index': 6, 'weight': 2.5, 'priority': 'CRITICAL'}
        }
        
        # Risk tolerance thresholds for safety classes
        self.safety_thresholds = {
            'acceptable_fnr': 0.10,  # Max 10% false negative rate
            'acceptable_fpr': 0.20,  # Max 20% false positive rate (more tolerant)
            'min_precision': 0.70,   # Minimum precision for safety classes
            'min_recall': 0.80,      # Minimum recall for safety classes (prioritize safety)
            'min_ap': 0.35          # Minimum average precision
        }
        
    def evaluate_safety_performance(self, 
                                  predictions: List[Dict], 
                                  ground_truth: List[Dict],
                                  confidence_threshold: float = 0.5) -> Dict:
        """
        Comprehensive safety-critical performance evaluation.
        
        Args:
            predictions: Model predictions
            ground_truth: Ground truth annotations
            confidence_threshold: Minimum confidence for positive predictions
            
        Returns:
            Dictionary containing safety-specific metrics and analysis
        """
        results = {}
        
        # Filter predictions by confidence threshold
        filtered_predictions = [p for p in predictions if p['score'] >= confidence_threshold]
        
        # Calculate per-class safety metrics
        per_class_safety = {}
        for class_name, class_info in self.safety_critical_classes.items():
            class_idx = class_info['index']
            
            # Extract class-specific predictions and ground truth
            class_preds = [p for p in filtered_predictions if p['category_id'] == class_idx]
            class_gt = [gt for gt in ground_truth if gt['category_id'] == class_idx]
            
            # Calculate safety metrics for this class
            class_safety_metrics = self._calculate_class_safety_metrics(
                class_preds, class_gt, class_name, class_info
            )
            per_class_safety[class_name] = class_safety_metrics
            
        results['per_class_safety'] = per_class_safety
        
        # Calculate overall safety score
        results['overall_safety_score'] = self._calculate_overall_safety_score(per_class_safety)
        
        # Safety risk assessment
        results['safety_risk_assessment'] = self._assess_safety_risks(per_class_safety)
        
        # Generate safety compliance report
        results['safety_compliance'] = self._evaluate_safety_compliance(per_class_safety)
        
        return results
    
    def _calculate_class_safety_metrics(self, 
                                       predictions: List[Dict], 
                                       ground_truth: List[Dict],
                                       class_name: str,
                                       class_info: Dict) -> Dict:
        """Calculate comprehensive safety metrics for a single class."""
        metrics = {}
        
        # Basic counts
        num_gt = len(ground_truth)
        num_pred = len(predictions)
        
        metrics['num_ground_truth'] = num_gt
        metrics['num_predictions'] = num_pred
        
        if num_gt == 0:
            # No ground truth objects for this class
            metrics.update({
                'precision': 0.0 if num_pred == 0 else 0.0,  # All predictions are false positives
                'recall': 0.0,  # Undefined, but set to 0
                'f1_score': 0.0,
                'false_negative_rate': 0.0,
                'false_positive_rate': 0.0,
                'true_positives': 0,
                'false_positives': num_pred,
                'false_negatives': 0
            })
            return metrics
            
        # Calculate IoU-based matches (using IoU threshold of 0.5)
        tp, fp, fn = self._calculate_tp_fp_fn(predictions, ground_truth, iou_threshold=0.5)
        
        metrics['true_positives'] = tp
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        
        # Calculate core metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1_score'] = f1_score
        
        # Safety-specific metrics
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0  # False Negative Rate
        fpr = fp / (fp + tp) if (fp + tp) > 0 else 0.0   # False Positive Rate (of predictions)
        
        metrics['false_negative_rate'] = fnr
        metrics['false_positive_rate'] = fpr
        
        # Risk assessment for this class
        metrics['safety_risk_level'] = self._assess_class_risk_level(
            fnr, fpr, precision, recall, class_info['priority']
        )
        
        # Compliance with safety thresholds
        metrics['meets_safety_standards'] = self._check_class_safety_compliance(
            fnr, fpr, precision, recall
        )
        
        return metrics
    
    def _calculate_tp_fp_fn(self, 
                           predictions: List[Dict], 
                           ground_truth: List[Dict],
                           iou_threshold: float = 0.5) -> Tuple[int, int, int]:
        """
        Calculate True Positives, False Positives, and False Negatives using IoU matching.
        
        Returns:
            Tuple of (true_positives, false_positives, false_negatives)
        """
        if len(ground_truth) == 0:
            return 0, len(predictions), 0
            
        if len(predictions) == 0:
            return 0, 0, len(ground_truth)
        
        # Calculate IoU matrix between all predictions and ground truth
        iou_matrix = np.zeros((len(predictions), len(ground_truth)))
        
        for i, pred in enumerate(predictions):
            for j, gt in enumerate(ground_truth):
                # Only calculate IoU if same image
                if pred['image_id'] == gt['image_id']:
                    iou_matrix[i, j] = self._calculate_bbox_iou(pred['bbox'], gt['bbox'])
        
        # Find best matches using greedy assignment
        matched_gt = set()
        matched_pred = set()
        
        # Sort predictions by confidence (highest first) for better matching
        pred_indices = sorted(range(len(predictions)), 
                             key=lambda i: predictions[i]['score'], reverse=True)
        
        for pred_idx in pred_indices:
            # Find best GT match for this prediction
            best_gt_idx = -1
            best_iou = 0.0
            
            for gt_idx in range(len(ground_truth)):
                if gt_idx not in matched_gt and iou_matrix[pred_idx, gt_idx] > best_iou:
                    best_iou = iou_matrix[pred_idx, gt_idx]
                    best_gt_idx = gt_idx
            
            # If IoU above threshold, it's a match
            if best_iou >= iou_threshold and best_gt_idx != -1:
                matched_pred.add(pred_idx)
                matched_gt.add(best_gt_idx)
        
        tp = len(matched_pred)
        fp = len(predictions) - tp
        fn = len(ground_truth) - len(matched_gt)
        
        return tp, fp, fn
    
    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
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
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def _calculate_overall_safety_score(self, per_class_safety: Dict) -> Dict:
        """Calculate weighted overall safety score."""
        total_weighted_score = 0.0
        total_weights = 0.0
        
        class_scores = {}
        
        for class_name, class_info in self.safety_critical_classes.items():
            if class_name in per_class_safety:
                class_metrics = per_class_safety[class_name]
                weight = class_info['weight']
                
                # Composite safety score for this class
                # Heavily weight recall (missing objects is worse than false alarms)
                recall_weight = 0.6
                precision_weight = 0.4
                
                class_score = (
                    recall_weight * class_metrics['recall'] + 
                    precision_weight * class_metrics['precision']
                )
                
                class_scores[class_name] = class_score
                total_weighted_score += class_score * weight
                total_weights += weight
        
        overall_score = total_weighted_score / total_weights if total_weights > 0 else 0.0
        
        return {
            'overall_safety_score': overall_score,
            'class_scores': class_scores,
            'interpretation': self._interpret_safety_score(overall_score)
        }
    
    def _assess_safety_risks(self, per_class_safety: Dict) -> Dict:
        """Assess safety risks based on performance metrics."""
        risk_assessment = {
            'high_risk_classes': [],
            'medium_risk_classes': [],
            'acceptable_classes': [],
            'critical_issues': []
        }
        
        for class_name, metrics in per_class_safety.items():
            if class_name not in self.safety_critical_classes:
                continue
                
            fnr = metrics['false_negative_rate']
            precision = metrics['precision']
            recall = metrics['recall']
            
            # Assess risk level
            if fnr > 0.15 or recall < 0.70:  # High risk thresholds
                risk_assessment['high_risk_classes'].append({
                    'class': class_name,
                    'fnr': fnr,
                    'recall': recall,
                    'issue': 'High miss rate - safety critical'
                })
            elif fnr > 0.10 or recall < 0.80:  # Medium risk thresholds
                risk_assessment['medium_risk_classes'].append({
                    'class': class_name,
                    'fnr': fnr,
                    'recall': recall,
                    'issue': 'Moderate miss rate - needs improvement'
                })
            else:
                risk_assessment['acceptable_classes'].append(class_name)
            
            # Check for critical issues
            if fnr > 0.25:
                risk_assessment['critical_issues'].append(
                    f"{class_name}: Extremely high false negative rate ({fnr:.2%}) - UNSAFE"
                )
            
            if precision < 0.30:
                risk_assessment['critical_issues'].append(
                    f"{class_name}: Very low precision ({precision:.2%}) - excessive false alarms"
                )
        
        return risk_assessment
    
    def _evaluate_safety_compliance(self, per_class_safety: Dict) -> Dict:
        """Evaluate compliance with safety standards."""
        compliance = {
            'overall_compliant': True,
            'class_compliance': {},
            'violations': [],
            'recommendations': []
        }
        
        for class_name, metrics in per_class_safety.items():
            if class_name not in self.safety_critical_classes:
                continue
                
            class_compliant = True
            violations = []
            
            # Check each safety threshold
            if metrics['false_negative_rate'] > self.safety_thresholds['acceptable_fnr']:
                violations.append(f"FNR ({metrics['false_negative_rate']:.2%}) > {self.safety_thresholds['acceptable_fnr']:.1%}")
                class_compliant = False
                
            if metrics['precision'] < self.safety_thresholds['min_precision']:
                violations.append(f"Precision ({metrics['precision']:.2%}) < {self.safety_thresholds['min_precision']:.1%}")
                class_compliant = False
                
            if metrics['recall'] < self.safety_thresholds['min_recall']:
                violations.append(f"Recall ({metrics['recall']:.2%}) < {self.safety_thresholds['min_recall']:.1%}")
                class_compliant = False
            
            compliance['class_compliance'][class_name] = {
                'compliant': class_compliant,
                'violations': violations
            }
            
            if not class_compliant:
                compliance['overall_compliant'] = False
                compliance['violations'].extend([f"{class_name}: {v}" for v in violations])
        
        # Generate recommendations
        if not compliance['overall_compliant']:
            compliance['recommendations'] = self._generate_safety_recommendations(per_class_safety)
        
        return compliance
    
    def _assess_class_risk_level(self, fnr: float, fpr: float, precision: float, recall: float, priority: str) -> str:
        """Assess risk level for a single class."""
        if fnr > 0.20 or recall < 0.70:
            return "HIGH_RISK"
        elif fnr > 0.10 or recall < 0.80:
            return "MEDIUM_RISK"  
        elif precision < 0.60:
            return "MEDIUM_RISK"  # Too many false alarms
        else:
            return "ACCEPTABLE"
    
    def _check_class_safety_compliance(self, fnr: float, fpr: float, precision: float, recall: float) -> bool:
        """Check if class meets safety compliance standards."""
        return (fnr <= self.safety_thresholds['acceptable_fnr'] and
                precision >= self.safety_thresholds['min_precision'] and
                recall >= self.safety_thresholds['min_recall'])
    
    def _interpret_safety_score(self, score: float) -> str:
        """Interpret overall safety score."""
        if score >= 0.80:
            return "EXCELLENT - Safe for production deployment"
        elif score >= 0.70:
            return "GOOD - Acceptable with monitoring"
        elif score >= 0.60:
            return "MARGINAL - Needs improvement before deployment"
        else:
            return "POOR - Not safe for deployment"
    
    def _generate_safety_recommendations(self, per_class_safety: Dict) -> List[str]:
        """Generate specific recommendations for improving safety performance."""
        recommendations = []
        
        for class_name, metrics in per_class_safety.items():
            if class_name not in self.safety_critical_classes:
                continue
                
            if metrics['false_negative_rate'] > 0.15:
                recommendations.append(
                    f"URGENT: Improve {class_name} recall - consider increasing class weights, "
                    f"adding hard negative mining, or collecting more training data"
                )
            
            if metrics['recall'] < 0.70:
                recommendations.append(
                    f"Improve {class_name} detection sensitivity - lower confidence thresholds "
                    f"or enhance training data with difficult examples"
                )
            
            if metrics['precision'] < 0.50:
                recommendations.append(
                    f"Reduce {class_name} false alarms - improve data quality or "
                    f"add negative examples to training data"
                )
        
        return recommendations
    
    def generate_safety_report(self, results: Dict) -> str:
        """Generate comprehensive safety analysis report."""
        report = []
        report.append("=" * 80)
        report.append("BDD100K MODEL - SAFETY-CRITICAL PERFORMANCE ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        # Overall Safety Assessment
        overall_safety = results['overall_safety_score']
        report.append("🚨 OVERALL SAFETY ASSESSMENT")
        report.append("-" * 50)
        report.append(f"Safety Score: {overall_safety['overall_safety_score']:.3f}")
        report.append(f"Status: {overall_safety['interpretation']}")
        report.append("")
        
        # Per-Class Safety Performance
        report.append("👥 VULNERABLE ROAD USER DETECTION PERFORMANCE")
        report.append("-" * 50)
        
        per_class_safety = results['per_class_safety']
        for class_name in ['pedestrian', 'rider', 'bicycle', 'motorcycle']:
            if class_name in per_class_safety:
                metrics = per_class_safety[class_name]
                risk_level = metrics.get('safety_risk_level', 'UNKNOWN')
                
                risk_emoji = {"HIGH_RISK": "🚨", "MEDIUM_RISK": "⚠️", "ACCEPTABLE": "✅"}
                emoji = risk_emoji.get(risk_level, "❓")
                
                report.append(f"{class_name.upper():12} {emoji}")
                report.append(f"  Recall:     {metrics['recall']:.3f} (Miss Rate: {metrics['false_negative_rate']:.1%})")
                report.append(f"  Precision:  {metrics['precision']:.3f}")
                report.append(f"  F1-Score:   {metrics['f1_score']:.3f}")
                report.append(f"  Risk Level: {risk_level}")
                report.append("")
        
        # Safety Risk Assessment
        risk_assessment = results['safety_risk_assessment']
        if risk_assessment['high_risk_classes'] or risk_assessment['critical_issues']:
            report.append("⚠️  SAFETY RISK ANALYSIS")
            report.append("-" * 50)
            
            if risk_assessment['critical_issues']:
                report.append("CRITICAL ISSUES:")
                for issue in risk_assessment['critical_issues']:
                    report.append(f"  ❌ {issue}")
                report.append("")
            
            if risk_assessment['high_risk_classes']:
                report.append("HIGH RISK CLASSES:")
                for risk_class in risk_assessment['high_risk_classes']:
                    report.append(f"  🚨 {risk_class['class']}: {risk_class['issue']}")
                report.append("")
        
        # Compliance Assessment
        compliance = results['safety_compliance']
        report.append("📋 SAFETY COMPLIANCE ASSESSMENT")
        report.append("-" * 50)
        
        if compliance['overall_compliant']:
            report.append("✅ COMPLIANT: Model meets safety standards")
        else:
            report.append("❌ NON-COMPLIANT: Model fails safety requirements")
            report.append("")
            report.append("Violations:")
            for violation in compliance['violations']:
                report.append(f"  • {violation}")
        
        report.append("")
        
        # Recommendations
        if compliance['recommendations']:
            report.append("🛠️  IMPROVEMENT RECOMMENDATIONS")
            report.append("-" * 50)
            for i, rec in enumerate(compliance['recommendations'], 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        return "\n".join(report)