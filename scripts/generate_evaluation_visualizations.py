#!/usr/bin/env python3
"""
Generate Comprehensive Evaluation Visualizations

This script creates visualizations from evaluation results including:
- Detection comparison samples
- Performance charts and metrics
- Confidence distributions
- Safety-critical analysis visualizations
"""

import argparse
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.evaluation.visualization import DetectionVisualizer


def load_evaluation_results(results_path: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def create_performance_visualizations(results: Dict, output_dir: Path) -> None:
    """Create performance visualization charts."""
    print("Creating performance visualizations...")
    
    visualizer = DetectionVisualizer()
    
    # 1. Overall Performance Chart
    if 'coco_metrics' in results and 'per_class_AP' in results['coco_metrics']:
        per_class_ap = results['coco_metrics']['per_class_AP']
        
        fig = visualizer.create_class_performance_chart(
            per_class_ap, 
            metric_name='Average Precision (mAP)',
            save_path=str(output_dir / "overall_performance_by_class.png")
        )
        plt.close(fig)
        print(f"✅ Overall performance chart saved")
    
    # 2. Safety-Critical Performance Chart
    if 'safety_metrics' in results and 'per_class_safety' in results['safety_metrics']:
        safety_metrics = results['safety_metrics']['per_class_safety']
        
        # Extract recall scores for safety-critical classes
        safety_recalls = {}
        safety_precisions = {}
        
        for class_name in ['pedestrian', 'rider', 'bicycle', 'motorcycle']:
            if class_name in safety_metrics:
                safety_recalls[class_name] = safety_metrics[class_name].get('recall', 0.0)
                safety_precisions[class_name] = safety_metrics[class_name].get('precision', 0.0)
        
        if safety_recalls:
            fig = visualizer.create_class_performance_chart(
                safety_recalls,
                metric_name='Recall (Safety-Critical Classes)',
                save_path=str(output_dir / "safety_critical_recall.png")
            )
            plt.close(fig)
            print(f"✅ Safety-critical recall chart saved")
        
        if safety_precisions:
            fig = visualizer.create_class_performance_chart(
                safety_precisions,
                metric_name='Precision (Safety-Critical Classes)',
                save_path=str(output_dir / "safety_critical_precision.png")
            )
            plt.close(fig)
            print(f"✅ Safety-critical precision chart saved")


def create_summary_dashboard(results: Dict, output_dir: Path) -> None:
    """Create a comprehensive summary dashboard."""
    print("Creating summary dashboard...")
    
    # Create a multi-panel summary figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('BDD100K Model Evaluation Dashboard', fontsize=20, fontweight='bold')
    
    # Panel 1: Overall Metrics
    ax1 = axes[0, 0]
    if 'coco_metrics' in results:
        metrics = results['coco_metrics']
        metric_names = ['mAP', 'mAP@0.5', 'mAP@0.75', 'mAP_small', 'mAP_medium', 'mAP_large']
        metric_values = [metrics.get(m, 0.0) for m in metric_names]
        
        bars = ax1.bar(range(len(metric_names)), metric_values, 
                      color=['skyblue', 'lightgreen', 'orange', 'pink', 'yellow', 'lightcoral'])
        ax1.set_title('Overall Detection Metrics', fontweight='bold')
        ax1.set_xticks(range(len(metric_names)))
        ax1.set_xticklabels(metric_names, rotation=45, ha='right')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, metric_values)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Panel 2: Safety Performance
    ax2 = axes[0, 1]
    if 'safety_metrics' in results and 'per_class_safety' in results['safety_metrics']:
        safety_classes = ['pedestrian', 'rider', 'bicycle', 'motorcycle']
        safety_scores = []
        safety_metrics = results['safety_metrics']['per_class_safety']
        
        for class_name in safety_classes:
            if class_name in safety_metrics:
                recall = safety_metrics[class_name].get('recall', 0.0)
                safety_scores.append(recall)
            else:
                safety_scores.append(0.0)
        
        bars = ax2.bar(safety_classes, safety_scores, color='red', alpha=0.7)
        ax2.set_title('Safety-Critical Class Recall', fontweight='bold')
        ax2.set_ylabel('Recall Score')
        ax2.set_xticklabels(safety_classes, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add threshold line
        ax2.axhline(y=0.8, color='green', linestyle='--', label='Target (0.8)')
        ax2.legend()
        
        # Add value labels
        for bar, value in zip(bars, safety_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Panel 3: Class Distribution (if available)
    ax3 = axes[0, 2]
    # This would show class distribution from the dataset
    ax3.text(0.5, 0.5, 'Class Distribution\n(From Training Data)', 
             ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    ax3.set_title('Dataset Class Distribution', fontweight='bold')
    
    # Panel 4: Performance by Size
    ax4 = axes[1, 0]
    if 'coco_metrics' in results:
        metrics = results['coco_metrics']
        size_categories = ['Small', 'Medium', 'Large']
        size_values = [metrics.get('mAP_small', 0.0), 
                      metrics.get('mAP_medium', 0.0), 
                      metrics.get('mAP_large', 0.0)]
        
        bars = ax4.bar(size_categories, size_values, 
                      color=['lightblue', 'lightgreen', 'lightsalmon'])
        ax4.set_title('Performance by Object Size', fontweight='bold')
        ax4.set_ylabel('mAP Score')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, size_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Panel 5: Model Status Summary
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    # Determine deployment status
    if 'coco_metrics' in results:
        overall_map = results['coco_metrics'].get('mAP', 0.0)
        safety_map = results['coco_metrics'].get('safety_critical_mAP', 0.0)
        
        if overall_map >= 0.45:
            deployment_status = "✅ READY"
            status_color = 'green'
        elif overall_map >= 0.35:
            deployment_status = "⚠️ NEEDS IMPROVEMENT"
            status_color = 'orange'
        else:
            deployment_status = "❌ NOT READY"
            status_color = 'red'
        
        if safety_map >= 0.35:
            safety_status = "✅ ACCEPTABLE"
            safety_color = 'green'
        else:
            safety_status = "❌ INSUFFICIENT"
            safety_color = 'red'
        
        # Create status text
        status_text = f"""
DEPLOYMENT STATUS
{deployment_status}

Overall mAP: {overall_map:.3f}
Safety mAP: {safety_map:.3f}

SAFETY STATUS
{safety_status}
        """.strip()
        
        ax5.text(0.1, 0.8, status_text, transform=ax5.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor='lightgray', alpha=0.8))
    
    # Panel 6: Key Recommendations
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    recommendations = []
    
    # Generate automatic recommendations based on results
    if 'coco_metrics' in results:
        metrics = results['coco_metrics']
        
        if metrics.get('mAP_small', 0.0) < 0.25:
            recommendations.append("• Improve small object detection")
        
        if metrics.get('safety_critical_mAP', 0.0) < 0.35:
            recommendations.append("• Focus on safety-critical classes")
        
        if metrics.get('mAP', 0.0) < 0.45:
            recommendations.append("• Overall performance needs improvement")
    
    if not recommendations:
        recommendations = ["• Continue monitoring performance", "• Consider additional validation"]
    
    rec_text = "KEY RECOMMENDATIONS:\\n" + "\\n".join(recommendations[:5])  # Limit to 5
    ax6.text(0.1, 0.9, rec_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
            facecolor='lightyellow', alpha=0.8))
    
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig(output_dir / "evaluation_dashboard.png", dpi=100, bbox_inches=None)
    plt.close(fig)
    print(f"✅ Summary dashboard saved")


def generate_report_summary(results: Dict, output_dir: Path) -> None:
    """Generate a text summary of key findings."""
    print("Generating report summary...")
    
    summary_lines = []
    summary_lines.append("# BDD100K Model Evaluation - Executive Summary")
    summary_lines.append("")
    
    # Overall Performance
    if 'coco_metrics' in results:
        metrics = results['coco_metrics']
        overall_map = metrics.get('mAP', 0.0)
        map_50 = metrics.get('mAP@0.5', 0.0)
        safety_map = metrics.get('safety_critical_mAP', 0.0)
        
        summary_lines.append("## Overall Performance")
        summary_lines.append(f"- **Overall mAP**: {overall_map:.3f}")
        summary_lines.append(f"- **mAP@0.5**: {map_50:.3f}")
        summary_lines.append(f"- **Safety-Critical mAP**: {safety_map:.3f}")
        summary_lines.append("")
        
        # Deployment Assessment
        if overall_map >= 0.45:
            deployment = "✅ **READY** for production deployment"
        elif overall_map >= 0.35:
            deployment = "⚠️ **NEEDS IMPROVEMENT** before deployment"
        else:
            deployment = "❌ **NOT READY** for deployment"
        
        summary_lines.append("## Deployment Readiness")
        summary_lines.append(f"- Status: {deployment}")
        summary_lines.append("")
    
    # Safety Assessment
    if 'safety_metrics' in results:
        safety_data = results['safety_metrics']
        if 'overall_safety_score' in safety_data:
            safety_score = safety_data['overall_safety_score'].get('overall_safety_score', 0.0)
            summary_lines.append("## Safety Assessment")
            summary_lines.append(f"- **Safety Score**: {safety_score:.3f}")
            
            if safety_score >= 0.80:
                safety_status = "✅ **EXCELLENT** - Safe for production"
            elif safety_score >= 0.70:
                safety_status = "✅ **GOOD** - Acceptable with monitoring"
            elif safety_score >= 0.60:
                safety_status = "⚠️ **MARGINAL** - Needs improvement"
            else:
                safety_status = "❌ **POOR** - Not safe for deployment"
            
            summary_lines.append(f"- Status: {safety_status}")
            summary_lines.append("")
    
    # Top Performing Classes
    if 'coco_metrics' in results and 'per_class_AP' in results['coco_metrics']:
        per_class_ap = results['coco_metrics']['per_class_AP']
        sorted_classes = sorted(per_class_ap.items(), key=lambda x: x[1], reverse=True)
        
        summary_lines.append("## Top Performing Classes")
        for class_name, ap in sorted_classes[:5]:
            summary_lines.append(f"- {class_name}: {ap:.3f}")
        summary_lines.append("")
        
        summary_lines.append("## Lowest Performing Classes") 
        for class_name, ap in sorted_classes[-3:]:
            summary_lines.append(f"- {class_name}: {ap:.3f}")
        summary_lines.append("")
    
    # Key Recommendations
    summary_lines.append("## Key Recommendations")
    
    if 'coco_metrics' in results:
        metrics = results['coco_metrics']
        
        if metrics.get('mAP_small', 0.0) < 0.25:
            summary_lines.append("- **HIGH PRIORITY**: Improve small object detection (traffic signs/lights)")
        
        if metrics.get('safety_critical_mAP', 0.0) < 0.35:
            summary_lines.append("- **CRITICAL**: Enhance safety-critical class detection (pedestrians, cyclists)")
        
        if metrics.get('mAP_large', 0.0) > metrics.get('mAP_small', 0.0) * 2:
            summary_lines.append("- Consider multi-scale training strategies")
        
        if metrics.get('mAP', 0.0) < 0.45:
            summary_lines.append("- Overall model performance requires improvement before production use")
    
    # Save summary
    summary_text = "\\n".join(summary_lines)
    with open(output_dir / "evaluation_summary.md", 'w') as f:
        f.write(summary_text)
    
    print(f"✅ Executive summary saved to evaluation_summary.md")


def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive evaluation visualizations')
    parser.add_argument('--results-path', type=str, 
                       default='evaluation_results/evaluation_results.json',
                       help='Path to evaluation results JSON file')
    parser.add_argument('--output-dir', type=str,
                       default='evaluation_results/visualizations',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Check if results file exists
    if not Path(args.results_path).exists():
        print(f"❌ Results file not found: {args.results_path}")
        print("Run comprehensive evaluation first:")
        print("  python3 scripts/run_comprehensive_evaluation.py --model-path <model_path>")
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("📊 Generating BDD100K Evaluation Visualizations")
    print("=" * 60)
    
    try:
        # Load results
        print("Loading evaluation results...")
        results = load_evaluation_results(args.results_path)
        
        # Create performance visualizations
        create_performance_visualizations(results, output_dir)
        
        # Create summary dashboard
        create_summary_dashboard(results, output_dir)
        
        # Generate report summary
        generate_report_summary(results, output_dir)
        
        # Create class legend
        visualizer = DetectionVisualizer()
        legend = visualizer.generate_legend()
        cv2.imwrite(str(output_dir / "class_legend.jpg"),
                   cv2.cvtColor(legend, cv2.COLOR_RGB2BGR))
        print(f"✅ Class legend saved")
        
        print("\\n" + "=" * 60)
        print(f"✅ All visualizations generated successfully!")
        print(f"📁 Check {output_dir} for all outputs:")
        print("  - evaluation_dashboard.png (Executive summary)")
        print("  - evaluation_summary.md (Text summary)")
        print("  - Various performance charts")
        print("  - class_legend.jpg (Color reference)")
        
    except Exception as e:
        print(f"\\n❌ Visualization generation failed: {e}")
        raise


if __name__ == "__main__":
    main()