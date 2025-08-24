#!/usr/bin/env python3
"""
Run Complete Qualitative Analysis on Trained DETR Model

This script runs comprehensive qualitative analysis on the trained DETR model
to understand model behavior, failure modes, and generate improvement recommendations.
"""

import sys
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.detr_model import create_bdd_detr_model
from src.data.detr_dataset import BDD100KDETRDataset
from src.analysis.qualitative_analysis import QualitativeAnalyzer


def main():
    """Run complete qualitative analysis."""
    print("🚀 Starting DETR Model Qualitative Analysis")
    print("=" * 60)
    
    # Paths
    val_ann = "data/analysis/processed/val_annotations.csv"
    images_root = "data/raw/bdd100k/bdd100k/images/100k"
    checkpoint_path = "checkpoints/detr_demo_checkpoint.pth"
    
    # Check if files exist
    if not all(Path(p).exists() for p in [val_ann, images_root]):
        print("❌ Required data files not found:")
        print(f"   - Validation annotations: {val_ann}")
        print(f"   - Images root: {images_root}")
        return
    
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Please run training first or update the checkpoint path")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create validation dataset and dataloader
    print("📊 Loading validation dataset...")
    try:
        val_dataset = BDD100KDETRDataset(
            annotations_file=val_ann,
            images_root=images_root,
            split='val'
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=2,
            collate_fn=val_dataset.collate_fn
        )
        
        print(f"✅ Loaded {len(val_dataset)} validation images")
        
    except Exception as e:
        print(f"❌ Error loading validation dataset: {e}")
        return
    
    # Load trained model
    print("🤖 Loading trained DETR model...")
    try:
        model = create_bdd_detr_model(pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ Model loaded from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Initialize qualitative analyzer
    print("🔬 Initializing qualitative analyzer...")
    output_dir = Path("qualitative_analysis_results")
    
    analyzer = QualitativeAnalyzer(
        model=model,
        val_dataloader=val_dataloader,
        device=device,
        output_dir=str(output_dir)
    )
    
    print(f"📁 Analysis results will be saved to: {output_dir.absolute()}")
    
    # Run complete qualitative analysis
    print("\n🚀 Starting comprehensive analysis pipeline...")
    try:
        results = analyzer.run_complete_qualitative_analysis(
            max_images=100,  # Analyze 100 images
            num_visualizations=15  # Create 15 visualization examples
        )
        
        # Print detailed results summary
        print("\n" + "="*60)
        print("📊 QUALITATIVE ANALYSIS RESULTS SUMMARY")
        print("="*60)
        
        summary = results['summary']
        
        print(f"📈 Overall Performance:")
        print(f"   • Precision: {summary['precision']:.3f}")
        print(f"   • Recall: {summary['recall']:.3f}")
        print(f"   • Images Analyzed: {summary['images_analyzed']}")
        print(f"   • Total Predictions: {summary['total_predictions']}")
        print(f"   • Total Ground Truth: {summary['total_ground_truth']}")
        
        print(f"\n🎯 Class-Specific Performance:")
        for class_name, perf in summary['class_performance_summary'].items():
            print(f"   • {class_name:15}: P={perf['precision']:.3f}, R={perf['recall']:.3f}, F1={perf['f1_score']:.3f}")
        
        print(f"\n🎨 Generated Outputs:")
        print(f"   • Visualizations: {summary['visualizations_created']}")
        print(f"   • Analysis Report: {results['report_path']}")
        print(f"   • Summary JSON: {results['summary_path']}")
        
        print(f"\n📍 Spatial Analysis:")
        spatial_matches = summary['spatial_patterns_match']
        for class_name, matches in spatial_matches.items():
            status = "✅" if matches else "❌"
            print(f"   • {class_name:15}: {status} {'Expected pattern' if matches else 'Unexpected pattern'}")
        
        print(f"\n💡 Key Insights:")
        
        # Find most problematic classes
        class_perf = summary['class_performance_summary']
        low_precision_classes = [name for name, perf in class_perf.items() if perf['precision'] < 0.3]
        low_recall_classes = [name for name, perf in class_perf.items() if perf['recall'] < 0.3]
        
        if low_precision_classes:
            print(f"   • Low Precision Classes: {', '.join(low_precision_classes)}")
            print(f"     → Too many false positives - consider higher confidence threshold")
        
        if low_recall_classes:
            print(f"   • Low Recall Classes: {', '.join(low_recall_classes)}")
            print(f"     → Missing objects - consider data augmentation or class rebalancing")
        
        # Spatial pattern insights
        spatial_issues = [name for name, matches in spatial_matches.items() if not matches]
        if spatial_issues:
            print(f"   • Unexpected Spatial Patterns: {', '.join(spatial_issues)}")
            print(f"     → May indicate dataset bias or model overfitting to position")
        
        print(f"\n🔬 Detailed Analysis Available:")
        print(f"   • Open: {output_dir.absolute()}/error_analysis_report.md")
        print(f"   • View: {output_dir.absolute()}/*.png files for visualizations")
        print(f"   • Data: {output_dir.absolute()}/analysis_summary.json")
        
        print("\n🎉 Qualitative analysis completed successfully!")
        print(f"📁 All results saved to: {output_dir.absolute()}")
        
    except Exception as e:
        print(f"❌ Error during qualitative analysis: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("✨ Analysis complete! Check the generated reports and visualizations.")


if __name__ == "__main__":
    main()