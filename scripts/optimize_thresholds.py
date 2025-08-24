#!/usr/bin/env python3
"""
Run Confidence Threshold Optimization

This script optimizes confidence thresholds for the trained DETR model
based on precision/recall trade-offs to find optimal deployment settings.
"""

import sys
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.detr_model import create_bdd_detr_model
from src.data.detr_dataset import BDD100KDETRDataset
from src.evaluation.threshold_optimization import ConfidenceThresholdOptimizer


def main():
    """Main threshold optimization function."""
    parser = argparse.ArgumentParser(description='Confidence Threshold Optimization')
    parser.add_argument('--checkpoint', type=str, 
                        default='improved_training_results/best_model.pth',
                        help='Path to trained model checkpoint')
    parser.add_argument('--max-images', type=int, default=500,
                        help='Maximum images to evaluate (default: 500)')
    parser.add_argument('--metric', type=str, default='f1_score',
                        choices=['f1_score', 'precision', 'recall'],
                        help='Metric to optimize (default: f1_score)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    print("🎯 Confidence Threshold Optimization")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Max images: {args.max_images}")
    print(f"Optimization metric: {args.metric}")
    print(f"Device: {args.device}")
    
    # Check paths
    checkpoint_path = Path(args.checkpoint)
    val_ann = "data/analysis/processed/val_annotations.csv"
    images_root = "data/raw/bdd100k/bdd100k/images/100k"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Available checkpoints:")
        for checkpoint_dir in [Path("checkpoints"), Path("improved_training_results"), Path("enhanced_training_results")]:
            if checkpoint_dir.exists():
                for ckpt in checkpoint_dir.glob("*.pth"):
                    print(f"  - {ckpt}")
        return
    
    if not all(Path(p).exists() for p in [val_ann, images_root]):
        print("❌ Required validation files not found:")
        print(f"   - Validation annotations: {val_ann}")
        print(f"   - Images root: {images_root}")
        return
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Load validation dataset
    print("📊 Loading validation dataset...")
    try:
        val_dataset = BDD100KDETRDataset(
            annotations_file=val_ann,
            images_root=images_root,
            split='val',
            use_enhanced_augmentation=False,
            augment=False
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
    print("🤖 Loading trained model...")
    try:
        model = create_bdd_detr_model(pretrained=False)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint.get('epoch', 'unknown')
            val_loss = checkpoint.get('val_loss', 'unknown')
            print(f"✅ Model loaded from epoch {epoch} (val_loss: {val_loss})")
        else:
            # Direct state dict
            model.load_state_dict(checkpoint)
            print("✅ Model loaded from checkpoint")
        
        model.to(args.device)
        model.eval()
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize threshold optimizer
    print("🔧 Initializing threshold optimizer...")
    optimizer = ConfidenceThresholdOptimizer(
        model=model,
        val_dataloader=val_dataloader,
        device=args.device,
        output_dir='threshold_optimization_results'
    )
    
    # Run complete optimization
    print("\n🚀 Starting threshold optimization...")
    print("This will:")
    print("  • Collect predictions from validation images")
    print("  • Test confidence thresholds from 0.05 to 0.95")
    print("  • Find optimal thresholds for each class")
    print("  • Generate performance curves and reports")
    
    try:
        results = optimizer.run_complete_optimization(
            max_images=args.max_images,
            optimization_metric=args.metric
        )
        
        # Print detailed results
        print("\n" + "=" * 60)
        print("📊 THRESHOLD OPTIMIZATION RESULTS")
        print("=" * 60)
        
        optimal_thresholds = results['optimal_thresholds']
        
        if 'overall' in optimal_thresholds:
            overall = optimal_thresholds['overall']
            print(f"🎯 Overall Optimal Threshold: {overall['threshold']:.3f}")
            print(f"🏆 Best {args.metric}: {overall['score']:.3f}")
        
        print(f"\n📋 Class-Specific Optimal Thresholds:")
        for class_name in model.config.class_names:
            if class_name in optimal_thresholds:
                opt = optimal_thresholds[class_name]
                print(f"   • {class_name:15}: {opt['threshold']:.3f} (score: {opt['score']:.3f})")
        
        print(f"\n🚀 Deployment Recommendations:")
        
        if 'overall' in optimal_thresholds:
            overall_threshold = optimal_thresholds['overall']['threshold']
            
            if overall_threshold < 0.3:
                print("   • Low threshold optimized - prioritizes recall over precision")
                print("   • Good for: Safety applications, object detection systems")
                print("   • Caution: May have more false positives")
            elif overall_threshold > 0.7:
                print("   • High threshold optimized - prioritizes precision over recall")
                print("   • Good for: Production systems, autonomous driving")
                print("   • Caution: May miss some objects (lower recall)")
            else:
                print("   • Balanced threshold - good precision/recall trade-off")
                print("   • Good for: General object detection applications")
        
        # Safety-critical recommendations
        safety_classes = ['train', 'rider', 'bus']
        safety_thresholds = {}
        for class_name in safety_classes:
            if class_name in optimal_thresholds:
                safety_thresholds[class_name] = optimal_thresholds[class_name]['threshold']
        
        if safety_thresholds:
            print(f"\n⚠️  Safety-Critical Class Recommendations:")
            for class_name, threshold in safety_thresholds.items():
                recommended_threshold = max(0.1, threshold - 0.1)  # Lower for safety
                print(f"   • {class_name}: Use {recommended_threshold:.2f} (vs optimal {threshold:.2f}) for safety")
        
        print(f"\n📁 Complete results saved to: threshold_optimization_results/")
        print(f"📊 View optimization curves: threshold_optimization_results/threshold_optimization_curves.png")
        print(f"📄 Read detailed report: threshold_optimization_results/threshold_optimization_report.md")
        
    except Exception as e:
        print(f"❌ Error during optimization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n🎉 Threshold optimization completed successfully!")


if __name__ == "__main__":
    main()