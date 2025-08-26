#!/usr/bin/env python3
"""
Run Enhanced DETR Training with Improved Class Rebalancing

This script runs enhanced DETR training based on qualitative analysis findings,
implementing stronger class rebalancing, proper convergence monitoring, and
advanced training techniques.
"""

import sys
import os
import argparse
from pathlib import Path
import torch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.enhanced_detr_trainer import create_enhanced_trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Enhanced DETR Training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=6,
                        help='Training batch size (default: 6)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--use-wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Training device (default: cuda)')
    
    args = parser.parse_args()
    
    print("🚀 Enhanced DETR Training Script")
    print("=" * 60)
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print(f"Use W&B: {args.use_wandb}")
    
    # Check paths
    annotations_dir = "data/analysis/processed"
    images_root = "data/raw/bdd100k/bdd100k/images/100k"
    
    train_ann = f"{annotations_dir}/train_annotations.csv"
    val_ann = f"{annotations_dir}/val_annotations.csv"
    
    if not all(Path(p).exists() for p in [train_ann, val_ann, images_root]):
        print("❌ Required files not found:")
        print(f"   - Training annotations: {train_ann}")
        print(f"   - Validation annotations: {val_ann}")
        print(f"   - Images root: {images_root}")
        return
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    # Create enhanced trainer
    print("🔧 Creating enhanced DETR trainer...")
    try:
        trainer = create_enhanced_trainer(
            annotations_dir=annotations_dir,
            images_root=images_root,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=args.device
        )
        
        print("✅ Enhanced trainer created successfully")
        
    except Exception as e:
        print(f"❌ Error creating trainer: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Install wandb if requested but not available
    if args.use_wandb:
        try:
            import wandb
        except ImportError:
            print("📦 Installing wandb for experiment tracking...")
            os.system("pip install wandb")
            import wandb
    
    # Start training
    print("\n🎯 Starting enhanced training with improved class rebalancing...")
    print("Key improvements implemented:")
    print("  • Enhanced class weights to prevent class collapse")
    print("  • Focal loss with increased gamma (2.5) for hard examples")
    print("  • Differential learning rates (backbone: 1e-5, head: 1e-4)")
    print("  • Gradient clipping and warmup scheduling")
    print("  • Early stopping with patience=10")
    print("  • Background penalty for better foreground detection")
    
    try:
        training_history = trainer.train(
            num_epochs=args.epochs,
            use_wandb=args.use_wandb
        )
        
        print("\n🎉 Enhanced training completed successfully!")
        print("📊 Key improvements expected:")
        print("  • Reduced class collapse (all classes should be detected)")
        print("  • Improved precision through better false positive control") 
        print("  • Better recall for rare classes (train, rider, bus)")
        print("  • More stable convergence with proper monitoring")
        
        # Print final class performance prediction
        print("\n🔮 Expected performance improvements:")
        print("  • Car class: F1 0.049 → 0.15+ (3x improvement)")
        print("  • Truck class: F1 0.000 → 0.05+ (detection activated)")
        print("  • Bus class: F1 0.000 → 0.03+ (detection activated)")
        print("  • Traffic signs: F1 0.000 → 0.08+ (detection activated)")
        print("  • Traffic lights: F1 0.000 → 0.06+ (detection activated)")
        print("  • Overall precision: 3.0% → 15%+ (5x improvement)")
        print("  • Overall recall: 13.1% → 30%+ (2.3x improvement)")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "=" * 60)
    print("✨ Enhanced training complete! Next steps:")
    print("1. Run qualitative analysis on new model")
    print("2. Compare with baseline results")
    print("3. Fine-tune confidence thresholds")
    print("4. Implement additional data augmentation if needed")


if __name__ == "__main__":
    main()