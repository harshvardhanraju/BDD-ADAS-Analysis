#!/usr/bin/env python3
"""
DETR Training Script for BDD100K Dataset

This script trains a DETR model on BDD100K dataset with proper configuration
and class imbalance handling based on our dataset analysis.
"""

import argparse
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.detr_trainer import train_bdd_detr


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train DETR on BDD100K dataset')
    
    # Data paths
    parser.add_argument(
        '--train-annotations', 
        type=str, 
        default='data/analysis/processed/train_annotations.csv',
        help='Path to training annotations CSV'
    )
    parser.add_argument(
        '--val-annotations',
        type=str,
        default='data/analysis/processed/val_annotations.csv', 
        help='Path to validation annotations CSV'
    )
    parser.add_argument(
        '--images-root',
        type=str,
        default='data/raw/bdd100k/bdd100k/images/100k',
        help='Root directory containing images'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        default=2,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int, 
        default=8,
        help='Batch size for training'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints',
        help='Directory to save checkpoints'
    )
    
    # Other options
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable Weights & Biases logging'
    )
    parser.add_argument(
        '--resume-from',
        type=str,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Check if data files exist
    train_path = Path(args.train_annotations)
    val_path = Path(args.val_annotations)
    images_path = Path(args.images_root)
    
    if not train_path.exists():
        print(f"❌ Training annotations not found: {train_path}")
        print("Please run the data analysis pipeline first.")
        return
    
    if not val_path.exists():
        print(f"❌ Validation annotations not found: {val_path}")
        print("Please run the data analysis pipeline first.")
        return
        
    if not images_path.exists():
        print(f"❌ Images directory not found: {images_path}")
        print("Please ensure BDD100K images are extracted to the correct location.")
        return
    
    print("🚀 Starting DETR training on BDD100K dataset...")
    print(f"📁 Training annotations: {train_path}")
    print(f"📁 Validation annotations: {val_path}")
    print(f"📁 Images root: {images_path}")
    print(f"🔧 Epochs: {args.epochs}")
    print(f"🔧 Batch size: {args.batch_size}")
    print(f"💾 Checkpoint directory: {args.checkpoint_dir}")
    print("-" * 60)
    
    try:
        # Start training
        history = train_bdd_detr(
            train_annotations=str(train_path),
            val_annotations=str(val_path),
            images_root=str(images_path),
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            use_wandb=not args.no_wandb,
            resume_from=args.resume_from
        )
        
        print("🎉 Training completed successfully!")
        print(f"📈 Final train loss: {history['train_loss'][-1]:.4f}")
        print(f"📈 Final val loss: {history['val_loss'][-1]:.4f}")
        print(f"💾 Checkpoints saved to: {args.checkpoint_dir}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()