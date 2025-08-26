#!/usr/bin/env python3
"""
Complete BDD100K 10-Class DETR Training Pipeline

This script demonstrates the complete training pipeline for all 10 BDD100K 
object detection classes with proper data loading, model configuration, and training.
"""

import os
import sys
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from pathlib import Path
import argparse
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.detr_dataset import BDD100KDETRDataset, create_bdd_dataloaders
from src.models.detr_model import BDD100KDETR, BDD100KDetrConfig
from src.training.enhanced_detr_trainer import EnhancedDETRTrainer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_complete_10class_model() -> BDD100KDETR:
    """Create DETR model configured for complete 10-class BDD100K."""
    config = BDD100KDetrConfig()
    
    logger.info(f"Creating DETR model for {config.num_classes} classes:")
    for i, class_name in enumerate(config.class_names):
        weight = config.class_weights[i]
        logger.info(f"  {i}: {class_name} (weight: {weight:.1f})")
    
    model = BDD100KDETR(config)
    return model


def create_demo_training_setup(data_dir: str, images_root: str, batch_size: int = 4):
    """Create training setup for demonstration."""
    
    train_annotations = os.path.join(data_dir, "train_annotations_10class.csv")
    val_annotations = os.path.join(data_dir, "val_annotations_10class.csv")
    
    logger.info("Creating datasets...")
    
    # Create datasets with smaller subset for demo
    train_dataset = BDD100KDETRDataset(
        annotations_file=train_annotations,
        images_root=images_root,
        split='train',
        image_size=(416, 416),  # Smaller for faster training
        augment=True,
        use_enhanced_augmentation=True,
        augmentation_strength='light'  # Light augmentation for demo
    )
    
    val_dataset = BDD100KDETRDataset(
        annotations_file=val_annotations,
        images_root=images_root,
        split='val',
        image_size=(416, 416),
        augment=False
    )
    
    # Create subset for demo (use first 1000 images)
    demo_size = min(1000, len(train_dataset))
    val_demo_size = min(200, len(val_dataset))
    
    train_subset = torch.utils.data.Subset(train_dataset, range(demo_size))
    val_subset = torch.utils.data.Subset(val_dataset, range(val_demo_size))
    
    logger.info(f"Demo training set: {len(train_subset)} images")
    logger.info(f"Demo validation set: {len(val_subset)} images")
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    return train_dataloader, val_dataloader, train_dataset.num_classes


def run_demo_training(
    model: BDD100KDETR,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    num_classes: int,
    max_epochs: int = 1,
    device: str = 'cuda'
):
    """Run demonstration training for 1 epoch."""
    
    logger.info("Setting up training...")
    
    model = model.to(device)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Setup scheduler
    num_training_steps = len(train_dataloader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, num_training_steps // 10),
        num_training_steps=num_training_steps
    )
    
    # Setup trainer
    trainer = EnhancedDETRTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_classes=num_classes,
        checkpoint_dir="checkpoints/complete_10class_demo"
    )
    
    logger.info(f"Starting demo training for {max_epochs} epoch(s)...")
    logger.info(f"Training batches: {len(train_dataloader)}")
    logger.info(f"Validation batches: {len(val_dataloader)}")
    
    try:
        # Train for specified epochs
        for epoch in range(max_epochs):
            logger.info(f"\n{'='*60}")
            logger.info(f"EPOCH {epoch + 1}/{max_epochs}")
            logger.info(f"{'='*60}")
            
            # Training
            train_metrics = trainer.train_epoch(train_dataloader, epoch)
            logger.info(f"Training metrics: {train_metrics}")
            
            # Validation
            val_metrics = trainer.validate_epoch(val_dataloader, epoch)
            logger.info(f"Validation metrics: {val_metrics}")
            
            # Save checkpoint
            checkpoint_path = trainer.save_checkpoint(epoch, train_metrics, val_metrics)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    logger.info("Demo training completed successfully!")
    return trainer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Complete BDD100K 10-class DETR training')
    parser.add_argument('--data-dir', type=str, 
                       default='data/analysis/processed_10class_corrected',
                       help='Directory with processed 10-class data')
    parser.add_argument('--images-root', type=str,
                       default='data/raw/bdd100k_labels_release/bdd100k/images/100k',
                       help='Root directory for images')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of epochs for demo training')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        args.device = 'cpu'
        args.batch_size = 2  # Reduce batch size for CPU
    
    logger.info("="*70)
    logger.info("BDD100K COMPLETE 10-CLASS DETR TRAINING PIPELINE")
    logger.info("="*70)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Images root: {args.images_root}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Demo epochs: {args.epochs}")
    
    try:
        # Create model
        logger.info("\n1. Creating complete 10-class DETR model...")
        model = create_complete_10class_model()
        
        # Setup data
        logger.info("\n2. Setting up training data...")
        train_dataloader, val_dataloader, num_classes = create_demo_training_setup(
            args.data_dir, args.images_root, args.batch_size
        )
        
        # Verify class count
        assert num_classes == 10, f"Expected 10 classes, got {num_classes}"
        logger.info("✅ All 10 classes confirmed in dataset")
        
        # Run training
        logger.info("\n3. Running demonstration training...")
        trainer = run_demo_training(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            num_classes=num_classes,
            max_epochs=args.epochs,
            device=args.device
        )
        
        logger.info("\n" + "="*70)
        logger.info("🎉 COMPLETE 10-CLASS DETR TRAINING PIPELINE SUCCESS!")
        logger.info("="*70)
        logger.info("✅ All 10 BDD100K classes successfully integrated")
        logger.info("✅ Enhanced augmentation pipeline active")
        logger.info("✅ Safety-critical class weighting applied")
        logger.info("✅ Complete training pipeline demonstrated")
        logger.info("="*70)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()