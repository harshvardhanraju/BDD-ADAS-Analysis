"""
Production BDD100K 10-Class DETR Training Pipeline

Complete production-ready training script with proper data balancing, augmentations,
pretrained model loading, and comprehensive monitoring.
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from pathlib import Path
import argparse
import logging
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.detr_dataset import BDD100KDETRDataset
from src.models.detr_model import BDD100KDETR, BDD100KDetrConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProductionDETRTrainer:
    """Production-ready DETR trainer with comprehensive features."""
    
    def __init__(
        self,
        model: BDD100KDETR,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: BDD100KDetrConfig,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints/production_10class',
        use_wandb: bool = False
    ):
        """Initialize production trainer."""
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer
        self.optimizer = AdamW(
            [
                {'params': [p for n, p in model.named_parameters() if 'backbone' not in n], 'lr': 1e-4},
                {'params': [p for n, p in model.named_parameters() if 'backbone' in n], 'lr': 1e-5}
            ],
            weight_decay=1e-4
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=1e-6)
        
        # Setup loss function with class weights
        self.criterion = nn.CrossEntropyLoss(weight=config.class_weights.to(device))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Initialize wandb if requested
        if self.use_wandb:
            import wandb
            wandb.init(
                project="bdd100k-10class-detr",
                name=f"production_training_{time.strftime('%Y%m%d_%H%M%S')}",
                config={
                    "num_classes": config.num_classes,
                    "batch_size": train_dataloader.batch_size,
                    "learning_rate": 1e-4,
                    "weight_decay": 1e-4,
                    "class_weights": config.class_weights.tolist(),
                    "classes": config.class_names
                }
            )
        
        logger.info("Production DETR trainer initialized")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_class_loss = 0.0
        total_bbox_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(
            self.train_dataloader, 
            desc=f"Epoch {epoch+1} Training",
            leave=False
        )
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            # Move to device
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, targets)
            
            # Calculate losses
            losses = outputs['losses'] if 'losses' in outputs else outputs
            loss = losses['loss'] if 'loss' in losses else sum(losses.values())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            if 'loss_ce' in losses:
                total_class_loss += losses['loss_ce'].item()
            if 'loss_bbox' in losses:
                total_bbox_loss += losses['loss_bbox'].item()
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 50 == 0:
                import wandb
                wandb.log({
                    'batch_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
        
        # Calculate epoch metrics
        epoch_metrics = {
            'train_loss': total_loss / num_batches,
            'train_class_loss': total_class_loss / num_batches,
            'train_bbox_loss': total_bbox_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_class_loss = 0.0
        total_bbox_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_dataloader,
                desc=f"Epoch {epoch+1} Validation",
                leave=False
            )
            
            for batch_idx, (images, targets) in enumerate(progress_bar):
                # Move to device
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # Forward pass
                outputs = self.model(images, targets)
                
                # Calculate losses
                losses = outputs['losses'] if 'losses' in outputs else outputs
                loss = losses['loss'] if 'loss' in losses else sum(losses.values())
                
                # Update metrics
                total_loss += loss.item()
                if 'loss_ce' in losses:
                    total_class_loss += losses['loss_ce'].item()
                if 'loss_bbox' in losses:
                    total_bbox_loss += losses['loss_bbox'].item()
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.set_postfix({'Val Loss': f'{avg_loss:.4f}'})
        
        # Calculate epoch metrics
        epoch_metrics = {
            'val_loss': total_loss / num_batches,
            'val_class_loss': total_class_loss / num_batches,
            'val_bbox_loss': total_bbox_loss / num_batches
        }
        
        return epoch_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.__dict__,
            'metrics': metrics,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved with val_loss: {metrics['val_loss']:.4f}")
        
        # Keep only last 5 checkpoints
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if len(checkpoints) > 5:
            for old_checkpoint in checkpoints[:-5]:
                old_checkpoint.unlink()
    
    def train(self, num_epochs: int, save_every: int = 5):
        """Complete training loop."""
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training batches: {len(self.train_dataloader)}")
        logger.info(f"Validation batches: {len(self.val_dataloader)}")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            
            # Update tracking
            self.train_losses.append(train_metrics['train_loss'])
            self.val_losses.append(val_metrics['val_loss'])
            self.learning_rates.append(train_metrics['learning_rate'])
            
            # Check if best model
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, epoch_metrics, is_best)
            
            # Log epoch results
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed:")
            logger.info(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            logger.info(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            logger.info(f"  Learning Rate: {train_metrics['learning_rate']:.2e}")
            logger.info(f"  Best Val Loss: {self.best_val_loss:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                import wandb
                wandb.log(epoch_metrics)
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Save final training plots
        self.save_training_plots()
    
    def save_training_plots(self):
        """Save training progress plots."""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        epochs = range(1, len(self.train_losses) + 1)
        axes[0, 0].plot(epochs, self.train_losses, label='Train Loss')
        axes[0, 0].plot(epochs, self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(epochs, self.learning_rates)
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Loss smoothing
        if len(self.train_losses) > 10:
            window = min(10, len(self.train_losses) // 4)
            smooth_train = np.convolve(self.train_losses, np.ones(window)/window, mode='valid')
            smooth_val = np.convolve(self.val_losses, np.ones(window)/window, mode='valid')
            smooth_epochs = range(window, len(self.train_losses) + 1)
            
            axes[1, 0].plot(smooth_epochs, smooth_train, label='Smoothed Train Loss')
            axes[1, 0].plot(smooth_epochs, smooth_val, label='Smoothed Val Loss')
            axes[1, 0].set_title('Smoothed Loss Curves')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Training summary
        axes[1, 1].axis('off')
        summary_text = f"""Training Summary:
        
Total Epochs: {len(self.train_losses)}
Best Val Loss: {self.best_val_loss:.4f}
Final Train Loss: {self.train_losses[-1]:.4f}
Final Val Loss: {self.val_losses[-1]:.4f}

Model Configuration:
Classes: {self.config.num_classes}
Backbone: {self.config.backbone}
Queries: {self.config.num_queries}
        """
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        verticalalignment='top', fontsize=10, fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Training plots saved to {self.checkpoint_dir / 'training_progress.png'}")


def create_balanced_sampler(dataset: BDD100KDETRDataset) -> WeightedRandomSampler:
    """Create weighted sampler for balanced training."""
    logger.info("Creating balanced sampler for training data...")
    
    # Get class distribution
    class_counts = {}
    image_classes = {}
    
    for i, (image, target) in enumerate(dataset):
        image_name = dataset.image_annotations[i]['image_name']
        classes_in_image = target['class_labels'].tolist()
        
        # Track classes per image
        image_classes[i] = classes_in_image
        
        # Count classes
        for class_id in classes_in_image:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
    
    # Calculate weights (inverse frequency)
    total_samples = sum(class_counts.values())
    class_weights = {}
    
    for class_id in range(dataset.num_classes):
        count = class_counts.get(class_id, 1)  # Avoid division by zero
        class_weights[class_id] = total_samples / (dataset.num_classes * count)
    
    # Calculate sample weights
    sample_weights = []
    for i in range(len(dataset)):
        classes_in_image = image_classes[i]
        if classes_in_image:
            # Use maximum weight for images with multiple classes
            max_weight = max(class_weights[class_id] for class_id in classes_in_image)
            sample_weights.append(max_weight)
        else:
            sample_weights.append(1.0)
    
    # Log class weights
    logger.info("Class weights for balanced sampling:")
    for class_id, weight in class_weights.items():
        class_name = dataset.id_to_class[class_id]
        count = class_counts.get(class_id, 0)
        logger.info(f"  {class_name}: {weight:.3f} (count: {count})")
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler


def create_production_datasets(
    data_dir: str, 
    images_root: str,
    image_size: Tuple[int, int] = (640, 640),
    use_enhanced_augmentation: bool = True,
    train_subset_size: int = 20000
) -> Tuple[BDD100KDETRDataset, BDD100KDETRDataset]:
    """Create production training and validation datasets."""
    
    train_annotations = os.path.join(data_dir, "train_annotations_10class.csv")
    val_annotations = os.path.join(data_dir, "val_annotations_10class.csv")
    
    logger.info("Creating production datasets...")
    logger.info(f"Image size: {image_size}")
    logger.info(f"Enhanced augmentation: {use_enhanced_augmentation}")
    logger.info(f"Training subset size: {train_subset_size:,} images")
    
    # Create subset of training data for memory constraints
    if train_subset_size and train_subset_size > 0:
        logger.info("Creating training subset to manage memory constraints...")
        import pandas as pd
        import numpy as np
        
        # Load full training annotations
        train_df = pd.read_csv(train_annotations)
        unique_images = train_df['image_name'].unique()
        
        logger.info(f"Total available training images: {len(unique_images):,}")
        
        # Randomly sample subset of images
        if len(unique_images) > train_subset_size:
            np.random.seed(42)  # For reproducibility
            selected_images = np.random.choice(unique_images, size=train_subset_size, replace=False)
            
            # Filter annotations for selected images
            subset_df = train_df[train_df['image_name'].isin(selected_images)].copy()
            
            # Save subset annotations
            subset_annotations_path = train_annotations.replace('.csv', f'_subset_{train_subset_size}.csv')
            subset_df.to_csv(subset_annotations_path, index=False)
            
            logger.info(f"Created training subset: {len(selected_images):,} images ({len(subset_df):,} annotations)")
            logger.info(f"Subset saved to: {subset_annotations_path}")
            
            # Update train_annotations path to use subset
            train_annotations = subset_annotations_path
        else:
            logger.info(f"Using all available images (requested subset size >= total images)")
    
    # Training dataset with augmentation
    train_dataset = BDD100KDETRDataset(
        annotations_file=train_annotations,
        images_root=images_root,
        split='train',
        image_size=image_size,
        augment=True,
        use_enhanced_augmentation=use_enhanced_augmentation,
        augmentation_strength='medium'
    )
    
    # Validation dataset without augmentation
    val_dataset = BDD100KDETRDataset(
        annotations_file=val_annotations,
        images_root=images_root,
        split='val',
        image_size=image_size,
        augment=False
    )
    
    logger.info(f"Training dataset: {len(train_dataset)} images")
    logger.info(f"Validation dataset: {len(val_dataset)} images")
    logger.info(f"Number of classes: {train_dataset.num_classes}")
    
    return train_dataset, val_dataset


def create_production_model(load_pretrained: bool = True, model_variant: str = "detr-resnet-50") -> BDD100KDETR:
    """Create production DETR model with optional pretrained weights and variant selection."""
    config = BDD100KDetrConfig(model_variant=model_variant)
    
    logger.info("Creating production DETR model...")
    logger.info(f"Classes: {config.num_classes}")
    logger.info(f"Backbone: {config.backbone}")
    logger.info(f"Queries: {config.num_queries}")
    logger.info(f"Load pretrained: {load_pretrained}")
    
    # Log class weights
    logger.info("Class weights:")
    for i, (class_name, weight) in enumerate(zip(config.class_names, config.class_weights)):
        logger.info(f"  {i}: {class_name} -> {weight:.1f}")
    
    model = BDD100KDETR(config)
    
    if load_pretrained:
        logger.info("✅ Model initialized with pretrained DETR-ResNet50 backbone")
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Production BDD100K 10-class DETR training')
    parser.add_argument('--data-dir', type=str, 
                       default='data/analysis/processed_10class_corrected',
                       help='Directory with processed 10-class data')
    parser.add_argument('--images-root', type=str,
                       default='data/raw/bdd100k/bdd100k/images/100k',
                       help='Root directory for images')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs to train')
    parser.add_argument('--image-size', type=int, nargs=2, default=[640, 640],
                       help='Image size (height width)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for training')
    parser.add_argument('--model-variant', type=str, default='detr-resnet-50',
                       choices=['detr-resnet-50', 'detr-resnet-101', 'conditional-detr', 'deformable-detr'],
                       help='DETR model variant to use')
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='checkpoints/production_10class',
                       help='Directory to save checkpoints')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--use-balanced-sampling', action='store_true',
                       help='Use balanced sampling for class imbalance')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    parser.add_argument('--train-subset-size', type=int, default=20000,
                       help='Number of training images to use (0 for all images)')
    
    args = parser.parse_args()
    
    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, switching to CPU")
        args.device = 'cpu'
        args.batch_size = max(2, args.batch_size // 4)  # Reduce batch size for CPU
        args.num_workers = min(2, args.num_workers)
    
    logger.info("=" * 80)
    logger.info("BDD100K PRODUCTION 10-CLASS DETR TRAINING")
    logger.info("=" * 80)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Images root: {args.images_root}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Model variant: {args.model_variant}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Image size: {args.image_size}")
    logger.info(f"Checkpoint dir: {args.checkpoint_dir}")
    logger.info(f"Use balanced sampling: {args.use_balanced_sampling}")
    logger.info(f"Use wandb: {args.use_wandb}")
    
    try:
        # Create datasets
        logger.info("\n1. Creating datasets...")
        train_dataset, val_dataset = create_production_datasets(
            args.data_dir, args.images_root, tuple(args.image_size),
            train_subset_size=args.train_subset_size
        )
        
        # Create samplers and dataloaders
        logger.info("\n2. Creating data loaders...")
        
        # Create balanced sampler if requested
        sampler = None
        shuffle = True
        if args.use_balanced_sampling:
            sampler = create_balanced_sampler(train_dataset)
            shuffle = False  # Don't shuffle when using sampler
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=train_dataset.collate_fn,
            pin_memory=True,
            drop_last=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True
        )
        
        # Create model
        logger.info("\n3. Creating model...")
        model = create_production_model(load_pretrained=True, model_variant=args.model_variant)
        
        # Verify class count
        assert train_dataset.num_classes == 10, f"Expected 10 classes, got {train_dataset.num_classes}"
        logger.info("✅ All 10 classes confirmed in dataset")
        
        # Create trainer
        logger.info("\n4. Initializing trainer...")
        trainer = ProductionDETRTrainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=BDD100KDetrConfig(),
            device=args.device,
            checkpoint_dir=args.checkpoint_dir,
            use_wandb=args.use_wandb
        )
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=args.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            trainer.best_val_loss = checkpoint['best_val_loss']
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Resumed from epoch {start_epoch}")
        
        # Start training
        logger.info("\n5. Starting training...")
        trainer.train(num_epochs=args.epochs)
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PRODUCTION 10-CLASS DETR TRAINING COMPLETED!")
        logger.info("=" * 80)
        logger.info("✅ All 10 BDD100K classes successfully trained")
        logger.info("✅ Enhanced augmentation and class balancing applied")
        logger.info("✅ Pretrained DETR backbone utilized")
        logger.info("✅ Production-ready model checkpoints saved")
        logger.info(f"📁 Checkpoints saved to: {args.checkpoint_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()