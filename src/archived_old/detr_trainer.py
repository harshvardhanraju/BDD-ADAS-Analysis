"""
DETR Training Pipeline for BDD100K Object Detection

This module implements the complete training pipeline for DETR on BDD100K dataset
with class imbalance handling and advanced training techniques.
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
import json

from ..models.detr_model import BDD100KDETR, BDD100KDetrConfig, create_bdd_detr_model
from ..data.detr_dataset import create_bdd_dataloaders


class DETRTrainer:
    """
    DETR Trainer with advanced features for BDD100K dataset.
    """
    
    def __init__(
        self,
        model: BDD100KDETR,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        config: BDD100KDetrConfig,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        use_wandb: bool = True
    ):
        """
        Initialize DETR trainer.
        
        Args:
            model: BDD100K DETR model
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            config: Model configuration
            device: Training device ('cuda' or 'cpu')
            checkpoint_dir: Directory to save checkpoints
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Initialize wandb if requested
        if self.use_wandb:
            wandb.init(
                project="bdd100k-detr",
                config=self._get_wandb_config(),
                name=f"detr_bdd100k_{int(time.time())}"
            )
            wandb.watch(self.model, log='all', log_freq=100)
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Different learning rates for backbone and other parameters
        param_dicts = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if "backbone" not in n and p.requires_grad]
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if "backbone" in n and p.requires_grad],
                "lr": self.config.lr_backbone,
            },
        ]
        
        self.optimizer = optim.AdamW(
            param_dicts,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=7,
            gamma=0.1
        )
        
        print(f"Setup optimizer with lr={self.config.lr}, backbone_lr={self.config.lr_backbone}")
    
    def _get_wandb_config(self) -> Dict:
        """Get configuration for wandb logging."""
        return {
            'model': 'DETR',
            'dataset': 'BDD100K',
            'num_classes': self.config.num_classes,
            'batch_size': self.config.batch_size,
            'learning_rate': self.config.lr,
            'backbone_lr': self.config.lr_backbone,
            'weight_decay': self.config.weight_decay,
            'num_queries': self.config.num_queries,
            'hidden_dim': self.config.hidden_dim,
            'num_encoder_layers': self.config.num_encoder_layers,
            'num_decoder_layers': self.config.num_decoder_layers,
            'focal_loss_alpha': self.config.focal_loss_alpha,
            'focal_loss_gamma': self.config.focal_loss_gamma
        }
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary containing training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch + 1} - Training",
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
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train_loss_step': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': self.current_epoch + 1,
                    'step': self.current_epoch * num_batches + batch_idx
                })
        
        avg_train_loss = total_loss / num_batches
        
        return {
            'train_loss': avg_train_loss,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        progress_bar = tqdm(
            self.val_dataloader,
            desc=f"Epoch {self.current_epoch + 1} - Validation",
            leave=False
        )
        
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(progress_bar):
                # Move to device
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                # Forward pass
                outputs = self.model(images, targets)
                loss = outputs.loss
                
                # Update metrics
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Val Loss': f'{loss.item():.4f}',
                    'Avg Val Loss': f'{avg_loss:.4f}'
                })
        
        avg_val_loss = total_loss / num_batches
        
        return {'val_loss': avg_val_loss}
    
    def save_checkpoint(
        self,
        filename: Optional[str] = None,
        is_best: bool = False
    ):
        """
        Save model checkpoint.
        
        Args:
            filename: Custom filename for checkpoint
            is_best: Whether this is the best model so far
        """
        if filename is None:
            filename = f"detr_epoch_{self.current_epoch + 1}.pth"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.__dict__,
            'training_history': self.training_history,
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"✅ Saved best model to {best_path}")
        
        print(f"💾 Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Checkpoint information
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint['training_history']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"✅ Loaded checkpoint from epoch {self.current_epoch}")
        return checkpoint
    
    def train(self, num_epochs: int, save_every: int = 1) -> Dict[str, List[float]]:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            
        Returns:
            Training history dictionary
        """
        print(f"🚀 Starting training for {num_epochs} epochs...")
        print(f"📊 Training samples: {len(self.train_dataloader.dataset):,}")
        print(f"📊 Validation samples: {len(self.val_dataloader.dataset):,}")
        print(f"💻 Device: {self.device}")
        print("-" * 60)
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            self.scheduler.step()
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['learning_rates'].append(train_metrics['learning_rate'])
            
            # Check if best model
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
            
            # Print epoch summary
            print(f"Epoch {epoch + 1}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  LR: {train_metrics['learning_rate']:.6f}")
            if is_best:
                print(f"  🎉 New best validation loss!")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss_epoch': train_metrics['train_loss'],
                    'val_loss_epoch': val_metrics['val_loss'],
                    'learning_rate_epoch': train_metrics['learning_rate'],
                    'best_val_loss': self.best_val_loss
                })
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(is_best=is_best)
            
            print("-" * 60)
        
        # Final checkpoint
        self.save_checkpoint(filename="final_model.pth", is_best=is_best)
        
        # Training summary
        total_time = time.time() - start_time
        print(f"🎉 Training completed in {total_time:.2f} seconds!")
        print(f"📈 Best validation loss: {self.best_val_loss:.4f}")
        
        if self.use_wandb:
            wandb.finish()
        
        return self.training_history


def train_bdd_detr(
    train_annotations: str,
    val_annotations: str,
    images_root: str,
    num_epochs: int = 2,
    batch_size: int = 8,
    checkpoint_dir: str = "checkpoints",
    use_wandb: bool = True,
    resume_from: Optional[str] = None
) -> Dict[str, List[float]]:
    """
    Complete training pipeline for BDD100K DETR.
    
    Args:
        train_annotations: Path to training annotations
        val_annotations: Path to validation annotations
        images_root: Root directory for images
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        checkpoint_dir: Directory to save checkpoints
        use_wandb: Whether to use wandb logging
        resume_from: Path to checkpoint to resume from
        
    Returns:
        Training history
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = create_bdd_detr_model(pretrained=True)
    config = BDD100KDetrConfig()
    config.batch_size = batch_size
    config.num_epochs = num_epochs
    
    # Create data loaders
    train_dataloader, val_dataloader = create_bdd_dataloaders(
        train_annotations=train_annotations,
        val_annotations=val_annotations,
        images_root=images_root,
        batch_size=batch_size,
        num_workers=4,
        image_size=(512, 512)
    )
    
    # Create trainer
    trainer = DETRTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        device=str(device),
        checkpoint_dir=checkpoint_dir,
        use_wandb=use_wandb
    )
    
    # Resume from checkpoint if specified
    if resume_from and Path(resume_from).exists():
        trainer.load_checkpoint(resume_from)
    
    # Train model
    history = trainer.train(num_epochs=num_epochs)
    
    return history


if __name__ == "__main__":
    # Test training setup
    print("Testing DETR training setup...")
    
    # Example paths (adjust as needed)
    train_ann = "data/analysis/processed/train_annotations.csv"
    val_ann = "data/analysis/processed/val_annotations.csv"
    images_root = "data/raw/bdd100k/bdd100k/images/100k"
    
    if all(Path(p).exists() for p in [train_ann, val_ann, images_root]):
        # Run a quick test (1 epoch, small batch)
        history = train_bdd_detr(
            train_annotations=train_ann,
            val_annotations=val_ann,
            images_root=images_root,
            num_epochs=1,
            batch_size=2,
            use_wandb=False  # Disable wandb for testing
        )
        
        print("Training test completed successfully!")
        print(f"Training history: {history}")
    else:
        print("Data files not found. Please ensure all paths are correct.")