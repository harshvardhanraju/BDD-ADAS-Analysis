#!/usr/bin/env python3
"""
Enhanced DETR Training Script with Improved Class Rebalancing

This script implements enhanced training strategies based on qualitative analysis
findings, including proper convergence monitoring, stronger class rebalancing,
and advanced training techniques.
"""

import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import wandb

from ..models.detr_model import BDD100KDETR, create_bdd_detr_model
from ..data.detr_dataset import BDD100KDETRDataset


class EnhancedDETRTrainer:
    """
    Enhanced DETR trainer with advanced techniques for class imbalance handling.
    """
    
    def __init__(
        self,
        model: BDD100KDETR,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        device: str = 'cuda',
        output_dir: str = 'enhanced_training_results'
    ):
        """
        Initialize enhanced trainer.
        
        Args:
            model: DETR model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            device: Training device
            output_dir: Output directory for results
        """
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Class information
        self.class_names = model.config.class_names
        self.num_classes = len(self.class_names)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_map = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'class_losses': {name: [] for name in self.class_names},
            'class_accuracies': {name: [] for name in self.class_names}
        }
        
        # Enhanced class weights based on analysis findings
        self.setup_enhanced_class_weights()
        
        # Setup optimizers and schedulers
        self.setup_optimizers()
        
        # Early stopping parameters
        self.patience = 10
        self.patience_counter = 0
        self.min_delta = 1e-4
        
        print(f"Enhanced DETR Trainer initialized")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Output directory: {self.output_dir.absolute()}")
    
    def setup_enhanced_class_weights(self):
        """Setup enhanced class weights to prevent class collapse."""
        # Base weights from qualitative analysis findings
        base_weights = torch.tensor([
            0.1,   # car (dominant class - reduce weight)
            5.0,   # truck (increase from 2.0)
            8.0,   # bus (increase from 3.0)
            100.0, # train (increase from 50.0)
            30.0,  # rider (increase from 15.0)
            2.0,   # traffic_sign (increase from 0.3)
            2.5    # traffic_light (increase from 0.4)
        ]).to(self.device)
        
        # Apply temperature scaling to prevent extreme weights
        temperature = 0.7  # Soften the weights
        self.class_weights = torch.pow(base_weights, temperature)
        
        # Normalize weights
        self.class_weights = self.class_weights / self.class_weights.mean()
        
        # Update model's class weights
        self.model.register_buffer('class_weights', self.class_weights)
        
        print("Enhanced class weights:")
        for i, (name, weight) in enumerate(zip(self.class_names, self.class_weights)):
            print(f"  {name:15}: {weight:.3f}")
    
    def setup_optimizers(self):
        """Setup optimizers with differential learning rates."""
        # Separate parameters for backbone and head
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name or 'model.backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Differential learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 1e-4},
            {'params': head_params, 'lr': 1e-4, 'weight_decay': 1e-4}
        ])
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=True
        )
        
        # Warmup scheduler for first few epochs
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=5
        )
        
        print("Optimizer setup complete:")
        print(f"  Backbone LR: 1e-5")
        print(f"  Head LR: 1e-4")
        print(f"  Scheduler: ReduceLROnPlateau")
    
    def compute_enhanced_loss(self, outputs, targets):
        """
        Compute enhanced loss with stronger class rebalancing.
        
        Args:
            outputs: Model outputs
            targets: Ground truth targets
            
        Returns:
            Enhanced total loss
        """
        # Get base DETR losses
        loss_dict = outputs.loss_dict if hasattr(outputs, 'loss_dict') else {}
        
        # Enhanced classification loss with focal loss and class weights
        if hasattr(outputs, 'logits'):
            logits = outputs.logits  # [batch_size, num_queries, num_classes + 1]
            batch_size, num_queries = logits.shape[:2]
            
            # Prepare targets for focal loss
            all_targets = []
            for batch_targets in targets:
                query_targets = torch.full((num_queries,), self.num_classes, 
                                         device=logits.device, dtype=torch.long)
                
                if 'class_labels' in batch_targets and len(batch_targets['class_labels']) > 0:
                    valid_labels = batch_targets['class_labels'][batch_targets['class_labels'] < self.num_classes]
                    query_targets[:len(valid_labels)] = valid_labels
                
                all_targets.append(query_targets)
            
            flat_targets = torch.cat(all_targets)
            flat_logits = logits.view(-1, logits.shape[-1])
            
            # Compute focal loss
            ce_loss = nn.functional.cross_entropy(flat_logits, flat_targets, reduction='none')
            pt = torch.exp(-ce_loss)
            
            # Apply class weights
            class_weights_expanded = torch.ones_like(flat_targets, dtype=torch.float)
            for i in range(self.num_classes):
                mask = flat_targets == i
                class_weights_expanded[mask] = self.class_weights[i]
            
            # Focal loss with class weights
            alpha = 0.25
            gamma = 2.5  # Increased gamma for harder focus
            focal_loss = alpha * class_weights_expanded * (1 - pt) ** gamma * ce_loss
            
            # Apply stronger penalty for background predictions on foreground classes
            background_mask = flat_targets < self.num_classes  # Real objects
            background_penalty = 2.0
            focal_loss[background_mask] *= background_penalty
            
            enhanced_class_loss = focal_loss.mean()
            loss_dict['loss_ce'] = enhanced_class_loss
        
        # Combine all losses with enhanced weights
        loss_weights = {
            'loss_ce': 2.0,      # Increased classification weight
            'loss_bbox': 5.0,    # Bounding box regression weight
            'loss_giou': 2.0     # GIoU loss weight
        }
        
        total_loss = 0
        for key, loss_value in loss_dict.items():
            weight = loss_weights.get(key, 1.0)
            total_loss += weight * loss_value
        
        return total_loss, loss_dict
    
    def train_epoch(self, epoch: int) -> Dict:
        """
        Train for one epoch with enhanced monitoring.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        epoch_losses = []
        class_losses = {name: [] for name in self.class_names}
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, targets) in enumerate(progress_bar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = self.model(images, targets)
                
                # Compute enhanced loss
                total_loss, loss_dict = self.compute_enhanced_loss(outputs, targets)
                
                # Backward pass with gradient clipping
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
                self.optimizer.step()
                
                # Track losses
                epoch_losses.append(total_loss.item())
                
                # Update progress bar
                avg_loss = np.mean(epoch_losses[-100:])  # Last 100 batches
                progress_bar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Avg': f'{avg_loss:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
                # Log to wandb if available
                if wandb.run is not None:
                    wandb.log({
                        'train/batch_loss': total_loss.item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        **{f'train/{k}': v.item() for k, v in loss_dict.items()}
                    })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Apply warmup scheduler for first 5 epochs
        if epoch < 5:
            self.warmup_scheduler.step()
        
        epoch_metrics = {
            'train_loss': np.mean(epoch_losses),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'loss_components': {k: np.mean(v) for k, v in class_losses.items() if v}
        }
        
        return epoch_metrics
    
    def validate_epoch(self, epoch: int) -> Dict:
        """
        Validate model performance for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        val_losses = []
        class_predictions = {name: {'correct': 0, 'total': 0} for name in self.class_names}
        
        with torch.no_grad():
            for images, targets in tqdm(self.val_dataloader, desc="Validation"):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                try:
                    outputs = self.model(images, targets)
                    val_loss, _ = self.compute_enhanced_loss(outputs, targets)
                    val_losses.append(val_loss.item())
                    
                    # Track class-wise accuracy
                    self._update_class_accuracy(outputs, targets, class_predictions)
                    
                except Exception as e:
                    continue
        
        # Calculate class accuracies
        class_accuracies = {}
        for class_name, stats in class_predictions.items():
            if stats['total'] > 0:
                class_accuracies[class_name] = stats['correct'] / stats['total']
            else:
                class_accuracies[class_name] = 0.0
        
        val_metrics = {
            'val_loss': np.mean(val_losses),
            'class_accuracies': class_accuracies,
            'overall_accuracy': np.mean(list(class_accuracies.values()))
        }
        
        return val_metrics
    
    def _update_class_accuracy(self, outputs, targets, class_predictions):
        """Update class-wise accuracy tracking."""
        if not hasattr(outputs, 'logits'):
            return
        
        logits = outputs.logits  # [batch_size, num_queries, num_classes + 1]
        
        for i, target in enumerate(targets):
            if 'class_labels' not in target or len(target['class_labels']) == 0:
                continue
            
            # Get predictions for this image
            image_logits = logits[i]  # [num_queries, num_classes + 1]
            probs = torch.softmax(image_logits, dim=-1)[:, :-1]  # Remove background
            pred_scores, pred_labels = probs.max(dim=-1)
            
            # Filter by confidence
            confident_preds = pred_scores > 0.1
            if not confident_preds.any():
                continue
            
            pred_labels = pred_labels[confident_preds]
            
            # Count ground truth classes
            for label in target['class_labels']:
                if label < self.num_classes:
                    class_name = self.class_names[label]
                    class_predictions[class_name]['total'] += 1
                    
                    # Check if predicted correctly
                    if label in pred_labels:
                        class_predictions[class_name]['correct'] += 1
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'metrics': metrics,
            'class_weights': self.class_weights,
            'best_val_loss': self.best_val_loss,
            'best_map': self.best_map
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved to {best_path}")
        
        # Keep only last 3 regular checkpoints
        checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()
    
    def plot_training_progress(self):
        """Plot training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Learning rate
        axes[0, 1].plot(self.training_history['learning_rates'])
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True)
        
        # Class accuracies (latest epoch)
        if self.training_history['class_accuracies']:
            latest_acc = {k: v[-1] if v else 0 for k, v in self.training_history['class_accuracies'].items()}
            classes = list(latest_acc.keys())
            accuracies = list(latest_acc.values())
            
            axes[1, 0].bar(classes, accuracies)
            axes[1, 0].set_title('Class Accuracies (Latest Epoch)')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Training convergence
        if len(self.training_history['train_loss']) > 10:
            # Moving average of last 10 epochs
            window = 10
            train_ma = np.convolve(self.training_history['train_loss'], 
                                 np.ones(window)/window, mode='valid')
            val_ma = np.convolve(self.training_history['val_loss'], 
                               np.ones(window)/window, mode='valid')
            
            axes[1, 1].plot(range(window-1, len(self.training_history['train_loss'])), 
                          train_ma, label='Train Loss (MA)')
            axes[1, 1].plot(range(window-1, len(self.training_history['val_loss'])), 
                          val_ma, label='Val Loss (MA)')
            axes[1, 1].set_title('Loss Convergence (Moving Average)')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'training_progress.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Training progress plot saved to {plot_path}")
    
    def train(self, num_epochs: int = 50, use_wandb: bool = False):
        """
        Main training loop with enhanced monitoring.
        
        Args:
            num_epochs: Number of epochs to train
            use_wandb: Whether to use Weights & Biases logging
        """
        print(f"🚀 Starting enhanced DETR training for {num_epochs} epochs...")
        
        # Initialize wandb if requested
        if use_wandb:
            wandb.init(
                project="bdd100k-detr-enhanced",
                config={
                    'epochs': num_epochs,
                    'class_weights': self.class_weights.cpu().tolist(),
                    'optimizer': 'AdamW',
                    'scheduler': 'ReduceLROnPlateau'
                }
            )
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_metrics = self.validate_epoch(epoch)
            
            # Update learning rate scheduler
            self.scheduler.step(val_metrics['val_loss'])
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['train_loss'])
            self.training_history['val_loss'].append(val_metrics['val_loss'])
            self.training_history['learning_rates'].append(train_metrics['learning_rate'])
            
            for class_name in self.class_names:
                self.training_history['class_accuracies'][class_name].append(
                    val_metrics['class_accuracies'].get(class_name, 0.0)
                )
            
            # Check for best model
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, {**train_metrics, **val_metrics}, is_best)
            
            # Print epoch results
            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"  Overall Accuracy: {val_metrics['overall_accuracy']:.3f}")
            print(f"  Learning Rate: {train_metrics['learning_rate']:.2e}")
            
            # Print class accuracies
            print("  Class Accuracies:")
            for class_name, acc in val_metrics['class_accuracies'].items():
                print(f"    {class_name:15}: {acc:.3f}")
            
            # Log to wandb
            if use_wandb:
                wandb.log({
                    'epoch': epoch,
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items()},
                    **{f'class_acc/{k}': v for k, v in val_metrics['class_accuracies'].items()}
                })
            
            # Plot progress every 10 epochs
            if epoch % 10 == 0:
                self.plot_training_progress()
            
            # Early stopping check
            if self.patience_counter >= self.patience:
                print(f"\n⏹️  Early stopping triggered after {self.patience} epochs without improvement")
                break
            
            print("-" * 60)
        
        # Final training summary
        training_time = time.time() - start_time
        print(f"\n🎉 Training completed!")
        print(f"⏱️  Training time: {training_time/3600:.2f} hours")
        print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
        print(f"📁 Results saved to: {self.output_dir.absolute()}")
        
        # Final plots
        self.plot_training_progress()
        
        if use_wandb:
            wandb.finish()
        
        return self.training_history


def create_enhanced_trainer(
    annotations_dir: str = "data/analysis/processed",
    images_root: str = "data/raw/bdd100k/bdd100k/images/100k",
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = 'cuda'
) -> EnhancedDETRTrainer:
    """
    Factory function to create enhanced DETR trainer.
    
    Args:
        annotations_dir: Directory containing processed annotations
        images_root: Root directory for images
        batch_size: Training batch size
        num_workers: Number of data loading workers
        device: Training device
        
    Returns:
        Configured enhanced trainer
    """
    # Create datasets with enhanced augmentation
    train_dataset = BDD100KDETRDataset(
        annotations_file=f"{annotations_dir}/train_annotations.csv",
        images_root=images_root,
        split='train',
        use_enhanced_augmentation=True,
        augmentation_strength='medium'
    )
    
    val_dataset = BDD100KDETRDataset(
        annotations_file=f"{annotations_dir}/val_annotations.csv", 
        images_root=images_root,
        split='val',
        use_enhanced_augmentation=True,
        augmentation_strength='light'  # Light augmentation for validation
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    # Create model
    model = create_bdd_detr_model(pretrained=True)
    
    # Create trainer
    trainer = EnhancedDETRTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        output_dir='enhanced_training_results'
    )
    
    return trainer


if __name__ == "__main__":
    print("Enhanced DETR Trainer Module")
    print("Use create_enhanced_trainer() to create a trainer instance")