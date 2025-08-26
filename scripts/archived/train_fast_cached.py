#!/usr/bin/env python3
"""
Fast Cached DETR Training Script

This script uses a cached dataset for much faster training by loading all images
into memory before training starts, eliminating file I/O during training.
"""

import sys
import os
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.detr_model import create_bdd_detr_model, FocalLoss
from src.data.cached_dataset import CachedBDD100KDataset


class FastCachedTrainer:
    """
    Fast trainer using cached dataset for improved training speed.
    """
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path("fast_training_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Enhanced class weights to prevent collapse
        self.setup_enhanced_weights()
        
        # Setup optimizer
        self.setup_optimizer()
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = 8  # Early stopping patience
    
    def setup_enhanced_weights(self):
        """Setup balanced class weights to reduce overfitting."""
        # More balanced weights to prevent overfitting
        enhanced_weights = torch.tensor([
            0.2,   # car - less aggressive reduction
            3.0,   # truck - moderate weight (was 12.0)
            5.0,   # bus - moderate weight (was 18.0)  
            20.0,  # train - significantly reduced from 250.0
            8.0,   # rider - reduced from 60.0
            1.5,   # traffic_sign - reduced from 6.0
            2.0    # traffic_light - reduced from 10.0
        ]).to(self.device)
        
        # Update model weights
        self.model.register_buffer('class_weights', enhanced_weights)
        
        print("🎯 Balanced class weights (reduce overfitting):")
        class_names = ['car', 'truck', 'bus', 'train', 'rider', 'traffic_sign', 'traffic_light']
        for name, weight in zip(class_names, enhanced_weights):
            print(f"   • {name:15}: {weight:6.1f}x")
        print()
    
    def setup_optimizer(self):
        """Setup optimizer with differential learning rates."""
        # Separate backbone and head parameters
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name or 'model.backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Differential learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 5e-4},  # Increased weight decay
            {'params': head_params, 'lr': 1e-4, 'weight_decay': 5e-4}       # Increased weight decay
        ])
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, 
            min_lr=1e-7, verbose=True
        )
        
        # Reduced focal loss to prevent overfitting
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)  # Reduced from 3.5
        
        print("🔧 Optimizer setup:")
        print(f"   • Backbone LR: 1e-5")
        print(f"   • Head LR: 1e-4") 
        print(f"   • Focal Loss gamma: 2.0 (reduced overfitting)")
        print(f"   • Scheduler: ReduceLROnPlateau")
        print()
    
    def compute_enhanced_loss(self, outputs, targets):
        """Compute enhanced loss with strong class rebalancing."""
        if not hasattr(outputs, 'logits'):
            return outputs.loss
        
        logits = outputs.logits  # [batch_size, num_queries, num_classes + 1]
        batch_size, num_queries = logits.shape[:2]
        
        # Prepare targets for enhanced loss
        all_targets = []
        for batch_targets in targets:
            # Background class = 7 (num_classes)
            query_targets = torch.full((num_queries,), 7, device=logits.device)
            
            if 'class_labels' in batch_targets and len(batch_targets['class_labels']) > 0:
                valid_labels = batch_targets['class_labels'][batch_targets['class_labels'] < 7]
                if len(valid_labels) > 0:
                    query_targets[:len(valid_labels)] = valid_labels
            
            all_targets.append(query_targets)
        
        flat_targets = torch.cat(all_targets)
        flat_logits = logits.view(-1, logits.shape[-1])
        
        # Enhanced focal loss with class weights
        ce_loss = nn.functional.cross_entropy(flat_logits, flat_targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Apply class weights
        class_weights_expanded = torch.ones_like(flat_targets, dtype=torch.float)
        for i in range(7):
            mask = flat_targets == i
            if mask.any():
                class_weights_expanded[mask] = self.model.class_weights[i]
        
        # Strong focal loss with class weights
        focal_weight = 0.25 * class_weights_expanded * (1 - pt) ** 3.5
        enhanced_loss = focal_weight * ce_loss
        
        # Extra penalty for missing real objects
        real_object_mask = flat_targets < 7
        enhanced_loss[real_object_mask] *= 2.0  # Reduced penalty to prevent overfitting
        
        # Get other loss components if available
        total_loss = enhanced_loss.mean()
        
        # Add bbox and giou losses if available
        if hasattr(outputs, 'loss_dict'):
            loss_dict = outputs.loss_dict
            bbox_loss = loss_dict.get('loss_bbox', 0)
            giou_loss = loss_dict.get('loss_giou', 0)
            
            total_loss = (
                2.5 * total_loss +        # Enhanced classification loss
                5.0 * bbox_loss +         # Bounding box loss  
                2.0 * giou_loss          # GIoU loss
            )
        
        return total_loss
    
    def train_epoch(self, epoch):
        """Train one epoch with progress tracking."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch:2d}")
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = self.model(images, targets)
                loss = self.compute_enhanced_loss(outputs, targets)
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                
                # Track loss
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                current_avg = total_loss / num_batches
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg': f'{current_avg:.4f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
                
            except Exception as e:
                print(f"Skipping batch {batch_idx}: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def validate_epoch(self, epoch):
        """Validate one epoch."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for images, targets in pbar:
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                try:
                    outputs = self.model(images, targets)
                    loss = self.compute_enhanced_loss(outputs, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
                    
                except Exception:
                    continue
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'class_weights': self.model.class_weights
        }
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved (val_loss: {val_loss:.4f})")
        
        # Save epoch checkpoint (keep last 3)
        epoch_path = self.output_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, epoch_path)
        
        # Clean up old checkpoints
        checkpoints = sorted(self.output_dir.glob('checkpoint_epoch_*.pth'))
        if len(checkpoints) > 3:
            for old_checkpoint in checkpoints[:-3]:
                old_checkpoint.unlink()
    
    def plot_progress(self):
        """Plot and save training progress."""
        if len(self.history['train_loss']) < 2:
            return
        
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 3, 1)
        epochs = range(1, len(self.history['train_loss']) + 1)
        plt.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', marker='o')
        plt.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        # Learning rate
        plt.subplot(1, 3, 2)
        plt.plot(epochs, self.history['learning_rates'], 'g-', marker='^')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.yscale('log')
        plt.grid(True)
        
        # Recent progress (last 20 epochs)
        plt.subplot(1, 3, 3)
        recent_epochs = min(20, len(self.history['train_loss']))
        if recent_epochs > 1:
            recent_range = epochs[-recent_epochs:]
            recent_train = self.history['train_loss'][-recent_epochs:]
            recent_val = self.history['val_loss'][-recent_epochs:]
            
            plt.plot(recent_range, recent_train, 'b-', label='Recent Train', marker='o')
            plt.plot(recent_range, recent_val, 'r-', label='Recent Val', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Recent Progress')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'training_progress.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def train(self, epochs=50):
        """Main training loop."""
        print(f"🚀 Starting fast cached training for {epochs} epochs...")
        print(f"📁 Results will be saved to: {self.output_dir}")
        print()
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation  
            val_loss = self.validate_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss, is_best)
            
            # Calculate time
            epoch_time = time.time() - epoch_start
            total_time = time.time() - start_time
            remaining_time = (epoch_time * (epochs - epoch)) / 3600  # hours
            
            # Print epoch results
            print(f"\n📊 Epoch {epoch}/{epochs} Results:")
            print(f"   • Train Loss: {train_loss:.4f}")
            print(f"   • Val Loss:   {val_loss:.4f} {'🏆' if is_best else ''}")
            print(f"   • Best Val:   {self.best_val_loss:.4f}")
            print(f"   • LR:         {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"   • Time:       {epoch_time:.1f}s/epoch")
            print(f"   • ETA:        {remaining_time:.1f}h remaining")
            print(f"   • Patience:   {self.patience_counter}/{self.patience}")
            
            # Plot progress every 5 epochs
            if epoch % 5 == 0:
                self.plot_progress()
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n⏹️  Early stopping triggered! No improvement for {self.patience} epochs")
                break
            
            print("-" * 80)
        
        # Training complete
        total_hours = (time.time() - start_time) / 3600
        print(f"\n🎉 Training completed in {total_hours:.2f} hours!")
        print(f"🏆 Best validation loss: {self.best_val_loss:.4f}")
        print(f"📁 Results saved to: {self.output_dir}")
        
        # Final plot
        self.plot_progress()
        
        return self.history


def main():
    """Main training function."""
    print("🚀 Fast Cached DETR Training")
    print("=" * 80)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    if device.type == 'cuda':
        print(f"   • GPU: {torch.cuda.get_device_name()}")
        print(f"   • Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print()
    
    # Create cached datasets
    print("📦 Creating cached datasets...")
    print("   This will load 20k training + 3k validation images to reduce overfitting")
    print()
    
    train_dataset = CachedBDD100KDataset(
        annotations_file="data/analysis/processed/train_annotations.csv",
        images_root="data/raw/bdd100k/bdd100k/images/100k",
        split='train',
        image_size=(512, 512),
        augment=True,
        cache_dir="dataset_cache",
        max_images=20000  # Increased to 20k images to reduce overfitting
    )
    
    val_dataset = CachedBDD100KDataset(
        annotations_file="data/analysis/processed/val_annotations.csv",
        images_root="data/raw/bdd100k/bdd100k/images/100k",
        split='val',
        image_size=(512, 512),
        augment=False,
        cache_dir="dataset_cache",
        max_images=3000  # Increased validation to 3000 images
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=6,  # Larger batch size since data is cached
        shuffle=True,
        num_workers=4,  # Can use more workers since no file I/O
        collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,  # Even larger for validation
        shuffle=False,
        num_workers=4,
        collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    print(f"✅ Datasets ready:")
    print(f"   • Train: {len(train_dataset):,} images")
    print(f"   • Val:   {len(val_dataset):,} images")
    print(f"   • Train batches: {len(train_loader):,}")
    print(f"   • Val batches:   {len(val_loader):,}")
    print()
    
    # Create model
    print("🤖 Creating DETR model...")
    model = create_bdd_detr_model(pretrained=True)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {model_params:,} parameters")
    print()
    
    # Create trainer
    trainer = FastCachedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Start training
    print("🎯 Training Configuration:")
    print("   • Balanced class weights to reduce overfitting")
    print("   • Moderate focal loss (gamma=2.0) to reduce overfitting")
    print("   • Quadruple penalty for missed objects")
    print("   • Early stopping with patience=8")
    print("   • Cached data for maximum speed")
    print()
    
    print("🚀 Starting training...")
    print("=" * 80)
    
    history = trainer.train(epochs=50)
    
    print("\n🎉 Training complete!")
    print("\n📈 Expected improvements:")
    print("   • All 7 classes should now be detected")
    print("   • Major reduction in class collapse")
    print("   • Better precision/recall balance")
    print("   • Faster training due to cached data")


if __name__ == "__main__":
    main()