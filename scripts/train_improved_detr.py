#!/usr/bin/env python3
"""
Improved DETR Training Script with Better Class Rebalancing

This script implements the key improvements from qualitative analysis:
- Stronger class weights to prevent class collapse
- More training epochs for proper convergence
- Better loss formulation
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
from src.data.detr_dataset import BDD100KDETRDataset


class ImprovedDETRTrainer:
    """
    Improved DETR trainer with key fixes from qualitative analysis.
    """
    
    def __init__(self, model, train_loader, val_loader, device='cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path("improved_training_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Enhanced class weights (stronger to prevent collapse)
        self.setup_enhanced_weights()
        
        # Setup optimizer with differential learning rates
        self.setup_optimizer()
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')
    
    def setup_enhanced_weights(self):
        """Setup enhanced class weights based on analysis findings."""
        # Much stronger weights to prevent class collapse
        enhanced_weights = torch.tensor([
            0.05,  # car - severely reduced weight
            10.0,  # truck - much higher weight
            15.0,  # bus - much higher weight  
            200.0, # train - extremely high weight
            50.0,  # rider - very high weight
            5.0,   # traffic_sign - higher weight
            8.0    # traffic_light - higher weight
        ]).to(self.device)
        
        # Update model weights
        self.model.register_buffer('class_weights', enhanced_weights)
        
        print("Enhanced class weights (prevent collapse):")
        class_names = ['car', 'truck', 'bus', 'train', 'rider', 'traffic_sign', 'traffic_light']
        for name, weight in zip(class_names, enhanced_weights):
            print(f"  {name:15}: {weight:.1f}x")
    
    def setup_optimizer(self):
        """Setup optimizer with proper learning rates."""
        # Differential learning rates
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': 1e-5, 'weight_decay': 1e-4},
            {'params': head_params, 'lr': 1e-4, 'weight_decay': 1e-4}
        ])
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Focal loss for hard example mining
        self.focal_loss = FocalLoss(alpha=0.25, gamma=3.0)  # Higher gamma
    
    def compute_enhanced_loss(self, outputs, targets):
        """Compute loss with enhanced class weighting."""
        if not hasattr(outputs, 'loss_dict'):
            # Standard loss computation
            return outputs.loss
        
        loss_dict = outputs.loss_dict
        
        # Replace classification loss with focal loss + class weights
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            batch_size, num_queries = logits.shape[:2]
            
            # Prepare targets
            all_targets = []
            for batch_targets in targets:
                query_targets = torch.full((num_queries,), 7, device=logits.device)  # Background = 7
                if 'class_labels' in batch_targets and len(batch_targets['class_labels']) > 0:
                    valid_labels = batch_targets['class_labels'][batch_targets['class_labels'] < 7]
                    if len(valid_labels) > 0:
                        query_targets[:len(valid_labels)] = valid_labels
                all_targets.append(query_targets)
            
            flat_targets = torch.cat(all_targets)
            flat_logits = logits.view(-1, logits.shape[-1])
            
            # Apply focal loss with class weights
            ce_loss = nn.functional.cross_entropy(flat_logits, flat_targets, reduction='none')
            pt = torch.exp(-ce_loss)
            
            # Apply class weights
            class_weights_expanded = torch.ones_like(flat_targets, dtype=torch.float)
            for i in range(7):
                mask = flat_targets == i
                class_weights_expanded[mask] = self.model.class_weights[i]
            
            # Enhanced focal loss
            focal_loss = 0.25 * class_weights_expanded * (1 - pt) ** 3.0 * ce_loss
            
            # Extra penalty for background predictions on real objects
            real_object_mask = flat_targets < 7
            focal_loss[real_object_mask] *= 3.0  # Triple penalty for missed objects
            
            loss_dict['loss_ce'] = focal_loss.mean()
        
        # Combine losses with enhanced weights
        total_loss = (
            2.0 * loss_dict.get('loss_ce', 0) +  # Enhanced classification weight
            5.0 * loss_dict.get('loss_bbox', 0) +  # Bbox loss
            2.0 * loss_dict.get('loss_giou', 0)   # GIoU loss
        )
        
        return total_loss
    
    def train_epoch(self, epoch):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                       for k, v in t.items()} for t in targets]
            
            self.optimizer.zero_grad()
            
            try:
                outputs = self.model(images, targets)
                loss = self.compute_enhanced_loss(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress
                avg_loss = total_loss / num_batches
                pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg': f'{avg_loss:.4f}'})
                
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
            for images, targets in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                try:
                    outputs = self.model(images, targets)
                    loss = self.compute_enhanced_loss(outputs, targets)
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                except Exception:
                    continue
        
        return total_loss / max(num_batches, 1)
    
    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'history': self.history
        }
        
        # Save checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"💾 Best model saved (val_loss: {val_loss:.4f})")
    
    def train(self, epochs=50):
        """Main training loop."""
        print(f"🚀 Starting improved DETR training for {epochs} epochs...")
        
        for epoch in range(1, epochs + 1):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation  
            val_loss = self.validate_epoch(epoch)
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_loss, val_loss, is_best)
            
            # Print epoch results
            print(f"\nEpoch {epoch}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Best Val Loss: {self.best_val_loss:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Plot progress every 10 epochs
            if epoch % 10 == 0:
                self.plot_progress()
            
            print("-" * 50)
        
        print("🎉 Training completed!")
        return self.history
    
    def plot_progress(self):
        """Plot training progress."""
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        # Plot recent losses (last 20 epochs)
        recent_epochs = min(20, len(self.history['train_loss']))
        recent_train = self.history['train_loss'][-recent_epochs:]
        recent_val = self.history['val_loss'][-recent_epochs:]
        
        plt.plot(range(len(self.history['train_loss'])-recent_epochs+1, len(self.history['train_loss'])+1), 
                recent_train, label='Recent Train Loss')
        plt.plot(range(len(self.history['val_loss'])-recent_epochs+1, len(self.history['val_loss'])+1), 
                recent_val, label='Recent Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Recent Training Progress')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_progress.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main function."""
    print("🚀 Improved DETR Training Script")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets with robust handling
    print("📊 Loading datasets...")
    train_dataset = BDD100KDETRDataset(
        annotations_file="data/analysis/processed/train_annotations.csv",
        images_root="data/raw/bdd100k/bdd100k/images/100k",
        split='train',
        use_enhanced_augmentation=False,  # Use basic augmentation to avoid issues
        augment=True
    )
    
    val_dataset = BDD100KDETRDataset(
        annotations_file="data/analysis/processed/val_annotations.csv",
        images_root="data/raw/bdd100k/bdd100k/images/100k", 
        split='val',
        use_enhanced_augmentation=False,
        augment=False
    )
    
    # Create data loaders with smaller batch size
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True,
        num_workers=2, collate_fn=train_dataset.collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False,
        num_workers=2, collate_fn=val_dataset.collate_fn,
        pin_memory=True
    )
    
    print(f"✅ Train dataset: {len(train_dataset)} images")
    print(f"✅ Val dataset: {len(val_dataset)} images")
    
    # Create model
    print("🤖 Creating DETR model...")
    model = create_bdd_detr_model(pretrained=True)
    
    # Create trainer
    trainer = ImprovedDETRTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    # Start training
    print("\n🎯 Key improvements implemented:")
    print("  • Enhanced class weights (200x for train, 50x for rider)")
    print("  • Stronger focal loss (gamma=3.0)")
    print("  • Differential learning rates")
    print("  • Triple penalty for missed objects")
    print("  • 50 epochs for proper convergence")
    
    history = trainer.train(epochs=50)
    
    print("\n🎉 Training completed!")
    print("📊 Expected improvements:")
    print("  • All classes should now be detected (no more collapse)")
    print("  • Better precision through enhanced loss formulation")
    print("  • Improved convergence over 50 epochs")


if __name__ == "__main__":
    main()