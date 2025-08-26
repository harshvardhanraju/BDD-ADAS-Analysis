#!/usr/bin/env python3
"""
DETR Training Demo Script for BDD100K Dataset

This script trains a DETR model on a subset of BDD100K dataset for demonstration
purposes, handling missing images gracefully.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import cv2

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.detr_model import create_bdd_detr_model


class BDDDemoDataset(Dataset):
    """Simplified BDD100K dataset for demo training."""
    
    def __init__(self, annotations_file, images_root, split='train', max_images=1000):
        self.annotations_file = Path(annotations_file)
        self.images_root = Path(images_root)
        self.split = split
        self.max_images = max_images
        
        # Class mapping
        self.class_mapping = {
            'car': 0, 'truck': 1, 'bus': 2, 'train': 3,
            'rider': 4, 'traffic sign': 5, 'traffic light': 6
        }
        
        # Load and filter data
        self._load_data()
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def _load_data(self):
        """Load and filter annotations."""
        print(f"Loading annotations from {self.annotations_file}")
        df = pd.read_csv(self.annotations_file, low_memory=False)
        df = df[df['split'] == self.split].copy()
        
        # Group by image and keep only images with valid objects
        self.image_data = []
        processed = 0
        
        for image_name, group in df.groupby('image_name'):
            if processed >= self.max_images:
                break
                
            # Check if image exists
            image_path = self.images_root / self.split / image_name
            if not image_path.exists():
                continue
                
            # Get valid objects
            objects = group[group['category'].notna()].copy()
            valid_objects = []
            
            for _, obj in objects.iterrows():
                category = obj['category']
                if category in self.class_mapping:
                    try:
                        bbox = [
                            float(obj['bbox_x1']), float(obj['bbox_y1']),
                            float(obj['bbox_x2']), float(obj['bbox_y2'])
                        ]
                        # Basic validation
                        if all(x >= 0 for x in bbox) and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                            valid_objects.append({
                                'class_id': self.class_mapping[category],
                                'bbox': bbox
                            })
                    except (ValueError, TypeError):
                        continue
            
            if valid_objects:
                self.image_data.append({
                    'image_name': image_name,
                    'objects': valid_objects
                })
                processed += 1
        
        print(f"Loaded {len(self.image_data)} images with valid annotations")
    
    def __len__(self):
        return len(self.image_data)
    
    def __getitem__(self, idx):
        item = self.image_data[idx]
        image_name = item['image_name']
        objects = item['objects']
        
        # Load image
        image_path = self.images_root / self.split / image_name
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        h, w = image.shape[:2]
        
        # Apply transforms
        image = self.transform(image)
        
        # Prepare targets in DETR format
        if objects:
            # Convert bboxes to normalized center format
            boxes = []
            labels = []
            
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                
                # Normalize and convert to center format
                center_x = (x1 + x2) / 2.0 / w
                center_y = (y1 + y2) / 2.0 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
                
                # Clamp values
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                width = max(0.01, min(1, width))
                height = max(0.01, min(1, height))
                
                boxes.append([center_x, center_y, width, height])
                labels.append(obj['class_id'])
            
            target = {
                'class_labels': torch.tensor(labels, dtype=torch.long),
                'boxes': torch.tensor(boxes, dtype=torch.float32),
                'image_id': torch.tensor([idx]),
                'area': torch.tensor([w*h*b[2]*b[3] for b in boxes], dtype=torch.float32),
                'iscrowd': torch.zeros(len(labels), dtype=torch.long),
                'orig_size': torch.tensor([h, w]),
                'size': torch.tensor([512, 512])
            }
        else:
            # Empty target
            target = {
                'class_labels': torch.zeros((0,), dtype=torch.long),
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'image_id': torch.tensor([idx]),
                'area': torch.zeros((0,), dtype=torch.float32),
                'iscrowd': torch.zeros((0,), dtype=torch.long),
                'orig_size': torch.tensor([h, w]),
                'size': torch.tensor([512, 512])
            }
        
        return image, target


def collate_fn(batch):
    """Custom collate function for batch processing."""
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    images = torch.stack(images, dim=0)
    return images, targets


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} - Training")
    
    for batch_idx, (images, targets) in enumerate(progress_bar):
        images = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        try:
            outputs = model(images, targets)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{avg_loss:.4f}'
            })
            
        except Exception as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue
    
    return total_loss / num_batches


def main():
    """Main training function."""
    print("🚀 Starting DETR Demo Training...")
    
    # Paths
    train_ann = "data/analysis/processed/train_annotations.csv"
    val_ann = "data/analysis/processed/val_annotations.csv"
    images_root = "data/raw/bdd100k/bdd100k/images/100k"
    
    # Check paths
    if not all(Path(p).exists() for p in [train_ann, val_ann, images_root]):
        print("❌ Required files not found")
        return
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets (small subset for demo)
    print("Creating datasets...")
    train_dataset = BDDDemoDataset(train_ann, images_root, 'train', max_images=100)
    val_dataset = BDDDemoDataset(val_ann, images_root, 'val', max_images=50)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, 
        collate_fn=collate_fn, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False, 
        collate_fn=collate_fn, num_workers=2
    )
    
    # Create model
    print("Creating DETR model...")
    model = create_bdd_detr_model(pretrained=True)
    model = model.to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Training loop
    num_epochs = 2
    print(f"Training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch + 1)
        
        # Validate (simplified)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch + 1} - Validation"):
                images = images.to(device)
                targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                           for k, v in t.items()} for t in targets]
                
                try:
                    outputs = model(images, targets)
                    val_loss += outputs.loss.item()
                except:
                    continue
        
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch + 1}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print("-" * 50)
    
    # Save checkpoint
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss
    }
    
    torch.save(checkpoint, checkpoint_dir / "detr_demo_checkpoint.pth")
    print(f"💾 Checkpoint saved to {checkpoint_dir}/detr_demo_checkpoint.pth")
    
    print("🎉 Demo training completed!")


if __name__ == "__main__":
    main()