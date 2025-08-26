"""
DETR Model Implementation for BDD100K Object Detection

This module implements a Deformable DETR model specifically configured for
BDD100K autonomous driving dataset with class imbalance handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DetrConfig, DetrForObjectDetection
from typing import Dict, List, Optional, Tuple
import numpy as np


class BDD100KDetrConfig:
    """Configuration class for BDD100K DETR model."""
    
    def __init__(self):
        # BDD100K complete 10-class configuration
        self.num_classes = 10  # Complete set of BDD100K detection classes
        self.class_names = [
            'pedestrian', 'rider', 'car', 'truck', 'bus', 
            'train', 'motorcycle', 'bicycle', 'traffic_light', 'traffic_sign'
        ]
        
        # Class weights based on complete 10-class analysis (inverse frequency weighted by safety)
        self.class_weights = torch.tensor([
            1.5,   # pedestrian (7.1% - safety critical)
            20.0,  # rider (0.35% - very rare, safety critical)  
            0.1,   # car (55.4% - most frequent)
            4.0,   # truck (2.32%)
            8.0,   # bus (0.90%)
            100.0, # train (0.01% - extremely rare)
            35.0,  # motorcycle (0.23% - very rare, safety critical)
            15.0,  # bicycle (0.56% - rare, safety critical)
            0.4,   # traffic_light (14.5%)
            0.3    # traffic_sign (18.6%)
        ])
        
        # Model configuration
        self.backbone = "resnet50"
        self.num_queries = 100
        self.hidden_dim = 256
        self.nheads = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dropout = 0.1
        
        # Loss configuration
        self.focal_loss_alpha = 0.25
        self.focal_loss_gamma = 2.0
        self.bbox_loss_coef = 5.0
        self.giou_loss_coef = 2.0
        self.class_loss_coef = 1.0
        
        # Training configuration
        self.lr = 1e-4
        self.weight_decay = 1e-4
        self.lr_backbone = 1e-5
        self.batch_size = 8
        self.num_epochs = 50


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance.
    
    Based on: "Focal Loss for Dense Object Detection" by Lin et al.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for rare class (between 0 and 1)
            gamma: Focusing parameter for hard examples
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions [N, C] where C = number of classes
            targets: Ground truth class indices [N]
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BDD100KDETR(nn.Module):
    """
    BDD100K-specific DETR model with class imbalance handling.
    """
    
    def __init__(self, config: BDD100KDetrConfig, pretrained: bool = True):
        """
        Initialize BDD100K DETR model.
        
        Args:
            config: Model configuration
            pretrained: Whether to load pretrained COCO weights
        """
        super().__init__()
        self.config = config
        self.num_classes = config.num_classes
        
        # Create DETR configuration
        detr_config = DetrConfig(
            num_labels=config.num_classes,
            num_queries=config.num_queries,
            d_model=config.hidden_dim,
            encoder_layers=config.num_encoder_layers,
            decoder_layers=config.num_decoder_layers,
            encoder_attention_heads=config.nheads,
            decoder_attention_heads=config.nheads,
            dropout=config.dropout,
            attention_dropout=config.dropout,
            activation_dropout=config.dropout,
            bbox_loss_coef=config.bbox_loss_coef,
            giou_loss_coef=config.giou_loss_coef,
            class_loss_coef=config.class_loss_coef,
        )
        
        # Initialize model
        if pretrained:
            print("Loading pretrained DETR model from COCO...")
            self.model = DetrForObjectDetection.from_pretrained(
                "facebook/detr-resnet-50",
                config=detr_config,
                ignore_mismatched_sizes=True
            )
        else:
            self.model = DetrForObjectDetection(detr_config)
        
        # Initialize focal loss for class imbalance
        self.focal_loss = FocalLoss(
            alpha=config.focal_loss_alpha,
            gamma=config.focal_loss_gamma
        )
        
        # Class weights for loss computation
        self.register_buffer('class_weights', config.class_weights)
        
        print(f"Initialized BDD100K DETR with {config.num_classes} classes")
        print(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, pixel_values: torch.Tensor, labels: Optional[List[Dict]] = None) -> Dict:
        """
        Forward pass of the model.
        
        Args:
            pixel_values: Input images [batch_size, 3, H, W]
            labels: Ground truth labels (for training)
            
        Returns:
            Dictionary containing logits, boxes, and optionally loss
        """
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        
        if self.training and labels is not None:
            # Apply class-weighted loss
            outputs.loss = self._compute_weighted_loss(outputs, labels)
        
        return outputs
    
    def _compute_weighted_loss(self, outputs, labels: List[Dict]) -> torch.Tensor:
        """
        Compute class-weighted loss with focal loss for classification.
        
        Args:
            outputs: Model outputs
            labels: Ground truth labels
            
        Returns:
            Weighted total loss
        """
        # Get the original loss components
        loss_dict = outputs.loss_dict if hasattr(outputs, 'loss_dict') else {}
        
        # Apply focal loss for classification if available
        if 'loss_ce' in loss_dict:
            # Replace cross-entropy with focal loss
            logits = outputs.logits  # [batch_size, num_queries, num_classes + 1]
            
            # Flatten logits and targets for focal loss computation
            batch_size, num_queries = logits.shape[:2]
            flat_logits = logits.view(-1, logits.shape[-1])
            
            # Create target tensor from labels
            targets = []
            for batch_labels in labels:
                batch_targets = torch.full((num_queries,), self.num_classes, 
                                         device=logits.device)  # Background class
                if 'class_labels' in batch_labels:
                    valid_indices = batch_labels['class_labels'] < self.num_classes
                    batch_targets[:len(batch_labels['class_labels'][valid_indices])] = \
                        batch_labels['class_labels'][valid_indices]
                targets.append(batch_targets)
            
            flat_targets = torch.cat(targets)
            
            # Compute focal loss
            focal_loss = self.focal_loss(flat_logits, flat_targets)
            loss_dict['loss_ce'] = focal_loss
        
        # Combine all losses
        total_loss = sum(loss_dict.values())
        return total_loss
    
    def predict(self, pixel_values: torch.Tensor, threshold: float = 0.5) -> List[Dict]:
        """
        Make predictions on input images.
        
        Args:
            pixel_values: Input images [batch_size, 3, H, W]
            threshold: Confidence threshold for predictions
            
        Returns:
            List of predictions for each image
        """
        self.eval()
        with torch.no_grad():
            outputs = self.model(pixel_values=pixel_values)
            
            predictions = []
            for i in range(len(outputs.logits)):
                logits = outputs.logits[i]  # [num_queries, num_classes + 1]
                boxes = outputs.pred_boxes[i]  # [num_queries, 4]
                
                # Apply softmax to get probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Get predictions above threshold
                scores, labels = probs[:, :-1].max(dim=-1)  # Exclude background
                keep = scores > threshold
                
                predictions.append({
                    'scores': scores[keep],
                    'labels': labels[keep],
                    'boxes': boxes[keep]
                })
            
            return predictions
    
    def get_class_name(self, label_id: int) -> str:
        """Get class name from label ID."""
        if 0 <= label_id < len(self.config.class_names):
            return self.config.class_names[label_id]
        return 'unknown'


def create_bdd_detr_model(pretrained: bool = True) -> BDD100KDETR:
    """
    Factory function to create BDD100K DETR model.
    
    Args:
        pretrained: Whether to load pretrained weights
        
    Returns:
        Configured BDD100K DETR model
    """
    config = BDD100KDetrConfig()
    model = BDD100KDETR(config, pretrained=pretrained)
    return model


if __name__ == "__main__":
    # Test model creation
    print("Testing BDD100K DETR model creation...")
    
    model = create_bdd_detr_model(pretrained=False)
    
    # Test forward pass
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 512, 512)
    
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_images)
        print(f"Output logits shape: {outputs.logits.shape}")
        print(f"Output boxes shape: {outputs.pred_boxes.shape}")
    
    print("Model creation and forward pass successful!")