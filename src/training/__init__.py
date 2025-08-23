"""
Training modules for DETR on BDD100K dataset.
"""

from .detr_trainer import DETRTrainer, train_bdd_detr

__all__ = ['DETRTrainer', 'train_bdd_detr']