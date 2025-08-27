# Archived Source Files

This directory contains source files that have been superseded or are no longer actively used in the comprehensive 6-phase evaluation framework.

## Archived on: August 27, 2025

## Archived Files:

### Training Modules (Superseded)
- `detr_trainer.py` - Original DETR trainer class, superseded by `enhanced_detr_trainer.py`

## Rationale for Archiving:
- `detr_trainer.py` was superseded by the enhanced trainer with better class weighting, checkpoint management, and training strategies

## Active Training Module:
- `enhanced_detr_trainer.py` - Enhanced trainer with advanced features:
  - Improved class imbalance handling
  - Better checkpoint management
  - Differential learning rates
  - Enhanced loss computation with focal loss
  - Configurable checkpoint retention

## Note:
The enhanced trainer provides all functionality of the original trainer plus significant improvements for handling the BDD100K dataset's class imbalance challenges.