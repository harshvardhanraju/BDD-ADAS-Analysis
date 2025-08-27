"""
BDD100K Model Evaluation Visualization Package

This package provides comprehensive visualization tools for BDD100K object detection models,
including ground truth vs prediction comparisons, failure analysis, and performance dashboards.
"""

from .detection_viz import DetectionVisualizer

__all__ = [
    'DetectionVisualizer'
]