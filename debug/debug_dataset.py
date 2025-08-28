#!/usr/bin/env python3
"""
Debug dataset to see raw annotation values before/after processing.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def debug_annotations():
    print("🔍 Debugging Raw Annotations")
    print("=" * 60)
    
    # Load raw annotations CSV
    csv_path = "data/analysis/processed_10class_corrected/val_annotations_10class.csv"
    df = pd.read_csv(csv_path)
    
    print(f"Loaded {len(df)} annotations from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    
    # Check first few rows
    print("\nFirst 5 annotations:")
    for i in range(5):
        row = df.iloc[i]
        print(f"  Row {i}: {row['image_name']}")
        print(f"    bbox: x1={row['bbox_x1']}, y1={row['bbox_y1']}, x2={row['bbox_x2']}, y2={row['bbox_y2']}")
        print(f"    dims: width={row['bbox_x2']-row['bbox_x1']:.1f}, height={row['bbox_y2']-row['bbox_y1']:.1f}")
        print(f"    class: {row['category']} (id: {row.get('class_id', 'N/A')})")
    
    # Check coordinate ranges
    print(f"\nCoordinate ranges across all annotations:")
    print(f"  x1: [{df['bbox_x1'].min():.1f}, {df['bbox_x1'].max():.1f}]")
    print(f"  y1: [{df['bbox_y1'].min():.1f}, {df['bbox_y1'].max():.1f}]")
    print(f"  x2: [{df['bbox_x2'].min():.1f}, {df['bbox_x2'].max():.1f}]")
    print(f"  y2: [{df['bbox_y2'].min():.1f}, {df['bbox_y2'].max():.1f}]")
    
    # Check box sizes
    widths = df['bbox_x2'] - df['bbox_x1']
    heights = df['bbox_y2'] - df['bbox_y1']
    print(f"  widths: [{widths.min():.1f}, {widths.max():.1f}], mean: {widths.mean():.1f}")
    print(f"  heights: [{heights.min():.1f}, {heights.max():.1f}], mean: {heights.mean():.1f}")
    
    # Check if coordinates are normalized or absolute
    max_coord = max(df['bbox_x2'].max(), df['bbox_y2'].max())
    print(f"\\nMax coordinate: {max_coord:.1f}")
    if max_coord <= 1.0:
        print("  ✅ Coordinates appear to be normalized (0-1)")
    elif max_coord <= 10:
        print("  ⚠️ Coordinates might be in unusual format")
    else:
        print("  ❌ Coordinates appear to be absolute pixels (not normalized)")
    
    # Check for any out-of-bounds boxes
    invalid_boxes = df[(df['bbox_x1'] >= df['bbox_x2']) | (df['bbox_y1'] >= df['bbox_y2'])]
    print(f"\\nInvalid boxes (x1>=x2 or y1>=y2): {len(invalid_boxes)}")
    
    # Check class distribution
    print(f"\\nClass distribution:")
    class_counts = df['category'].value_counts()
    for class_name, count in class_counts.head(10).items():
        print(f"  {class_name}: {count}")
    
    print("\\n" + "=" * 60)

if __name__ == "__main__":
    debug_annotations()