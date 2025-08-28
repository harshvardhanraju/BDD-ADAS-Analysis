# Complete BDD100K Dataset Analysis Report (10 Classes)
======================================================================

Generated from comprehensive analysis of BDD100K dataset with all 10 object detection classes.

## Dataset Overview
- **Total Annotations**: 1,472,397
- **Total Images**: 79,863
- **Number of Classes**: 10
- **Classes Found**: 10
- **Missing Classes**: []

## Complete 10-Class Distribution

| Rank | Class | Count | Percentage | Safety Weight | Category |
|------|-------|-------|------------|---------------|----------|
| 1 | car | 815,717 | 55.40% | 6.0 | vehicle |
| 2 | traffic sign | 274,594 | 18.65% | 6.0 | infrastructure |
| 3 | traffic light | 213,002 | 14.47% | 8.0 | infrastructure |
| 4 | pedestrian | 104,611 | 7.10% | 10.0 | safety_critical |
| 5 | truck | 34,216 | 2.32% | 7.0 | vehicle |
| 6 | bus | 13,269 | 0.90% | 7.0 | vehicle |
| 7 | bicycle | 8,217 | 0.56% | 9.0 | safety_critical |
| 8 | rider | 5,166 | 0.35% | 9.0 | safety_critical |
| 9 | motorcycle | 3,454 | 0.23% | 8.0 | safety_critical |
| 10 | train | 151 | 0.01% | 5.0 | vehicle |

## Class Imbalance Analysis
- **Imbalance Ratio**: 5402.10:1 (car vs train)
- **Most Frequent**: car (815,717 instances)
- **Least Frequent**: train (151 instances)
- **Gini Coefficient**: 0.719 (0=equal, 1=maximum inequality)
- **Entropy**: 1.878 (higher = more balanced)

## Safety-Critical Classes Analysis

Critical for autonomous driving safety:

- **pedestrian**:
  - Total instances: 104,611
  - Images with class: 25,296
  - Average per image: 4.14
  - Occlusion rate: 58.0%
  - Truncation rate: 3.5%

- **rider**:
  - Total instances: 5,166
  - Images with class: 4,101
  - Average per image: 1.26
  - Occlusion rate: 89.1%
  - Truncation rate: 4.9%

- **bicycle**:
  - Total instances: 8,217
  - Images with class: 4,921
  - Average per image: 1.67
  - Occlusion rate: 84.3%
  - Truncation rate: 8.5%

- **motorcycle**:
  - Total instances: 3,454
  - Images with class: 2,618
  - Average per image: 1.32
  - Occlusion rate: 76.3%
  - Truncation rate: 9.3%

## Generated Visualizations

- **Class Distribution**: `data/analysis/complete_analysis_10class/complete_10class_distribution.png`
- **Safety Analysis**: `data/analysis/complete_analysis_10class/safety_focused_analysis.png`
- **Environmental Analysis**: `data/analysis/complete_analysis_10class/environmental_analysis.png`
- **Imbalance Analysis**: `data/analysis/complete_analysis_10class/class_imbalance_analysis.png`
- **Split Comparison**: `data/analysis/complete_analysis_10class/split_comparison_analysis.png`