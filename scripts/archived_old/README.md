# Archived Scripts and Files

This directory contains scripts and files that have been superseded by the comprehensive 6-phase evaluation framework or are no longer actively used in the production pipeline.

## Archived on: August 27, 2025

## Archived Scripts:

### Training Scripts (Superseded)
- `train_detr.py` - Basic DETR training script, superseded by `train_complete_10class_detr.py`
- `train_production_10class_detr.py` - Production training script, integrated into complete version

### Analysis Scripts (Integrated into 6-Phase Framework)
- `run_data_analysis.py` - Data analysis runner, superseded by comprehensive evaluation pipeline
- `run_qualitative_analysis.py` - Qualitative analysis, integrated into Phase 3 failure analysis
- `comprehensive_outlier_analysis.py` - Outlier analysis, specialized script not in main pipeline
- `enhanced_pattern_analysis.py` - Pattern analysis, integrated into Phase 4
- `create_actionable_visualizations.py` - Visualization creation, integrated into Phase 2
- `generate_10class_plots.py` - Class-specific plotting, superseded by comprehensive visualization

### Evaluation Scripts (Integrated)
- `evaluate_detr.py` - Basic model evaluation, superseded by comprehensive evaluation
- `optimize_thresholds.py` - Threshold optimization, specialized tool not in main pipeline

### Demo/Tutorial Scripts
- `example_usage.py` - Usage examples, tutorial script not needed in production

## Current Active Pipeline Scripts:
1. `train_complete_10class_detr.py` - Complete training pipeline with checkpoint fixes
2. `run_comprehensive_evaluation.py` - Phase 1: Quantitative evaluation
3. `generate_evaluation_visualizations.py` - Phase 2: Visualizations
4. `test_failure_analysis.py` - Phase 3: Failure analysis
5. `run_phase4_analysis.py` - Phase 4: Advanced clustering
6. `generate_comprehensive_report.py` - Phase 5: Comprehensive reporting
7. `generate_improvement_recommendations.py` - Phase 6: Improvement recommendations
8. `verify_complete_framework.py` - Framework verification

## Rationale for Archiving:
- Scripts were either superseded by more comprehensive versions
- Individual analysis components were integrated into the unified 6-phase evaluation framework
- Scripts were demo/tutorial in nature and not part of the production pipeline
- Scripts represented earlier iterations of functionality now handled by enhanced versions

## Note:
These files are preserved for reference but are not maintained or updated. The comprehensive 6-phase evaluation framework provides all functionality in a unified, production-ready manner.