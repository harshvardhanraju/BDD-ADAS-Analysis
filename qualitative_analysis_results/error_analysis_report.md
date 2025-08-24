# DETR Model Error Analysis Report
============================================================
Analysis Date: 2025-08-24 11:24:51
Images Analyzed: 100

## Overall Statistics
- Total Predictions: 9,998
- Total Ground Truth: 1,582
- Precision: 0.030
- Recall: 0.131

## Error Breakdown
- False Positives: 9,699 (97.0%)
- False Negatives: 1,375 (86.9%)
- Classification Errors: 28
- Localization Errors: 256

## False Positives by Class
*(What model incorrectly detects)*

- **car**: 9699 false positives (avg confidence: 0.153)

## False Negatives by Class
*(What model fails to detect)*

- **car**: 781 missed detections
- **traffic_sign**: 294 missed detections
- **traffic_light**: 224 missed detections
- **truck**: 53 missed detections
- **bus**: 16 missed detections
- **rider**: 7 missed detections

## Most Common Classification Confusions
*(What classes get confused with each other)*

- truck → car: 15 times
- traffic_sign → car: 6 times
- bus → car: 6 times
- traffic_light → car: 1 times

## Key Insights & Recommendations

### Model is Over-Detecting
- **Issue**: More false positives than false negatives
- **Recommendation**: Increase confidence threshold or apply stricter NMS

### Low Precision Alert
- **Current Precision**: 0.030
- **Recommendation**: Focus on reducing false positives

### Low Recall Alert
- **Current Recall**: 0.131
- **Recommendation**: Focus on reducing false negatives
