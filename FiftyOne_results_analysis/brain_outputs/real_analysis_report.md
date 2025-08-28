# Real FiftyOne Brain Analysis Report

**Generated:** 2025-08-28 15:24:38.335187
**Dataset:** bdd100k_real_analysis

## Dataset Statistics

- **Total Images:** 3
- **Total Objects:** 52
- **Safety-Critical Objects:** 1
- **Classes Present:** car, motorcycle, rider, traffic light, traffic sign

## Analysis Features

### Real Data Processing
- ✅ Actual BDD100K images loaded
- ✅ Real model predictions generated
- ✅ BLIP2 embeddings computed
- ✅ FiftyOne Brain analysis performed

### Brain Analysis Results
- ✅ gt_viz
- ✅ pred_viz

### Available Views
- 👁️ safety_critical
- 👁️ high_conf_predictions
- 👁️ low_conf_predictions
- 👁️ car_objects
- 👁️ pedestrian_objects
- 👁️ traffic_sign_objects
- 👁️ traffic_light_objects

## Interactive Exploration

The FiftyOne App provides interactive exploration with:
- Ground truth vs predictions comparison
- BLIP2 embedding-based clustering
- Safety-critical object filtering
- Brain-powered similarity search
- Mistake detection and analysis

## Usage

```python
import fiftyone as fo
dataset = fo.load_dataset('bdd100k_real_analysis')
session = fo.launch_app(dataset)
```
