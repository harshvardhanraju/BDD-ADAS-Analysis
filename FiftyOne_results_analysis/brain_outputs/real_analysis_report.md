# Real FiftyOne Brain Analysis Report

**Generated:** 2025-08-29 14:15:20.691399
**Dataset:** bdd100k_real_analysis

## Dataset Statistics

- **Total Images:** 10
- **Total Objects:** 170
- **Safety-Critical Objects:** 5
- **Classes Present:** bus, car, motorcycle, pedestrian, rider, traffic light, traffic sign

## Analysis Features

### Real Data Processing
- ✅ Actual BDD100K images loaded
- ✅ Real model predictions generated
- ✅ BLIP2 embeddings computed
- ✅ FiftyOne Brain analysis performed

### Brain Analysis Results
- ✅ gt_viz
- ✅ pred_viz
- ✅ similarity

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
