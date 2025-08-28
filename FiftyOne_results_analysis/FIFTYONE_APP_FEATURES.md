# FiftyOne App Interactive Features

This document explains what you can do with the FiftyOne App for interactive analysis of your BDD100K object detection results.

## 🌐 Accessing the FiftyOne App

After running the real analysis script:
```bash
python3 FiftyOne_results_analysis/scripts/real_analysis_with_blip2.py --subset 20
```

The FiftyOne App launches automatically at: **http://localhost:5151**

## 🎯 Interactive Analysis Features

### 1. **Ground Truth vs Predictions Visualization**
- **Toggle Views**: Switch between ground truth and model predictions
- **Side-by-Side Comparison**: See both GT and predictions simultaneously
- **Confidence Overlay**: View prediction confidence scores
- **Class-Specific Filtering**: Focus on specific object types

### 2. **Embedding-Based Clustering**
- **2D Visualization**: UMAP embeddings displayed as interactive scatter plot
- **Semantic Clusters**: Objects grouped by visual/semantic similarity
- **Click to Explore**: Click points to see corresponding images
- **Cluster Analysis**: Understand natural groupings in your data

### 3. **Advanced Filtering and Search**
- **Safety-Critical Filter**: Show only pedestrians, riders, bicycles, motorcycles
- **Confidence Filtering**: Adjust confidence thresholds interactively
- **Size-Based Filtering**: Focus on small, medium, or large objects
- **Class-Specific Views**: Filter by object category

### 4. **Similarity Search**
- **Visual Similarity**: Find objects that look similar to selected example
- **Performance Similarity**: Group objects with similar detection accuracy
- **Hard Example Mining**: Identify challenging cases for improvement
- **Pattern Discovery**: Reveal systematic model behaviors

### 5. **Mistake Detection and Analysis**
- **False Positive Highlighting**: Predictions without ground truth matches
- **False Negative Detection**: Ground truth objects not detected by model
- **Localization Errors**: Correct class but wrong bounding box location
- **Confidence Calibration**: Compare predicted vs actual confidence

## 🔍 How to Use the Interactive Features

### Basic Navigation
1. **Image Grid**: Browse through your dataset images
2. **Sample View**: Click any image to see detailed annotations
3. **Sidebar Filters**: Apply various filters to focus your analysis
4. **Tag System**: Mark interesting examples for further investigation

### Clustering Exploration
1. **Embeddings Tab**: Switch to embedding visualization
2. **2D Plot**: Explore UMAP clusters of your objects
3. **Point Selection**: Click points to highlight corresponding images
4. **Zoom and Pan**: Navigate through the embedding space

### Performance Analysis
1. **Filter by Confidence**: Use slider to adjust confidence thresholds
2. **Class Performance**: Compare performance across different object types
3. **Error Analysis**: Focus on false positives/negatives
4. **Safety Metrics**: Analyze critical object detection accuracy

### Comparative Analysis
1. **Ground Truth View**: See actual annotations
2. **Predictions View**: View model predictions
3. **Overlay Mode**: Show both simultaneously
4. **Diff Analysis**: Highlight differences between GT and predictions

## 📊 Specific Views Available

### Pre-configured Views
- **`safety_critical`**: Only pedestrians, riders, bicycles, motorcycles
- **`high_conf_predictions`**: Predictions with confidence > 0.5
- **`low_conf_predictions`**: Predictions with confidence < 0.1
- **`car_objects`**: Car detections only
- **`pedestrian_objects`**: Pedestrian detections only
- **`traffic_sign_objects`**: Traffic sign detections only
- **`traffic_light_objects`**: Traffic light detections only

### Custom Views
Create your own views using FiftyOne's query language:
```python
# Find small objects
small_objects = dataset.filter_labels(
    "ground_truth",
    fo.ViewField("bounding_box")[2] * fo.ViewField("bounding_box")[3] < 0.01
)

# Find low-confidence safety-critical predictions
risky_predictions = dataset.filter_labels(
    "predictions",
    (fo.ViewField("is_safety_critical") == True) & (fo.ViewField("confidence") < 0.3)
)
```

## 🧠 Brain Analysis Integration

### UMAP Embeddings
- **Interactive Scatter Plot**: 2D visualization of high-dimensional embeddings
- **Cluster Coloring**: Points colored by object class or performance
- **Selection Tools**: Select regions to filter corresponding images
- **Zoom Capabilities**: Explore dense regions in detail

### Similarity Analysis
- **Query by Example**: Select an object to find similar ones
- **Similarity Scores**: Quantitative similarity metrics
- **Ranked Results**: Similar objects sorted by similarity score
- **Visual Comparison**: Side-by-side view of similar objects

### Mistake Detection
- **Mistakenness Scores**: Automated error likelihood assessment
- **Ranked Error List**: Potential mistakes sorted by likelihood
- **Visual Verification**: Quick review of flagged cases
- **Annotation Correction**: Tools for fixing identified errors

## 💡 Analysis Workflows

### 1. **Model Performance Review**
1. Start with high-confidence predictions view
2. Compare against ground truth
3. Identify systematic errors
4. Focus on safety-critical misses

### 2. **Hard Example Mining**
1. Use similarity search on challenging examples
2. Find similar difficult cases
3. Understand common failure patterns
4. Target these patterns for improvement

### 3. **Data Quality Assessment**
1. Use mistake detection to find annotation errors
2. Review flagged examples manually
3. Correct or remove problematic annotations
4. Improve dataset quality

### 4. **Safety Analysis**
1. Filter to safety-critical objects only
2. Analyze miss rates for vulnerable road users
3. Identify high-risk scenarios
4. Adjust model thresholds for safety

### 5. **Cluster Analysis**
1. Explore embedding clusters
2. Understand natural data groupings
3. Identify outliers and edge cases
4. Discover data distribution patterns

## 🎮 Interactive Controls

### Mouse Controls
- **Click**: Select objects/images
- **Drag**: Pan in embedding view
- **Scroll**: Zoom in/out
- **Double-click**: Open detailed view

### Keyboard Shortcuts
- **Arrow Keys**: Navigate between images
- **Space**: Toggle prediction overlay
- **Enter**: Open sample details
- **Esc**: Close detailed views

### Filter Controls
- **Sliders**: Adjust confidence thresholds
- **Checkboxes**: Toggle object classes
- **Text Search**: Find specific samples
- **Date/Time**: Filter by metadata

## 🔧 Advanced Features

### Export Capabilities
- **Filtered Datasets**: Export subsets for further analysis
- **Annotation Formats**: Convert to various annotation formats
- **Image Crops**: Extract object patches for training
- **Metadata Export**: Save analysis results as CSV/JSON

### Integration Options
- **Jupyter Notebooks**: Embed FiftyOne in notebooks
- **Python Scripts**: Programmatic access to all features
- **Custom Plugins**: Extend functionality with plugins
- **API Access**: REST API for external integration

### Collaboration Features
- **Shared Sessions**: Multiple users can view same dataset
- **Annotations**: Add notes and tags collaboratively
- **Export Reports**: Generate analysis summaries
- **Version Control**: Track dataset and annotation changes

## 🚀 Getting Started Tips

1. **Start Small**: Begin with 10-20 images to familiarize yourself
2. **Explore Views**: Try different pre-configured views to understand capabilities
3. **Use Similarity**: Select interesting objects and find similar ones
4. **Focus on Errors**: Use mistake detection to find systematic issues
5. **Iterate**: Apply insights to improve your model, then re-analyze

---

The FiftyOne App transforms static analysis into an interactive exploration experience, enabling you to discover insights that traditional metrics miss and make targeted improvements to your object detection model.

**Ready to explore? Launch the app and start discovering! 🎉**