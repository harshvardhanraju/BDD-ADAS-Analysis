"""
Interactive Dashboard for BDD100K Dataset Analysis

This module provides a comprehensive interactive dashboard using Streamlit
for exploring and visualizing the BDD100K dataset analysis results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from parsers.bdd_parser import BDDParser
from analysis.class_analysis import ClassDistributionAnalyzer
from analysis.spatial_analysis import SpatialAnalyzer
from analysis.image_analysis import ImageCharacteristicsAnalyzer

# Page configuration
st.set_page_config(
    page_title="BDD100K Dataset Explorer",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .analysis-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_processed_data():
    """Load processed annotation data."""
    try:
        train_data = pd.read_csv("data/processed/train_annotations.csv")
        val_data = pd.read_csv("data/processed/val_annotations.csv")
        combined_data = pd.concat([train_data, val_data], ignore_index=True)
        return combined_data, train_data, val_data
    except FileNotFoundError:
        return None, None, None

@st.cache_data
def load_analysis_results():
    """Load pre-computed analysis results."""
    results = {}
    try:
        with open("data/analysis/reports/parsing_statistics.json", "r") as f:
            results['parsing_stats'] = json.load(f)
    except FileNotFoundError:
        results['parsing_stats'] = None
    
    return results

def create_overview_metrics(data):
    """Create overview metrics cards."""
    if data is None:
        st.error("No data available. Please run the BDD parser first.")
        return
    
    # Calculate key metrics
    total_images = data['image_name'].nunique()
    total_objects = len(data[data['category'].notna()])
    num_classes = data['category'].nunique()
    avg_objects_per_image = total_objects / total_images if total_images > 0 else 0
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{total_images:,}</h3>
                <p>Total Images</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{total_objects:,}</h3>
                <p>Total Objects</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{num_classes}</h3>
                <p>Object Classes</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            f"""
            <div class="metric-card">
                <h3>{avg_objects_per_image:.1f}</h3>
                <p>Avg Objects/Image</p>
            </div>
            """, 
            unsafe_allow_html=True
        )

def create_class_distribution_plots(data):
    """Create interactive class distribution plots."""
    st.subheader("🎯 Class Distribution Analysis")
    
    if data is None or data['category'].isna().all():
        st.warning("No class data available.")
        return
    
    # Filter out null categories
    class_data = data[data['category'].notna()]
    
    # Overall class distribution
    class_counts = class_data['category'].value_counts()
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Interactive bar plot
        fig = px.bar(
            x=class_counts.index,
            y=class_counts.values,
            title="Object Class Distribution",
            labels={'x': 'Object Class', 'y': 'Count'},
            color=class_counts.values,
            color_continuous_scale="viridis"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Pie chart for proportions
        fig_pie = px.pie(
            values=class_counts.values,
            names=class_counts.index,
            title="Class Proportions"
        )
        fig_pie.update_layout(height=500)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    # Split comparison
    st.subheader("📊 Train vs Validation Split Comparison")
    
    split_comparison = pd.crosstab(class_data['category'], class_data['split'])
    
    # Stacked bar chart
    fig_split = px.bar(
        split_comparison,
        title="Class Distribution by Split",
        labels={'value': 'Count', 'index': 'Class'},
        color_discrete_map={'train': '#1f77b4', 'val': '#ff7f0e'}
    )
    fig_split.update_layout(height=400)
    st.plotly_chart(fig_split, use_container_width=True)
    
    # Class imbalance analysis
    st.subheader("⚖️ Class Imbalance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Imbalance ratio
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        st.metric("Imbalance Ratio (Max/Min)", f"{imbalance_ratio:.2f}")
        
        # Gini coefficient
        counts = class_counts.values
        mean_count = np.mean(counts)
        gini = np.sum(np.abs(counts[:, None] - counts[None, :])) / (2 * len(counts)**2 * mean_count)
        st.metric("Gini Coefficient", f"{gini:.3f}")
    
    with col2:
        # Show most and least frequent classes
        st.write("**Most Frequent Classes:**")
        for i, (class_name, count) in enumerate(class_counts.head(3).items()):
            percentage = count / len(class_data) * 100
            st.write(f"{i+1}. {class_name}: {count:,} ({percentage:.1f}%)")
        
        st.write("**Least Frequent Classes:**")
        for i, (class_name, count) in enumerate(class_counts.tail(3).items()):
            percentage = count / len(class_data) * 100
            st.write(f"{i+1}. {class_name}: {count:,} ({percentage:.1f}%)")

def create_spatial_analysis_plots(data):
    """Create interactive spatial analysis plots."""
    st.subheader("🌐 Spatial Distribution Analysis")
    
    # Filter data with bounding boxes
    bbox_data = data.dropna(subset=['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])
    
    if len(bbox_data) == 0:
        st.warning("No bounding box data available.")
        return
    
    # Add computed columns
    bbox_data = bbox_data.copy()
    bbox_data['bbox_center_x'] = (bbox_data['bbox_x1'] + bbox_data['bbox_x2']) / 2
    bbox_data['bbox_center_y'] = (bbox_data['bbox_y1'] + bbox_data['bbox_y2']) / 2
    
    # Normalize coordinates (assuming typical BDD100K dimensions)
    bbox_data['norm_center_x'] = bbox_data['bbox_center_x'] / 1280
    bbox_data['norm_center_y'] = bbox_data['bbox_center_y'] / 720
    
    # Sample data for performance
    if len(bbox_data) > 5000:
        bbox_sample = bbox_data.sample(n=5000, random_state=42)
        st.info(f"Showing sample of 5,000 objects from {len(bbox_data):,} total objects")
    else:
        bbox_sample = bbox_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Spatial heatmap
        fig_heatmap = px.density_heatmap(
            bbox_sample,
            x='norm_center_x',
            y='norm_center_y',
            title='Object Center Heatmap',
            labels={'norm_center_x': 'Normalized X', 'norm_center_y': 'Normalized Y'},
            nbinsx=20,
            nbinsy=15
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # Scatter plot colored by class
        top_classes = bbox_data['category'].value_counts().head(5).index
        sample_top_classes = bbox_sample[bbox_sample['category'].isin(top_classes)]
        
        fig_scatter = px.scatter(
            sample_top_classes,
            x='norm_center_x',
            y='norm_center_y',
            color='category',
            title='Object Centers by Class (Top 5)',
            labels={'norm_center_x': 'Normalized X', 'norm_center_y': 'Normalized Y'},
            opacity=0.6
        )
        fig_scatter.update_layout(height=400)
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Bounding box dimension analysis
    st.subheader("📏 Bounding Box Dimensions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Width distribution
        fig_width = px.histogram(
            bbox_data,
            x='bbox_width',
            title='Bounding Box Width Distribution',
            nbins=50
        )
        fig_width.update_layout(height=300)
        st.plotly_chart(fig_width, use_container_width=True)
    
    with col2:
        # Height distribution
        fig_height = px.histogram(
            bbox_data,
            x='bbox_height',
            title='Bounding Box Height Distribution',
            nbins=50
        )
        fig_height.update_layout(height=300)
        st.plotly_chart(fig_height, use_container_width=True)
    
    with col3:
        # Area distribution
        fig_area = px.histogram(
            bbox_data,
            x='bbox_area',
            title='Bounding Box Area Distribution',
            nbins=50,
            log_y=True
        )
        fig_area.update_layout(height=300)
        st.plotly_chart(fig_area, use_container_width=True)

def create_class_specific_analysis(data):
    """Create class-specific analysis interface."""
    st.subheader("🔍 Class-Specific Analysis")
    
    if data is None or data['category'].isna().all():
        st.warning("No class data available.")
        return
    
    # Class selection
    available_classes = sorted(data['category'].dropna().unique())
    selected_class = st.selectbox("Select a class to analyze:", available_classes)
    
    if selected_class:
        class_data = data[data['category'] == selected_class]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Objects", f"{len(class_data):,}")
        
        with col2:
            st.metric("Unique Images", f"{class_data['image_name'].nunique():,}")
        
        with col3:
            avg_per_image = len(class_data) / class_data['image_name'].nunique()
            st.metric("Avg Objects/Image", f"{avg_per_image:.2f}")
        
        # Split distribution for selected class
        if 'split' in class_data.columns:
            split_dist = class_data['split'].value_counts()
            
            fig_split = px.bar(
                x=split_dist.index,
                y=split_dist.values,
                title=f"{selected_class} Distribution Across Splits",
                labels={'x': 'Split', 'y': 'Count'}
            )
            st.plotly_chart(fig_split, use_container_width=True)
        
        # Spatial analysis for selected class
        bbox_class_data = class_data.dropna(subset=['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2'])
        
        if len(bbox_class_data) > 0:
            bbox_class_data = bbox_class_data.copy()
            bbox_class_data['norm_center_x'] = (bbox_class_data['bbox_x1'] + bbox_class_data['bbox_x2']) / 2 / 1280
            bbox_class_data['norm_center_y'] = (bbox_class_data['bbox_y1'] + bbox_class_data['bbox_y2']) / 2 / 720
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Spatial heatmap for class
                fig_class_heat = px.density_heatmap(
                    bbox_class_data,
                    x='norm_center_x',
                    y='norm_center_y',
                    title=f'{selected_class} Spatial Distribution',
                    labels={'norm_center_x': 'Normalized X', 'norm_center_y': 'Normalized Y'}
                )
                st.plotly_chart(fig_class_heat, use_container_width=True)
            
            with col2:
                # Box dimensions for class
                dimensions_df = pd.DataFrame({
                    'Dimension': ['Width', 'Height', 'Area'],
                    'Mean': [
                        bbox_class_data['bbox_width'].mean(),
                        bbox_class_data['bbox_height'].mean(),
                        bbox_class_data['bbox_area'].mean()
                    ],
                    'Std': [
                        bbox_class_data['bbox_width'].std(),
                        bbox_class_data['bbox_height'].std(),
                        bbox_class_data['bbox_area'].std()
                    ]
                })
                
                fig_dims = px.bar(
                    dimensions_df,
                    x='Dimension',
                    y='Mean',
                    error_y='Std',
                    title=f'{selected_class} Average Dimensions'
                )
                st.plotly_chart(fig_dims, use_container_width=True)

def create_image_analysis_section(data):
    """Create image characteristics analysis section."""
    st.subheader("🖼️ Image Characteristics")
    
    # Basic image statistics
    unique_images = data[['split', 'image_name']].drop_duplicates()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Unique Images", f"{len(unique_images):,}")
        
        if 'split' in unique_images.columns:
            split_dist = unique_images['split'].value_counts()
            fig_img_split = px.pie(
                values=split_dist.values,
                names=split_dist.index,
                title="Images per Split"
            )
            st.plotly_chart(fig_img_split, use_container_width=True)
    
    with col2:
        # Objects per image distribution
        objects_per_image = data.groupby(['split', 'image_name']).size().reset_index(name='object_count')
        
        fig_obj_per_img = px.histogram(
            objects_per_image,
            x='object_count',
            title='Objects per Image Distribution',
            nbins=30
        )
        fig_obj_per_img.update_layout(height=400)
        st.plotly_chart(fig_obj_per_img, use_container_width=True)
    
    # Scene attributes analysis (if available)
    attr_columns = [col for col in data.columns if col.startswith('img_attr_')]
    
    if attr_columns:
        st.subheader("🌅 Scene Attributes")
        
        # Select attribute to analyze
        selected_attr = st.selectbox("Select scene attribute:", 
                                   [col.replace('img_attr_', '') for col in attr_columns])
        
        if selected_attr:
            attr_col = f'img_attr_{selected_attr}'
            attr_data = data[data[attr_col].notna()]
            
            if len(attr_data) > 0:
                attr_dist = attr_data[attr_col].value_counts()
                
                fig_attr = px.bar(
                    x=attr_dist.index,
                    y=attr_dist.values,
                    title=f'{selected_attr.title()} Distribution',
                    labels={'x': selected_attr.title(), 'y': 'Count'}
                )
                st.plotly_chart(fig_attr, use_container_width=True)

def create_data_quality_section(data):
    """Create data quality analysis section."""
    st.subheader("🔍 Data Quality Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Missing data analysis
        st.write("**Missing Data Analysis**")
        
        essential_columns = ['image_name', 'category', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
        available_columns = [col for col in essential_columns if col in data.columns]
        
        missing_stats = {}
        for col in available_columns:
            missing_count = data[col].isna().sum()
            missing_pct = missing_count / len(data) * 100
            missing_stats[col] = {'count': missing_count, 'percentage': missing_pct}
        
        missing_df = pd.DataFrame(missing_stats).T
        missing_df = missing_df.reset_index()
        missing_df.columns = ['Column', 'Missing Count', 'Missing Percentage']
        
        fig_missing = px.bar(
            missing_df,
            x='Column',
            y='Missing Percentage',
            title='Missing Data by Column',
            labels={'Missing Percentage': 'Missing %'}
        )
        st.plotly_chart(fig_missing, use_container_width=True)
    
    with col2:
        # Annotation consistency
        st.write("**Annotation Consistency**")
        
        # Images with/without objects
        images_with_objects = data[data['category'].notna()]['image_name'].nunique()
        total_images = data['image_name'].nunique()
        images_without_objects = total_images - images_with_objects
        
        consistency_data = pd.DataFrame({
            'Category': ['With Objects', 'Without Objects'],
            'Count': [images_with_objects, images_without_objects]
        })
        
        fig_consistency = px.pie(
            consistency_data,
            values='Count',
            names='Category',
            title='Images with/without Objects'
        )
        st.plotly_chart(fig_consistency, use_container_width=True)
    
    with col3:
        # Bounding box validity
        st.write("**Bounding Box Validity**")
        
        bbox_columns = ['bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
        if all(col in data.columns for col in bbox_columns):
            bbox_data = data.dropna(subset=bbox_columns)
            
            # Check for invalid bounding boxes
            invalid_bbox = bbox_data[
                (bbox_data['bbox_x2'] <= bbox_data['bbox_x1']) |
                (bbox_data['bbox_y2'] <= bbox_data['bbox_y1']) |
                (bbox_data['bbox_x1'] < 0) |
                (bbox_data['bbox_y1'] < 0)
            ]
            
            validity_data = pd.DataFrame({
                'Category': ['Valid', 'Invalid'],
                'Count': [len(bbox_data) - len(invalid_bbox), len(invalid_bbox)]
            })
            
            fig_validity = px.pie(
                validity_data,
                values='Count',
                names='Category',
                title='Bounding Box Validity'
            )
            st.plotly_chart(fig_validity, use_container_width=True)

def create_interactive_filters(data):
    """Create interactive filters for data exploration."""
    st.sidebar.subheader("🔧 Interactive Filters")
    
    # Split filter
    if 'split' in data.columns:
        splits = ['All'] + sorted(data['split'].unique().tolist())
        selected_split = st.sidebar.selectbox("Filter by Split:", splits)
        
        if selected_split != 'All':
            data = data[data['split'] == selected_split]
    
    # Class filter
    if 'category' in data.columns:
        classes = ['All'] + sorted(data['category'].dropna().unique().tolist())
        selected_classes = st.sidebar.multiselect("Filter by Classes:", classes, default=['All'])
        
        if 'All' not in selected_classes and selected_classes:
            data = data[data['category'].isin(selected_classes)]
    
    # Object count filter
    objects_per_image = data.groupby(['split', 'image_name']).size()
    if len(objects_per_image) > 0:
        min_objects = int(objects_per_image.min())
        max_objects = int(objects_per_image.max())
        
        if max_objects > min_objects:
            object_range = st.sidebar.slider(
                "Objects per Image Range:",
                min_value=min_objects,
                max_value=max_objects,
                value=(min_objects, max_objects)
            )
            
            # Filter data based on object count
            valid_images = objects_per_image[
                (objects_per_image >= object_range[0]) & 
                (objects_per_image <= object_range[1])
            ].index
            
            data = data.set_index(['split', 'image_name'])
            data = data.loc[data.index.isin(valid_images)]
            data = data.reset_index()
    
    return data

def main():
    """Main dashboard application."""
    # Header
    st.markdown('<h1 class="main-header">🚗 BDD100K Dataset Explorer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    combined_data, train_data, val_data = load_processed_data()
    
    if combined_data is None:
        st.error("""
        ❌ **No processed data found!**
        
        Please run the BDD parser first:
        1. Ensure the BDD100K dataset is downloaded
        2. Run the parser: `python src/parsers/bdd_parser.py`
        3. Refresh this dashboard
        """)
        st.stop()
    
    # Apply filters
    filtered_data = create_interactive_filters(combined_data)
    
    # Show data info
    st.sidebar.info(f"""
    **Filtered Dataset Info:**
    - Images: {filtered_data['image_name'].nunique():,}
    - Objects: {len(filtered_data[filtered_data['category'].notna()]):,}
    - Classes: {filtered_data['category'].nunique()}
    """)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🎯 Class Analysis", 
        "🌐 Spatial Analysis", 
        "🖼️ Image Analysis", 
        "🔍 Data Quality"
    ])
    
    with tab1:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        create_overview_metrics(filtered_data)
        
        # Quick insights
        st.subheader("📈 Quick Insights")
        
        if filtered_data['category'].notna().sum() > 0:
            class_data = filtered_data[filtered_data['category'].notna()]
            most_common_class = class_data['category'].mode().iloc[0]
            class_count = (class_data['category'] == most_common_class).sum()
            class_pct = class_count / len(class_data) * 100
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"🏆 **Most Common Class**: {most_common_class} ({class_count:,} objects, {class_pct:.1f}%)")
                
                # Top 5 classes
                top_classes = class_data['category'].value_counts().head(5)
                st.write("**Top 5 Classes:**")
                for i, (cls, count) in enumerate(top_classes.items()):
                    pct = count / len(class_data) * 100
                    st.write(f"{i+1}. {cls}: {count:,} ({pct:.1f}%)")
            
            with col2:
                # Dataset balance insights
                imbalance_ratio = top_classes.iloc[0] / top_classes.iloc[-1] if len(top_classes) > 1 else 1
                
                if imbalance_ratio > 10:
                    st.warning(f"⚠️ **High Class Imbalance**: {imbalance_ratio:.1f}:1 ratio")
                elif imbalance_ratio > 5:
                    st.info(f"ℹ️ **Moderate Class Imbalance**: {imbalance_ratio:.1f}:1 ratio")
                else:
                    st.success(f"✅ **Well Balanced Classes**: {imbalance_ratio:.1f}:1 ratio")
                
                # Objects per image stats
                obj_per_img = filtered_data.groupby(['split', 'image_name']).size()
                st.metric("Avg Objects/Image", f"{obj_per_img.mean():.1f}")
                st.metric("Max Objects/Image", f"{obj_per_img.max()}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        create_class_distribution_plots(filtered_data)
        create_class_specific_analysis(filtered_data)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        create_spatial_analysis_plots(filtered_data)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        create_image_analysis_section(filtered_data)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        create_data_quality_section(filtered_data)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: #666;'>
            BDD100K Dataset Explorer | Built with Streamlit | 
            <a href='https://github.com/your-repo/bdd-analysis' target='_blank'>View Source</a>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()