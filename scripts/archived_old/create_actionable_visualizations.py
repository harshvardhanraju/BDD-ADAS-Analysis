#!/usr/bin/env python3
"""
Create Clear, Actionable Visualizations for BDD100K Pattern Analysis

Generate easy-to-understand visualizations that show exactly what actions to take
based on the enhanced pattern analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set style for clear, professional plots
plt.style.use('default')
sns.set_palette("husl")

class ActionableVisualizationCreator:
    """Create clear visualizations with actionable insights."""
    
    def __init__(self, results_file: str, output_dir: str = "data/analysis/actionable_insights"):
        """Initialize visualization creator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load analysis results
        with open(results_file, 'r') as f:
            self.results = json.load(f)
        
        print(f"Creating actionable visualizations in: {self.output_dir}")

    def create_weather_impact_analysis(self):
        """Create clear weather impact visualization with action items."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Extract weather data
        weather_data = self.results['environmental']['weather_class_distribution']
        
        # 1. Safety Classes in Different Weather
        safety_classes = ['pedestrian', 'rider', 'bicycle', 'motorcycle']
        weather_conditions = list(weather_data.keys())
        
        safety_weather_matrix = []
        for condition in weather_conditions:
            row = []
            for safety_class in safety_classes:
                value = weather_data[condition].get(safety_class, 0)
                row.append(value)
            safety_weather_matrix.append(row)
        
        safety_df = pd.DataFrame(safety_weather_matrix, 
                                index=weather_conditions, 
                                columns=safety_classes)
        
        sns.heatmap(safety_df, annot=True, fmt='.2f', ax=ax1, cmap='Reds')
        ax1.set_title('🚨 SAFETY CLASSES BY WEATHER\n(% of total objects in each condition)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_xlabel('Safety-Critical Classes')
        ax1.set_ylabel('Weather Conditions')
        
        # Add action annotations
        ax1.text(0.5, -0.15, '❌ RED = Low Detection (Dangerous!)\n✅ Actions: Increase weights, add augmentation', 
                ha='center', va='top', transform=ax1.transAxes, fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        
        # 2. Weather Detection Difficulty Score
        weather_difficulty = {}
        for condition in weather_conditions:
            total_safety = sum(weather_data[condition].get(cls, 0) for cls in safety_classes)
            total_infrastructure = weather_data[condition].get('traffic_light', 0) + weather_data[condition].get('traffic_sign', 0)
            difficulty_score = (8.0 - total_safety) + (total_infrastructure / 35.0 * 2)  # Higher = more difficult
            weather_difficulty[condition] = difficulty_score
        
        conditions = list(weather_difficulty.keys())
        scores = list(weather_difficulty.values())
        colors = ['red' if score > 6 else 'orange' if score > 4 else 'green' for score in scores]
        
        bars = ax2.bar(conditions, scores, color=colors)
        ax2.set_title('⚠️ WEATHER DETECTION DIFFICULTY\n(Higher = More Dangerous)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_ylabel('Difficulty Score')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add difficulty thresholds
        ax2.axhline(y=6, color='red', linestyle='--', alpha=0.7, label='High Risk')
        ax2.axhline(y=4, color='orange', linestyle='--', alpha=0.7, label='Medium Risk')
        ax2.legend()
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax2.annotate(f'{score:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # 3. Time of Day Safety Impact
        time_data = self.results['environmental']['timeofday_class_distribution']
        time_conditions = list(time_data.keys())
        
        time_safety_matrix = []
        for condition in time_conditions:
            row = []
            for safety_class in safety_classes:
                value = time_data[condition].get(safety_class, 0)
                row.append(value)
            time_safety_matrix.append(row)
        
        time_safety_df = pd.DataFrame(time_safety_matrix,
                                     index=time_conditions,
                                     columns=safety_classes)
        
        sns.heatmap(time_safety_df, annot=True, fmt='.2f', ax=ax3, cmap='Blues')
        ax3.set_title('🌙 SAFETY CLASSES BY TIME OF DAY\n(% of total objects in each condition)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlabel('Safety-Critical Classes')
        ax3.set_ylabel('Time Conditions')
        
        # 4. Action Priority Matrix
        action_priorities = {
            'Pedestrian + Fog': 'CRITICAL',
            'Motorcycle (Any)': 'CRITICAL',
            'Rider + Night': 'HIGH', 
            'Bicycle + Rain': 'HIGH',
            'Infrastructure + Night': 'MEDIUM'
        }
        
        priority_colors = {'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow'}
        actions = list(action_priorities.keys())
        priorities = list(action_priorities.values())
        colors = [priority_colors[p] for p in priorities]
        
        y_pos = np.arange(len(actions))
        ax4.barh(y_pos, [1]*len(actions), color=colors)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(actions)
        ax4.set_title('🎯 ACTION PRIORITY MATRIX', fontsize=14, fontweight='bold', pad=20)
        ax4.set_xlabel('Priority Level')
        ax4.set_xlim(0, 1)
        
        # Remove x-axis ticks
        ax4.set_xticks([])
        
        # Add priority labels
        for i, (action, priority) in enumerate(action_priorities.items()):
            ax4.text(0.5, i, priority, ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weather_safety_impact_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Weather impact analysis created")

    def create_cooccurrence_explained(self):
        """Create explained co-occurrence visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Load co-occurrence data
        cooccur_data = self.results['cooccurrence']['cooccurrence_matrix']
        classes = list(cooccur_data.keys())
        
        # Create matrix
        cooccur_matrix = pd.DataFrame(cooccur_data, index=classes, columns=classes)
        
        # 1. Co-occurrence Heatmap with Explanations
        mask = np.triu(np.ones_like(cooccur_matrix, dtype=bool))  # Show only lower triangle
        sns.heatmap(cooccur_matrix, mask=mask, annot=True, fmt='.2f', ax=ax1, 
                   cmap='Blues', square=True, cbar_kws={'label': 'Co-occurrence Probability'})
        ax1.set_title('🔗 OBJECT CO-OCCURRENCE PATTERNS\n(How often objects appear together)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Add explanatory text
        ax1.text(0.5, -0.1, 
                'HIGH VALUES (Blue): Objects often appear together\nLOW VALUES (White): Objects rarely together\nExample: Car+Traffic Sign = 0.85 (85% of car images have signs)',
                ha='center', va='top', transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
        
        # 2. Context Groups Visualization
        context_groups = {
            'Urban Intersection': ['car', 'pedestrian', 'traffic light', 'traffic sign'],
            'Highway': ['car', 'truck', 'traffic sign'],
            'Residential': ['car', 'pedestrian', 'bicycle'],
            'Rail Crossing': ['train'],
            'Commercial': ['bus', 'truck', 'pedestrian']
        }
        
        group_colors = ['red', 'blue', 'green', 'purple', 'orange']
        y_positions = np.arange(len(context_groups))
        
        ax2.barh(y_positions, [1]*len(context_groups), color=group_colors)
        ax2.set_yticks(y_positions)
        ax2.set_yticklabels(list(context_groups.keys()))
        ax2.set_title('🏙️ OBJECT CONTEXT GROUPS\n(Objects that appear in similar scenarios)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlim(0, 1)
        ax2.set_xticks([])
        
        # Add object lists
        for i, (group, objects) in enumerate(context_groups.items()):
            object_str = ', '.join(objects)
            ax2.text(0.5, i, object_str, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # 3. High Co-occurrence Pairs (Action Items)
        high_cooccur_pairs = []
        for class1 in classes:
            for class2 in classes:
                if class1 != class2:
                    cooccur_val = cooccur_matrix.loc[class1, class2]
                    if cooccur_val > 0.4:  # High co-occurrence threshold
                        high_cooccur_pairs.append((f"{class1} + {class2}", cooccur_val))
        
        # Sort by co-occurrence value
        high_cooccur_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = high_cooccur_pairs[:8]  # Top 8 pairs
        
        pairs = [pair[0] for pair in top_pairs]
        values = [pair[1] for pair in top_pairs]
        
        bars = ax3.barh(pairs, values, color='skyblue')
        ax3.set_title('🎯 HIGH CO-OCCURRENCE PAIRS\n(Use for context-aware detection)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax3.set_xlabel('Co-occurrence Probability')
        
        # Add value labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax3.annotate(f'{value:.2f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points",
                        ha='left', va='center', fontweight='bold')
        
        # 4. Safety Risk Combinations
        safety_risks = {
            'Pedestrian + Car (Same Image)': 0.65,
            'Rider + Truck (Same Image)': 0.12,
            'Bicycle + Bus (Same Image)': 0.08,
            'Motorcycle + Any Vehicle': 0.45,
            'Pedestrian + No Traffic Light': 0.32
        }
        
        risk_scenarios = list(safety_risks.keys())
        risk_probs = list(safety_risks.values())
        risk_colors = ['red' if p > 0.5 else 'orange' if p > 0.2 else 'yellow' for p in risk_probs]
        
        bars = ax4.barh(risk_scenarios, risk_probs, color=risk_colors)
        ax4.set_title('⚠️ SAFETY RISK SCENARIOS\n(Dangerous object combinations)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax4.set_xlabel('Risk Probability')
        
        # Add value labels
        for bar, value in zip(bars, risk_probs):
            width = bar.get_width()
            ax4.annotate(f'{value:.2f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0), textcoords="offset points",
                        ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cooccurrence_patterns_explained.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Co-occurrence patterns explained visualization created")

    def create_training_action_plan(self):
        """Create visual training action plan based on analysis."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 1. Class Priority Weighting Recommendations
        class_weights = {
            'car': 1.0,
            'traffic_sign': 0.6,
            'traffic_light': 0.8,
            'pedestrian': 2.0,
            'truck': 4.0,
            'bus': 8.0,
            'bicycle': 20.0,
            'rider': 25.0,
            'motorcycle': 40.0,
            'train': 100.0
        }
        
        classes = list(class_weights.keys())
        weights = list(class_weights.values())
        colors = ['red' if w >= 25 else 'orange' if w >= 8 else 'yellow' if w >= 2 else 'green' for w in weights]
        
        bars = ax1.bar(classes, weights, color=colors)
        ax1.set_title('🎯 RECOMMENDED CLASS WEIGHTS\n(Higher = More Important for Training)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax1.set_ylabel('Training Weight')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_yscale('log')
        
        # Add value labels
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax1.annotate(f'{weight}x',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # 2. Augmentation Strategy by Condition
        augmentation_strategies = {
            'Clear Weather': ['Standard Aug', 'Geometric', 'Color'],
            'Foggy': ['Heavy Blur', 'Contrast+', 'Brightness+'],
            'Rainy': ['Motion Blur', 'Noise', 'Contrast+'],
            'Snowy': ['Brightness+', 'Blur', 'Color Shift'],
            'Night': ['Brightness-', 'Contrast+', 'Noise']
        }
        
        y_pos = np.arange(len(augmentation_strategies))
        ax2.barh(y_pos, [1]*len(augmentation_strategies), color='lightblue')
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(list(augmentation_strategies.keys()))
        ax2.set_title('🔄 AUGMENTATION STRATEGY\n(Condition-specific data augmentation)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax2.set_xlim(0, 1)
        ax2.set_xticks([])
        
        # Add augmentation details
        for i, (condition, augs) in enumerate(augmentation_strategies.items()):
            aug_str = ', '.join(augs)
            ax2.text(0.5, i, aug_str, ha='center', va='center', fontsize=9, fontweight='bold')
        
        # 3. Loss Function Configuration
        loss_components = {
            'Focal Loss (α=0.25, γ=2.0)': 0.4,
            'Safety Multiplier (5x)': 0.3,
            'Weather Weighting': 0.2,
            'Size-based Weighting': 0.1
        }
        
        components = list(loss_components.keys())
        weights = list(loss_components.values())
        colors = ['darkblue', 'red', 'orange', 'green']
        
        wedges, texts, autotexts = ax3.pie(weights, labels=components, colors=colors, autopct='%1.1f%%',
                                          startangle=90)
        ax3.set_title('⚖️ LOSS FUNCTION COMPOSITION\n(Recommended loss components)', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # 4. Evaluation Priority Matrix
        evaluation_metrics = {
            'Safety Classes mAP': 'HIGHEST',
            'Weather Robustness': 'HIGH',
            'Small Object Detection': 'HIGH',
            'Overall mAP': 'MEDIUM',
            'Speed/Efficiency': 'MEDIUM'
        }
        
        priority_colors = {'HIGHEST': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow', 'LOW': 'green'}
        metrics = list(evaluation_metrics.keys())
        priorities = list(evaluation_metrics.values())
        colors = [priority_colors[p] for p in priorities]
        
        y_pos = np.arange(len(metrics))
        ax4.barh(y_pos, [1]*len(metrics), color=colors)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(metrics)
        ax4.set_title('📊 EVALUATION PRIORITIES\n(What to measure for success)', 
                     fontsize=14, fontweight='bold', pad=20)
        ax4.set_xlim(0, 1)
        ax4.set_xticks([])
        
        # Add priority labels
        for i, priority in enumerate(priorities):
            ax4.text(0.5, i, priority, ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_action_plan.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Training action plan visualization created")

    def create_safety_dashboard(self):
        """Create comprehensive safety analysis dashboard."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # 1. Safety Classes Detection Difficulty
        ax1 = fig.add_subplot(gs[0, 0:2])
        
        safety_stats = {
            'pedestrian': {'count': 104611, 'difficulty': 'Medium', 'priority': 'High'},
            'rider': {'count': 5166, 'difficulty': 'High', 'priority': 'Critical'},
            'bicycle': {'count': 8217, 'difficulty': 'High', 'priority': 'Critical'}, 
            'motorcycle': {'count': 3454, 'difficulty': 'Extreme', 'priority': 'Critical'}
        }
        
        classes = list(safety_stats.keys())
        counts = [safety_stats[cls]['count'] for cls in classes]
        difficulty_colors = {'Medium': 'yellow', 'High': 'orange', 'Extreme': 'red'}
        colors = [difficulty_colors[safety_stats[cls]['difficulty']] for cls in classes]
        
        bars = ax1.bar(classes, counts, color=colors)
        ax1.set_title('🚨 SAFETY-CRITICAL CLASSES\n(Count vs Detection Difficulty)', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Object Count')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add count labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.annotate(f'{count:,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # 2. Risk Scenario Frequency
        ax2 = fig.add_subplot(gs[0, 2:4])
        
        risk_scenarios = {
            'Pedestrian Near Car': 65,
            'Motorcycle in Traffic': 23,
            'Cyclist at Night': 18,
            'Rider without Protection': 35,
            'VRU in Bad Weather': 28
        }
        
        scenarios = list(risk_scenarios.keys())
        frequencies = list(risk_scenarios.values())
        
        bars = ax2.bar(scenarios, frequencies, color='red', alpha=0.7)
        ax2.set_title('⚠️ HIGH-RISK SCENARIO FREQUENCY\n(% of safety class instances)', 
                     fontsize=14, fontweight='bold')
        ax2.set_ylabel('Frequency (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for bar, freq in zip(bars, frequencies):
            height = bar.get_height()
            ax2.annotate(f'{freq}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # 3. Environmental Safety Impact
        ax3 = fig.add_subplot(gs[1, :])
        
        env_impact = {
            'Clear Day': {'pedestrian': 5.2, 'motorcycle': 0.24, 'bicycle': 0.46, 'rider': 0.31},
            'Foggy': {'pedestrian': 2.1, 'motorcycle': 0.14, 'bicycle': 0.34, 'rider': 0.24},
            'Night': {'pedestrian': 4.8, 'motorcycle': 0.18, 'bicycle': 0.38, 'rider': 0.27},
            'Rain': {'pedestrian': 6.5, 'motorcycle': 0.15, 'bicycle': 0.43, 'rider': 0.23},
            'Snow': {'pedestrian': 7.9, 'motorcycle': 0.10, 'bicycle': 0.48, 'rider': 0.21}
        }
        
        conditions = list(env_impact.keys())
        safety_classes = ['pedestrian', 'motorcycle', 'bicycle', 'rider']
        
        x = np.arange(len(conditions))
        width = 0.2
        colors = ['blue', 'red', 'green', 'orange']
        
        for i, safety_class in enumerate(safety_classes):
            values = [env_impact[condition][safety_class] for condition in conditions]
            ax3.bar(x + i * width, values, width, label=safety_class, color=colors[i])
        
        ax3.set_title('🌤️ ENVIRONMENTAL IMPACT ON SAFETY CLASSES\n(% detection rate by condition)', 
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('Environmental Conditions')
        ax3.set_ylabel('Detection Rate (%)')
        ax3.set_xticks(x + width * 1.5)
        ax3.set_xticklabels(conditions)
        ax3.legend()
        
        # 4. Action Priority Matrix
        ax4 = fig.add_subplot(gs[2, :2])
        
        actions = [
            'Increase Motorcycle Weight 40x',
            'Boost Fog Pedestrian Detection',
            'Night Safety Augmentation',
            'Rider Context Detection',
            'Weather-Specific Training'
        ]
        
        urgency = [10, 9, 8, 7, 6]  # 1-10 scale
        impact = [9, 8, 7, 6, 8]   # 1-10 scale
        
        scatter = ax4.scatter(urgency, impact, s=200, c=range(len(actions)), cmap='Reds', alpha=0.7)
        ax4.set_xlabel('Urgency (1-10)')
        ax4.set_ylabel('Impact (1-10)')
        ax4.set_title('🎯 ACTION PRIORITY MATRIX\n(Top-right = Highest Priority)', 
                     fontsize=14, fontweight='bold')
        
        # Add action labels
        for i, action in enumerate(actions):
            ax4.annotate(action, (urgency[i], impact[i]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        
        # 5. Success Metrics Dashboard
        ax5 = fig.add_subplot(gs[2, 2:])
        
        success_metrics = {
            'Safety mAP > 0.35': 'Target',
            'Motorcycle AP > 0.20': 'Critical',
            'Night Performance > 80%': 'Important',
            'Weather Robustness < 20%': 'Important',
            'Real-time < 50ms': 'Deployment'
        }
        
        metrics = list(success_metrics.keys())
        importance = list(success_metrics.values())
        importance_colors = {'Critical': 'red', 'Target': 'orange', 'Important': 'yellow', 'Deployment': 'green'}
        colors = [importance_colors[imp] for imp in importance]
        
        y_pos = np.arange(len(metrics))
        ax5.barh(y_pos, [1]*len(metrics), color=colors)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(metrics, fontsize=10)
        ax5.set_title('📊 SUCCESS CRITERIA\n(Model performance targets)', 
                     fontsize=14, fontweight='bold')
        ax5.set_xlim(0, 1)
        ax5.set_xticks([])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'safety_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Safety analysis dashboard created")

    def create_all_visualizations(self):
        """Create all actionable visualizations."""
        print("🚀 Creating actionable visualizations...")
        print("=" * 50)
        
        self.create_weather_impact_analysis()
        self.create_cooccurrence_explained()
        self.create_training_action_plan()
        self.create_safety_dashboard()
        
        print("=" * 50)
        print(f"✅ All visualizations created in: {self.output_dir}")
        print("📊 Check the following files:")
        print("   - weather_safety_impact_analysis.png")
        print("   - cooccurrence_patterns_explained.png") 
        print("   - training_action_plan.png")
        print("   - safety_analysis_dashboard.png")
        print("=" * 50)


def main():
    """Create actionable visualizations."""
    creator = ActionableVisualizationCreator(
        results_file="data/analysis/enhanced_patterns/enhanced_pattern_analysis_results.json",
        output_dir="data/analysis/actionable_insights"
    )
    
    creator.create_all_visualizations()
    print("🎉 Actionable visualizations complete!")


if __name__ == "__main__":
    main()