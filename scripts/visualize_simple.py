#!/usr/bin/env python3
"""
Simple visualization script for Rocket TVC training analysis.
Fixed version that uses modern TensorBoard EventAccumulator instead of tensorflow.compat
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def read_tensorboard_logs(log_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    """
    Read TensorBoard logs using modern EventAccumulator (no tensorflow.compat needed).
    This fixes the "module 'tensorflow' has no attribute 'compat'" error.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("❌ TensorBoard not installed. Install with: pip install tensorboard")
        return {}
    
    log_path = Path(log_dir)
    tb_files = list(log_path.glob("**/events.out.tfevents.*"))
    
    if not tb_files:
        print(f"❌ No TensorBoard files found in {log_dir}")
        return {}
    
    print(f"📊 Found {len(tb_files)} TensorBoard files")
    
    metrics = {}
    
    for tb_file in tb_files:
        print(f"📖 Reading {tb_file.name}...")
        
        try:
            # Configure EventAccumulator - modern approach, no tensorflow.compat needed
            size_guidance = {
                'compressedHistograms': 0,
                'images': 0,
                'audio': 0,
                'scalars': 1000,
                'histograms': 0,
                'tensors': 0
            }
            
            ea = EventAccumulator(str(tb_file), size_guidance=size_guidance)
            ea.Reload()
            
            scalar_tags = ea.Tags()['scalars']
            print(f"✅ Available metrics: {scalar_tags}")
            
            for tag in scalar_tags:
                if tag not in metrics:
                    metrics[tag] = []
                
                scalar_events = ea.Scalars(tag)
                for event in scalar_events:
                    metrics[tag].append((event.step, event.value))
        
        except Exception as e:
            print(f"❌ Error reading {tb_file.name}: {e}")
            continue
    
    # Sort metrics by step
    for tag in metrics:
        metrics[tag] = sorted(metrics[tag], key=lambda x: x[0])
    
    return metrics

def create_simple_plots(metrics: Dict[str, List[Tuple[int, float]]], output_dir: str):
    """Create simple training plots from metrics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not metrics:
        print("❌ No metrics to plot")
        return
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create subplots based on available metrics
    available_metrics = list(metrics.keys())
    num_plots = min(len(available_metrics), 6)  # Limit to 6 plots for simplicity
    
    if num_plots == 0:
        print("❌ No valid metrics found")
        return
    
    # Calculate grid layout
    cols = 2 if num_plots > 1 else 1
    rows = (num_plots + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6 * rows))
    fig.suptitle('Simple TVC Training Analysis', fontsize=16, fontweight='bold')
    
    # Handle single subplot case
    if num_plots == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, list) else [axes]
    else:
        axes = axes.flatten()
    
    # Plot each metric
    for i, metric_key in enumerate(available_metrics[:num_plots]):
        ax = axes[i]
        
        if metric_key in metrics and metrics[metric_key]:
            steps, values = zip(*metrics[metric_key])
            
            # Plot the data
            ax.plot(steps, values, linewidth=2, alpha=0.8)
            ax.set_title(metric_key.replace('/', ' ').title(), fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
            # Add final value annotation
            if values:
                final_val = values[-1]
                ax.text(0.02, 0.98, f'Final: {final_val:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'No data for\n{metric_key}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.set_title(metric_key.replace('/', ' ').title(), fontweight='bold')
    
    # Hide unused subplots
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plot_path = output_path / 'simple_training_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Simple training analysis saved to {plot_path}")
    plt.close()


def main():
    """Main function for simple visualization."""
    parser = argparse.ArgumentParser(description='Simple Rocket TVC Training Visualization Tool')
    parser.add_argument('--log-dir', type=str, required=True,
                       help='Path to TensorBoard logs directory')
    parser.add_argument('--output-dir', type=str, default='visualization_output',
                       help='Output directory for generated plots')
    
    args = parser.parse_args()
    
    print("🚀 Simple TVC Training Visualization Tool")
    print("=" * 50)
    print(f"🔍 Analyzing logs in: {args.log_dir}")
    
    # Validate input directory
    if not Path(args.log_dir).exists():
        print(f"❌ Log directory does not exist: {args.log_dir}")
        sys.exit(1)
    
    # Read metrics from TensorBoard logs using modern approach
    metrics = read_tensorboard_logs(args.log_dir)
    
    if not metrics:
        print("❌ No valid metrics found in the logs")
        # Create output directory and save a "no data" report
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No Training Data Found\n\nCheck that:\n• TensorBoard files exist in log directory\n• Training has been run\n• Logs were saved correctly', 
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax.set_title('TVC-AI Training Visualization Report', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        no_data_path = output_path / 'no_data_report.png'
        plt.savefig(no_data_path, dpi=300, bbox_inches='tight')
        print(f"ℹ️ No data report saved to {no_data_path}")
        plt.close()
        return
    
    print(f"✅ Found {len(metrics)} metric types with data")
    
    # Create simple visualizations
    create_simple_plots(metrics, args.output_dir)
    
    print(f"🎉 Visualization complete! Check output directory: {args.output_dir}")


if __name__ == "__main__":
    main()