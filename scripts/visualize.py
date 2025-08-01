#!/usr/bin/env python3
"""
Rocket TVC Training Visualization Tool

Clean, focused visualization tool for analyzing rocket TVC control performance.
Features:
- TensorBoard log analysis and plotting
- Training metrics visualization
- Performance analysis
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
    Read TensorBoard logs using EventAccumulator.
    Clean implementation without fallbacks.
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
            # Configure EventAccumulator
            size_guidance = {
                'compressedHistograms': 0,
                'images': 0,
                'audio': 0,
                'scalars': 10000,
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


def create_training_plots(metrics: Dict[str, List[Tuple[int, float]]], output_dir: str):
    """Create comprehensive training plots from metrics."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not metrics:
        print("❌ No metrics to plot")
        return
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create main analysis figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SAC Training Analysis - Rocket TVC Control', fontsize=16, fontweight='bold')
    
    # Define key metrics to plot
    key_metrics = [
        ('episode/reward', 'Episode Reward', axes[0, 0], 'blue'),
        ('episode/length', 'Episode Length', axes[0, 1], 'green'),
        ('episode/success', 'Success Rate', axes[0, 2], 'orange'),
        ('training/actor_loss', 'Actor Loss', axes[1, 0], 'red'),
        ('training/critic1_loss', 'Critic Loss', axes[1, 1], 'purple'),
        ('training/alpha', 'Entropy Coefficient', axes[1, 2], 'brown')
    ]
    
    # Plot each metric
    for metric_key, title, ax, color in key_metrics:
        if metric_key in metrics and metrics[metric_key]:
            steps, values = zip(*metrics[metric_key])
            
            # Plot raw data
            ax.plot(steps, values, alpha=0.6, color=color, linewidth=1)
            
            # Add smoothed line if enough data
            if len(values) > 20:
                window_size = max(1, len(values) // 50)
                smoothed = []
                for i in range(len(values)):
                    start_idx = max(0, i - window_size)
                    end_idx = min(len(values), i + window_size + 1)
                    smoothed.append(np.mean(values[start_idx:end_idx]))
                
                ax.plot(steps, smoothed, color=color, linewidth=2, label='Smoothed')
                ax.legend()
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            if values:
                final_val = values[-1]
                mean_val = np.mean(values)
                stats_text = f'Final: {final_val:.3f}\nMean: {mean_val:.3f}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, f'No data for\n{title}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
            ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    plot_path = output_path / 'training_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training analysis saved to {plot_path}")
    plt.close()
    
    # Create detailed reward analysis
    create_reward_analysis(metrics, output_path)


def create_reward_analysis(metrics: Dict[str, List[Tuple[int, float]]], output_path: Path):
    """Create detailed reward progression analysis."""
    if 'episode/reward' not in metrics or not metrics['episode/reward']:
        print("⚠️ No reward data available for detailed analysis")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Detailed Reward Analysis', fontsize=14, fontweight='bold')
    
    steps, rewards = zip(*metrics['episode/reward'])
    
    # Reward progression
    axes[0, 0].plot(steps, rewards, alpha=0.6, color='blue')
    if len(rewards) > 20:
        window_size = max(1, len(rewards) // 50)
        smoothed = []
        for i in range(len(rewards)):
            start_idx = max(0, i - window_size)
            end_idx = min(len(rewards), i + window_size + 1)
            smoothed.append(np.mean(rewards[start_idx:end_idx]))
        axes[0, 0].plot(steps, smoothed, color='darkblue', linewidth=2)
    
    axes[0, 0].set_title('Reward Progression')
    axes[0, 0].set_xlabel('Training Steps')
    axes[0, 0].set_ylabel('Episode Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[0, 1].hist(rewards, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(np.mean(rewards), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(rewards):.2f}')
    axes[0, 1].axvline(np.median(rewards), color='orange', linestyle='--', 
                       label=f'Median: {np.median(rewards):.2f}')
    axes[0, 1].set_title('Reward Distribution')
    axes[0, 1].set_xlabel('Episode Reward')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning progress (reward improvement over time)
    if len(rewards) > 100:
        chunk_size = len(rewards) // 10
        chunk_means = []
        chunk_steps = []
        for i in range(0, len(rewards), chunk_size):
            chunk_end = min(i + chunk_size, len(rewards))
            chunk_means.append(np.mean(rewards[i:chunk_end]))
            chunk_steps.append(steps[i + chunk_size // 2] if chunk_end < len(steps) else steps[-1])
        
        axes[1, 0].plot(chunk_steps, chunk_means, 'o-', color='purple', linewidth=2, markersize=6)
        axes[1, 0].set_title('Learning Progress (Chunked Means)')
        axes[1, 0].set_xlabel('Training Steps')
        axes[1, 0].set_ylabel('Mean Reward')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'Not enough data\nfor progress analysis', 
                        ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Learning Progress')
    
    # Performance statistics
    stats_text = f"""
    Total Episodes: {len(rewards)}
    Mean Reward: {np.mean(rewards):.3f}
    Std Deviation: {np.std(rewards):.3f}
    Min Reward: {np.min(rewards):.3f}
    Max Reward: {np.max(rewards):.3f}
    Final 10% Mean: {np.mean(rewards[-len(rewards)//10:]):.3f}
    """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 1].set_title('Performance Statistics')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    reward_plot_path = output_path / 'reward_analysis.png'
    plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Reward analysis saved to {reward_plot_path}")
    plt.close()


def generate_training_report(log_dir: str, output_dir: str):
    """Generate comprehensive training visualization report."""
    print("🚀 Rocket TVC Training Visualization Tool")
    print("=" * 50)
    print(f"🔍 Analyzing training logs in: {log_dir}")
    
    # Read metrics from TensorBoard logs
    metrics = read_tensorboard_logs(log_dir)
    
    if not metrics:
        print("❌ No valid metrics found in the logs")
        # Create output directory and save a "no data" report
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No Training Data Found\n\nCheck that:\n• TensorBoard files exist in log directory\n• Training has been run\n• Logs were saved correctly', 
                ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        ax.set_title('TVC-AI Training Visualization Report', fontsize=16, fontweight='bold')
        ax.axis('off')
        
        no_data_path = output_path / 'no_data_report.png'
        plt.savefig(no_data_path, dpi=300, bbox_inches='tight')
        print(f"ℹ️ Report saved to {no_data_path}")
        plt.close()
        return
    
    print(f"✅ Found {len(metrics)} metric types with data")
    
    # Create visualizations
    create_training_plots(metrics, output_dir)
    
    print(f"🎉 Visualization complete! Check output directory: {output_dir}")


def main():
    """Main function to run the visualization tool."""
    parser = argparse.ArgumentParser(description='Rocket TVC Training Visualization Tool')
    parser.add_argument('--log-dir', type=str, required=True,
                       help='Path to TensorBoard logs directory')
    parser.add_argument('--output-dir', type=str, default='visualization_output',
                       help='Output directory for generated plots')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not Path(args.log_dir).exists():
        print(f"❌ Log directory does not exist: {args.log_dir}")
        sys.exit(1)
    
    # Generate visualization report
    generate_training_report(args.log_dir, args.output_dir)


if __name__ == "__main__":
    main()
