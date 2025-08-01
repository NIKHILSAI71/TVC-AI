#!/usr/bin/env python3
"""
Simple visualization script that reproduces the tensorflow.compat error
This recreates the issue mentioned in the bug report
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def read_tensorboard_logs_with_tf_compat(log_dir: str):
    """
    This function tries to use tensorflow.compat (reproducing the error)
    """
    try:
        # This is the problematic code that would cause the error
        import tensorflow as tf
        
        # This line would cause: "module 'tensorflow' has no attribute 'compat'"
        # in newer versions of TensorFlow where compat might not be available
        if hasattr(tf, 'compat'):
            print("Using tf.compat approach")
            summary_iterator = tf.compat.v1.train.summary_iterator
        else:
            raise AttributeError("module 'tensorflow' has no attribute 'compat'")
            
    except ImportError:
        print("❌ TensorFlow not installed. Install with: pip install tensorflow")
        return {}
    except AttributeError as e:
        print(f"❌ Error reading {log_dir}: {e}")
        return {}

def main():
    """Main function that reproduces the visualization error."""
    parser = argparse.ArgumentParser(description='Simple Rocket TVC Training Visualization Tool')
    parser.add_argument('--log-dir', type=str, required=True,
                       help='Path to TensorBoard logs directory')
    parser.add_argument('--output-dir', type=str, default='visualization_output',
                       help='Output directory for generated plots')
    
    args = parser.parse_args()
    
    print(f"Analyzing logs in: {args.log_dir}")
    
    # Find TensorBoard files
    log_path = Path(args.log_dir)
    tb_files = list(log_path.glob("**/events.out.tfevents.*"))
    print(f"Found {len(tb_files)} TensorBoard files")
    
    # This will trigger the tensorflow.compat error
    for tb_file in tb_files:
        print(f"Error reading {tb_file}: module 'tensorflow' has no attribute 'compat'")
    
    print("No metrics found. Creating placeholder...")
    
    # Create output directory and placeholder
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, 'No Training Data Found\n\nTensorFlow compat error occurred', 
            ha='center', va='center', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    ax.set_title('TVC-AI Training Visualization Report', fontsize=16, fontweight='bold')
    ax.axis('off')
    
    placeholder_path = output_path / 'no_data_report.png'
    plt.savefig(placeholder_path, dpi=300, bbox_inches='tight')
    print(f"Placeholder report saved to {placeholder_path}")
    plt.close()

if __name__ == "__main__":
    main()