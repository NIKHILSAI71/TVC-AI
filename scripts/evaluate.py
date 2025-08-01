#!/usr/bin/env python3
# Copyright (c) 2025 NIKHILSAIPAGIDIMARRI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Model Evaluation Script for Rocket TVC Control

This script evaluates trained SAC models on various test scenarios including:
- Standard evaluation with nominal parameters
- Robustness testing with domain randomization
- Performance analysis and visualization
- Model comparison across different checkpoints
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from agent import SACAgent, SACConfig
from env import make_evaluation_env, make_debug_env, RocketTVCEnv, RocketConfig


class EvaluationResults:
    """Container for evaluation results and analysis."""
    
    def __init__(self):
        self.episode_data = []
        self.trajectory_data = []
        self.performance_metrics = {}
    
    def add_episode(self, episode_data: Dict[str, Any]):
        """Add episode results."""
        self.episode_data.append(episode_data)
    
    def add_trajectory(self, trajectory_data: Dict[str, Any]):
        """Add trajectory data for detailed analysis."""
        self.trajectory_data.append(trajectory_data)
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute comprehensive performance metrics."""
        if not self.episode_data:
            return {}
        
        df = pd.DataFrame(self.episode_data)
        
        metrics = {
            # Basic performance
            'mean_reward': df['reward'].mean(),
            'std_reward': df['reward'].std(),
            'min_reward': df['reward'].min(),
            'max_reward': df['reward'].max(),
            
            # Episode characteristics
            'mean_length': df['length'].mean(),
            'std_length': df['length'].std(),
            
            # Success metrics
            'success_rate': df['success'].mean(),
            'crash_rate': df['crashed'].mean(),
            'timeout_rate': df['timeout'].mean(),
            
            # Stability metrics
            'mean_final_tilt': df['final_tilt_deg'].mean(),
            'std_final_tilt': df['final_tilt_deg'].std(),
            'mean_final_altitude': df['final_altitude'].mean(),
            'mean_max_tilt': df['max_tilt_deg'].mean(),
            'mean_max_angular_vel': df['max_angular_vel'].mean(),
            
            # Control effort
            'mean_control_effort': df['control_effort'].mean(),
            'mean_fuel_used': df['fuel_used'].mean(),
        }
        
        # Percentile metrics
        for p in [25, 50, 75, 90, 95]:
            metrics[f'reward_p{p}'] = np.percentile(df['reward'], p)
            metrics[f'tilt_p{p}'] = np.percentile(df['final_tilt_deg'], p)
        
        self.performance_metrics = metrics
        return metrics
    
    def save_results(self, output_path: str):
        """Save evaluation results to file."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save episode data
        if self.episode_data:
            df = pd.DataFrame(self.episode_data)
            df.to_csv(output_path / 'episode_results.csv', index=False)
        
        # Save metrics
        if self.performance_metrics:
            metrics_df = pd.DataFrame([self.performance_metrics])
            metrics_df.to_csv(output_path / 'performance_metrics.csv', index=False)
        
        # Save trajectory data if available
        if self.trajectory_data:
            trajectory_df = pd.DataFrame(self.trajectory_data)
            trajectory_df.to_csv(output_path / 'trajectory_data.csv', index=False)


def evaluate_single_episode(agent: SACAgent, env: RocketTVCEnv, 
                          deterministic: bool = True, 
                          record_trajectory: bool = False) -> Dict[str, Any]:
    """
    Evaluate agent on a single episode.
    
    Args:
        agent: SAC agent to evaluate
        env: Environment instance
        deterministic: Whether to use deterministic policy
        record_trajectory: Whether to record detailed trajectory
        
    Returns:
        Dictionary containing episode results
    """
    obs, info = env.reset()
    episode_reward = 0
    episode_length = 0
    done = False
    
    # Tracking variables
    trajectory = []
    max_tilt = 0
    max_angular_vel = 0
    control_effort = 0
    
    while not done:
        action = agent.select_action(obs, deterministic=deterministic)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
        
        # Track metrics
        current_tilt = info.get('tilt_angle_deg', 0)
        max_tilt = max(max_tilt, current_tilt)
        
        angular_vel_mag = np.linalg.norm(info.get('angular_velocity', [0, 0, 0]))
        max_angular_vel = max(max_angular_vel, angular_vel_mag)
        
        action_mag = np.linalg.norm(action)
        control_effort += action_mag
        
        # Record trajectory if requested
        if record_trajectory:
            trajectory.append({
                'step': episode_length,
                'position': info.get('position', [0, 0, 0]),
                'orientation_euler': info.get('orientation_euler', [0, 0, 0]),
                'linear_velocity': info.get('linear_velocity', [0, 0, 0]),
                'angular_velocity': info.get('angular_velocity', [0, 0, 0]),
                'action': action.tolist(),
                'reward': reward,
                'tilt_deg': current_tilt,
                'altitude': info.get('altitude', 0),
                'fuel_remaining': info.get('fuel_remaining', 0)
            })
        
        obs = next_obs
    
    # Determine episode outcome
    final_tilt = info.get('tilt_angle_deg', 180)
    final_altitude = info.get('altitude', 0)
    fuel_used = 1.0 - info.get('fuel_remaining', 0)
    
    success = final_tilt < 20 and episode_length > 200 and final_altitude > 0.5
    crashed = final_altitude < 0.2
    timeout = episode_length >= env.max_episode_steps - 1
    
    episode_result = {
        'reward': episode_reward,
        'length': episode_length,
        'success': success,
        'crashed': crashed,
        'timeout': timeout,
        'final_tilt_deg': final_tilt,
        'final_altitude': final_altitude,
        'max_tilt_deg': max_tilt,
        'max_angular_vel': max_angular_vel,
        'control_effort': control_effort / episode_length,
        'fuel_used': fuel_used,
        'trajectory': trajectory if record_trajectory else None
    }
    
    return episode_result


def run_standard_evaluation(agent: SACAgent, num_episodes: int = 100) -> EvaluationResults:
    """Run standard evaluation with nominal parameters."""
    env = make_evaluation_env(
        domain_randomization=False,
        sensor_noise=False,
        max_episode_steps=2000
    )
    
    results = EvaluationResults()
    
    for episode in tqdm(range(num_episodes), desc="Standard Evaluation"):
        episode_result = evaluate_single_episode(
            agent, env, deterministic=True, 
            record_trajectory=(episode < 5)  # Record first 5 trajectories
        )
        results.add_episode(episode_result)
        
        if episode_result['trajectory']:
            for step_data in episode_result['trajectory']:
                step_data['episode'] = episode
                results.add_trajectory(step_data)
    
    env.close()
    return results


def run_robustness_evaluation(agent: SACAgent, num_episodes: int = 200) -> EvaluationResults:
    """Run robustness evaluation with domain randomization."""
    env = make_evaluation_env(
        domain_randomization=True,
        sensor_noise=True,
        max_episode_steps=2000
    )
    
    results = EvaluationResults()
    
    for episode in tqdm(range(num_episodes), desc="Robustness Evaluation"):
        episode_result = evaluate_single_episode(
            agent, env, deterministic=True,
            record_trajectory=(episode < 3)  # Record first 3 trajectories
        )
        results.add_episode(episode_result)
        
        if episode_result['trajectory']:
            for step_data in episode_result['trajectory']:
                step_data['episode'] = episode
                step_data['test_type'] = 'robustness'
                results.add_trajectory(step_data)
    
    env.close()
    return results


def run_stress_test(agent: SACAgent, num_episodes: int = 50) -> EvaluationResults:
    """Run stress test with extreme conditions."""
    # Create environment with more extreme randomization
    config = RocketConfig()
    config.mass_variation = 0.5  # ±50% mass variation
    config.thrust_variation = 0.5  # ±50% thrust variation
    config.cg_offset_max = 0.1  # 10cm CG offset
    
    env = RocketTVCEnv(
        config=config,
        domain_randomization=True,
        sensor_noise=True,
        max_episode_steps=1500,
        debug=False
    )
    
    results = EvaluationResults()
    
    for episode in tqdm(range(num_episodes), desc="Stress Test"):
        episode_result = evaluate_single_episode(
            agent, env, deterministic=True
        )
        episode_result['test_type'] = 'stress'
        results.add_episode(episode_result)
    
    env.close()
    return results


def create_evaluation_plots(results_dict: Dict[str, EvaluationResults], 
                          output_dir: str):
    """Create comprehensive evaluation plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Reward Distribution Comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    reward_data = []
    for test_name, results in results_dict.items():
        if results.episode_data:
            df = pd.DataFrame(results.episode_data)
            for reward in df['reward']:
                reward_data.append({'Test': test_name, 'Reward': reward})
    
    if reward_data:
        reward_df = pd.DataFrame(reward_data)
        
        # Box plot
        sns.boxplot(data=reward_df, x='Test', y='Reward', ax=axes[0])
        axes[0].set_title('Reward Distribution by Test Type')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Histogram
        for test_name in reward_df['Test'].unique():
            test_rewards = reward_df[reward_df['Test'] == test_name]['Reward']
            axes[1].hist(test_rewards, alpha=0.7, label=test_name, bins=20)
        axes[1].set_xlabel('Reward')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Reward Histograms')
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'reward_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Success Rate Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    success_data = []
    for test_name, results in results_dict.items():
        metrics = results.compute_metrics()
        if metrics:
            success_data.append({
                'Test': test_name,
                'Success Rate': metrics.get('success_rate', 0),
                'Crash Rate': metrics.get('crash_rate', 0),
                'Timeout Rate': metrics.get('timeout_rate', 0)
            })
    
    if success_data:
        success_df = pd.DataFrame(success_data)
        
        x = range(len(success_df))
        width = 0.25
        
        ax.bar([i - width for i in x], success_df['Success Rate'], 
               width, label='Success', color='green', alpha=0.7)
        ax.bar(x, success_df['Crash Rate'], 
               width, label='Crash', color='red', alpha=0.7)
        ax.bar([i + width for i in x], success_df['Timeout Rate'], 
               width, label='Timeout', color='orange', alpha=0.7)
        
        ax.set_xlabel('Test Type')
        ax.set_ylabel('Rate')
        ax.set_title('Episode Outcome Rates')
        ax.set_xticks(x)
        ax.set_xticklabels(success_df['Test'])
        ax.legend()
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'success_rates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Trajectory Visualization (if available)
    for test_name, results in results_dict.items():
        if results.trajectory_data:
            plot_trajectories(results.trajectory_data, 
                            output_dir / f'{test_name}_trajectories.png')


def plot_trajectories(trajectory_data: List[Dict], output_path: str):
    """Plot rocket trajectories."""
    df = pd.DataFrame(trajectory_data)
    
    if df.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Group by episode
    episodes = df['episode'].unique()[:5]  # Plot first 5 episodes
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(episodes)))
    
    for i, episode in enumerate(episodes):
        episode_data = df[df['episode'] == episode]
        color = colors[i]
        
        # Extract trajectory data
        positions = np.array([pos for pos in episode_data['position']])
        orientations = np.array([ori for ori in episode_data['orientation_euler']])
        
        if len(positions) == 0:
            continue
        
        # 3D Trajectory
        axes[0, 0].plot(positions[:, 0], positions[:, 1], 
                       color=color, alpha=0.7, label=f'Episode {episode}')
        
        # Altitude vs Time
        axes[0, 1].plot(episode_data['step'], episode_data['altitude'], 
                       color=color, alpha=0.7, label=f'Episode {episode}')
        
        # Tilt Angle vs Time
        axes[1, 0].plot(episode_data['step'], episode_data['tilt_deg'], 
                       color=color, alpha=0.7, label=f'Episode {episode}')
        
        # Control Actions vs Time
        actions = np.array([act for act in episode_data['action']])
        if len(actions) > 0:
            axes[1, 1].plot(episode_data['step'], actions[:, 0], 
                           color=color, alpha=0.7, linestyle='-', 
                           label=f'Episode {episode} (Pitch)')
            axes[1, 1].plot(episode_data['step'], actions[:, 1], 
                           color=color, alpha=0.4, linestyle='--', 
                           label=f'Episode {episode} (Yaw)')
    
    # Format plots
    axes[0, 0].set_xlabel('X Position (m)')
    axes[0, 0].set_ylabel('Y Position (m)')
    axes[0, 0].set_title('Horizontal Trajectory')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axis('equal')
    
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Altitude (m)')
    axes[0, 1].set_title('Altitude vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Tilt Angle (degrees)')
    axes[1, 0].set_title('Tilt Angle vs Time')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Failure Threshold')
    
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Gimbal Angle (normalized)')
    axes[1, 1].set_title('Control Actions vs Time')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add legends to first plot only to avoid clutter
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained SAC agent')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--num_episodes', type=int, default=100,
                       help='Number of episodes per test')
    parser.add_argument('--tests', nargs='+', 
                       choices=['standard', 'robustness', 'stress', 'all'],
                       default=['all'], help='Tests to run')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes (only for small number of episodes)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load agent
    logger.info(f"Loading model from {args.model_path}")
    
    # Create dummy environment to get dimensions
    temp_env = make_evaluation_env()
    obs_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()
    
    # Create and load agent
    agent = SACAgent(obs_dim, action_dim)
    agent.load(args.model_path)
    
    logger.info("Model loaded successfully")
    
    # Determine which tests to run
    tests_to_run = []
    if 'all' in args.tests:
        tests_to_run = ['standard', 'robustness', 'stress']
    else:
        tests_to_run = args.tests
    
    # Run evaluations
    results_dict = {}
    
    if 'standard' in tests_to_run:
        logger.info("Running standard evaluation...")
        results_dict['Standard'] = run_standard_evaluation(agent, args.num_episodes)
    
    if 'robustness' in tests_to_run:
        logger.info("Running robustness evaluation...")
        results_dict['Robustness'] = run_robustness_evaluation(agent, args.num_episodes)
    
    if 'stress' in tests_to_run:
        logger.info("Running stress test...")
        results_dict['Stress Test'] = run_stress_test(agent, max(50, args.num_episodes // 2))
    
    # Compute and print metrics
    logger.info("\\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    
    for test_name, results in results_dict.items():
        metrics = results.compute_metrics()
        logger.info(f"\\n{test_name}:")
        logger.info(f"  Mean Reward: {metrics.get('mean_reward', 0):.2f} ± {metrics.get('std_reward', 0):.2f}")
        logger.info(f"  Success Rate: {metrics.get('success_rate', 0):.2%}")
        logger.info(f"  Crash Rate: {metrics.get('crash_rate', 0):.2%}")
        logger.info(f"  Mean Final Tilt: {metrics.get('mean_final_tilt', 0):.1f}°")
        logger.info(f"  Mean Episode Length: {metrics.get('mean_length', 0):.0f}")
    
    # Save results
    logger.info(f"\\nSaving results to {args.output_dir}")
    for test_name, results in results_dict.items():
        test_output_dir = Path(args.output_dir) / test_name.lower().replace(' ', '_')
        results.save_results(test_output_dir)
    
    # Create plots
    logger.info("Creating evaluation plots...")
    create_evaluation_plots(results_dict, args.output_dir)
    
    logger.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()
