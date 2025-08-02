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
Enhanced Model Evaluation Script for Rocket TVC Control

This script evaluates trained SAC models with:
- Robust model loading without fallbacks
- Comprehensive performance analysis
- Enhanced visualization with all graphs visible
- Detailed metrics for rocket control assessment
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
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from agent import SACAgent, SACConfig
from env import make_evaluation_env, make_debug_env, RocketTVCEnv, RocketConfig

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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


def load_trained_agent(model_path: str) -> SACAgent:
    """
    Load a trained SAC agent from checkpoint with robust error handling.
    No fallbacks - proper error handling only.
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    logger.info(f"Loading model from {model_path}")
    
    # Get environment dimensions
    temp_env = make_evaluation_env()
    obs_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    temp_env.close()
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            # Extract configuration if available
            if 'config' in checkpoint:
                config = SACConfig(**checkpoint['config'])
                logger.info("✅ Loaded configuration from checkpoint")
            else:
                # Use optimized config for rocket control stability
                config = SACConfig(
                    hidden_dims=[512, 512, 256],  # Larger network for better representation
                    lr_actor=1e-4,  # Lower learning rate for stability
                    lr_critic=3e-4,
                    lr_alpha=1e-4,  # Lower alpha learning rate
                    gamma=0.995,  # Higher discount factor for long-term stability
                    tau=0.001,  # Slower target network updates
                    alpha=0.1,  # Lower initial entropy for less exploration
                    automatic_entropy_tuning=True,
                    target_entropy=-2.0,  # Appropriate for 2D action space
                    batch_size=512,  # Larger batch size for stable gradients
                    gradient_clip_norm=1.0,  # Stricter gradient clipping
                    action_noise=0.05,  # Lower action noise for stability
                    curriculum_learning=True
                )
                logger.info("🔧 Using optimized configuration for rocket control stability")
            
            # Create agent with configuration
            agent = SACAgent(obs_dim, action_dim, config)
            
            # Load state dict
            if 'agent_state_dict' in checkpoint:
                agent.load_state_dict(checkpoint['agent_state_dict'])
                logger.info("✅ Loaded agent state from checkpoint")
            elif 'actor_state_dict' in checkpoint:
                # Legacy format - load individual components
                agent.actor.load_state_dict(checkpoint['actor_state_dict'])
                agent.critic1.load_state_dict(checkpoint['critic1_state_dict'])
                agent.critic2.load_state_dict(checkpoint['critic2_state_dict'])
                if 'log_alpha' in checkpoint:
                    agent.log_alpha.data = checkpoint['log_alpha']
                logger.info("✅ Loaded agent components from legacy checkpoint")
            else:
                # Direct model state dict
                agent.load_state_dict(checkpoint)
                logger.info("✅ Loaded agent state directly")
        else:
            # Legacy single model format
            config = SACConfig(
                hidden_dims=[512, 512, 256],
                lr_actor=1e-4,
                lr_critic=3e-4,
                lr_alpha=1e-4,
                gamma=0.995,
                tau=0.001,
                alpha=0.1,
                automatic_entropy_tuning=True,
                target_entropy=-2.0,
                batch_size=512,
                gradient_clip_norm=1.0,
                action_noise=0.05,
                curriculum_learning=True
            )
            agent = SACAgent(obs_dim, action_dim, config)
            agent.load_state_dict(checkpoint)
            logger.info("✅ Loaded agent from direct state dict")
        
        # Set agent to evaluation mode
        agent.eval_mode()
        logger.info("🎯 Model loaded successfully and set to evaluation mode")
        return agent
        
    except Exception as e:
        logger.error(f"❌ Failed to load model checkpoint: {e}")
        raise RuntimeError(f"Could not load model from {model_path}: {e}")


def evaluate_single_episode(agent: SACAgent, env: RocketTVCEnv, 
                          deterministic: bool = True, 
                          record_trajectory: bool = False) -> Dict[str, Any]:
    """Evaluate agent on a single episode with detailed tracking."""
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
    
    for episode in tqdm(range(num_episodes), desc="🎯 Standard Evaluation"):
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
    
    for episode in tqdm(range(num_episodes), desc="🛡️ Robustness Evaluation"):
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
    
    for episode in tqdm(range(num_episodes), desc="⚡ Stress Test"):
        episode_result = evaluate_single_episode(
            agent, env, deterministic=True
        )
        episode_result['test_type'] = 'stress'
        results.add_episode(episode_result)
    
    env.close()
    return results


def create_enhanced_evaluation_plots(results_dict: Dict[str, EvaluationResults], 
                                   output_dir: str):
    """Create comprehensive evaluation plots with enhanced visibility."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set enhanced style for better visibility
    plt.style.use('default')  # Use default style for better compatibility
    sns.set_palette("husl")
    
    # Enhanced color scheme
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    
    # 1. Comprehensive Performance Dashboard
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # Collect all data
    all_data = []
    for test_name, results in results_dict.items():
        if results.episode_data:
            df = pd.DataFrame(results.episode_data)
            df['test_type'] = test_name
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Reward Distribution (Box + Violin)
        ax1 = fig.add_subplot(gs[0, :2])
        sns.boxplot(data=combined_df, x='test_type', y='reward', ax=ax1, palette=colors)
        sns.stripplot(data=combined_df, x='test_type', y='reward', ax=ax1, 
                     size=3, alpha=0.6, color='black')
        ax1.set_title('📊 Reward Distribution by Test Type', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Test Type', fontsize=12)
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Success Rate Comparison
        ax2 = fig.add_subplot(gs[0, 2:])
        success_data = combined_df.groupby('test_type').agg({
            'success': 'mean',
            'crashed': 'mean',
            'timeout': 'mean'
        }).round(3)
        
        success_data.plot(kind='bar', ax=ax2, color=colors[:3], alpha=0.8)
        ax2.set_title('🎯 Success/Failure Rates by Test Type', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Test Type', fontsize=12)
        ax2.set_ylabel('Rate', fontsize=12)
        ax2.legend(['Success', 'Crashed', 'Timeout'])
        ax2.grid(True, alpha=0.3)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # Stability Analysis - Final Tilt
        ax3 = fig.add_subplot(gs[1, :2])
        sns.histplot(data=combined_df, x='final_tilt_deg', hue='test_type', 
                    bins=30, alpha=0.7, ax=ax3, palette=colors)
        ax3.axvline(x=20, color='red', linestyle='--', alpha=0.8, label='Success Threshold')
        ax3.set_title('📐 Final Tilt Angle Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Final Tilt Angle (degrees)', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Control Effort Analysis
        ax4 = fig.add_subplot(gs[1, 2:])
        sns.scatterplot(data=combined_df, x='control_effort', y='reward', 
                       hue='test_type', alpha=0.7, ax=ax4, palette=colors)
        ax4.set_title('⚡ Control Effort vs Reward', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Mean Control Effort', fontsize=12)
        ax4.set_ylabel('Episode Reward', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Episode Length Analysis
        ax5 = fig.add_subplot(gs[2, :2])
        sns.boxplot(data=combined_df, x='test_type', y='length', ax=ax5, palette=colors)
        ax5.set_title('⏱️ Episode Length Distribution', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Test Type', fontsize=12)
        ax5.set_ylabel('Episode Length (steps)', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        # Fuel Usage Analysis
        ax6 = fig.add_subplot(gs[2, 2:])
        sns.histplot(data=combined_df, x='fuel_used', hue='test_type', 
                    bins=20, alpha=0.7, ax=ax6, palette=colors)
        ax6.set_title('⛽ Fuel Usage Distribution', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Fuel Used (fraction)', fontsize=12)
        ax6.set_ylabel('Count', fontsize=12)
        ax6.grid(True, alpha=0.3)
        
        # Performance Correlation Matrix
        ax7 = fig.add_subplot(gs[3, :2])
        corr_data = combined_df[['reward', 'final_tilt_deg', 'control_effort', 
                                'fuel_used', 'length', 'max_tilt_deg']].corr()
        sns.heatmap(corr_data, annot=True, cmap='RdBu_r', center=0, ax=ax7,
                   square=True, fmt='.2f')
        ax7.set_title('🔗 Performance Metrics Correlation', fontsize=14, fontweight='bold')
        
        # Stability Timeline (if trajectory data available)
        ax8 = fig.add_subplot(gs[3, 2:])
        if any(results.trajectory_data for results in results_dict.values()):
            # Plot trajectory data
            traj_data = []
            for test_name, results in results_dict.items():
                if results.trajectory_data:
                    traj_df = pd.DataFrame(results.trajectory_data)
                    traj_df['test_type'] = test_name
                    traj_data.append(traj_df)
            
            if traj_data:
                traj_combined = pd.concat(traj_data, ignore_index=True)
                for test_name in traj_combined['test_type'].unique():
                    test_traj = traj_combined[traj_combined['test_type'] == test_name]
                    episodes = test_traj['episode'].unique()[:3]  # Show first 3 episodes
                    
                    for ep in episodes:
                        ep_data = test_traj[test_traj['episode'] == ep]
                        ax8.plot(ep_data['step'], ep_data['tilt_deg'], 
                               alpha=0.6, label=f'{test_name} Ep{ep}')
                
                ax8.axhline(y=20, color='red', linestyle='--', alpha=0.8, label='Success Threshold')
                ax8.set_title('🚀 Rocket Tilt Trajectories', fontsize=14, fontweight='bold')
                ax8.set_xlabel('Time Steps', fontsize=12)
                ax8.set_ylabel('Tilt Angle (degrees)', fontsize=12)
                ax8.grid(True, alpha=0.3)
                ax8.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax8.text(0.5, 0.5, 'No trajectory data available', 
                    ha='center', va='center', transform=ax8.transAxes,
                    fontsize=12, style='italic')
            ax8.set_title('🚀 Rocket Trajectories', fontsize=14, fontweight='bold')
    
    plt.suptitle('🚀 TVC-AI Comprehensive Evaluation Dashboard', 
                fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'comprehensive_evaluation_dashboard.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. Create individual detailed plots for each test type
    for test_name, results in results_dict.items():
        if results.trajectory_data:
            create_trajectory_plot(results.trajectory_data, test_name, 
                                 output_dir / f'{test_name.lower().replace(" ", "_")}_trajectories.png')
    
    logger.info(f"📈 Enhanced evaluation plots saved to {output_dir}")


def create_trajectory_plot(trajectory_data: List[Dict], test_name: str, output_path: Path):
    """Create detailed trajectory plots for a specific test."""
    if not trajectory_data:
        return
    
    df = pd.DataFrame(trajectory_data)
    episodes = df['episode'].unique()[:5]  # Show first 5 episodes
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'🚀 {test_name} - Detailed Trajectory Analysis', fontsize=16, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(episodes)))
    
    for i, episode in enumerate(episodes):
        ep_data = df[df['episode'] == episode]
        color = colors[i]
        
        # Position trajectory
        axes[0, 0].plot(ep_data['step'], [pos[2] for pos in ep_data['position']], 
                       color=color, alpha=0.8, label=f'Episode {episode}')
        
        # Tilt angle
        axes[0, 1].plot(ep_data['step'], ep_data['tilt_deg'], 
                       color=color, alpha=0.8, label=f'Episode {episode}')
        
        # Control actions
        gimbal_x = [act[0] for act in ep_data['action']]
        gimbal_y = [act[1] for act in ep_data['action']]
        axes[1, 0].plot(ep_data['step'], gimbal_x, color=color, alpha=0.8, 
                       linestyle='-', label=f'Gimbal X - Ep {episode}')
        axes[1, 0].plot(ep_data['step'], gimbal_y, color=color, alpha=0.8, 
                       linestyle='--', label=f'Gimbal Y - Ep {episode}')
        
        # Reward accumulation
        rewards = np.cumsum(ep_data['reward'])
        axes[1, 1].plot(ep_data['step'], rewards, color=color, alpha=0.8, 
                       label=f'Episode {episode}')
    
    # Configure subplots
    axes[0, 0].set_title('📍 Altitude Over Time')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Altitude (m)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].set_title('📐 Tilt Angle Over Time')
    axes[0, 1].axhline(y=20, color='red', linestyle='--', alpha=0.8, label='Success Threshold')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Tilt Angle (degrees)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].set_title('🎮 Control Actions Over Time')
    axes[1, 0].set_xlabel('Time Steps')
    axes[1, 0].set_ylabel('Gimbal Angle (normalized)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].set_title('📈 Cumulative Reward Over Time')
    axes[1, 1].set_xlabel('Time Steps')
    axes[1, 1].set_ylabel('Cumulative Reward')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main():
    """Main evaluation function with enhanced reporting."""
    parser = argparse.ArgumentParser(description='Enhanced TVC-AI Model Evaluation')
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
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    logger.info("🚀 Starting TVC-AI Enhanced Evaluation")
    logger.info(f"📍 Model: {args.model_path}")
    logger.info(f"📁 Output: {args.output_dir}")
    logger.info(f"🎯 Episodes per test: {args.num_episodes}")
    
    # Load agent using robust loading function (no fallbacks)
    agent = load_trained_agent(args.model_path)
    
    # Determine which tests to run
    tests_to_run = []
    if 'all' in args.tests:
        tests_to_run = ['standard', 'robustness', 'stress']
    else:
        tests_to_run = args.tests
    
    logger.info(f"🧪 Running tests: {', '.join(tests_to_run)}")
    
    # Run evaluations
    results_dict = {}
    
    if 'standard' in tests_to_run:
        logger.info("🎯 Running standard evaluation...")
        results_dict['Standard'] = run_standard_evaluation(agent, args.num_episodes)
    
    if 'robustness' in tests_to_run:
        logger.info("🛡️ Running robustness evaluation...")
        results_dict['Robustness'] = run_robustness_evaluation(agent, args.num_episodes)
    
    if 'stress' in tests_to_run:
        logger.info("⚡ Running stress test...")
        results_dict['Stress Test'] = run_stress_test(agent, max(50, args.num_episodes // 2))
    
    # Compute and print comprehensive metrics
    logger.info("\n" + "="*80)
    logger.info("🚀 COMPREHENSIVE EVALUATION RESULTS")
    logger.info("="*80)
    
    for test_name, results in results_dict.items():
        metrics = results.compute_metrics()
        logger.info(f"\n📊 {test_name} Results:")
        logger.info(f"  🎯 Performance Metrics:")
        logger.info(f"    Mean Reward: {metrics.get('mean_reward', 0):.2f} ± {metrics.get('std_reward', 0):.2f}")
        logger.info(f"    Reward Range: [{metrics.get('min_reward', 0):.2f}, {metrics.get('max_reward', 0):.2f}]")
        logger.info(f"    Median Reward: {metrics.get('reward_p50', 0):.2f}")
        
        logger.info(f"  ✅ Success Metrics:")
        logger.info(f"    Success Rate: {metrics.get('success_rate', 0):.2%}")
        logger.info(f"    Crash Rate: {metrics.get('crash_rate', 0):.2%}")
        logger.info(f"    Timeout Rate: {metrics.get('timeout_rate', 0):.2%}")
        
        logger.info(f"  🚀 Stability Metrics:")
        logger.info(f"    Mean Final Tilt: {metrics.get('mean_final_tilt', 0):.1f}° ± {metrics.get('std_final_tilt', 0):.1f}°")
        logger.info(f"    Max Tilt 95th percentile: {metrics.get('tilt_p95', 0):.1f}°")
        logger.info(f"    Mean Final Altitude: {metrics.get('mean_final_altitude', 0):.2f}m")
        
        logger.info(f"  ⚡ Control Metrics:")
        logger.info(f"    Mean Control Effort: {metrics.get('mean_control_effort', 0):.3f}")
        logger.info(f"    Mean Fuel Usage: {metrics.get('mean_fuel_used', 0):.2%}")
        logger.info(f"    Mean Episode Length: {metrics.get('mean_length', 0):.0f} steps")
    
    # Save results
    logger.info(f"\n💾 Saving results to {args.output_dir}")
    for test_name, results in results_dict.items():
        test_output_dir = Path(args.output_dir) / test_name.lower().replace(' ', '_')
        results.save_results(test_output_dir)
    
    # Create enhanced plots with better visibility
    logger.info("📈 Creating enhanced evaluation plots...")
    create_enhanced_evaluation_plots(results_dict, args.output_dir)
    
    logger.info("✅ Evaluation completed successfully!")
    logger.info(f"📁 Results saved to: {Path(args.output_dir).absolute()}")
    logger.info("📊 All evaluation graphs and metrics are now available!")


if __name__ == "__main__":
    main()
