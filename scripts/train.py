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
SAC Training Script for Rocket TVC Control - Optimized

This script trains a Soft Actor-Critic agent to control rocket attitude using
thrust vector control in a PyBullet-based simulation environment.

Optimized for fast learning:
- Essential logging with TensorBoard only
- Streamlined configuration
- Focused on core SAC training
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from tqdm import tqdm

# Optional wandb import with graceful fallback
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available, continuing without it...")

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from agent.sac_agent import SACAgent, SACConfig
from env.rocket_tvc_env import RocketTVCEnv, RocketConfig


class TensorBoardLogger:
    """TensorBoard logging utility."""
    
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.step = 0
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """Log scalar value."""
        if step is None:
            step = self.step
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: Optional[int] = None):
        """Log multiple scalar values."""
        if step is None:
            step = self.step
        self.writer.add_scalars(tag, values, step)
    
    def increment_step(self):
        """Increment global step counter."""
        self.step += 1
    
    def close(self):
        """Close writer."""
        self.writer.close()


class TrainingMetrics:
    """Training metrics tracker."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_rates = []
        self.training_metrics = {}
    
    def add_episode(self, reward: float, length: int, success: bool):
        """Add episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.success_rates.append(float(success))
        
        # Keep only recent episodes
        if len(self.episode_rewards) > self.window_size:
            self.episode_rewards.pop(0)
            self.episode_lengths.pop(0)
            self.success_rates.pop(0)
    
    def add_training_metrics(self, metrics: Dict[str, float]):
        """Add training metrics."""
        for key, value in metrics.items():
            if key not in self.training_metrics:
                self.training_metrics[key] = []
            self.training_metrics[key].append(value)
            
            # Keep only recent values
            if len(self.training_metrics[key]) > self.window_size:
                self.training_metrics[key].pop(0)
    
    def get_summary(self) -> Dict[str, float]:
        """Get metrics summary."""
        summary = {}
        
        if self.episode_rewards:
            summary.update({
                'episode_reward_mean': np.mean(self.episode_rewards),
                'episode_reward_std': np.std(self.episode_rewards),
                'episode_reward_min': np.min(self.episode_rewards),
                'episode_reward_max': np.max(self.episode_rewards),
                'episode_length_mean': np.mean(self.episode_lengths),
                'success_rate': np.mean(self.success_rates),
                'num_episodes': len(self.episode_rewards)
            })
        
        # Add training metrics
        for key, values in self.training_metrics.items():
            if values:
                summary[f'training_{key}_mean'] = np.mean(values)
        
        return summary


def evaluate_agent(agent: SACAgent, eval_env, num_episodes: int = 10, 
                  render: bool = False) -> Dict[str, float]:
    """
    Evaluate the agent's performance.
    
    Args:
        agent: The SAC agent to evaluate
        eval_env: Evaluation environment
        num_episodes: Number of episodes to evaluate
        render: Whether to render during evaluation
        
    Returns:
        Dictionary of evaluation metrics
    """
    eval_rewards = []
    eval_lengths = []
    success_count = 0
    
    for episode in range(num_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
            
            if render:
                eval_env.render()
        
        eval_rewards.append(episode_reward)
        eval_lengths.append(episode_length)
        
        # Check success (rocket stayed stable)
        final_tilt = info.get('tilt_angle_deg', 180)
        if final_tilt < 20 and episode_length > 200:  # Stable for sufficient time
            success_count += 1
    
    eval_metrics = {
        'eval_reward_mean': np.mean(eval_rewards),
        'eval_reward_std': np.std(eval_rewards),
        'eval_length_mean': np.mean(eval_lengths),
        'eval_success_rate': success_count / num_episodes
    }
    
    return eval_metrics


@hydra.main(version_base=None, config_path="../config", config_name="minimal")
def train_agent(cfg: DictConfig) -> None:
    """
    Main training function.
    
    Args:
        cfg: Hydra configuration object
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Print configuration
    logger.info("Training Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Set random seeds for reproducibility
    if cfg.globals.seed is not None:
        np.random.seed(cfg.globals.seed)
        torch.manual_seed(cfg.globals.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.globals.seed)
    
    # Create output directories
    output_dir = Path(cfg.globals.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_dir = output_dir / "models"
    model_dir.mkdir(exist_ok=True)
    
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Initialize Weights & Biases if enabled and available
    # Fix: Only initialize wandb if explicitly enabled AND API key is available
    wandb_enabled = False
    if cfg.wandb.get('enabled', False) and WANDB_AVAILABLE:
        try:
            # Only initialize if API key is explicitly available
            if os.getenv('WANDB_API_KEY'):
                wandb.init(
                    project=cfg.wandb.project,
                    name=cfg.wandb.run_name,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    tags=cfg.wandb.tags,
                    mode=cfg.wandb.get('mode', 'online')
                )
                wandb_enabled = True
                logger.info("Wandb initialized successfully")
            elif cfg.wandb.get('mode', 'online') == 'offline':
                # Only allow offline mode if explicitly requested
                wandb.init(
                    project=cfg.wandb.project,
                    name=cfg.wandb.run_name,
                    config=OmegaConf.to_container(cfg, resolve=True),
                    tags=cfg.wandb.tags,
                    mode='offline'
                )
                wandb_enabled = True
                logger.info("Wandb initialized in offline mode")
            else:
                logger.info("Wandb disabled: No API key found and offline mode not requested")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}. Continuing without wandb...")
            wandb_enabled = False
    else:
        if not WANDB_AVAILABLE:
            logger.info("Wandb not available (package not installed)")
        else:
            logger.info("Wandb disabled in configuration")
    
    # Update config to reflect actual wandb status
    cfg.wandb.enabled = wandb_enabled
    
    # Initialize TensorBoard logger
    tb_logger = TensorBoardLogger(str(log_dir))
    
    # Create environments with improved configuration
    logger.info("Creating environments...")
    
    # Create rocket configuration from Hydra config with all improvements
    rocket_config = RocketConfig(
        # Basic physics
        gravity=cfg.get('gravity', -9.81),
        physics_timestep=cfg.get('timestep', 0.004167),
        
        # Rocket properties (using improved values)
        mass=cfg.get('mass', 2.5),
        length=cfg.get('length', 1.0),
        radius=cfg.get('radius', 0.05),
        thrust_mean=cfg.get('thrust_max', 50.0),
        thrust_std=cfg.get('thrust_std', 2.0),
        burn_time=cfg.get('burn_time', 5.0),
        
        # Control system - using improved values
        max_gimbal_angle=cfg.get('max_gimbal_angle', 25.0),  # Increased from default 15
        max_gimbal_rate=cfg.get('max_gimbal_rate', 100.0),  # Rate limiting
        
        # Domain randomization with configurable parameters
        mass_variation=cfg.get('mass_variation', 0.3),
        thrust_variation=cfg.get('thrust_variation', 0.2),
        cg_offset_max=cfg.get('cg_offset_max', 0.05),
        max_initial_tilt=cfg.get('max_initial_tilt', 5.0),
        max_initial_angular_vel=cfg.get('max_initial_angular_vel', 0.5),
        
        # Initial conditions
        initial_altitude=cfg.get('initial_height_range', [1.0, 3.0])[0],  # Use min of range
        target_altitude=cfg.get('initial_height_range', [1.0, 3.0])[1],   # Use max of range
        
        # Wind and disturbances
        wind_force_max=cfg.get('wind_force_max', 2.0),
        
        # Sensor noise
        gyro_noise_std=cfg.get('gyro_noise_std', 0.1),
        quaternion_noise_std=cfg.get('quaternion_noise_std', 0.01),
        
        # Reward function parameters (using improved values)
        attitude_penalty_gain=cfg.get('attitude_penalty_gain', 15.0),
        angular_velocity_penalty_gain=cfg.get('angular_velocity_penalty_gain', 0.2),
        control_effort_penalty_gain=cfg.get('control_effort_penalty_gain', 0.02),
        saturation_threshold=cfg.get('saturation_threshold', 0.8),
        saturation_penalty=cfg.get('saturation_penalty', 2.0),
        saturation_bonus=cfg.get('saturation_bonus', 0.1),
        stability_angle_threshold=cfg.get('stability_angle_threshold', 3.0),
        stability_angular_vel_threshold=cfg.get('stability_angular_vel_threshold', 0.5),
        stability_bonus=cfg.get('stability_bonus', 2.0),
        tilt_improvement_threshold=cfg.get('tilt_improvement_threshold', 0.1),
        tilt_improvement_bonus=cfg.get('tilt_improvement_bonus', 1.0),
        tilt_degradation_penalty=cfg.get('tilt_degradation_penalty', 2.0),
        efficiency_bonus=cfg.get('efficiency_bonus', 0.5),
        
        # Altitude management
        min_safe_altitude=cfg.get('min_safe_altitude', 0.5),
        max_safe_altitude=cfg.get('max_safe_altitude', 15.0),
        altitude_penalty_gain=cfg.get('altitude_penalty_gain', 5.0),
        nominal_altitude_bonus=cfg.get('nominal_altitude_bonus', 0.2),
        
        # Termination conditions
        ground_termination_height=cfg.get('ground_termination_height', 0.1),
        max_tilt_degrees=cfg.get('max_tilt_degrees', 45.0),
        max_horizontal_distance=cfg.get('max_horizontal_distance', 50.0),
        max_termination_altitude=cfg.get('max_termination_altitude', 20.0),
        max_angular_velocity=cfg.get('max_angular_velocity', 10.0),
        
        # Termination penalties
        crash_penalty=cfg.get('crash_penalty', -50.0),
        tilt_penalty=cfg.get('tilt_penalty', -30.0),
        altitude_penalty=cfg.get('altitude_penalty', -20.0),
        angular_velocity_penalty=cfg.get('angular_velocity_penalty', -25.0),
        
        # Physics parameters
        aerodynamic_damping_coefficient=cfg.get('aerodynamic_damping_coefficient', 0.02),
        minimum_thrust=cfg.get('minimum_thrust', 10.0),
        lateral_friction=cfg.get('lateral_friction', 0.1),
        spinning_friction=cfg.get('spinning_friction', 0.01),
        rolling_friction=cfg.get('rolling_friction', 0.01),
        restitution=cfg.get('restitution', 0.1),
        linear_damping=cfg.get('linear_damping', 0.01),
        angular_damping=cfg.get('angular_damping', 0.01)
    )
    
    # Training environment
    train_env = RocketTVCEnv(
        config=rocket_config,
        max_episode_steps=cfg.env.max_episode_steps,
        domain_randomization=cfg.env.domain_randomization,
        sensor_noise=cfg.env.sensor_noise,
        render_mode=cfg.env.get('render_mode'),
        debug=cfg.globals.debug
    )
    
    # Evaluation environment (no randomization for consistent evaluation)
    eval_env = RocketTVCEnv(
        config=rocket_config,
        max_episode_steps=cfg.env.max_episode_steps * 2,  # Longer evaluation episodes
        domain_randomization=False,  # No randomization for evaluation
        sensor_noise=False,  # No noise for evaluation
        render_mode=None,  # No rendering during evaluation
        debug=cfg.globals.debug
    )
    
    # Get environment dimensions
    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    
    logger.info(f"Observation dimension: {obs_dim}")
    logger.info(f"Action dimension: {action_dim}")
    
    # Create SAC agent with configuration
    sac_config = SACConfig(
        # Network architecture from config
        hidden_dims=cfg.agent.get('hidden_dims', [256, 256]),
        lr_actor=cfg.agent.get('lr_actor', 1e-3),
        lr_critic=cfg.agent.get('lr_critic', 1e-3),
        lr_alpha=cfg.agent.get('lr_alpha', 1e-3),
        gamma=cfg.agent.get('gamma', 0.95),
        tau=cfg.agent.get('tau', 0.01),
        alpha=cfg.agent.get('alpha', 0.1),
        automatic_entropy_tuning=cfg.agent.get('automatic_entropy_tuning', True),
        batch_size=cfg.agent.get('batch_size', 256),
        buffer_size=cfg.agent.get('buffer_size', 50000),
        learning_starts=cfg.agent.get('learning_starts', 1000),
        train_freq=cfg.agent.get('train_freq', 1),
        gradient_steps=cfg.agent.get('gradient_steps', 1),
        layer_norm=cfg.agent.get('layer_norm', True),
    )
    
    agent = SACAgent(obs_dim, action_dim, sac_config)
    logger.info("SAC agent created successfully")
    
    # Load checkpoint if specified
    if hasattr(cfg, 'checkpoint_path') and cfg.checkpoint_path:
        logger.info(f"Loading checkpoint from {cfg.checkpoint_path}")
        agent.load(cfg.checkpoint_path)
    
    # Training metrics
    metrics_window = getattr(cfg.logging, 'metrics_window', 100)
    metrics = TrainingMetrics(window_size=metrics_window)
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    best_eval_reward = float('-inf')
    episodes_without_improvement = 0
    
    obs, _ = train_env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    for step in tqdm(range(cfg.training.total_steps), desc="Training Steps"):
        # Select action
        if step < sac_config.learning_starts:
            action = train_env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)
        
        # Take environment step
        next_obs, reward, terminated, truncated, info = train_env.step(action)
        done = terminated or truncated
        
        # Store transition in replay buffer
        agent.replay_buffer.add(obs, action, reward, next_obs, terminated)
        
        # Update observation and episode metrics
        obs = next_obs
        episode_reward += reward
        episode_length += 1
        
        # Train agent
        if step >= sac_config.learning_starts and step % sac_config.train_freq == 0:
            training_info = agent.train()
            if training_info:
                metrics.add_training_metrics(training_info)
                
                # Log training metrics to TensorBoard
                for key, value in training_info.items():
                    tb_logger.log_scalar(f"training/{key}", value, step)
        
        # Handle episode end
        if done:
            # Determine if episode was successful
            final_tilt = info.get('tilt_angle_deg', 180)
            success = final_tilt < 20 and episode_length > 200
            
            # Add episode metrics
            metrics.add_episode(episode_reward, episode_length, success)
            episode_count += 1
            
            # Log episode metrics
            tb_logger.log_scalar("episode/reward", episode_reward, episode_count)
            tb_logger.log_scalar("episode/length", episode_length, episode_count)
            tb_logger.log_scalar("episode/success", float(success), episode_count)
            tb_logger.log_scalar("episode/final_tilt_deg", final_tilt, episode_count)
            tb_logger.log_scalar("episode/final_altitude", info.get('altitude', 0), episode_count)
            
            if cfg.wandb.enabled and wandb_enabled:
                try:
                    wandb.log({
                        "episode/reward": episode_reward,
                        "episode/length": episode_length,
                        "episode/success": float(success),
                        "episode/final_tilt_deg": final_tilt,
                        "step": step
                    })
                except Exception:
                    pass  # Fail silently if wandb logging fails
            
            # Reset environment
            obs, _ = train_env.reset()
            episode_reward = 0
            episode_length = 0
            
            # Log progress
            if episode_count % cfg.logging.console_log_interval == 0:
                summary = metrics.get_summary()
                logger.info(f"Episode {episode_count}, Step {step}")
                logger.info(f"Reward: {summary.get('episode_reward_mean', 0):.2f} ± {summary.get('episode_reward_std', 0):.2f}")
                logger.info(f"Success Rate: {summary.get('success_rate', 0):.2%}")
                logger.info(f"Buffer Size: {len(agent.replay_buffer)}")
        
        # Evaluation
        if step % cfg.training.eval_freq == 0 and step > 0:
            logger.info("Running evaluation...")
            eval_metrics = evaluate_agent(
                agent, eval_env, 
                num_episodes=cfg.training.eval_episodes,
                render=cfg.globals.debug
            )
            
            # Log evaluation metrics
            for key, value in eval_metrics.items():
                tb_logger.log_scalar(f"eval/{key}", value, step)
            
            if cfg.wandb.enabled and wandb_enabled:
                wandb.log({f"eval/{k}": v for k, v in eval_metrics.items()})
            
            logger.info(f"Evaluation - Reward: {eval_metrics['eval_reward_mean']:.2f}, "
                       f"Success Rate: {eval_metrics['eval_success_rate']:.2%}")
            
            # Save best model
            eval_reward = eval_metrics['eval_reward_mean']
            if eval_reward > best_eval_reward:
                best_eval_reward = eval_reward
                episodes_without_improvement = 0
                best_model_path = model_dir / "best_model.pth"
                agent.save(str(best_model_path))
                logger.info(f"New best model saved with reward: {eval_reward:.2f}")
            else:
                episodes_without_improvement += 1
            
            # Early stopping
            if cfg.training.early_stopping.enabled:
                if episodes_without_improvement >= cfg.training.early_stopping.patience:
                    logger.info(f"Early stopping triggered after {episodes_without_improvement} evaluations without improvement")
                    break
        
        # Save checkpoint
        if step % cfg.training.checkpoint_freq == 0 and step > 0:
            checkpoint_path = model_dir / f"checkpoint_{step}.pth"
            agent.save(str(checkpoint_path))
            logger.info(f"Checkpoint saved at step {step}")
        
        tb_logger.increment_step()
    
    # Final evaluation and model saving
    logger.info("Training completed. Running final evaluation...")
    final_eval_metrics = evaluate_agent(
        agent, eval_env, 
        num_episodes=cfg.training.final_eval_episodes,
        render=False
    )
    
    logger.info("Final Evaluation Results:")
    for key, value in final_eval_metrics.items():
        logger.info(f"{key}: {value:.4f}")
    
    # Save final model
    final_model_path = model_dir / "final_model.pth"
    agent.save(str(final_model_path))
    
    # Log training summary
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f} seconds")
    logger.info(f"Total episodes: {episode_count}")
    logger.info(f"Best evaluation reward: {best_eval_reward:.2f}")
    
    if cfg.wandb.enabled and wandb_enabled:
        wandb.log({
            "final/total_time": total_time,
            "final/total_episodes": episode_count,
            "final/best_eval_reward": best_eval_reward,
            **{f"final/{k}": v for k, v in final_eval_metrics.items()}
        })
        wandb.finish()
    
    # Cleanup
    tb_logger.close()
    train_env.close()
    eval_env.close()
    
    logger.info("Training session finished successfully!")


if __name__ == "__main__":
    train_agent()
