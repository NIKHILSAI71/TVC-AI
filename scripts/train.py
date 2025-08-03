"""
State-of-the-Art TVC Training Script
Comprehensive system using latest 2024-2025 deep RL research

This is the complete integration file that brings together:
- Enhanced environment with mission success detection
- Multi-algorithm ensemble agent (PPO+SAC+TD3) 
- Transformer-based policy networks
- Hierarchical RL with skill discovery
- Physics-informed neural networks
- Curiosity-driven exploration
- Adaptive curriculum learning
- Reward hacking detection and prevention
- Real-time safety monitoring
"""

import sys
import os
import yaml
import os
import torch
import numpy as np
import gymnasium as gym
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import wandb
from collections import deque
import time
import argparse
from dataclasses import dataclass
import json
import warnings
warnings.filterwarnings('ignore')

# Set wandb to offline mode to avoid interactive prompts
os.environ['WANDB_MODE'] = 'offline'

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import our state-of-the-art components
from env.enhanced_rocket_tvc_env import EnhancedRocketTVCEnv, MissionPhase
from agent.multi_algorithm_agent import MultiAlgorithmAgent
from scripts.curriculum_manager import CurriculumManager

@dataclass
class TrainingMetrics:
    """Comprehensive training metrics tracking"""
    episode_rewards: List[float] = None
    episode_lengths: List[int] = None
    success_rates: List[float] = None
    mission_completion_rates: List[float] = None
    safety_violations: List[int] = None
    algorithm_performance: Dict[str, List[float]] = None
    curriculum_stage: List[int] = None
    physics_losses: List[float] = None
    hacking_scores: List[float] = None
    
    def __post_init__(self):
        if self.episode_rewards is None:
            self.episode_rewards = []
        if self.episode_lengths is None:
            self.episode_lengths = []
        if self.success_rates is None:
            self.success_rates = []
        if self.mission_completion_rates is None:
            self.mission_completion_rates = []
        if self.safety_violations is None:
            self.safety_violations = []
        if self.algorithm_performance is None:
            self.algorithm_performance = {'ppo': [], 'sac': [], 'td3': []}
        if self.curriculum_stage is None:
            self.curriculum_stage = []
        if self.physics_losses is None:
            self.physics_losses = []
        if self.hacking_scores is None:
            self.hacking_scores = []

class RewardHackingDetector:
    """Advanced reward hacking detection system"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.reward_history = deque(maxlen=window_size)
        self.success_history = deque(maxlen=window_size)
        self.episode_length_history = deque(maxlen=window_size)
        
    def add_episode(self, total_reward: float, success: bool, episode_length: int):
        """Add episode data for analysis"""
        self.reward_history.append(total_reward)
        self.success_history.append(success)
        self.episode_length_history.append(episode_length)
    
    def detect_hacking(self) -> Dict[str, float]:
        """Detect potential reward hacking with multiple indicators"""
        if len(self.reward_history) < 50:
            return {'hacking_score': 0.0, 'confidence': 0.0}
        
        rewards = np.array(list(self.reward_history))
        successes = np.array(list(self.success_history))
        lengths = np.array(list(self.episode_length_history))
        
        indicators = {}
        
        # 1. High reward with low success rate
        mean_reward = np.mean(rewards[-20:])
        success_rate = np.mean(successes[-20:])
        
        if mean_reward > 1000 and success_rate < 0.1:
            indicators['reward_success_mismatch'] = 1.0
        else:
            indicators['reward_success_mismatch'] = 0.0
        
        # 2. Excessive episode length without success
        mean_length = np.mean(lengths[-20:])
        if mean_length > 900 and success_rate < 0.1:
            indicators['excessive_episode_length'] = 1.0
        else:
            indicators['excessive_episode_length'] = 0.0
        
        # 3. Reward variance without success variance
        reward_variance = np.var(rewards[-20:])
        success_variance = np.var(successes[-20:])
        
        if reward_variance > 10000 and success_variance < 0.01:
            indicators['reward_variance_mismatch'] = 1.0
        else:
            indicators['reward_variance_mismatch'] = 0.0
        
        # 4. Sudden reward spikes without completion
        recent_rewards = rewards[-10:]
        if len(recent_rewards) >= 10:
            if np.max(recent_rewards) > 2 * np.mean(rewards[:-10]) and success_rate < 0.2:
                indicators['reward_spike'] = 1.0
            else:
                indicators['reward_spike'] = 0.0
        else:
            indicators['reward_spike'] = 0.0
        
        # 5. Consistent high rewards with zero success
        if mean_reward > 2000 and success_rate == 0.0 and len(rewards) >= 50:
            indicators['impossible_performance'] = 1.0
        else:
            indicators['impossible_performance'] = 0.0
        
        # Overall hacking score
        hacking_score = np.mean(list(indicators.values()))
        confidence = min(1.0, len(self.reward_history) / self.window_size)
        
        return {
            'hacking_score': hacking_score,
            'confidence': confidence,
            'indicators': indicators,
            'mean_reward': mean_reward,
            'success_rate': success_rate,
            'mean_episode_length': mean_length
        }

class StateOfTheArtTrainer:
    """State-of-the-art TVC trainer implementing all modern techniques"""
    
    def __init__(self, config_path: str, debug: bool = False):
        self.config_path = config_path
        self.debug = debug
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging and output
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.setup_environment()
        self.setup_agent()
        self.setup_training()
        
        # Metrics and monitoring
        self.metrics = TrainingMetrics()
        self.reward_hacking_detector = RewardHackingDetector()
        
        # Training state
        self.current_episode = 0
        self.total_timesteps = 0
        self.best_success_rate = 0.0
        self.training_start_time = time.time()
        self.no_improvement_steps = 0
        
    def setup_logging(self):
        """Setup comprehensive logging with W&B integration"""
        # Create timestamped output directory
        base_output_dir = Path(self.config['globals']['output_dir'])
        timestamp = datetime.now().strftime('%Y-%m-%d/%H-%M-%S')
        output_dir = base_output_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Update config with actual output directory
        self.config['globals']['actual_output_dir'] = str(output_dir)
        
        # Configure logging with UTF-8 encoding for Windows compatibility
        log_level = getattr(logging, self.config.get('logging', {}).get('level', 'INFO'))
        
        # Setup UTF-8 encoding for Windows console
        import sys
        if sys.platform.startswith('win'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except AttributeError:
                # Fallback for older Python versions
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(output_dir / 'sota_training.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        
        # Setup Weights & Biases
        wandb_config = self.config.get('logging', {}).get('wandb', {})
        if wandb_config.get('enabled', False):
            wandb.init(
                project=wandb_config.get('project', 'tvc-ai-sota'),
                name=f"{self.config['globals']['experiment_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config,
                tags=['state-of-the-art', 'multi-algorithm', 'transformer', 'hierarchical']
            )
    
    def setup_environment(self):
        """Setup enhanced environment with all features"""
        env_config = self.config.get('env', {})
        
        self.env = EnhancedRocketTVCEnv(
            config=self.config,
            max_episode_steps=env_config.get('max_episode_steps', 1000),
            render_mode=None,
            enable_hierarchical=self.config.get('hierarchical_rl', {}).get('enabled', True),
            enable_curiosity=self.config.get('exploration', {}).get('curiosity', {}).get('enabled', True),
            enable_physics_informed=self.config.get('physics_informed', {}).get('enabled', True),
            debug=self.debug
        )
        
        # Create evaluation environment (deterministic)
        self.eval_env = EnhancedRocketTVCEnv(
            config=self.config,
            max_episode_steps=env_config.get('max_episode_steps', 1000),
            render_mode=None,
            enable_hierarchical=False,
            enable_curiosity=False,
            enable_physics_informed=False,
            debug=False
        )
        
        self.logger.info("Enhanced environment created with state-of-the-art features:")
        self.logger.info(f"  Observation space: {self.env.observation_space}")
        self.logger.info(f"  Action space: {self.env.action_space}")
        self.logger.info(f"  Max episode steps: {env_config.get('max_episode_steps', 1000)}")
    
    def setup_agent(self):
        """Setup multi-algorithm agent ensemble"""
        obs_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        
        self.agent = MultiAlgorithmAgent(obs_dim, action_dim, self.config)
        
        self.logger.info("Multi-algorithm ensemble agent created:")
        algorithms = self.config.get('algorithms', {})
        for alg_name, alg_config in algorithms.items():
            if alg_config.get('enabled', False) and alg_name != 'ensemble':
                self.logger.info(f"  - {alg_name.upper()} with transformer networks")
    
    def setup_training(self):
        """Setup training configuration and components"""
        training_config = self.config.get('training', {})
        
        # Training parameters
        self.total_timesteps_target = training_config.get('total_timesteps', 2000000)
        self.eval_freq = training_config.get('eval_freq', 5000)
        self.save_freq = training_config.get('save_freq', 10000)
        
        # Curriculum manager
        if self.config.get('curriculum', {}).get('enabled', False):
            self.curriculum_manager = CurriculumManager(
                self.config.get('curriculum', {}),
                self.logger
            )
            self.logger.info("Adaptive curriculum learning enabled")
        else:
            self.curriculum_manager = None
        
        # Early stopping
        early_stopping = training_config.get('early_stopping', {})
        self.early_stopping_enabled = early_stopping.get('enabled', True)
        self.early_stopping_patience = early_stopping.get('patience', 200000)
        self.early_stopping_min_improvement = early_stopping.get('min_improvement', 0.01)
        
        self.logger.info(f"Training configured for {self.total_timesteps_target:,} timesteps")
        
    def train(self):
        """Main state-of-the-art training loop"""
        self.logger.info(">>> Starting State-of-the-Art TVC Training!")
        self.logger.info("="*80)
        self.logger.info("Features enabled:")
        self.logger.info("  [X] Multi-algorithm ensemble (PPO+SAC+TD3)")
        self.logger.info("  [X] Transformer-based policy networks")
        self.logger.info("  [X] Hierarchical RL with skill discovery")
        self.logger.info("  [X] Physics-informed neural networks")
        self.logger.info("  [X] Curiosity-driven exploration")
        self.logger.info("  [X] Adaptive curriculum learning")
        self.logger.info("  [X] Real-time reward hacking detection")
        self.logger.info("  [X] Mission success detection with real landing criteria")
        self.logger.info("="*80)
        
        episode_rewards = deque(maxlen=100)
        episode_lengths = deque(maxlen=100)
        episode_successes = deque(maxlen=100)
        
        # Training loop
        while self.total_timesteps < self.total_timesteps_target:
            # Run episode
            episode_info = self.run_episode()
            
            # Update metrics
            episode_rewards.append(episode_info['reward'])
            episode_lengths.append(episode_info['length'])
            episode_successes.append(episode_info['success'])
            
            self.metrics.episode_rewards.append(episode_info['reward'])
            self.metrics.episode_lengths.append(episode_info['length'])
            
            # Add to reward hacking detector
            self.reward_hacking_detector.add_episode(
                episode_info['reward'], 
                episode_info['success'], 
                episode_info['length']
            )
            
            # Update curriculum if enabled
            if self.curriculum_manager is not None:
                success_rate = np.mean(list(episode_successes)[-10:]) if len(episode_successes) >= 10 else 0.0
                self.curriculum_manager.update(success_rate, episode_info)
            
            # Log progress
            if self.current_episode % self.config.get('logging', {}).get('log_frequency', 10) == 0:
                self.log_training_progress(episode_rewards, episode_lengths, episode_successes)
            
            # Evaluate periodically
            if self.total_timesteps % self.eval_freq == 0:
                eval_results = self.evaluate()
                self.log_evaluation_results(eval_results)
                
                # Check for improvement
                if eval_results['success_rate'] > self.best_success_rate + self.early_stopping_min_improvement:
                    self.best_success_rate = eval_results['success_rate']
                    self.no_improvement_steps = 0
                    self.save_checkpoint('best_model.pth')
                    self.logger.info(f">>> New best success rate: {self.best_success_rate:.1%}")
                else:
                    self.no_improvement_steps += self.eval_freq
                
                # Early stopping check
                if (self.early_stopping_enabled and 
                    self.no_improvement_steps >= self.early_stopping_patience):
                    self.logger.info(f">>> Early stopping triggered after {self.no_improvement_steps} steps without improvement")
                    break
            
            # Save periodic checkpoint
            if self.total_timesteps % self.save_freq == 0:
                self.save_checkpoint(f'checkpoint_{self.total_timesteps}.pth')
            
            # Check for reward hacking
            if self.current_episode % 50 == 0:
                hacking_info = self.reward_hacking_detector.detect_hacking()
                self.metrics.hacking_scores.append(hacking_info['hacking_score'])
                
                if hacking_info['hacking_score'] > 0.7:
                    self.logger.warning(f"!!! Potential reward hacking detected! Score: {hacking_info['hacking_score']:.3f}")
                    self.logger.warning(f"Indicators: {hacking_info['indicators']}")
                    
                    # Log to wandb if enabled
                    if wandb.run is not None:
                        wandb.log({
                            'reward_hacking/hacking_score': hacking_info['hacking_score'],
                            'reward_hacking/confidence': hacking_info['confidence'],
                            **{f'reward_hacking/{k}': v for k, v in hacking_info['indicators'].items()}
                        })
            
            self.current_episode += 1
        
        # Final evaluation
        self.logger.info(">>> Training completed. Running final evaluation...")
        final_eval = self.evaluate(episodes=50)
        self.log_evaluation_results(final_eval, final=True)
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        # Training summary
        self.log_training_summary()
    
    def run_episode(self) -> Dict[str, Any]:
        """Run a single training episode with all enhancements"""
        obs, info = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_success = False
        safety_violations = 0
        
        # Select algorithm for this episode
        algorithm = self.agent.select_algorithm()
        
        while True:
            # Get action from ensemble
            action, action_info = self.agent.get_action(torch.FloatTensor(obs).unsqueeze(0))
            action = action[0]  # Remove batch dimension - already numpy
            
            # Step environment
            next_obs, reward, terminated, truncated, step_info = self.env.step(action)
            
            # Check for mission success and safety violations
            mission_success = step_info.get('mission_successful', False)
            safety_violation = self._check_safety_violation(step_info)
            if safety_violation:
                safety_violations += 1
            
            # Update agent
            if self.total_timesteps >= 1000:  # Start training after warmup
                try:
                    # Create batch dictionary for agent update
                    batch = {
                        'states': torch.FloatTensor(obs).unsqueeze(0).to(self.agent.device),
                        'actions': torch.FloatTensor(action).unsqueeze(0).to(self.agent.device),
                        'rewards': torch.FloatTensor([reward]).to(self.agent.device),
                        'next_states': torch.FloatTensor(next_obs).unsqueeze(0).to(self.agent.device),
                        'dones': torch.BoolTensor([terminated or truncated]).to(self.agent.device)
                    }
                    losses = self.agent.update(batch)
                    
                    # Log losses periodically
                    if wandb.run is not None and self.total_timesteps % 100 == 0:
                        wandb.log({f'losses/{k}': v for k, v in losses.items()}, step=self.total_timesteps)
                except Exception as e:
                    self.logger.warning(f"Agent update failed: {e}. Skipping this update step.")
                    losses = {}
            
            # Update metrics
            episode_reward += reward
            episode_length += 1
            self.total_timesteps += 1
            
            # Check termination
            if terminated or truncated:
                episode_success = step_info.get('mission_successful', False)
                break
            
            obs = next_obs
        
        # Update algorithm performance
        self.agent.update_performance(algorithm, episode_reward)
        
        episode_info = {
            'reward': episode_reward,
            'length': episode_length,
            'success': episode_success,
            'algorithm_used': algorithm,
            'final_altitude': step_info.get('altitude', 0.0),
            'final_tilt': step_info.get('tilt_angle_deg', 0.0),
            'mission_phase': step_info.get('mission_phase', 'unknown'),
            'fuel_remaining': step_info.get('fuel_remaining', 0.0),
            'safety_violations': safety_violations
        }
        
        return episode_info
    
    def _check_safety_violation(self, info: Dict) -> bool:
        """Check if safety constraints were violated"""
        safety_constraints = self.config.get('safety', {})
        
        # Tilt constraint (30 degrees = 0.52 radians)
        max_tilt = safety_constraints.get('max_tilt', 0.52)
        if info.get('tilt_angle_deg', 0.0) > np.degrees(max_tilt):
            return True
        
        # Angular velocity constraint
        max_angular_vel = safety_constraints.get('max_angular_velocity', 5.0)
        if info.get('angular_velocity_mag', 0.0) > max_angular_vel:
            return True
        
        # Altitude constraints
        min_altitude = safety_constraints.get('min_altitude', 0.1)
        max_altitude = safety_constraints.get('max_altitude', 20.0)
        altitude = info.get('altitude', 0.0)
        if altitude < min_altitude or altitude > max_altitude:
            return True
        
        return False
    
    def evaluate(self, episodes: int = 20) -> Dict[str, float]:
        """Comprehensive policy evaluation"""
        eval_rewards = []
        eval_lengths = []
        eval_successes = []
        eval_safety_violations = []
        
        for _ in range(episodes):
            obs, _ = self.eval_env.reset()
            episode_reward = 0.0
            episode_length = 0
            safety_violations = 0
            
            while True:
                # Get deterministic action
                action, _ = self.agent.get_action(
                    torch.FloatTensor(obs).unsqueeze(0), 
                    deterministic=True
                )
                action = action[0].cpu().numpy()
                
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                if self._check_safety_violation(info):
                    safety_violations += 1
                
                if terminated or truncated:
                    eval_successes.append(info.get('mission_successful', False))
                    break
            
            eval_rewards.append(episode_reward)
            eval_lengths.append(episode_length)
            eval_safety_violations.append(safety_violations)
        
        return {
            'reward_mean': np.mean(eval_rewards),
            'reward_std': np.std(eval_rewards),
            'length_mean': np.mean(eval_lengths),
            'length_std': np.std(eval_lengths),
            'success_rate': np.mean(eval_successes),
            'safety_violation_rate': np.mean([v > 0 for v in eval_safety_violations]),
            'avg_safety_violations': np.mean(eval_safety_violations)
        }
    
    def log_training_progress(self, episode_rewards, episode_lengths, episode_successes):
        """Log detailed training progress"""
        recent_reward = np.mean(list(episode_rewards)[-10:]) if len(episode_rewards) >= 10 else 0.0
        recent_length = np.mean(list(episode_lengths)[-10:]) if len(episode_lengths) >= 10 else 0.0
        recent_success = np.mean(list(episode_successes)[-10:]) if len(episode_successes) >= 10 else 0.0
        
        # Calculate training speed
        elapsed_time = time.time() - self.training_start_time
        steps_per_second = self.total_timesteps / elapsed_time if elapsed_time > 0 else 0
        
        # Get latest reward hacking score
        hacking_info = self.reward_hacking_detector.detect_hacking()
        hacking_status = "[SAFE]" if hacking_info['hacking_score'] < 0.3 else "[CAUTION]" if hacking_info['hacking_score'] < 0.7 else "[DANGER]"
        
        self.logger.info(
            f"Episode {self.current_episode:>5}, Step {self.total_timesteps:>8,} | "
            f"Reward: {recent_reward:>7.2f} ± {np.std(list(episode_rewards)[-10:]):>5.2f} | "
            f"Success: {recent_success:>5.1%} | "
            f"Length: {recent_length:>5.0f} | "
            f"Speed: {steps_per_second:>4.0f} steps/s | "
            f"Hacking: {hacking_status}"
        )
        
        # Log to wandb
        if wandb.run is not None:
            wandb.log({
                'training/episode': self.current_episode,
                'training/timesteps': self.total_timesteps,
                'training/reward_mean': recent_reward,
                'training/reward_std': np.std(list(episode_rewards)[-10:]),
                'training/success_rate': recent_success,
                'training/episode_length': recent_length,
                'training/steps_per_second': steps_per_second,
                'training/hacking_score': hacking_info['hacking_score'],
                'curriculum/stage': self.curriculum_manager.get_current_stage().name if self.curriculum_manager and self.curriculum_manager.get_current_stage() else 'none'
            }, step=self.total_timesteps)
    
    def log_evaluation_results(self, eval_results: Dict[str, float], final: bool = False):
        """Log comprehensive evaluation results"""
        prefix = ">>> FINAL" if final else ">>> EVAL"
        
        self.logger.info(f"{prefix} Evaluation Results:")
        self.logger.info(f"  Success Rate: {eval_results['success_rate']:>6.1%}")
        self.logger.info(f"  Reward: {eval_results['reward_mean']:>10.2f} ± {eval_results['reward_std']:>6.2f}")
        self.logger.info(f"  Episode Length: {eval_results['length_mean']:>7.0f} ± {eval_results['length_std']:>5.0f}")
        self.logger.info(f"  Safety Violations: {eval_results['safety_violation_rate']:>5.1%}")
        
        # Log to wandb
        if wandb.run is not None:
            prefix_lower = "final_eval" if final else "eval"
            wandb.log({
                f'{prefix_lower}/reward_mean': eval_results['reward_mean'],
                f'{prefix_lower}/reward_std': eval_results['reward_std'],
                f'{prefix_lower}/success_rate': eval_results['success_rate'],
                f'{prefix_lower}/length_mean': eval_results['length_mean'],
                f'{prefix_lower}/safety_violation_rate': eval_results['safety_violation_rate']
            }, step=self.total_timesteps)
    
    def log_training_summary(self):
        """Log comprehensive training summary"""
        total_time = time.time() - self.training_start_time
        
        self.logger.info("="*80)
        self.logger.info(">>> TRAINING COMPLETED!")
        self.logger.info("="*80)
        self.logger.info(f"Total Episodes: {self.current_episode:,}")
        self.logger.info(f"Total Timesteps: {self.total_timesteps:,}")
        self.logger.info(f"Training Time: {total_time/3600:.2f} hours")
        self.logger.info(f"Best Success Rate: {self.best_success_rate:.1%}")
        
        if len(self.metrics.episode_rewards) > 0:
            self.logger.info(f"Final Reward: {np.mean(self.metrics.episode_rewards[-100:]):.2f}")
            if len(self.metrics.episode_rewards) >= 200:
                improvement = np.mean(self.metrics.episode_rewards[-100:]) - np.mean(self.metrics.episode_rewards[:100])
                self.logger.info(f"Reward Improvement: {improvement:.2f}")
        
        # Algorithm performance summary
        self.logger.info("Algorithm Performance:")
        for alg_name, performance in self.agent.performance_history.items():
            if len(performance) > 0:
                avg_perf = np.mean(list(performance))
                self.logger.info(f"  {alg_name.upper()}: {avg_perf:.2f}")
        
        # Reward hacking summary
        final_hacking_info = self.reward_hacking_detector.detect_hacking()
        hacking_status = "[CLEAN]" if final_hacking_info['hacking_score'] < 0.3 else "[SUSPICIOUS]"
        self.logger.info(f"Final Reward Hacking Status: {hacking_status} (Score: {final_hacking_info['hacking_score']:.3f})")
        
        # Save metrics
        self.save_training_metrics()
        self.logger.info("="*80)
    
    def save_training_metrics(self):
        """Save comprehensive training metrics"""
        output_dir = Path(self.config['globals']['actual_output_dir'])
        metrics_file = output_dir / 'sota_training_metrics.json'
        
        metrics_dict = {
            'training_summary': {
                'total_episodes': self.current_episode,
                'total_timesteps': self.total_timesteps,
                'training_time_hours': (time.time() - self.training_start_time) / 3600,
                'best_success_rate': self.best_success_rate
            },
            'episode_data': {
                'rewards': self.metrics.episode_rewards,
                'lengths': self.metrics.episode_lengths,
                'success_rates': self.metrics.success_rates,
                'hacking_scores': self.metrics.hacking_scores
            },
            'algorithm_performance': {k: list(v) for k, v in self.agent.performance_history.items()},
            'final_hacking_analysis': self.reward_hacking_detector.detect_hacking(),
            'config_used': self.config
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        self.logger.info(f"Training metrics saved to {metrics_file}")
    
    def save_checkpoint(self, filename: str):
        """Save comprehensive training checkpoint"""
        output_dir = Path(self.config['globals']['actual_output_dir'])
        models_dir = output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        checkpoint_path = models_dir / filename
        self.agent.save_checkpoint(str(checkpoint_path))
        
        # Save training state
        training_state = {
            'current_episode': self.current_episode,
            'total_timesteps': self.total_timesteps,
            'best_success_rate': self.best_success_rate,
            'no_improvement_steps': self.no_improvement_steps,
            'curriculum_stage': self.curriculum_manager.get_current_stage().name if self.curriculum_manager and self.curriculum_manager.get_current_stage() else 'none',
            'metrics': {
                'episode_rewards': self.metrics.episode_rewards,
                'episode_lengths': self.metrics.episode_lengths,
                'hacking_scores': self.metrics.hacking_scores
            }
        }
        
        training_state_path = models_dir / f'training_state_{filename.replace(".pth", ".json")}'
        with open(training_state_path, 'w') as f:
            json.dump(training_state, f, indent=2)

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='State-of-the-Art TVC AI Training')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug mode')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Available configs:")
        config_dir = Path('config')
        if config_dir.exists():
            for config_file in config_dir.glob('*.yaml'):
                print(f"  - {config_file}")
        return
    
    # Create trainer
    try:
        trainer = StateOfTheArtTrainer(args.config, debug=args.debug)
        
        # Resume from checkpoint if specified
        if args.resume:
            trainer.logger.info(f"Resuming training from {args.resume}")
            # Resume logic would go here
        
        # Start training
        trainer.train()
        
    except KeyboardInterrupt:
        trainer.logger.info("Training interrupted by user")
        trainer.save_checkpoint('interrupted_model.pth')
        
    except Exception as e:
        if 'trainer' in locals():
            trainer.logger.error(f"Training failed with error: {e}")
            trainer.save_checkpoint('error_model.pth')
        raise
    
    finally:
        # Cleanup
        if 'trainer' in locals():
            if hasattr(trainer, 'env'):
                trainer.env.close()
            if hasattr(trainer, 'eval_env'):
                trainer.eval_env.close()
            
            if wandb.run is not None:
                wandb.finish()

if __name__ == "__main__":
    main()
