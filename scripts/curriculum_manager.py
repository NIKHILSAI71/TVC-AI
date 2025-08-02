"""
Curriculum Learning Manager for TVC-AI Training

Implements progressive difficulty scaling for stable SAC training
based on research findings for continuous control tasks.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CurriculumStage:
    """Configuration for a curriculum learning stage."""
    name: str
    duration_steps: int
    conditions: Dict
    success_criteria: Dict
    completed: bool = False
    performance_history: List[float] = None
    
    def __post_init__(self):
        if self.performance_history is None:
            self.performance_history = []


class CurriculumManager:
    """
    Manages curriculum learning progression for rocket TVC training.
    
    Implements research-backed curriculum learning with:
    - Staged difficulty progression
    - Performance-based stage transitions
    - Adaptive difficulty adjustment
    - Comprehensive monitoring
    """
    
    def __init__(self, curriculum_config: Dict, logger: Optional[logging.Logger] = None):
        self.config = curriculum_config
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize curriculum stages
        self.stages = self._initialize_stages()
        self.current_stage_idx = 0
        self.current_step = 0
        self.total_training_steps = 0
        
        # Performance tracking
        self.evaluation_history = []
        self.stage_transition_steps = []
        
        # Configuration tracking
        self.config_history = []
        
        self.logger.info(f"Initialized curriculum with {len(self.stages)} stages")
        self._log_curriculum_overview()
    
    def _initialize_stages(self) -> List[CurriculumStage]:
        """Initialize curriculum stages from configuration."""
        stages = []
        
        if not self.config.get('enabled', False):
            self.logger.warning("Curriculum learning is disabled!")
            return stages
        
        # Use configured stages if provided, otherwise use defaults
        stage_configs = self.config.get('stages', [
            {
                'name': 'basic_stabilization',
                'duration_steps': 150000,
                'conditions': {
                    'max_initial_tilt': 0.5,
                    'max_initial_angular_vel': 0.1,
                    'domain_randomization': False,
                    'sensor_noise': False,
                    'max_gimbal_angle': 10.0,
                    'wind_enabled': False
                },
                'success_criteria': {
                    'min_success_rate': 0.6,
                    'min_avg_reward': 100.0,
                    'evaluation_episodes': 50
                }
            },
            {
                'name': 'moderate_disturbances',
                'duration_steps': 200000,
                'conditions': {
                    'max_initial_tilt': 2.0,
                    'max_initial_angular_vel': 0.3,
                    'domain_randomization': True,
                    'randomization_strength': 0.3,
                    'sensor_noise': False,
                    'max_gimbal_angle': 15.0,
                    'wind_enabled': False
                },
                'success_criteria': {
                    'min_success_rate': 0.5,
                    'min_avg_reward': 200.0,
                    'evaluation_episodes': 50
                }
            },
            {
                'name': 'full_challenge',
                'duration_steps': 300000,
                'conditions': {
                    'max_initial_tilt': 5.0,
                    'max_initial_angular_vel': 0.5,
                    'domain_randomization': True,
                    'randomization_strength': 1.0,
                    'sensor_noise': True,
                    'max_gimbal_angle': 25.0,
                    'wind_enabled': True
                },
                'success_criteria': {
                    'min_success_rate': 0.4,
                    'min_avg_reward': 300.0,
                    'evaluation_episodes': 100
                }
            }
        ])
        
        for config in stage_configs:
            stage = CurriculumStage(
                name=config['name'],
                duration_steps=config['duration_steps'],
                conditions=config['conditions'],
                success_criteria=config['success_criteria']
            )
            stages.append(stage)
        
        return stages
    
    def _log_curriculum_overview(self):
        """Log overview of curriculum stages."""
        self.logger.info("=== CURRICULUM LEARNING OVERVIEW ===")
        for i, stage in enumerate(self.stages):
            self.logger.info(f"Stage {i+1}: {stage.name}")
            self.logger.info(f"  Duration: {stage.duration_steps:,} steps")
            self.logger.info(f"  Success criteria: {stage.success_criteria}")
            self.logger.info(f"  Key conditions: max_tilt={stage.conditions.get('max_initial_tilt', 'N/A')}°, "
                           f"randomization={stage.conditions.get('domain_randomization', False)}")
        self.logger.info("=" * 40)
    
    def get_current_stage(self) -> Optional[CurriculumStage]:
        """Get the current curriculum stage."""
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None
    
    def get_environment_config(self) -> Dict:
        """Get environment configuration for current stage."""
        current_stage = self.get_current_stage()
        if current_stage is None:
            self.logger.warning("No active curriculum stage - using default config")
            return {}
        
        return current_stage.conditions
    
    def should_advance_stage(self, eval_metrics: Dict) -> bool:
        """
        Check if current stage should advance based on performance.
        
        Args:
            eval_metrics: Dictionary containing evaluation metrics
            
        Returns:
            True if should advance to next stage
        """
        current_stage = self.get_current_stage()
        if current_stage is None or current_stage.completed:
            return False
        
        # Check if minimum duration has passed
        stage_steps = self.current_step - sum(s.duration_steps for s in self.stages[:self.current_stage_idx])
        if stage_steps < current_stage.duration_steps * 0.5:  # Must complete at least 50% of stage
            return False
        
        # Check success criteria
        criteria = current_stage.success_criteria
        success_rate = eval_metrics.get('eval_success_rate', 0.0)
        avg_reward = eval_metrics.get('eval_reward_mean', -float('inf'))
        
        meets_success_rate = success_rate >= criteria['min_success_rate']
        meets_avg_reward = avg_reward >= criteria['min_avg_reward']
        
        self.logger.info(f"Stage advancement check - Success rate: {success_rate:.3f} "
                        f"(need {criteria['min_success_rate']:.3f}), "
                        f"Avg reward: {avg_reward:.1f} (need {criteria['min_avg_reward']:.1f})")
        
        return meets_success_rate and meets_avg_reward
    
    def advance_stage(self) -> bool:
        """
        Advance to the next curriculum stage.
        
        Returns:
            True if advanced successfully, False if no more stages
        """
        current_stage = self.get_current_stage()
        if current_stage:
            current_stage.completed = True
            self.stage_transition_steps.append(self.current_step)
            self.logger.info(f"✅ Completed stage: {current_stage.name} at step {self.current_step}")
        
        self.current_stage_idx += 1
        next_stage = self.get_current_stage()
        
        if next_stage:
            self.logger.info(f"🚀 Advancing to stage: {next_stage.name}")
            self.logger.info(f"New conditions: {next_stage.conditions}")
            return True
        else:
            self.logger.info("🏁 All curriculum stages completed!")
            return False
    
    def update(self, step: int, eval_metrics: Optional[Dict] = None) -> Dict:
        """
        Update curriculum state and return current configuration.
        
        Args:
            step: Current training step
            eval_metrics: Latest evaluation metrics (if available)
            
        Returns:
            Current environment configuration
        """
        self.current_step = step
        current_stage = self.get_current_stage()
        
        if current_stage is None:
            return {}
        
        # Record evaluation metrics
        if eval_metrics:
            current_stage.performance_history.append(eval_metrics.get('eval_reward_mean', 0.0))
            self.evaluation_history.append({
                'step': step,
                'stage': current_stage.name,
                'metrics': eval_metrics.copy()
            })
            
            # Check for stage advancement
            if self.should_advance_stage(eval_metrics):
                self.advance_stage()
        
        # Get current configuration
        config = self.get_environment_config()
        
        # Add progress information
        stage_start_step = sum(s.duration_steps for s in self.stages[:self.current_stage_idx])
        stage_progress = (step - stage_start_step) / current_stage.duration_steps
        config['_curriculum_info'] = {
            'stage_name': current_stage.name,
            'stage_index': self.current_stage_idx,
            'stage_progress': min(stage_progress, 1.0),
            'total_stages': len(self.stages)
        }
        
        return config
    
    def get_training_stats(self) -> Dict:
        """Get comprehensive curriculum training statistics."""
        current_stage = self.get_current_stage()
        completed_stages = [s for s in self.stages if s.completed]
        
        stats = {
            'curriculum_enabled': len(self.stages) > 0,
            'total_stages': len(self.stages),
            'completed_stages': len(completed_stages),
            'current_stage': current_stage.name if current_stage else 'completed',
            'current_stage_index': self.current_stage_idx,
            'stage_transition_steps': self.stage_transition_steps.copy(),
            'total_evaluations': len(self.evaluation_history)
        }
        
        # Performance summary for each stage
        stage_performance = {}
        for stage in self.stages:
            if stage.performance_history:
                stage_performance[stage.name] = {
                    'evaluations': len(stage.performance_history),
                    'best_reward': max(stage.performance_history),
                    'final_reward': stage.performance_history[-1],
                    'improvement': stage.performance_history[-1] - stage.performance_history[0] if len(stage.performance_history) > 1 else 0.0,
                    'completed': stage.completed
                }
        
        stats['stage_performance'] = stage_performance
        return stats
    
    def save_curriculum_data(self, output_dir: Path):
        """Save curriculum learning data for analysis."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save curriculum statistics
        stats = self.get_training_stats()
        stats_file = output_dir / "curriculum_stats.json"
        
        import json
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save evaluation history
        if self.evaluation_history:
            import pandas as pd
            df = pd.DataFrame(self.evaluation_history)
            df.to_csv(output_dir / "curriculum_evaluation_history.csv", index=False)
        
        self.logger.info(f"Saved curriculum data to {output_dir}")
    
    def is_curriculum_complete(self) -> bool:
        """Check if all curriculum stages are completed."""
        return self.current_stage_idx >= len(self.stages)
    
    def get_adaptive_hyperparameters(self) -> Dict:
        """Get stage-adaptive hyperparameters."""
        current_stage = self.get_current_stage()
        if current_stage is None:
            return {}
        
        # Adaptive hyperparameters based on stage
        stage_hyperparams = {
            'basic_stabilization': {
                'batch_size': 128,  # Smaller batches for initial learning
                'learning_starts': 1000,  # Less exploration needed
                'train_freq': 4,  # More frequent training
                'exploration_noise': 0.3  # Higher exploration
            },
            'moderate_disturbances': {
                'batch_size': 256,  # Medium batches
                'learning_starts': 2000,
                'train_freq': 8,
                'exploration_noise': 0.2  # Reduced exploration
            },
            'full_challenge': {
                'batch_size': 512,  # Larger batches for stable learning
                'learning_starts': 5000,  # More exploration for complex environment
                'train_freq': 10,
                'exploration_noise': 0.1  # Minimal exploration
            }
        }
        
        return stage_hyperparams.get(current_stage.name, {})
