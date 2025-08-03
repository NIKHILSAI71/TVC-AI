#!/usr/bin/env python3
"""
Training Stability Manager
Implements state-of-the-art techniques to prevent catastrophic forgetting and improve training stability:
- Learning rate scheduling
- Primacy bias mitigation  
- Loss of plasticity prevention
- Reward hacking detection
- Dormant neuron reinitialization
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
from dataclasses import dataclass
from enum import Enum
import math
import warnings

logger = logging.getLogger(__name__)

class SchedulerType(Enum):
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    PLATEAU = "plateau"
    WARMUP_COSINE = "warmup_cosine"

@dataclass
class StabilityConfig:
    # Learning rate scheduling
    enable_lr_scheduling: bool = True
    scheduler_type: SchedulerType = SchedulerType.WARMUP_COSINE
    initial_lr_factor: float = 0.1  # Start with 10% of base LR
    warmup_steps: int = 10000
    decay_factor: float = 0.5
    patience: int = 20000  # For plateau scheduler
    
    # Plasticity preservation
    enable_plasticity_preservation: bool = True
    use_crelu: bool = True  # Concatenated ReLU
    dormant_threshold: float = 0.01  # Threshold for dormant neurons
    dormant_check_interval: int = 5000
    reinit_dormant_ratio: float = 0.1  # Ratio of dormant neurons to reinit
    
    # Primacy bias mitigation
    enable_primacy_mitigation: bool = True
    reset_interval: int = 50000  # Periodic weight reset interval
    reset_ratio: float = 0.05  # Ratio of weights to reset
    
    # Target network updates
    adaptive_tau: bool = True
    tau_min: float = 0.001
    tau_max: float = 0.01
    tau_decay: float = 0.999
    
    # Reward hacking detection
    enable_reward_hacking_detection: bool = True
    reward_window_size: int = 100
    hacking_threshold: float = 0.7
    success_reward_threshold: float = 0.3  # Min success rate for valid rewards
    
    # Gradient management  
    enable_gradient_surgery: bool = True
    gradient_conflict_threshold: float = 0.1
    gradient_clip_value: float = 1.0
    
    # Regularization
    enable_init_regularization: bool = True
    init_reg_weight: float = 0.001

class LearningRateScheduler:
    """Advanced learning rate scheduler with multiple strategies"""
    
    def __init__(self, optimizer: torch.optim.Optimizer, config: StabilityConfig, 
                 total_steps: int):
        self.optimizer = optimizer
        self.config = config
        self.total_steps = total_steps
        self.current_step = 0
        self.best_metric = float('-inf')
        self.patience_counter = 0
        
        # Store initial learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        logger.info(f"📈 LR Scheduler initialized: {config.scheduler_type.value}")
        logger.info(f"   Base LRs: {self.base_lrs}")
        logger.info(f"   Total steps: {total_steps}")
    
    def step(self, metric: Optional[float] = None) -> Dict[str, float]:
        """Update learning rates"""
        self.current_step += 1
        
        if self.config.scheduler_type == SchedulerType.LINEAR:
            factor = self._linear_schedule()
        elif self.config.scheduler_type == SchedulerType.EXPONENTIAL:
            factor = self._exponential_schedule()
        elif self.config.scheduler_type == SchedulerType.COSINE:
            factor = self._cosine_schedule()
        elif self.config.scheduler_type == SchedulerType.WARMUP_COSINE:
            factor = self._warmup_cosine_schedule()
        elif self.config.scheduler_type == SchedulerType.PLATEAU:
            factor = self._plateau_schedule(metric)
        else:
            factor = 1.0
        
        # Apply factor to all parameter groups
        current_lrs = []
        for i, group in enumerate(self.optimizer.param_groups):
            new_lr = self.base_lrs[i] * factor
            group['lr'] = new_lr
            current_lrs.append(new_lr)
        
        return {
            'lr_factor': factor,
            'current_lrs': current_lrs,
            'step': self.current_step
        }
    
    def _linear_schedule(self) -> float:
        """Linear decay schedule"""
        progress = self.current_step / self.total_steps
        return max(0.1, 1.0 - progress)
    
    def _exponential_schedule(self) -> float:
        """Exponential decay schedule"""
        decay_steps = self.total_steps // 10
        return self.config.decay_factor ** (self.current_step // decay_steps)
    
    def _cosine_schedule(self) -> float:
        """Cosine annealing schedule"""
        progress = self.current_step / self.total_steps
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    def _warmup_cosine_schedule(self) -> float:
        """Warmup followed by cosine annealing"""
        if self.current_step < self.config.warmup_steps:
            # Warmup phase
            warmup_factor = self.config.initial_lr_factor
            progress = self.current_step / self.config.warmup_steps
            return warmup_factor + (1.0 - warmup_factor) * progress
        else:
            # Cosine annealing phase
            steps_after_warmup = self.current_step - self.config.warmup_steps
            total_cosine_steps = self.total_steps - self.config.warmup_steps
            progress = steps_after_warmup / total_cosine_steps
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    def _plateau_schedule(self, metric: Optional[float]) -> float:
        """Reduce on plateau schedule"""
        if metric is not None:
            if metric > self.best_metric:
                self.best_metric = metric
                self.patience_counter = 0
            else:
                self.patience_counter += 1
        
        num_reductions = self.patience_counter // self.config.patience
        return self.config.decay_factor ** num_reductions

class PlasticityPreserver:
    """Prevents loss of plasticity using multiple techniques"""
    
    def __init__(self, config: StabilityConfig):
        self.config = config
        self.neuron_activations = {}  # Track neuron activation statistics
        self.weight_magnitudes = {}   # Track weight magnitude statistics
        
        logger.info("🧠 Plasticity Preserver initialized")
    
    def apply_crelu(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Concatenated ReLU to prevent dead neurons"""
        if self.config.use_crelu:
            positive = torch.relu(x)
            negative = torch.relu(-x)
            return torch.cat([positive, negative], dim=-1)
        return torch.relu(x)
    
    def track_activations(self, model: nn.Module, input_batch: torch.Tensor):
        """Track neuron activations to detect dormant neurons"""
        if not self.config.enable_plasticity_preservation:
            return
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation_mean = output.abs().mean().item()
                    if name not in self.neuron_activations:
                        self.neuron_activations[name] = deque(maxlen=100)
                    self.neuron_activations[name].append(activation_mean)
            return hook
        
        # Register hooks for all linear and conv layers
        handles = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                handle = module.register_forward_hook(hook_fn(name))
                handles.append(handle)
        
        # Forward pass to collect activations
        with torch.no_grad():
            model(input_batch)
        
        # Remove hooks
        for handle in handles:
            handle.remove()
    
    def detect_dormant_neurons(self) -> Dict[str, List[int]]:
        """Detect dormant neurons across the network"""
        dormant_neurons = {}
        
        for layer_name, activations in self.neuron_activations.items():
            if len(activations) > 10:  # Need sufficient history
                mean_activation = np.mean(activations)
                if mean_activation < self.config.dormant_threshold:
                    dormant_neurons[layer_name] = [0]  # Simplified detection
        
        return dormant_neurons
    
    def reinitialize_dormant_neurons(self, model: nn.Module, 
                                   dormant_neurons: Dict[str, List[int]]):
        """Reinitialize weights of dormant neurons"""
        if not dormant_neurons:
            return
        
        reinitialized_count = 0
        for name, module in model.named_modules():
            if name in dormant_neurons and isinstance(module, nn.Linear):
                # Reinitialize random subset of weights
                with torch.no_grad():
                    weight_shape = module.weight.shape
                    num_to_reinit = int(weight_shape[0] * self.config.reinit_dormant_ratio)
                    
                    if num_to_reinit > 0:
                        indices = torch.randperm(weight_shape[0])[:num_to_reinit]
                        nn.init.xavier_uniform_(module.weight[indices])
                        if module.bias is not None:
                            nn.init.zeros_(module.bias[indices])
                        reinitialized_count += num_to_reinit
        
        if reinitialized_count > 0:
            logger.info(f"🔄 Reinitialized {reinitialized_count} dormant neurons")

class PrimacyBiasMitigator:
    """Mitigates primacy bias through periodic weight resets"""
    
    def __init__(self, config: StabilityConfig):
        self.config = config
        self.step_count = 0
        self.initial_weights = {}
        
        logger.info("🎯 Primacy Bias Mitigator initialized")
    
    def store_initial_weights(self, model: nn.Module):
        """Store initial weights for potential reset"""
        self.initial_weights = {}
        for name, param in model.named_parameters():
            self.initial_weights[name] = param.data.clone()
    
    def check_and_reset(self, model: nn.Module) -> bool:
        """Check if reset is needed and perform it"""
        self.step_count += 1
        
        if (self.config.enable_primacy_mitigation and 
            self.step_count % self.config.reset_interval == 0):
            
            self.partial_weight_reset(model)
            return True
        return False
    
    def partial_weight_reset(self, model: nn.Module):
        """Perform partial weight reset to combat primacy bias"""
        reset_count = 0
        
        for name, param in model.named_parameters():
            if name in self.initial_weights and 'weight' in name:
                # Reset a random subset of weights
                with torch.no_grad():
                    flat_param = param.data.flatten()
                    num_weights = len(flat_param)
                    num_to_reset = int(num_weights * self.config.reset_ratio)
                    
                    if num_to_reset > 0:
                        indices = torch.randperm(num_weights)[:num_to_reset]
                        flat_initial = self.initial_weights[name].flatten()
                        flat_param[indices] = flat_initial[indices]
                        param.data = flat_param.reshape(param.shape)
                        reset_count += num_to_reset
        
        logger.info(f"🔄 Reset {reset_count} weights to combat primacy bias")

class RewardHackingDetector:
    """Detects and mitigates reward hacking patterns"""
    
    def __init__(self, config: StabilityConfig):
        self.config = config
        self.reward_history = deque(maxlen=config.reward_window_size)
        self.success_history = deque(maxlen=config.reward_window_size)
        self.episode_length_history = deque(maxlen=config.reward_window_size)
        
        logger.info("🛡️ Reward Hacking Detector initialized")
    
    def add_episode(self, total_reward: float, success: bool, episode_length: int):
        """Add episode data for analysis"""
        self.reward_history.append(total_reward)
        self.success_history.append(success)
        self.episode_length_history.append(episode_length)
    
    def detect_hacking(self) -> Dict[str, float]:
        """Detect potential reward hacking with multiple indicators"""
        if len(self.reward_history) < 20:
            return {'hacking_score': 0.0, 'confidence': 0.0}
        
        rewards = np.array(list(self.reward_history))
        successes = np.array(list(self.success_history))
        lengths = np.array(list(self.episode_length_history))
        
        indicators = {}
        
        # 1. High reward with low success rate
        recent_rewards = rewards[-20:]
        recent_successes = successes[-20:]
        
        mean_reward = np.mean(recent_rewards)
        success_rate = np.mean(recent_successes)
        
        if mean_reward > 1000 and success_rate < self.config.success_reward_threshold:
            indicators['reward_success_mismatch'] = min(1.0, mean_reward / 2000)
        else:
            indicators['reward_success_mismatch'] = 0.0
        
        # 2. Excessive episode length without success
        mean_length = np.mean(lengths[-20:])
        if mean_length > 800 and success_rate < 0.1:
            indicators['length_exploitation'] = min(1.0, mean_length / 1000)
        else:
            indicators['length_exploitation'] = 0.0
        
        # 3. Reward variance without success variance
        reward_variance = np.var(recent_rewards)
        success_variance = np.var(recent_successes)
        
        if reward_variance > 5000 and success_variance < 0.01:
            indicators['variance_mismatch'] = min(1.0, reward_variance / 10000)
        else:
            indicators['variance_mismatch'] = 0.0
        
        # 4. Sudden reward spikes
        if len(recent_rewards) >= 10:
            reward_diff = np.diff(recent_rewards)
            max_spike = np.max(reward_diff) if len(reward_diff) > 0 else 0
            if max_spike > 500 and success_rate < 0.2:
                indicators['reward_spikes'] = min(1.0, max_spike / 1000)
            else:
                indicators['reward_spikes'] = 0.0
        else:
            indicators['reward_spikes'] = 0.0
        
        # Overall hacking score
        hacking_score = np.mean(list(indicators.values()))
        confidence = min(1.0, len(self.reward_history) / self.config.reward_window_size)
        
        return {
            'hacking_score': hacking_score,
            'confidence': confidence,
            'indicators': indicators,
            'mean_reward': mean_reward,
            'success_rate': success_rate,
            'mean_episode_length': mean_length
        }
    
    def get_penalty_factor(self) -> float:
        """Get penalty factor based on hacking detection"""
        detection_result = self.detect_hacking()
        hacking_score = detection_result['hacking_score']
        
        if hacking_score > self.config.hacking_threshold:
            # Apply penalty to learning rate or entropy
            penalty = 0.5 + 0.5 * (1.0 - hacking_score)
            logger.warning(f"⚠️ Reward hacking detected (score: {hacking_score:.3f}), applying penalty: {penalty:.3f}")
            return penalty
        
        return 1.0

class TrainingStabilityManager:
    """Main manager coordinating all stability techniques"""
    
    def __init__(self, config: StabilityConfig, total_training_steps: int):
        self.config = config
        self.total_steps = total_training_steps
        
        # Initialize components
        self.lr_scheduler = None  # Will be set when optimizer is provided
        self.plasticity_preserver = PlasticityPreserver(config)
        self.primacy_mitigator = PrimacyBiasMitigator(config)
        self.reward_detector = RewardHackingDetector(config)
        
        # Adaptive parameters
        self.current_tau = config.tau_max
        self.step_count = 0
        
        logger.info("🎛️ Training Stability Manager initialized")
        self._log_config()
    
    def _log_config(self):
        """Log configuration details"""
        logger.info("="*60)
        logger.info("STABILITY CONFIGURATION")
        logger.info("="*60)
        logger.info(f"LR Scheduling: {self.config.enable_lr_scheduling}")
        logger.info(f"Plasticity Preservation: {self.config.enable_plasticity_preservation}")
        logger.info(f"Primacy Mitigation: {self.config.enable_primacy_mitigation}")
        logger.info(f"Reward Hacking Detection: {self.config.enable_reward_hacking_detection}")
        logger.info(f"Total Training Steps: {self.total_steps}")
        logger.info("="*60)
    
    def setup_optimizer(self, optimizer: torch.optim.Optimizer):
        """Setup learning rate scheduler with optimizer"""
        if self.config.enable_lr_scheduling:
            self.lr_scheduler = LearningRateScheduler(
                optimizer, self.config, self.total_steps
            )
    
    def setup_model(self, model: nn.Module):
        """Initial model setup"""
        if self.config.enable_primacy_mitigation:
            self.primacy_mitigator.store_initial_weights(model)
    
    def step(self, model: nn.Module, optimizer: torch.optim.Optimizer,
             episode_reward: float, episode_success: bool, episode_length: int,
             sample_batch: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Perform stability management step"""
        self.step_count += 1
        
        results = {
            'step': self.step_count,
            'stability_actions': []
        }
        
        # Update learning rate
        if self.lr_scheduler is not None:
            lr_info = self.lr_scheduler.step(episode_reward)
            results['lr_info'] = lr_info
        
        # Track episode for reward hacking detection
        self.reward_detector.add_episode(episode_reward, episode_success, episode_length)
        hacking_info = self.reward_detector.detect_hacking()
        results['hacking_info'] = hacking_info
        
        # Check for primacy bias reset
        if self.primacy_mitigator.check_and_reset(model):
            results['stability_actions'].append('primacy_reset')
        
        # Update adaptive tau
        if self.config.adaptive_tau:
            self.current_tau = max(
                self.config.tau_min,
                self.current_tau * self.config.tau_decay
            )
            results['current_tau'] = self.current_tau
        
        # Check plasticity every interval
        if (self.config.enable_plasticity_preservation and 
            sample_batch is not None and 
            self.step_count % self.config.dormant_check_interval == 0):
            
            self.plasticity_preserver.track_activations(model, sample_batch)
            dormant_neurons = self.plasticity_preserver.detect_dormant_neurons()
            
            if dormant_neurons:
                self.plasticity_preserver.reinitialize_dormant_neurons(model, dormant_neurons)
                results['stability_actions'].append('dormant_reinit')
                results['dormant_neurons'] = len(dormant_neurons)
        
        return results
    
    def get_tau(self) -> float:
        """Get current tau value for target network updates"""
        return self.current_tau
    
    def should_stop_training(self) -> Tuple[bool, str]:
        """Check if training should be stopped due to stability issues"""
        hacking_info = self.reward_detector.detect_hacking()
        
        # Stop if severe reward hacking is detected
        if (hacking_info['hacking_score'] > 0.9 and 
            hacking_info['confidence'] > 0.8):
            return True, f"Severe reward hacking detected (score: {hacking_info['hacking_score']:.3f})"
        
        return False, ""
    
    def get_penalty_factor(self) -> float:
        """Get penalty factor for training based on stability analysis"""
        return self.reward_detector.get_penalty_factor()

# Utility function to create stability manager with recommended settings
def create_stability_manager(total_steps: int, 
                           conservative: bool = True) -> TrainingStabilityManager:
    """
    Create stability manager with recommended settings.
    
    Args:
        total_steps: Total training steps
        conservative: Whether to use conservative (more stable) settings
    """
    if conservative:
        config = StabilityConfig(
            # Conservative LR scheduling
            scheduler_type=SchedulerType.WARMUP_COSINE,
            initial_lr_factor=0.05,  # Start very low
            warmup_steps=total_steps // 20,  # 5% warmup
            
            # Aggressive plasticity preservation
            dormant_threshold=0.005,  # Lower threshold
            dormant_check_interval=2000,  # More frequent checks
            reinit_dormant_ratio=0.15,  # More reinitialization
            
            # Conservative primacy mitigation
            reset_interval=25000,  # More frequent resets
            reset_ratio=0.03,  # Smaller resets
            
            # Strict reward hacking detection
            hacking_threshold=0.6,  # Lower threshold
            success_reward_threshold=0.4,  # Higher success requirement
        )
    else:
        config = StabilityConfig()  # Default settings
    
    return TrainingStabilityManager(config, total_steps)
