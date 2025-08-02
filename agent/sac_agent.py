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
Soft Actor-Critic (SAC) Implementation for Rocket TVC Control

This module implements the SAC algorithm optimized for continuous control
of thrust vector control systems. SAC uses maximum entropy reinforcement
learning to encourage exploration while learning an optimal policy.

Key Features:
- Continuous action space control
- Entropy regularization for robust exploration
- Separate actor and critic networks
- Target networks for stable learning
- Experience replay buffer
- Automatic entropy temperature tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import logging
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import os
import json

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class SACConfig:
    """Configuration parameters for SAC algorithm."""
    # Network architecture
    hidden_dims: List[int] = None  # Will default to [256, 256]
    activation: str = "relu"  # "relu", "tanh", "elu", "swish"
    layer_norm: bool = False  # Use layer normalization
    dropout: float = 0.0  # Dropout probability
    
    # Learning parameters
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    
    # SAC hyperparameters
    gamma: float = 0.99  # Discount factor
    tau: float = 0.005   # Soft update coefficient
    alpha: float = 0.2   # Initial entropy coefficient
    automatic_entropy_tuning: bool = True
    target_entropy: float = None  # Will be set to -action_dim if None
    
    # Training parameters
    batch_size: int = 256
    buffer_size: int = 1000000
    learning_starts: int = 1000  # Steps before training begins
    train_freq: int = 1  # Training frequency
    gradient_steps: int = 1  # Gradient steps per training step
    
    # Exploration and stability improvements
    action_noise: float = 0.1
    exploration_noise_decay: float = 0.995  # Decay rate for exploration noise
    min_exploration_noise: float = 0.01  # Minimum exploration noise
    gradient_clip_norm: float = 10.0  # Gradient clipping norm
    
    # Curriculum learning parameters
    curriculum_learning: bool = True
    curriculum_stages: int = 4  # Number of curriculum stages
    curriculum_steps_per_stage: int = 50000  # Steps per curriculum stage
    
    # Performance monitoring
    eval_freq: int = 5000  # Evaluation frequency during training
    save_freq: int = 10000  # Model saving frequency
    early_stopping_patience: int = 10  # Early stopping patience (in eval cycles)
    target_success_rate: float = 0.8  # Target success rate for curriculum progression
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]  # Default to minimal for faster training
        if self.target_entropy is None:
            self.target_entropy = -2.0  # Will be set properly in agent initialization


class ReplayBuffer:
    """Experience replay buffer for SAC."""
    
    def __init__(self, obs_dim: int, action_dim: int, max_size: int = 1000000):
        """
        Initialize replay buffer.
        
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension  
            max_size: Maximum buffer size
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        # Allocate memory
        self.observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=np.float32)
    
    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, 
            next_obs: np.ndarray, done: bool):
        """Add experience to buffer."""
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_observations[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        
        batch = (
            torch.FloatTensor(self.observations[idxs]).to(device),
            torch.FloatTensor(self.actions[idxs]).to(device),
            torch.FloatTensor(self.rewards[idxs]).to(device),
            torch.FloatTensor(self.next_observations[idxs]).to(device),
            torch.FloatTensor(self.dones[idxs]).to(device)
        )
        
        return batch
    
    def __len__(self):
        return self.size


class MLP(nn.Module):
    """Multi-Layer Perceptron with configurable architecture and improved stability."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], 
                 activation: str = "relu", output_activation: Optional[str] = None,
                 use_layer_norm: bool = False, dropout: float = 0.0):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            output_activation: Output layer activation (None for linear)
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "swish":
            self.activation = nn.SiLU()  # Swish/SiLU activation
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList() if use_layer_norm else None
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
            # Initialize weights with proper scaling
            if i < len(dims) - 2:  # Not the output layer
                nn.init.xavier_uniform_(self.layers[-1].weight)
                nn.init.zeros_(self.layers[-1].bias)
            
            if use_layer_norm and i < len(dims) - 2:  # Layer norm for hidden layers only
                self.layer_norms.append(nn.LayerNorm(dims[i + 1]))
        
        # Output activation
        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation is None:
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            if self.layer_norms is not None:
                x = self.layer_norms[i](x)
            x = self.activation(x)
            if self.dropout is not None:
                x = self.dropout(x)
        
        x = self.layers[-1](x)
        return self.output_activation(x)


class Actor(nn.Module):
    """SAC Actor network with squashed Gaussian policy and improved stability."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], 
                 activation: str = "relu", action_scale: float = 1.0,
                 use_layer_norm: bool = False, dropout: float = 0.0):
        """
        Initialize Actor network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            action_scale: Scaling factor for actions
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.action_scale = action_scale
        
        # Shared network
        self.shared_net = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1], activation,
                             use_layer_norm=use_layer_norm, dropout=dropout)
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # Initialize output layers with smaller weights for stability
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)
        
        # Constrain log_std to reasonable range
        self.log_std_min = -20
        self.log_std_max = 2
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to get action distribution parameters.
        
        Args:
            obs: Observation tensor
            
        Returns:
            mean: Action mean
            log_std: Action log standard deviation
        """
        shared = self.shared_net(obs)
        
        mean = self.mean_head(shared)
        log_std = self.log_std_head(shared)
        
        # Constrain log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample actions from the policy.
        
        Args:
            obs: Observation tensor
            
        Returns:
            action: Sampled action (squashed)
            log_prob: Log probability of the action
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Sample from normal distribution
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        
        # Squash action using tanh
        action = torch.tanh(x_t) * self.action_scale
        
        # Compute log probability with change of variables formula
        log_prob = normal.log_prob(x_t)
        # Subtract log derivative of tanh to account for squashing
        log_prob -= torch.log(self.action_scale * (1 - action.pow(2) / self.action_scale**2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """
        Get action for inference.
        
        Args:
            obs: Observation tensor
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Action tensor
        """
        with torch.no_grad():
            if deterministic:
                mean, _ = self.forward(obs)
                action = torch.tanh(mean) * self.action_scale
            else:
                action, _ = self.sample(obs)
        
        return action


class Critic(nn.Module):
    """SAC Critic network (Q-function) with improved stability."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], 
                 activation: str = "relu", use_layer_norm: bool = False, 
                 dropout: float = 0.0):
        """
        Initialize Critic network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.q_net = MLP(obs_dim + action_dim, 1, hidden_dims, activation,
                        use_layer_norm=use_layer_norm, dropout=dropout)
        
        # Initialize output layer with smaller weights
        if hasattr(self.q_net.layers[-1], 'weight'):
            nn.init.uniform_(self.q_net.layers[-1].weight, -3e-3, 3e-3)
            nn.init.uniform_(self.q_net.layers[-1].bias, -3e-3, 3e-3)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            obs: Observation tensor
            action: Action tensor
            
        Returns:
            Q-value tensor
        """
        x = torch.cat([obs, action], dim=1)
        return self.q_net(x)


class SACAgent:
    """Enhanced Soft Actor-Critic Agent for TVC rocket control."""
    
    def __init__(self, obs_dim: int, action_dim: int, config: SACConfig = None):
        """
        Initialize SAC agent.
        
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            config: SAC configuration
        """
        self.config = config or SACConfig()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Set target entropy if not specified
        if self.config.target_entropy is None:
            self.config.target_entropy = -action_dim
        
        # Initialize networks with configuration parameters
        network_kwargs = {
            'use_layer_norm': getattr(self.config, 'layer_norm', False),
            'dropout': getattr(self.config, 'dropout', 0.0)
        }
        
        self.actor = Actor(obs_dim, action_dim, self.config.hidden_dims, 
                          self.config.activation, **network_kwargs).to(device)
        
        self.critic1 = Critic(obs_dim, action_dim, self.config.hidden_dims, 
                             self.config.activation, **network_kwargs).to(device)
        self.critic2 = Critic(obs_dim, action_dim, self.config.hidden_dims, 
                             self.config.activation, **network_kwargs).to(device)
        
        # Target networks
        self.target_critic1 = Critic(obs_dim, action_dim, self.config.hidden_dims, 
                                   self.config.activation, **network_kwargs).to(device)
        self.target_critic2 = Critic(obs_dim, action_dim, self.config.hidden_dims, 
                                   self.config.activation, **network_kwargs).to(device)
        
        # Initialize target networks
        self._hard_update(self.target_critic1, self.critic1)
        self._hard_update(self.target_critic2, self.critic2)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config.lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.lr_critic)
        
        # Automatic entropy tuning
        if self.config.automatic_entropy_tuning:
            self.target_entropy = self.config.target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr_alpha)
        else:
            self.alpha = self.config.alpha
            self.log_alpha = None
            self.alpha_optimizer = None
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(obs_dim, action_dim, self.config.buffer_size)
        
        # Training statistics and performance monitoring
        self.total_steps = 0
        self.training_steps = 0
        self.current_exploration_noise = self.config.action_noise
        self.curriculum_stage = 0
        self.best_eval_reward = -float('inf')
        self.eval_rewards_history = []
        self.episodes_since_improvement = 0
        
        # Performance tracking
        self.training_metrics = {
            'actor_loss': [],
            'critic1_loss': [],
            'critic2_loss': [],
            'alpha_loss': [],
            'alpha_value': [],
            'q_value_mean': [],
            'policy_entropy': []
        }
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Enhanced SAC Agent initialized:")
        self.logger.info(f"  Observation dim: {obs_dim}")
        self.logger.info(f"  Action dim: {action_dim}")  
        self.logger.info(f"  Hidden dims: {self.config.hidden_dims}")
        self.logger.info(f"  Curriculum learning: {self.config.curriculum_learning}")
        self.logger.info(f"  Device: {device}")
        self._build_networks()
        
        # Initialize optimizers
        self._build_optimizers()
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(obs_dim, action_dim, self.config.buffer_size)
        
        # Training statistics
        self.total_steps = 0
        self.training_steps = 0
        
        # Logging
        self.logger = logging.getLogger(__name__)
    
    def _build_networks(self):
        """Build actor and critic networks."""
        # Actor network
        self.actor = Actor(
            self.obs_dim, 
            self.action_dim, 
            self.config.hidden_dims,
            self.config.activation
        ).to(device)
        
        # Two Q-networks (double Q-learning)
        self.critic1 = Critic(
            self.obs_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation
        ).to(device)
        
        self.critic2 = Critic(
            self.obs_dim,
            self.action_dim, 
            self.config.hidden_dims,
            self.config.activation
        ).to(device)
        
        # Target networks
        self.target_critic1 = Critic(
            self.obs_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation
        ).to(device)
        
        self.target_critic2 = Critic(
            self.obs_dim,
            self.action_dim,
            self.config.hidden_dims,
            self.config.activation
        ).to(device)
        
        # Initialize target networks
        self._hard_update(self.target_critic1, self.critic1)
        self._hard_update(self.target_critic2, self.critic2)
        
        # Entropy temperature
        if self.config.automatic_entropy_tuning:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = self.config.alpha
    
    def _build_optimizers(self):
        """Build optimizers for networks."""
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), 
            lr=self.config.lr_actor
        )
        
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(),
            lr=self.config.lr_critic
        )
        
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(),
            lr=self.config.lr_critic
        )
        
        if self.config.automatic_entropy_tuning:
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha],
                lr=self.config.lr_alpha
            )
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False, 
                     add_noise: bool = True) -> np.ndarray:
        """
        Select action with improved exploration and curriculum learning.
        
        Args:
            obs: Observation array
            deterministic: Whether to use deterministic policy
            add_noise: Whether to add exploration noise
            
        Returns:
            action: Selected action
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action_tensor = self.actor.get_action(obs_tensor, deterministic)
        action = action_tensor.cpu().numpy().flatten()
        
        # Add exploration noise during training with adaptive noise
        if not deterministic and add_noise and self.current_exploration_noise > 0:
            noise = np.random.normal(0, self.current_exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def update_exploration_noise(self):
        """Update exploration noise with decay."""
        if self.current_exploration_noise > self.config.min_exploration_noise:
            self.current_exploration_noise *= self.config.exploration_noise_decay
            self.current_exploration_noise = max(
                self.current_exploration_noise, 
                self.config.min_exploration_noise
            )
    
    def update_curriculum_stage(self, success_rate: float):
        """
        Update curriculum learning stage based on performance.
        
        Args:
            success_rate: Current success rate from evaluation
        """
        if not self.config.curriculum_learning:
            return
        
        # Check if we should advance to the next curriculum stage
        if (success_rate >= self.config.target_success_rate and 
            self.curriculum_stage < self.config.curriculum_stages - 1):
            self.curriculum_stage += 1
            self.logger.info(f"Advanced to curriculum stage {self.curriculum_stage}")
            self.logger.info(f"Success rate: {success_rate:.2%}")
        
        # Check if we should progress based on steps
        stage_from_steps = min(
            self.total_steps // self.config.curriculum_steps_per_stage,
            self.config.curriculum_stages - 1
        )
        
        if stage_from_steps > self.curriculum_stage:
            self.curriculum_stage = stage_from_steps
            self.logger.info(f"Advanced to curriculum stage {self.curriculum_stage} (time-based)")
    
    def get_curriculum_difficulty(self) -> float:
        """
        Get current curriculum difficulty level.
        
        Returns:
            Difficulty level from 0.2 (easy) to 1.0 (full difficulty)
        """
        if not self.config.curriculum_learning:
            return 1.0
        
        # Progressive difficulty increase
        base_difficulty = 0.2
        max_difficulty = 1.0
        
        difficulty_per_stage = (max_difficulty - base_difficulty) / max(self.config.curriculum_stages - 1, 1)
        current_difficulty = base_difficulty + self.curriculum_stage * difficulty_per_stage
        
        return min(current_difficulty, max_difficulty)
    
    def store_transition(self, obs: np.ndarray, action: np.ndarray, reward: float,
                        next_obs: np.ndarray, done: bool):
        """Store transition in replay buffer."""
        self.replay_buffer.add(obs, action, reward, next_obs, done)
        self.total_steps += 1
    
    def update(self, batch_size: int = None) -> Dict[str, float]:
        """
        Perform one training update step.
        
        Args:
            batch_size: Size of the batch to sample (defaults to config batch_size)
            
        Returns:
            training_info: Dictionary with training statistics
        """
        if len(self.replay_buffer) < self.config.learning_starts:
            return {}
        
        if batch_size is None:
            batch_size = self.config.batch_size
            
        batch = self.replay_buffer.sample(batch_size)
        training_info = self._update_networks(batch)
        
        self.training_steps += 1
        
        return training_info
    
    def train_step(self) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Returns:
            Training information dictionary
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        # Perform one gradient step
        batch = self.replay_buffer.sample(self.config.batch_size)
        training_info = self._update_networks(batch)
        self.training_steps += 1
        
        return training_info

    def train(self) -> Dict[str, float]:
        """
        Train the agent.
        
        Returns:
            training_info: Dictionary with training statistics
        """
        if len(self.replay_buffer) < self.config.learning_starts:
            return {}
        
        if self.total_steps % self.config.train_freq != 0:
            return {}
        
        training_info = {}
        
        for _ in range(self.config.gradient_steps):
            batch = self.replay_buffer.sample(self.config.batch_size)
            info = self._update_networks(batch)
            
            # Accumulate training info
            for key, value in info.items():
                if key not in training_info:
                    training_info[key] = 0
                training_info[key] += value
        
        # Average over gradient steps
        for key in training_info:
            training_info[key] /= self.config.gradient_steps
        
        self.training_steps += 1
        
        return training_info
    
    def _update_networks(self, batch: Tuple[torch.Tensor, ...]) -> Dict[str, float]:
        """Update actor and critic networks."""
        obs, actions, rewards, next_obs, dones = batch
        
        # Update critics
        critic_info = self._update_critics(obs, actions, rewards, next_obs, dones)
        
        # Update actor
        actor_info = self._update_actor(obs)
        
        # Update entropy temperature
        alpha_info = self._update_alpha(obs)
        
        # Soft update target networks
        self._soft_update(self.target_critic1, self.critic1)
        self._soft_update(self.target_critic2, self.critic2)
        
        # Update exploration noise
        self.update_exploration_noise()
        
        # Store training metrics
        self.training_metrics['actor_loss'].append(actor_info.get('actor_loss', 0))
        self.training_metrics['critic1_loss'].append(critic_info.get('critic1_loss', 0))
        self.training_metrics['critic2_loss'].append(critic_info.get('critic2_loss', 0))
        self.training_metrics['alpha_loss'].append(alpha_info.get('alpha_loss', 0))
        self.training_metrics['alpha_value'].append(alpha_info.get('alpha', 0))
        self.training_metrics['q_value_mean'].append(critic_info.get('q_value_mean', 0))
        self.training_metrics['policy_entropy'].append(actor_info.get('policy_entropy', 0))
        
        # Combine info
        training_info = {**critic_info, **actor_info, **alpha_info}
        training_info['exploration_noise'] = self.current_exploration_noise
        training_info['curriculum_stage'] = self.curriculum_stage
        
        return training_info
    
    def _update_critics(self, obs: torch.Tensor, actions: torch.Tensor, 
                       rewards: torch.Tensor, next_obs: torch.Tensor, 
                       dones: torch.Tensor) -> Dict[str, float]:
        """Update critic networks."""
        with torch.no_grad():
            # Sample next actions from current policy
            next_actions, next_log_probs = self.actor.sample(next_obs)
            
            # Compute target Q-values
            target_q1 = self.target_critic1(next_obs, next_actions)
            target_q2 = self.target_critic2(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            
            target_q = rewards.unsqueeze(1) + self.config.gamma * (1 - dones.unsqueeze(1)) * target_q
        
        # Current Q-values
        current_q1 = self.critic1(obs, actions)
        current_q2 = self.critic2(obs, actions)
        
        # Critic losses
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.config.gradient_clip_norm)
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.config.gradient_clip_norm)
        self.critic2_optimizer.step()
        
        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "q_value_mean": torch.mean(current_q1).item()
        }
    
    def _update_actor(self, obs: torch.Tensor) -> Dict[str, float]:
        """Update actor network."""
        # Sample actions from current policy
        actions, log_probs = self.actor.sample(obs)
        
        # Compute Q-values for sampled actions
        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        q = torch.min(q1, q2)
        
        # Actor loss (maximize Q-value while minimizing entropy)
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip_norm)
        self.actor_optimizer.step()
        
        return {
            "actor_loss": actor_loss.item(),
            "policy_entropy": -log_probs.mean().item(),
            "q_actor_mean": q.mean().item()
        }
    
    def _update_alpha(self, obs: torch.Tensor) -> Dict[str, float]:
        """Update entropy temperature."""
        if not self.config.automatic_entropy_tuning:
            return {"alpha": self.alpha}
        
        with torch.no_grad():
            _, log_probs = self.actor.sample(obs)
        
        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_probs + self.config.target_entropy)).mean()
        
        # Update alpha
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        return {
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item()
        }
    
    def _soft_update(self, target: nn.Module, source: nn.Module):
        """Soft update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.tau) + param.data * self.config.tau
            )
    
    def _hard_update(self, target: nn.Module, source: nn.Module):
        """Hard update target network."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save(self, filepath: str):
        """Save agent state with complete configuration."""
        state = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "training_steps": self.training_steps,
            "config": self.config.__dict__,
            # Save architecture info for compatibility checking
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.config.hidden_dims,
            "activation": self.config.activation
        }
        
        if self.config.automatic_entropy_tuning:
            state["log_alpha"] = self.log_alpha.data
            state["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        
        torch.save(state, filepath)
        self.logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str, strict: bool = False):
        """
        Load agent state with architecture compatibility checking.
        
        Args:
            filepath: Path to saved model
            strict: Whether to enforce strict architecture matching
        """
        state = torch.load(filepath, map_location=device)
        
        # Check architecture compatibility
        saved_obs_dim = state.get("obs_dim", self.obs_dim)
        saved_action_dim = state.get("action_dim", self.action_dim)
        saved_hidden_dims = state.get("hidden_dims", self.config.hidden_dims)
        saved_activation = state.get("activation", self.config.activation)
        
        architecture_mismatch = (
            saved_obs_dim != self.obs_dim or
            saved_action_dim != self.action_dim or
            saved_hidden_dims != self.config.hidden_dims or
            saved_activation != self.config.activation
        )
        
        if architecture_mismatch:
            self.logger.warning("Architecture mismatch detected:")
            self.logger.warning(f"  Saved: obs_dim={saved_obs_dim}, action_dim={saved_action_dim}")
            self.logger.warning(f"         hidden_dims={saved_hidden_dims}, activation={saved_activation}")
            self.logger.warning(f"  Current: obs_dim={self.obs_dim}, action_dim={self.action_dim}")
            self.logger.warning(f"           hidden_dims={self.config.hidden_dims}, activation={self.config.activation}")
            
            if strict:
                raise ValueError("Architecture mismatch in strict mode")
            
            # Try to load with compatible architecture
            if "config" in state:
                try:
                    # Recreate networks with saved architecture
                    saved_config = SACConfig(**state["config"])
                    self._recreate_networks_with_config(saved_config)
                    self.logger.info("Recreated networks with saved configuration")
                except Exception as e:
                    self.logger.error(f"Failed to recreate networks: {e}")
                    if strict:
                        raise
                    else:
                        self.logger.warning("Proceeding with current architecture (some weights may not load)")
        
        # Load state dictionaries with error handling
        try:
            self.actor.load_state_dict(state["actor"], strict=strict)
            self.critic1.load_state_dict(state["critic1"], strict=strict)
            self.critic2.load_state_dict(state["critic2"], strict=strict)
            self.target_critic1.load_state_dict(state["target_critic1"], strict=strict)
            self.target_critic2.load_state_dict(state["target_critic2"], strict=strict)
            
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic1_optimizer.load_state_dict(state["critic1_optimizer"])
            self.critic2_optimizer.load_state_dict(state["critic2_optimizer"])
        except Exception as e:
            self.logger.error(f"Error loading state dict: {e}")
            if strict:
                raise
            else:
                self.logger.warning("Some weights may not have been loaded due to architecture mismatch")
        
        # Load entropy tuning parameters
        if self.config.automatic_entropy_tuning and "log_alpha" in state:
            try:
                self.log_alpha.data = state["log_alpha"]
                self.alpha = self.log_alpha.exp()
                self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
            except Exception as e:
                self.logger.warning(f"Failed to load alpha parameters: {e}")
        
        # Load training statistics
        self.total_steps = state.get("total_steps", 0)
        self.training_steps = state.get("training_steps", 0)
        
        self.logger.info(f"Agent loaded from {filepath}")
        self.logger.info(f"  Total steps: {self.total_steps}")
        self.logger.info(f"  Training steps: {self.training_steps}")
    
    def _recreate_networks_with_config(self, config: SACConfig):
        """Recreate networks with different configuration."""
        # Save current config and update with saved config
        old_config = self.config
        self.config = config
        
        # Recreate networks
        network_kwargs = {
            'use_layer_norm': getattr(self.config, 'layer_norm', False),
            'dropout': getattr(self.config, 'dropout', 0.0)
        }
        
        self.actor = Actor(self.obs_dim, self.action_dim, self.config.hidden_dims, 
                          self.config.activation, **network_kwargs).to(device)
        
        self.critic1 = Critic(self.obs_dim, self.action_dim, self.config.hidden_dims, 
                             self.config.activation, **network_kwargs).to(device)
        self.critic2 = Critic(self.obs_dim, self.action_dim, self.config.hidden_dims, 
                             self.config.activation, **network_kwargs).to(device)
        
        self.target_critic1 = Critic(self.obs_dim, self.action_dim, self.config.hidden_dims, 
                                   self.config.activation, **network_kwargs).to(device)
        self.target_critic2 = Critic(self.obs_dim, self.action_dim, self.config.hidden_dims, 
                                   self.config.activation, **network_kwargs).to(device)
        
        # Recreate optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.config.lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.config.lr_critic)
        
        # Handle entropy tuning parameters
        if self.config.automatic_entropy_tuning:
            if self.log_alpha is None:
                self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.config.lr_alpha)
            self.alpha = self.log_alpha.exp()
    
    @classmethod
    def load_from_checkpoint(cls, filepath: str, obs_dim: int = None, action_dim: int = None):
        """
        Load agent from checkpoint, automatically determining architecture.
        
        Args:
            filepath: Path to saved model
            obs_dim: Override observation dimension (if None, use saved value)
            action_dim: Override action dimension (if None, use saved value)
        
        Returns:
            Loaded SACAgent instance
        """
        state = torch.load(filepath, map_location=device)
        
        # Extract architecture info from saved state
        saved_obs_dim = state.get("obs_dim")
        saved_action_dim = state.get("action_dim")
        saved_config = state.get("config", {})
        
        # Use provided dimensions or fall back to saved ones
        final_obs_dim = obs_dim or saved_obs_dim
        final_action_dim = action_dim or saved_action_dim
        
        if final_obs_dim is None or final_action_dim is None:
            raise ValueError("Could not determine observation or action dimensions")
        
        # Create config from saved state
        config = SACConfig(**saved_config)
        
        # Create agent with correct architecture
        agent = cls(final_obs_dim, final_action_dim, config)
        
        # Load the state
        agent.load(filepath, strict=False)
        
        return agent


if __name__ == "__main__":
    # Test the SAC agent
    obs_dim = 8  # Rocket TVC observation dimension
    action_dim = 2  # Gimbal angles [pitch, yaw]
    
    config = SACConfig()
    agent = SACAgent(obs_dim, action_dim, config)
    
    print(f"Actor network: {agent.actor}")
    print(f"Critic network: {agent.critic1}")
    print(f"Device: {device}")
    
    # Test action selection
    obs = np.random.randn(obs_dim)
    action = agent.select_action(obs)
    print(f"Test action: {action}")
