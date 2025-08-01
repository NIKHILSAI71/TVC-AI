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
    activation: str = "relu"  # "relu", "tanh", "elu"
    
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
    
    # Exploration
    action_noise: float = 0.1
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 256]


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
    """Multi-Layer Perceptron with configurable architecture."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], 
                 activation: str = "relu", output_activation: Optional[str] = None):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
            activation: Activation function name
            output_activation: Output layer activation (None for linear)
        """
        super().__init__()
        
        # Choose activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Build layers
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()
        
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))
        
        # Output activation
        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        elif output_activation is None:
            self.output_activation = nn.Identity()
        else:
            raise ValueError(f"Unknown output activation: {output_activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        
        x = self.layers[-1](x)
        return self.output_activation(x)


class Actor(nn.Module):
    """SAC Actor network with squashed Gaussian policy."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], 
                 activation: str = "relu", action_scale: float = 1.0):
        """
        Initialize Actor network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
            action_scale: Scaling factor for actions
        """
        super().__init__()
        
        self.action_dim = action_dim
        self.action_scale = action_scale
        
        # Shared network
        self.shared_net = MLP(obs_dim, hidden_dims[-1], hidden_dims[:-1], activation)
        
        # Mean and log_std heads
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
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
    """SAC Critic network (Q-function)."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims: List[int], 
                 activation: str = "relu"):
        """
        Initialize Critic network.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            activation: Activation function
        """
        super().__init__()
        
        input_dim = obs_dim + action_dim
        self.q_net = MLP(input_dim, 1, hidden_dims, activation)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Q-value.
        
        Args:
            obs: Observation tensor
            action: Action tensor
            
        Returns:
            q_value: Q-value tensor
        """
        q_input = torch.cat([obs, action], dim=1)
        return self.q_net(q_input)


class SACAgent:
    """Soft Actor-Critic Agent for continuous control."""
    
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
        
        # Initialize networks
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
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Select action given observation.
        
        Args:
            obs: Observation array
            deterministic: Whether to use deterministic policy
            
        Returns:
            action: Selected action
        """
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        action_tensor = self.actor.get_action(obs_tensor, deterministic)
        action = action_tensor.cpu().numpy().flatten()
        
        # Add exploration noise during training
        if not deterministic and self.config.action_noise > 0:
            noise = np.random.normal(0, self.config.action_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
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
        
        # Combine info
        training_info = {**critic_info, **actor_info, **alpha_info}
        
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
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "q1_mean": current_q1.mean().item(),
            "q2_mean": current_q2.mean().item()
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
        self.actor_optimizer.step()
        
        return {
            "actor_loss": actor_loss.item(),
            "entropy": -log_probs.mean().item()
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
        """Save agent state."""
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
            "config": self.config.__dict__
        }
        
        if self.config.automatic_entropy_tuning:
            state["log_alpha"] = self.log_alpha.data
            state["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        
        torch.save(state, filepath)
        self.logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state."""
        state = torch.load(filepath, map_location=device)
        
        self.actor.load_state_dict(state["actor"])
        self.critic1.load_state_dict(state["critic1"])
        self.critic2.load_state_dict(state["critic2"])
        self.target_critic1.load_state_dict(state["target_critic1"])
        self.target_critic2.load_state_dict(state["target_critic2"])
        
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(state["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(state["critic2_optimizer"])
        
        if self.config.automatic_entropy_tuning and "log_alpha" in state:
            self.log_alpha.data = state["log_alpha"]
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        
        self.total_steps = state.get("total_steps", 0)
        self.training_steps = state.get("training_steps", 0)
        
        self.logger.info(f"Agent loaded from {filepath}")


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
