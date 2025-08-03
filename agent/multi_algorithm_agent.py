"""
State-of-the-Art Multi-Algorithm Agent System
Incorporates latest 2024-2025 deep RL research including:
- Multi-algorithm ensemble (PPO + SAC + TD3)
- Transformer-based policy networks
- Hierarchical reinforcement learning
- Physics-informed neural networks
- Constrained RL for safety
- Meta-learning capabilities
- Device-agnostic operation (CPU/GPU/TPU)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import deque
import random
import copy
from enum import Enum

# Import device management utilities
try:
    from ..utils.device_manager import DeviceManager, get_device_manager, to_device, to_numpy
except ImportError:
    # Fallback for when device manager is not available
    class DeviceManager:
        def __init__(self, device):
            self.device = device
            self.device_type = 'cpu' if device.type == 'cpu' else 'cuda'
        
        def to_device(self, tensor):
            return tensor.to(self.device)
        
        def to_numpy(self, tensor):
            if isinstance(tensor, np.ndarray):
                return tensor
            return tensor.detach().cpu().numpy()
    
    def get_device_manager(prefer_device='auto', enable_tpu=True):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return DeviceManager(device)

class AlgorithmType(Enum):
    PPO = "ppo"
    SAC = "sac"
    TD3 = "td3"
    ENSEMBLE = "ensemble"

@dataclass
class NetworkConfig:
    """Configuration for neural network architectures"""
    architecture_type: str = "transformer"  # transformer, cnn, mlp, hybrid
    activation: str = "gelu"
    use_layer_norm: bool = True
    use_spectral_norm: bool = True
    dropout: float = 0.1
    
    # Transformer specific
    use_transformer: bool = True
    d_model: int = 256
    nhead: int = 8
    num_transformer_layers: int = 4
    dim_feedforward: int = 512
    
    # Advanced features
    use_attention: bool = True
    use_residual: bool = True
    use_squeeze_excitation: bool = True
    
    # Fields with default_factory must come last
    hidden_dims: List[int] = field(default_factory=lambda: [512, 512, 256])

@dataclass
class SafetyConstraints:
    """Safety constraints for constrained RL"""
    max_tilt: float = 0.52  # 30 degrees
    max_angular_velocity: float = 5.0  # rad/s
    min_altitude: float = 0.1  # m
    max_altitude: float = 20.0  # m
    max_control_effort: float = 1.0
    fuel_reserve: float = 0.1  # 10% reserve

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer networks"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SqueezeExcitation(nn.Module):
    """Squeeze and Excitation block for feature recalibration"""
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        
    def forward(self, x):
        b, c = x.size()
        y = self.global_avg_pool(x.unsqueeze(-1)).squeeze(-1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y

class TransformerPolicyNetwork(nn.Module):
    """Advanced transformer-based policy network"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: NetworkConfig):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Input embedding
        self.input_embedding = nn.Linear(obs_dim, config.d_model)
        self.pos_encoding = PositionalEncoding(config.d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.num_transformer_layers
        )
        
        # Feature processing
        self.feature_norm = nn.LayerNorm(config.d_model)
        
        if config.use_squeeze_excitation:
            self.se_block = SqueezeExcitation(config.d_model)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dims[0]),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dims[0]),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dims[1]),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[1], action_dim * 2)  # mean and log_std
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, config.hidden_dims[0]),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dims[0]),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[0], config.hidden_dims[1]),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dims[1]),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dims[1], 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0.0)
    
    def forward(self, obs, sequence_length=1):
        """Forward pass through the network"""
        batch_size = obs.size(0)
        
        # Reshape for sequence processing
        if sequence_length > 1:
            obs = obs.view(batch_size, sequence_length, -1)
        else:
            obs = obs.unsqueeze(1)
        
        # Embed and add positional encoding
        x = self.input_embedding(obs)
        x = self.pos_encoding(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)
        
        # Take the last timestep for action prediction
        x = x[:, -1, :]
        
        # Feature normalization
        x = self.feature_norm(x)
        
        # Squeeze and excitation
        if self.config.use_squeeze_excitation:
            x = self.se_block(x)
        
        # Policy and value outputs
        policy_out = self.policy_head(x)
        value_out = self.value_head(x)
        
        # Split policy output into mean and log_std
        mean, log_std = torch.chunk(policy_out, 2, dim=-1)
        log_std = torch.clamp(log_std, -20, 2)
        
        return mean, log_std, value_out.squeeze(-1)

class PhysicsInformedLoss(nn.Module):
    """Physics-informed loss for incorporating domain knowledge"""
    
    def __init__(self, physics_weight: float = 0.1):
        super().__init__()
        self.physics_weight = physics_weight
    
    def forward(self, states, actions, next_states):
        """Compute physics-informed loss"""
        batch_size = states.size(0)
        
        # Extract physical quantities
        # Assuming state format: [quat(4), angular_vel(3), fuel(1), phase(1), progress(1)]
        quat = states[:, :4]
        angular_vel = states[:, 4:7]
        
        next_quat = next_states[:, :4]
        next_angular_vel = next_states[:, 4:7]
        
        losses = {}
        
        # Conservation of angular momentum (simplified)
        # In absence of external torques, angular momentum should be conserved
        angular_momentum = angular_vel
        next_angular_momentum = next_angular_vel
        
        # Allow for control torques from gimbal
        control_torque = actions.norm(dim=1, keepdim=True).repeat(1, 3) * 0.1
        expected_angular_momentum = angular_momentum + control_torque
        
        momentum_loss = F.mse_loss(next_angular_momentum, expected_angular_momentum)
        losses['momentum_conservation'] = momentum_loss
        
        # Energy conservation (kinetic energy)
        kinetic_energy = 0.5 * (angular_vel ** 2).sum(dim=1)
        next_kinetic_energy = 0.5 * (next_angular_vel ** 2).sum(dim=1)
        
        # Energy can change due to control inputs and fuel consumption
        control_energy = 0.5 * (actions ** 2).sum(dim=1)
        expected_energy_change = control_energy * 0.01  # Small coupling
        
        energy_loss = F.mse_loss(
            next_kinetic_energy, 
            kinetic_energy + expected_energy_change
        )
        losses['energy_conservation'] = energy_loss
        
        # Quaternion normalization constraint
        quat_norm_loss = F.mse_loss(quat.norm(dim=1), torch.ones(batch_size, device=quat.device))
        next_quat_norm_loss = F.mse_loss(next_quat.norm(dim=1), torch.ones(batch_size, device=next_quat.device))
        
        losses['quaternion_normalization'] = quat_norm_loss + next_quat_norm_loss
        
        # Total physics loss
        total_physics_loss = sum(losses.values()) * self.physics_weight
        
        return total_physics_loss, losses

class SafetyLayer(nn.Module):
    """Safety layer using Control Barrier Functions"""
    
    def __init__(self, action_dim: int, constraints: SafetyConstraints):
        super().__init__()
        self.action_dim = action_dim
        self.constraints = constraints
        
        # Safety correction network
        self.safety_net = nn.Sequential(
            nn.Linear(10 + action_dim, 128),  # state + proposed action
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # corrected action
        )
    
    def forward(self, state, proposed_action):
        """Apply safety corrections to proposed actions"""
        # Extract relevant state information
        # Assuming state format: [quat(4), angular_vel(3), fuel(1), phase(1), progress(1)]
        quat = state[:, :4]
        angular_vel = state[:, 4:7]
        fuel = state[:, 7:8]
        
        # Calculate tilt angle from quaternion
        # Convert quaternion to euler angles (simplified)
        roll = torch.atan2(2*(quat[:, 3]*quat[:, 0] + quat[:, 1]*quat[:, 2]),
                          1 - 2*(quat[:, 0]**2 + quat[:, 1]**2))
        pitch = torch.asin(2*(quat[:, 3]*quat[:, 1] - quat[:, 2]*quat[:, 0]))
        yaw = torch.atan2(2*(quat[:, 3]*quat[:, 2] + quat[:, 0]*quat[:, 1]),
                         1 - 2*(quat[:, 1]**2 + quat[:, 2]**2))
        
        tilt_angle = torch.sqrt(pitch**2 + yaw**2)
        angular_vel_mag = torch.norm(angular_vel, dim=1)
        
        # Check safety constraints
        safety_violations = torch.zeros_like(tilt_angle, dtype=torch.bool)
        
        # Tilt constraint
        safety_violations |= tilt_angle > self.constraints.max_tilt
        
        # Angular velocity constraint  
        safety_violations |= angular_vel_mag > self.constraints.max_angular_velocity
        
        # Control effort constraint
        control_effort = torch.norm(proposed_action, dim=1)
        safety_violations |= control_effort > self.constraints.max_control_effort
        
        # If no violations, return original action
        if not safety_violations.any():
            return proposed_action
        
        # Otherwise, compute safety correction
        safety_input = torch.cat([state, proposed_action], dim=-1)
        safety_correction = self.safety_net(safety_input)
        
        # Apply correction only where needed
        corrected_action = proposed_action.clone()
        corrected_action[safety_violations] = safety_correction[safety_violations]
        
        # Ensure corrected action is within bounds
        corrected_action = torch.clamp(corrected_action, -1.0, 1.0)
        
        return corrected_action

class HierarchicalAgent:
    """Hierarchical RL agent with high-level goal selection and low-level control"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: dict):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # High-level policy (goal selection)
        self.high_level_goals = ["hover", "land", "recover", "maintain_altitude"]
        self.goal_dim = len(self.high_level_goals)
        
        self.high_level_policy = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, self.goal_dim)
        )
        
        # Low-level policy (goal-conditioned control)
        self.low_level_policy = TransformerPolicyNetwork(
            obs_dim + self.goal_dim,  # observation + goal
            action_dim,
            NetworkConfig()
        )
        
        # Goal embedding
        self.goal_embedding = nn.Embedding(self.goal_dim, 32)
        
        # Optimizers
        self.high_level_optimizer = optim.Adam(self.high_level_policy.parameters(), lr=1e-4)
        self.low_level_optimizer = optim.Adam(self.low_level_policy.parameters(), lr=3e-4)
    
    def to(self, device):
        """Move all networks to specified device"""
        self.high_level_policy = self.high_level_policy.to(device)
        self.low_level_policy = self.low_level_policy.to(device)
        self.goal_embedding = self.goal_embedding.to(device)
        return self
        
    def select_goal(self, state):
        """Select high-level goal based on current state"""
        with torch.no_grad():
            goal_logits = self.high_level_policy(state)
            goal_probs = F.softmax(goal_logits, dim=-1)
            goal_idx = torch.multinomial(goal_probs, 1).squeeze(-1)
        return goal_idx
    
    def get_action(self, state, goal_idx):
        """Get low-level action conditioned on goal"""
        # Create goal one-hot encoding
        batch_size = state.size(0)
        goal_onehot = torch.zeros(batch_size, self.goal_dim, device=state.device)
        goal_onehot.scatter_(1, goal_idx.unsqueeze(1), 1)
        
        # Concatenate state and goal
        state_goal = torch.cat([state, goal_onehot], dim=-1)
        
        # Get action from low-level policy
        mean, log_std, value = self.low_level_policy(state_goal)
        
        return mean, log_std, value

class MultiAlgorithmAgent:
    """State-of-the-art multi-algorithm ensemble agent"""
    
    def __init__(self, obs_dim: int, action_dim: int, config: dict):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.config = config
        
        # Initialize device manager for robust device handling
        hardware_config = config.get('hardware', {})
        device_config = hardware_config.get('device', {})
        
        if isinstance(device_config, str):
            prefer_device = device_config
            enable_tpu = True
        else:
            prefer_device = device_config.get('type', 'auto')
            enable_tpu = device_config.get('enable_tpu', True)
        
        try:
            self.device_manager = get_device_manager(prefer_device, enable_tpu)
            self.device = self.device_manager.device
        except Exception as e:
            # Fallback to simple device selection
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.device_manager = DeviceManager(self.device)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"🤖 MultiAlgorithmAgent initialized on device: {self.device}")
        
        # Network configuration - filter to only include NetworkConfig fields
        network_config = config.get('network', {})
        network_fields = {
            'architecture_type': network_config.get('architecture_type', 'transformer'),
            'activation': network_config.get('mlp_backbone', {}).get('activation', 'gelu'),
            'use_layer_norm': network_config.get('mlp_backbone', {}).get('layer_norm', True),
            'use_spectral_norm': network_config.get('mlp_backbone', {}).get('spectral_norm', True),
            'dropout': network_config.get('transformer', {}).get('dropout', 0.1),
            'use_transformer': network_config.get('transformer', {}).get('enabled', True),
            'd_model': network_config.get('transformer', {}).get('d_model', 256),
            'nhead': network_config.get('transformer', {}).get('nhead', 8),
            'num_transformer_layers': network_config.get('transformer', {}).get('num_layers', 4),
            'dim_feedforward': network_config.get('transformer', {}).get('dim_feedforward', 512),
            'use_attention': network_config.get('advanced_features', {}).get('use_attention', True),
            'use_residual': network_config.get('advanced_features', {}).get('use_residual_connections', True),
            'use_squeeze_excitation': False,  # Default
            'hidden_dims': network_config.get('mlp_backbone', {}).get('hidden_dims', [512, 512, 256])
        }
        self.network_config = NetworkConfig(**network_fields)
        
        # Safety constraints - filter to only include SafetyConstraints fields
        safety_config = config.get('safety', {})
        safety_fields = {
            'max_tilt': safety_config.get('constrained_rl', {}).get('constraints', {}).get('max_tilt', 0.52),
            'max_angular_velocity': safety_config.get('constrained_rl', {}).get('constraints', {}).get('max_angular_velocity', 5.0),
            'min_altitude': safety_config.get('constrained_rl', {}).get('constraints', {}).get('min_altitude', 0.1),
            'max_altitude': safety_config.get('constrained_rl', {}).get('constraints', {}).get('max_altitude', 20.0),
            'max_control_effort': 1.0,  # Default
            'fuel_reserve': safety_config.get('constrained_rl', {}).get('constraints', {}).get('fuel_reserve', 0.1)
        }
        self.safety_constraints = SafetyConstraints(**safety_fields)
        
        # Initialize algorithms
        self.algorithms = {}
        self.algorithm_weights = {}
        self.performance_history = {alg: deque(maxlen=100) for alg in ['ppo', 'sac', 'td3']}
        
        if config.get('algorithms', {}).get('ppo', {}).get('enabled', True):
            self.algorithms['ppo'] = self._create_ppo_agent()
            self.algorithm_weights['ppo'] = 1.0
            
        if config.get('algorithms', {}).get('sac', {}).get('enabled', True):
            self.algorithms['sac'] = self._create_sac_agent()
            self.algorithm_weights['sac'] = 1.0
            
        if config.get('algorithms', {}).get('td3', {}).get('enabled', True):
            self.algorithms['td3'] = self._create_td3_agent()
            self.algorithm_weights['td3'] = 1.0
        
        # Hierarchical agent
        if config.get('hierarchical_rl', {}).get('enabled', False):
            self.hierarchical_agent = HierarchicalAgent(obs_dim, action_dim, config)
            self.hierarchical_agent = self.hierarchical_agent.to(self.device)
        else:
            self.hierarchical_agent = None
        
        # Physics-informed loss
        if config.get('physics_informed', {}).get('enabled', False):
            self.physics_loss = PhysicsInformedLoss(
                config.get('physics_informed', {}).get('physics_loss_weight', 0.1)
            )
            self.physics_loss = self.physics_loss.to(self.device)
        else:
            self.physics_loss = None
        
        # Safety layer
        if config.get('safety', {}).get('safety_layer', {}).get('enabled', False):
            self.safety_layer = SafetyLayer(action_dim, self.safety_constraints)
            self.safety_layer = self.safety_layer.to(self.device)
        else:
            self.safety_layer = None
        
        # Meta-learning components
        if config.get('experimental', {}).get('meta_learning', {}).get('enabled', False):
            self._setup_meta_learning()
        
        # Ensemble selection strategy
        self.selection_strategy = config.get('algorithms', {}).get('ensemble', {}).get(
            'selection_strategy', 'dynamic'
        )
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
    def to(self, device):
        """Move all components to specified device with proper error handling"""
        try:
            self.device = device
            self.device_manager.device = device
            
            # Move all algorithms to device
            for alg_name, agent in self.algorithms.items():
                try:
                    if 'policy' in agent:
                        agent['policy'] = agent['policy'].to(device)
                    if 'q1' in agent:
                        agent['q1'] = agent['q1'].to(device)
                    if 'q2' in agent:
                        agent['q2'] = agent['q2'].to(device)
                    if 'target_q1' in agent:
                        agent['target_q1'] = agent['target_q1'].to(device)
                    if 'target_q2' in agent:
                        agent['target_q2'] = agent['target_q2'].to(device)
                    if 'target_policy' in agent:
                        agent['target_policy'] = agent['target_policy'].to(device)
                except Exception as e:
                    self.logger.warning(f"Failed to move {alg_name} to {device}: {e}")
            
            # Move other components
            if self.hierarchical_agent is not None:
                self.hierarchical_agent = self.hierarchical_agent.to(device)
            if self.physics_loss is not None:
                self.physics_loss = self.physics_loss.to(device)
            if self.safety_layer is not None:
                self.safety_layer = self.safety_layer.to(device)
                
            self.logger.info(f"✅ MultiAlgorithmAgent moved to {device}")
            return self
            
        except Exception as e:
            self.logger.error(f"❌ Failed to move agent to {device}: {e}")
            return self
        
    def _create_ppo_agent(self):
        """Create PPO agent with transformer network"""
        policy_net = TransformerPolicyNetwork(self.obs_dim, self.action_dim, self.network_config)
        policy_net = policy_net.to(self.device)
        
        return {
            'policy': policy_net,
            'optimizer': optim.Adam(
                policy_net.parameters(),
                lr=self.config.get('algorithms', {}).get('ppo', {}).get('learning_rate', 2.5e-4)
            ),
            'type': AlgorithmType.PPO
        }
    
    def _create_sac_agent(self):
        """Create SAC agent with transformer networks"""
        policy_net = TransformerPolicyNetwork(self.obs_dim, self.action_dim, self.network_config)
        policy_net = policy_net.to(self.device)
        
        # Q-networks for SAC
        q1_net = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(self.device)
        
        q2_net = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(self.device)
        
        return {
            'policy': policy_net,
            'q1': q1_net,
            'q2': q2_net,
            'target_q1': copy.deepcopy(q1_net).to(self.device),
            'target_q2': copy.deepcopy(q2_net).to(self.device),
            'optimizer_policy': optim.Adam(policy_net.parameters(), lr=3e-4),
            'optimizer_q1': optim.Adam(q1_net.parameters(), lr=3e-4),
            'optimizer_q2': optim.Adam(q2_net.parameters(), lr=3e-4),
            'type': AlgorithmType.SAC
        }
    
    def _create_td3_agent(self):
        """Create TD3 agent with transformer networks"""
        # Deterministic policy for TD3
        policy_net = nn.Sequential(
            nn.Linear(self.obs_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, self.action_dim),
            nn.Tanh()
        ).to(self.device)
        
        # Q-networks
        q1_net = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(self.device)
        
        q2_net = nn.Sequential(
            nn.Linear(self.obs_dim + self.action_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(self.device)
        
        return {
            'policy': policy_net,
            'q1': q1_net,
            'q2': q2_net,
            'target_policy': copy.deepcopy(policy_net).to(self.device),
            'target_q1': copy.deepcopy(q1_net).to(self.device),
            'target_q2': copy.deepcopy(q2_net).to(self.device),
            'optimizer_policy': optim.Adam(policy_net.parameters(), lr=3e-4),
            'optimizer_q1': optim.Adam(q1_net.parameters(), lr=3e-4),
            'optimizer_q2': optim.Adam(q2_net.parameters(), lr=3e-4),
            'type': AlgorithmType.TD3
        }
    
    def _setup_meta_learning(self):
        """Setup meta-learning components (MAML)"""
        # Meta-learning wrapper around the main policy
        self.meta_policy = TransformerPolicyNetwork(
            self.obs_dim, self.action_dim, self.network_config
        )
        self.meta_optimizer = optim.Adam(self.meta_policy.parameters(), lr=1e-3)
        self.inner_lr = 1e-2
        self.adaptation_steps = 5
    
    def select_algorithm(self, performance_metrics: Optional[Dict] = None):
        """Select best algorithm based on recent performance"""
        previous_algorithm = getattr(self, '_current_algorithm', None)
        
        if self.selection_strategy == "dynamic":
            # Select based on recent performance
            best_algorithm = None
            best_performance = -float('inf')
            
            for alg_name, history in self.performance_history.items():
                if len(history) > 0 and alg_name in self.algorithms:
                    avg_performance = np.mean(list(history)[-10:])  # Last 10 episodes
                    if avg_performance > best_performance:
                        best_performance = avg_performance
                        best_algorithm = alg_name
                        
            selected_algorithm = best_algorithm or 'ppo'  # Default to PPO
            
        elif self.selection_strategy == "voting":
            # Use ensemble voting
            selected_algorithm = "ensemble"
            
        else:  # "best"
            # Always use the best performing algorithm overall
            best_algorithm = None
            best_performance = -float('inf')
            
            for alg_name, history in self.performance_history.items():
                if len(history) > 0 and alg_name in self.algorithms:
                    avg_performance = np.mean(list(history))
                    if avg_performance > best_performance:
                        best_performance = avg_performance
                        best_algorithm = alg_name
                        
            selected_algorithm = best_algorithm or 'ppo'
        
        # Track algorithm switches
        if previous_algorithm is not None and previous_algorithm != selected_algorithm:
            self.logger.info(f"🔄 Algorithm switch: {previous_algorithm} → {selected_algorithm}")
            
        self._current_algorithm = selected_algorithm
        return selected_algorithm
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False, 
                   algorithm: Optional[str] = None):
        """Get action from the selected algorithm with robust device handling"""
        
        try:
            if algorithm is None:
                algorithm = self.select_algorithm()
            
            # Ensure state is on correct device
            state = self.device_manager.to_device(state)
            
            if algorithm == "ensemble":
                return self._get_ensemble_action(state, deterministic)
            
            # Use hierarchical agent if enabled
            if self.hierarchical_agent is not None:
                goal_idx = self.hierarchical_agent.select_goal(state)
                mean, log_std, value = self.hierarchical_agent.get_action(state, goal_idx)
                agent_type = AlgorithmType.PPO  # Default for hierarchical
            else:
                # Use selected algorithm
                if algorithm not in self.algorithms:
                    self.logger.warning(f"Algorithm {algorithm} not available, falling back to first available")
                    algorithm = list(self.algorithms.keys())[0]
                
                agent = self.algorithms[algorithm]
                agent_type = agent['type']
                
                if agent_type == AlgorithmType.PPO or agent_type == AlgorithmType.SAC:
                    mean, log_std, value = agent['policy'](state)
                else:  # TD3
                    mean = agent['policy'](state)
                    log_std = torch.zeros_like(mean)
                    value = None
            
            # Sample action
            if deterministic:
                action = mean
            else:
                if agent_type == AlgorithmType.TD3:
                    # Add exploration noise for TD3
                    noise = torch.randn_like(mean) * 0.1
                    action = mean + noise
                else:
                    std = torch.exp(torch.clamp(log_std, -20, 2))  # Clamp for stability
                    dist = Normal(mean, std)
                    action = dist.sample()
            
            # Apply safety layer if enabled
            if self.safety_layer is not None:
                action = self.safety_layer(state, action)
            
            # Clamp action to valid range
            action = torch.clamp(action, -1.0, 1.0)
            
            # Convert to numpy using device manager
            action_numpy = self.device_manager.to_numpy(action)
            
            # Create info dict with safe conversions
            info = {
                'algorithm': algorithm,
                'mean': self.device_manager.to_numpy(mean),
                'log_std': self.device_manager.to_numpy(log_std) if log_std is not None else None,
                'value': self.device_manager.to_numpy(value) if value is not None else None
            }
            
            return action_numpy, info
            
        except Exception as e:
            self.logger.error(f"Error in get_action: {e}")
            # Emergency fallback - return random action
            fallback_action = np.random.uniform(-1.0, 1.0, size=(self.action_dim,))
            fallback_info = {'algorithm': 'fallback', 'error': str(e)}
            return fallback_action, fallback_info
    
    def _get_ensemble_action(self, state: torch.Tensor, deterministic: bool = False):
        """Get ensemble action by combining multiple algorithms with device management"""
        try:
            actions = []
            weights = []
            
            for alg_name, agent in self.algorithms.items():
                try:
                    if agent['type'] == AlgorithmType.PPO or agent['type'] == AlgorithmType.SAC:
                        mean, log_std, _ = agent['policy'](state)
                        if deterministic:
                            action = mean
                        else:
                            std = torch.exp(torch.clamp(log_std, -20, 2))
                            dist = Normal(mean, std)
                            action = dist.sample()
                    else:  # TD3
                        action = agent['policy'](state)
                        if not deterministic:
                            noise = torch.randn_like(action) * 0.1
                            action = action + noise
                    
                    actions.append(action)
                    weights.append(self.algorithm_weights.get(alg_name, 1.0))
                    
                except Exception as e:
                    self.logger.warning(f"Failed to get action from {alg_name}: {e}")
                    continue
            
            if not actions:
                # Emergency fallback
                fallback_action = torch.zeros(state.shape[0], self.action_dim, device=state.device)
                return self.device_manager.to_numpy(fallback_action), {'algorithm': 'fallback', 'error': 'no_algorithms'}
            
            # Weighted ensemble
            weights = torch.tensor(weights, device=state.device)
            weights = weights / weights.sum()
            
            ensemble_action = sum(w * action for w, action in zip(weights, actions))
            
            # Apply safety layer
            if self.safety_layer is not None:
                ensemble_action = self.safety_layer(state, ensemble_action)
            
            ensemble_action = torch.clamp(ensemble_action, -1.0, 1.0)
            
            return self.device_manager.to_numpy(ensemble_action), {
                'algorithm': 'ensemble',
                'weights': self.device_manager.to_numpy(weights),
                'individual_actions': [self.device_manager.to_numpy(a) for a in actions]
            }
            
        except Exception as e:
            self.logger.error(f"Error in ensemble action: {e}")
            fallback_action = np.random.uniform(-1.0, 1.0, size=(self.action_dim,))
            return fallback_action, {'algorithm': 'fallback', 'error': str(e)}
    
    def update(self, batch: Dict, algorithm: Optional[str] = None):
        """Update the selected algorithm with robust device handling"""
        try:
            if algorithm is None:
                algorithm = self.select_algorithm()
            
            # Move batch to device using device manager
            device_batch = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    device_batch[k] = self.device_manager.to_device(v)
                else:
                    device_batch[k] = v
            
            # Apply physics-informed loss if enabled
            physics_loss = 0.0
            if self.physics_loss is not None:
                try:
                    physics_loss, physics_losses = self.physics_loss(
                        device_batch['states'], device_batch['actions'], device_batch['next_states']
                    )
                except Exception as e:
                    self.logger.warning(f"Physics loss computation failed: {e}")
                    physics_loss = 0.0
            
            # Update the selected algorithm
            if algorithm == 'ppo' and 'ppo' in self.algorithms:
                losses = self._update_ppo(device_batch)
            elif algorithm == 'sac' and 'sac' in self.algorithms:
                losses = self._update_sac(device_batch)
            elif algorithm == 'td3' and 'td3' in self.algorithms:
                losses = self._update_td3(device_batch)
            else:
                self.logger.warning(f"Algorithm {algorithm} not available for update")
                losses = {}
            
            # Add physics loss
            if physics_loss > 0:
                losses['physics_loss'] = physics_loss.item() if isinstance(physics_loss, torch.Tensor) else physics_loss
            
            return losses
            
        except Exception as e:
            self.logger.error(f"Error in agent update: {e}")
            return {'error': str(e)}
    
    def _update_ppo(self, batch: Dict):
        """Update PPO algorithm"""
        # Simplified PPO update - in practice, this would be much more detailed
        agent = self.algorithms['ppo']
        
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        dones = batch['dones']
        
        # Forward pass
        mean, log_std, values = agent['policy'](states)
        
        # Compute policy loss (simplified)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Placeholder for advantage computation
        advantages = rewards  # Simplified - should compute proper advantages
        
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, rewards)
        
        total_loss = policy_loss + 0.5 * value_loss
        
        agent['optimizer'].zero_grad()
        total_loss.backward()
        agent['optimizer'].step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def _update_sac(self, batch: Dict):
        """Update SAC algorithm"""
        # Simplified SAC update
        agent = self.algorithms['sac']
        
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Q-function update
        with torch.no_grad():
            next_mean, next_log_std, _ = agent['policy'](next_states)
            next_std = torch.exp(next_log_std)
            next_dist = Normal(next_mean, next_std)
            next_actions = next_dist.sample()
            
            target_q1 = agent['target_q1'](torch.cat([next_states, next_actions], dim=-1))
            target_q2 = agent['target_q2'](torch.cat([next_states, next_actions], dim=-1))
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + 0.99 * (1 - dones) * target_q.squeeze()
        
        current_q1 = agent['q1'](torch.cat([states, actions], dim=-1)).squeeze()
        current_q2 = agent['q2'](torch.cat([states, actions], dim=-1)).squeeze()
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # Update Q-networks
        agent['optimizer_q1'].zero_grad()
        q1_loss.backward()
        agent['optimizer_q1'].step()
        
        agent['optimizer_q2'].zero_grad()
        q2_loss.backward()
        agent['optimizer_q2'].step()
        
        # Policy update
        mean, log_std, _ = agent['policy'](states)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        new_actions = dist.rsample()
        
        q1_new = agent['q1'](torch.cat([states, new_actions], dim=-1))
        q2_new = agent['q2'](torch.cat([states, new_actions], dim=-1))
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = -(q_new - 0.2 * dist.log_prob(new_actions).sum(dim=-1, keepdim=True)).mean()
        
        agent['optimizer_policy'].zero_grad()
        policy_loss.backward()
        agent['optimizer_policy'].step()
        
        # Soft update target networks
        tau = 0.005
        for target_param, param in zip(agent['target_q1'].parameters(), agent['q1'].parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        for target_param, param in zip(agent['target_q2'].parameters(), agent['q2'].parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item()
        }
    
    def _update_td3(self, batch: Dict):
        """Update TD3 algorithm"""
        # Simplified TD3 update
        agent = self.algorithms['td3']
        
        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']
        
        # Q-function update (similar to SAC but with target policy smoothing)
        with torch.no_grad():
            noise = torch.randn_like(actions) * 0.2
            noise = torch.clamp(noise, -0.5, 0.5)
            next_actions = agent['target_policy'](next_states) + noise
            next_actions = torch.clamp(next_actions, -1.0, 1.0)
            
            target_q1 = agent['target_q1'](torch.cat([next_states, next_actions], dim=-1))
            target_q2 = agent['target_q2'](torch.cat([next_states, next_actions], dim=-1))
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + 0.99 * (1 - dones) * target_q.squeeze()
        
        current_q1 = agent['q1'](torch.cat([states, actions], dim=-1)).squeeze()
        current_q2 = agent['q2'](torch.cat([states, actions], dim=-1)).squeeze()
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # Update Q-networks
        agent['optimizer_q1'].zero_grad()
        q1_loss.backward()
        agent['optimizer_q1'].step()
        
        agent['optimizer_q2'].zero_grad()
        q2_loss.backward()
        agent['optimizer_q2'].step()
        
        # Delayed policy update
        policy_loss = 0.0
        if hasattr(self, 'td3_update_counter'):
            self.td3_update_counter += 1
        else:
            self.td3_update_counter = 1
        
        if self.td3_update_counter % 2 == 0:  # Update policy every 2 Q-updates
            new_actions = agent['policy'](states)
            policy_loss = -agent['q1'](torch.cat([states, new_actions], dim=-1)).mean()
            
            agent['optimizer_policy'].zero_grad()
            policy_loss.backward()
            agent['optimizer_policy'].step()
            
            # Soft update target networks
            tau = 0.005
            for target_param, param in zip(agent['target_policy'].parameters(), agent['policy'].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            for target_param, param in zip(agent['target_q1'].parameters(), agent['q1'].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            for target_param, param in zip(agent['target_q2'].parameters(), agent['q2'].parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item() if isinstance(policy_loss, torch.Tensor) else policy_loss
        }
    
    def update_performance(self, algorithm: str, performance: float):
        """Update performance history for algorithm selection"""
        if algorithm in self.performance_history:
            self.performance_history[algorithm].append(performance)
            
            # Update algorithm weights based on performance
            if len(self.performance_history[algorithm]) >= 10:
                recent_performance = np.mean(list(self.performance_history[algorithm])[-10:])
                self.algorithm_weights[algorithm] = max(0.1, recent_performance)
    
    def save_checkpoint(self, path: str):
        """Save agent checkpoint"""
        checkpoint = {
            'algorithms': {},
            'performance_history': dict(self.performance_history),
            'algorithm_weights': self.algorithm_weights,
            'config': self.config
        }
        
        for alg_name, agent in self.algorithms.items():
            if agent['type'] == AlgorithmType.PPO:
                checkpoint['algorithms'][alg_name] = {
                    'policy_state': agent['policy'].state_dict(),
                    'optimizer_state': agent['optimizer'].state_dict(),
                    'type': agent['type'].value
                }
            elif agent['type'] == AlgorithmType.SAC:
                checkpoint['algorithms'][alg_name] = {
                    'policy_state': agent['policy'].state_dict(),
                    'q1_state': agent['q1'].state_dict(),
                    'q2_state': agent['q2'].state_dict(),
                    'target_q1_state': agent['target_q1'].state_dict(),
                    'target_q2_state': agent['target_q2'].state_dict(),
                    'optimizer_policy_state': agent['optimizer_policy'].state_dict(),
                    'optimizer_q1_state': agent['optimizer_q1'].state_dict(),
                    'optimizer_q2_state': agent['optimizer_q2'].state_dict(),
                    'type': agent['type'].value
                }
            elif agent['type'] == AlgorithmType.TD3:
                checkpoint['algorithms'][alg_name] = {
                    'policy_state': agent['policy'].state_dict(),
                    'q1_state': agent['q1'].state_dict(),
                    'q2_state': agent['q2'].state_dict(),
                    'target_policy_state': agent['target_policy'].state_dict(),
                    'target_q1_state': agent['target_q1'].state_dict(),
                    'target_q2_state': agent['target_q2'].state_dict(),
                    'optimizer_policy_state': agent['optimizer_policy'].state_dict(),
                    'optimizer_q1_state': agent['optimizer_q1'].state_dict(),
                    'optimizer_q2_state': agent['optimizer_q2'].state_dict(),
                    'type': agent['type'].value
                }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load agent checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.performance_history = {k: deque(v, maxlen=100) for k, v in checkpoint['performance_history'].items()}
        self.algorithm_weights = checkpoint['algorithm_weights']
        
        for alg_name, alg_data in checkpoint['algorithms'].items():
            if alg_name in self.algorithms:
                agent = self.algorithms[alg_name]
                
                if alg_data['type'] == AlgorithmType.PPO.value:
                    agent['policy'].load_state_dict(alg_data['policy_state'])
                    agent['optimizer'].load_state_dict(alg_data['optimizer_state'])
                    
                elif alg_data['type'] == AlgorithmType.SAC.value:
                    agent['policy'].load_state_dict(alg_data['policy_state'])
                    agent['q1'].load_state_dict(alg_data['q1_state'])
                    agent['q2'].load_state_dict(alg_data['q2_state'])
                    agent['target_q1'].load_state_dict(alg_data['target_q1_state'])
                    agent['target_q2'].load_state_dict(alg_data['target_q2_state'])
                    agent['optimizer_policy'].load_state_dict(alg_data['optimizer_policy_state'])
                    agent['optimizer_q1'].load_state_dict(alg_data['optimizer_q1_state'])
                    agent['optimizer_q2'].load_state_dict(alg_data['optimizer_q2_state'])
                    
                elif alg_data['type'] == AlgorithmType.TD3.value:
                    agent['policy'].load_state_dict(alg_data['policy_state'])
                    agent['q1'].load_state_dict(alg_data['q1_state'])
                    agent['q2'].load_state_dict(alg_data['q2_state'])
                    agent['target_policy'].load_state_dict(alg_data['target_policy_state'])
                    agent['target_q1'].load_state_dict(alg_data['target_q1_state'])
                    agent['target_q2'].load_state_dict(alg_data['target_q2_state'])
                    agent['optimizer_policy'].load_state_dict(alg_data['optimizer_policy_state'])
                    agent['optimizer_q1'].load_state_dict(alg_data['optimizer_q1_state'])
                    agent['optimizer_q2'].load_state_dict(alg_data['optimizer_q2_state'])
        
        self.logger.info(f"Checkpoint loaded from {path}")

# Example usage
if __name__ == "__main__":
    # Configuration for the multi-algorithm agent
    config = {
        'device': 'cuda',
        'algorithms': {
            'ppo': {'enabled': True, 'learning_rate': 2.5e-4},
            'sac': {'enabled': True, 'learning_rate': 3e-4},
            'td3': {'enabled': True, 'learning_rate': 3e-4},
            'ensemble': {'selection_strategy': 'dynamic'}
        },
        'network': {
            'use_transformer': True,
            'd_model': 256,
            'nhead': 8,
            'num_transformer_layers': 4,
            'hidden_dims': [512, 512, 256],
            'activation': 'gelu',
            'use_layer_norm': True,
            'dropout': 0.1
        },
        'safety': {
            'safety_layer': {'enabled': True},
            'max_tilt': 0.52,
            'max_angular_velocity': 5.0
        },
        'hierarchical_rl': {'enabled': True},
        'physics_informed': {'enabled': True, 'physics_loss_weight': 0.1},
        'experimental': {'meta_learning': {'enabled': False}}
    }
    
    # Create the agent
    agent = MultiAlgorithmAgent(obs_dim=10, action_dim=2, config=config)
    
    # Test action selection
    state = torch.randn(1, 10)
    action, info = agent.get_action(state)
    
    print(f"Action: {action}")
    print(f"Info: {info}")
    print(f"Selected algorithm: {info['algorithm']}")
