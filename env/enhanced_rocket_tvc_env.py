"""
Enhanced TVC Environment with State-of-the-Art Features
Incorporates latest research in RL for continuous control and real-world deployment
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import math
import random
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
from collections import deque

class MissionPhase(Enum):
    """Mission phases for hierarchical control"""
    BOOST = "boost"
    COAST = "coast" 
    LANDING = "landing"
    TOUCHDOWN = "touchdown"
    HOVER = "hover"
    COMPLETE = "complete"
    FAILED = "failed"

class SuccessCriteria(Enum):
    """Success criteria based on real-world requirements"""
    ATTITUDE = "attitude"      # < 5 degrees tilt
    VELOCITY = "velocity"      # < 2 m/s vertical, < 0.5 m/s horizontal
    POSITION = "position"      # Within landing zone
    STABILITY = "stability"    # Sustained stable flight
    FUEL = "fuel"             # Fuel efficiency

@dataclass
class MissionSuccess:
    """Mission success detection system"""
    # Real-world landing thresholds (based on SpaceX Falcon 9)
    max_tilt_angle: float = 0.087  # 5 degrees
    max_angular_velocity: float = 0.1  # rad/s
    max_horizontal_velocity: float = 0.5  # m/s
    max_vertical_velocity: float = 2.0  # m/s
    min_altitude: float = 0.2  # m
    max_altitude: float = 2.0  # m
    position_tolerance: float = 1.0  # m
    success_duration: int = 100  # steps (2 seconds at 50Hz)
    
    # Mission phase durations
    boost_duration: int = 100
    coast_duration: int = 200
    landing_duration: int = 300
    touchdown_duration: int = 100
    
    def __post_init__(self):
        self.success_counter = 0
        self.phase_counters = {phase: 0 for phase in MissionPhase}
        self.criteria_history = deque(maxlen=self.success_duration)

@dataclass
class PhysicsInformedConfig:
    """Physics-informed neural network configuration"""
    enabled: bool = True
    physics_loss_weight: float = 0.1
    conservation_laws: List[str] = field(default_factory=lambda: [
        "momentum", "energy", "angular_momentum"
    ])
    dynamics_prediction: bool = True
    
class MultiObjectiveReward:
    """Multi-objective reward function with anti-hacking measures"""
    
    def __init__(self, config: dict):
        self.config = config
        self.primary_weight = 0.8
        self.secondary_weight = 0.2
        
        # Anti-hacking measures
        self.reward_history = deque(maxlen=1000)
        self.gradient_penalty_weight = config.get('gradient_penalty', 0.1)
        self.diversity_bonus_weight = config.get('diversity_bonus', 0.05)
        
    def compute_reward(self, state: dict, action: np.ndarray, 
                      mission_success: MissionSuccess) -> Tuple[float, dict]:
        """Compute multi-objective reward with anti-hacking measures"""
        
        rewards = {}
        
        # Primary objectives (mission-critical)
        mission_completion = self._compute_mission_completion(state, mission_success)
        safety_compliance = self._compute_safety_compliance(state)
        fuel_efficiency = self._compute_fuel_efficiency(state, action)
        
        rewards['mission_completion'] = mission_completion * 100.0
        rewards['safety_compliance'] = safety_compliance * 50.0
        rewards['fuel_efficiency'] = fuel_efficiency * 20.0
        
        # Secondary objectives (performance optimization)
        stability_bonus = self._compute_stability_bonus(state)
        control_smoothness = self._compute_control_smoothness(action)
        altitude_maintenance = self._compute_altitude_maintenance(state)
        
        rewards['stability_bonus'] = stability_bonus * 10.0
        rewards['control_smoothness'] = control_smoothness * 5.0
        rewards['altitude_maintenance'] = altitude_maintenance * 5.0
        
        # Penalties
        penalties = self._compute_penalties(state, action)
        rewards.update(penalties)
        
        # Anti-hacking measures
        anti_hack_adjustment = self._apply_anti_hacking_measures(rewards)
        
        # Total reward
        total_reward = sum(rewards.values()) + anti_hack_adjustment
        
        # Clip to prevent extreme values
        total_reward = np.clip(total_reward, -1000.0, 200.0)
        
        self.reward_history.append(total_reward)
        
        return total_reward, rewards
    
    def _compute_mission_completion(self, state: dict, success: MissionSuccess) -> float:
        """Sparse reward only given on actual mission success"""
        if state.get('mission_successful', False):
            return 1.0
        elif state.get('mission_phase') == MissionPhase.LANDING.value:
            # Small progress reward during landing phase
            return 0.1
        return 0.0
    
    def _compute_safety_compliance(self, state: dict) -> float:
        """Safety-based reward using real-world thresholds"""
        tilt = state.get('tilt_angle', 0.0)
        angular_vel = state.get('angular_velocity_mag', 0.0)
        altitude = state.get('altitude', 0.0)
        
        # Exponential penalty for exceeding safety thresholds
        tilt_penalty = np.exp(-10 * max(0, tilt - 0.087))  # 5 degrees
        angular_penalty = np.exp(-5 * max(0, angular_vel - 0.1))
        altitude_penalty = 1.0 if 0.2 <= altitude <= 20.0 else 0.5
        
        return (tilt_penalty + angular_penalty + altitude_penalty) / 3.0
    
    def _compute_fuel_efficiency(self, state: dict, action: np.ndarray) -> float:
        """Reward fuel-efficient control"""
        fuel_remaining = state.get('fuel_remaining', 1.0)
        control_effort = np.linalg.norm(action)
        
        # Bonus for maintaining control with less fuel usage
        if fuel_remaining > 0.1 and control_effort < 0.5:
            return fuel_remaining * (1.0 - control_effort)
        return 0.0
    
    def _compute_stability_bonus(self, state: dict) -> float:
        """Bonus for stable, controlled flight"""
        tilt = state.get('tilt_angle', 0.0)
        angular_vel = state.get('angular_velocity_mag', 0.0)
        
        if tilt < 0.05 and angular_vel < 0.1:  # Very stable
            return 1.0
        elif tilt < 0.1 and angular_vel < 0.2:  # Moderately stable
            return 0.5
        return 0.0
    
    def _compute_control_smoothness(self, action: np.ndarray) -> float:
        """Penalize jerky, discontinuous control"""
        if hasattr(self, 'previous_action'):
            action_diff = np.linalg.norm(action - self.previous_action)
            smoothness = np.exp(-5 * action_diff)
        else:
            smoothness = 1.0
        
        self.previous_action = action.copy()
        return smoothness
    
    def _compute_altitude_maintenance(self, state: dict) -> float:
        """Reward maintaining target altitude"""
        altitude = state.get('altitude', 0.0)
        target_altitude = state.get('target_altitude', 3.0)
        
        altitude_error = abs(altitude - target_altitude)
        return np.exp(-2 * altitude_error)
    
    def _compute_penalties(self, state: dict, action: np.ndarray) -> dict:
        """Compute various penalty terms"""
        penalties = {}
        
        # Crash penalty
        if state.get('crashed', False):
            penalties['crash_penalty'] = -1000.0
        
        # Excessive tilt penalty
        tilt = state.get('tilt_angle', 0.0)
        if tilt > 0.52:  # 30 degrees
            penalties['excessive_tilt'] = -500.0 * (tilt - 0.52)
        
        # Control saturation penalty
        control_mag = np.linalg.norm(action)
        if control_mag > 0.9:
            penalties['control_saturation'] = -50.0 * (control_mag - 0.9)
        
        return penalties
    
    def _apply_anti_hacking_measures(self, rewards: dict) -> float:
        """Apply anti-reward-hacking measures"""
        adjustment = 0.0
        
        # Gradient penalty for extreme rewards
        if len(self.reward_history) > 10:
            recent_rewards = list(self.reward_history)[-10:]
            reward_variance = np.var(recent_rewards)
            if reward_variance > 10000:  # High variance indicates potential hacking
                adjustment -= self.gradient_penalty_weight * reward_variance
        
        # Diversity bonus for exploration
        if len(set(self.reward_history)) > len(self.reward_history) * 0.8:
            adjustment += self.diversity_bonus_weight
        
        return adjustment

class CuriosityModule:
    """Intrinsic Curiosity Module for better exploration"""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Inverse dynamics model: predicts action from state transitions
        self.inverse_model = nn.Sequential(
            nn.Linear(obs_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Forward dynamics model: predicts next state from current state and action
        self.forward_model = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim)
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.inverse_model.parameters()) + 
            list(self.forward_model.parameters()),
            lr=1e-4
        )
        
    def compute_intrinsic_reward(self, state: np.ndarray, action: np.ndarray, 
                                next_state: np.ndarray) -> float:
        """Compute intrinsic reward based on prediction error"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Forward model prediction error
        state_action = torch.cat([state_tensor, action_tensor], dim=1)
        predicted_next_state = self.forward_model(state_action)
        prediction_error = nn.MSELoss()(predicted_next_state, next_state_tensor)
        
        return prediction_error.item() * 0.01  # Scale intrinsic reward

class EnhancedRocketTVCEnv(gym.Env):
    """
    State-of-the-Art Rocket TVC Environment
    Incorporates latest 2024-2025 RL research for continuous control
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        config: Optional[dict] = None,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        enable_hierarchical: bool = True,
        enable_curiosity: bool = True,
        enable_physics_informed: bool = True,
        debug: bool = False
    ):
        super().__init__()
        
        self.config = config or {}
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.enable_hierarchical = enable_hierarchical
        self.enable_curiosity = enable_curiosity
        self.enable_physics_informed = enable_physics_informed
        self.debug = debug
        
        # Initialize components
        self.mission_success = MissionSuccess()
        self.multi_objective_reward = MultiObjectiveReward(self.config.get('reward_function', {}))
        
        if enable_curiosity:
            self.curiosity_module = CuriosityModule(obs_dim=8, action_dim=2)
        
        # Mission state
        self.current_phase = MissionPhase.BOOST
        self.mission_successful = False
        self.phase_start_time = 0
        
        # Enhanced state tracking
        self.state_history = deque(maxlen=100)
        self.action_history = deque(maxlen=100)
        self.reward_components_history = deque(maxlen=100)
        
        # Physics simulation setup
        self._setup_physics()
        self._setup_spaces()
        
        # Logging
        logging.basicConfig(level=logging.DEBUG if debug else logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _setup_physics(self):
        """Enhanced physics setup with real-world parameters"""
        if hasattr(self, 'physics_client') and self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        # Connect to physics server
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Enhanced physics parameters
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(
            fixedTimeStep=0.02,  # 50Hz
            numSubSteps=4,
            enableConeFriction=1,
            contactBreakingThreshold=0.001,
            enableFileCaching=0
        )
        
        # Ground plane with realistic friction
        self.ground_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.ground_id, -1, 
                        lateralFriction=0.8,
                        spinningFriction=0.1,
                        rollingFriction=0.05)
    
    def _setup_spaces(self):
        """Define observation and action spaces"""
        # Enhanced observation space
        # [quaternion(4), angular_velocity(3), fuel(1), phase(1), mission_progress(1)]
        obs_low = np.array([
            -1.0, -1.0, -1.0, -1.0,  # Quaternion
            -10.0, -10.0, -10.0,      # Angular velocity
            0.0,                      # Fuel remaining
            0.0,                      # Mission phase (normalized)
            0.0                       # Mission progress
        ], dtype=np.float32)
        
        obs_high = np.array([
            1.0, 1.0, 1.0, 1.0,      # Quaternion
            10.0, 10.0, 10.0,        # Angular velocity
            1.0,                     # Fuel remaining
            1.0,                     # Mission phase (normalized)
            1.0                      # Mission progress
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Action space: continuous gimbal angles
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Enhanced reset with mission phase initialization"""
        super().reset(seed=seed)
        
        # Reset mission state
        self.current_phase = MissionPhase.BOOST
        self.mission_successful = False
        self.phase_start_time = 0
        self.current_step = 0
        
        # Reset physics
        p.resetSimulation(self.physics_client)
        self._setup_physics()
        
        # Create rocket with enhanced parameters
        self._create_enhanced_rocket()
        
        # Clear histories
        self.state_history.clear()
        self.action_history.clear()
        self.reward_components_history.clear()
        
        # Get initial observation
        obs = self._get_enhanced_observation()
        info = self._get_enhanced_info()
        
        return obs, info
    
    def _create_enhanced_rocket(self):
        """Create rocket with realistic parameters"""
        # Enhanced rocket parameters based on model rockets
        mass = 2.0
        length = 1.0
        radius = 0.05
        
        # Create rocket body with realistic inertia
        rocket_collision = p.createCollisionShape(
            p.GEOM_CYLINDER, 
            radius=radius, 
            height=length
        )
        
        rocket_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=radius,
            length=length,
            rgbaColor=[0.8, 0.2, 0.2, 1.0]
        )
        
        # Calculate realistic moments of inertia
        I_xx = I_yy = (1/12) * mass * (3 * radius**2 + length**2)
        I_zz = (1/2) * mass * radius**2
        
        self.rocket_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=rocket_collision,
            baseVisualShapeIndex=rocket_visual,
            basePosition=[0, 0, 1.0],
            baseOrientation=[0, 0, 0, 1],
            baseInertialFramePosition=[0, 0, 0],
            baseInertialFrameOrientation=[0, 0, 0, 1]
        )
        
        # Set the inertia after creation
        p.changeDynamics(
            self.rocket_id, -1,
            localInertiaDiagonal=[I_xx, I_yy, I_zz]
        )
        
        # Enhanced dynamics properties
        p.changeDynamics(
            self.rocket_id, -1,
            linearDamping=0.01,
            angularDamping=0.02,
            restitution=0.1,
            lateralFriction=0.3,
            spinningFriction=0.1,
            rollingFriction=0.05
        )
        
        # Initialize rocket state
        self.fuel_remaining = 1.0
        self.thrust_profile = 35.0  # N
        self.gimbal_angles = np.zeros(2)
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Enhanced step function with multi-objective rewards and mission tracking"""
        
        # Clip and process action
        action = np.clip(action, -1.0, 1.0)
        self.gimbal_angles = action * np.radians(18.0)  # Convert to radians
        
        # Apply enhanced control forces
        self._apply_enhanced_control()
        
        # Step physics
        p.stepSimulation(self.physics_client)
        self.current_step += 1
        
        # Get enhanced state information
        state_dict = self._get_state_dict()
        obs = self._get_enhanced_observation()
        
        # Update mission phase
        self._update_mission_phase(state_dict)
        
        # Check mission success
        self._check_mission_success(state_dict)
        
        # Compute multi-objective reward
        reward, reward_components = self.multi_objective_reward.compute_reward(
            state_dict, action, self.mission_success
        )
        
        # Add intrinsic curiosity reward if enabled
        if self.enable_curiosity and len(self.state_history) > 0:
            prev_obs = self.state_history[-1]
            intrinsic_reward = self.curiosity_module.compute_intrinsic_reward(
                prev_obs, action, obs[:8]  # Only use core observation
            )
            reward += intrinsic_reward
            reward_components['curiosity'] = intrinsic_reward
        
        # Store history
        self.state_history.append(obs[:8].copy())
        self.action_history.append(action.copy())
        self.reward_components_history.append(reward_components.copy())
        
        # Check termination
        terminated, truncated = self._check_termination(state_dict)
        
        # Enhanced info
        info = self._get_enhanced_info()
        info['reward_components'] = reward_components
        info['mission_phase'] = self.current_phase.value
        info['mission_successful'] = self.mission_successful
        
        return obs, reward, terminated, truncated, info
    
    def _apply_enhanced_control(self):
        """Apply enhanced thrust vector control with realistic physics"""
        pos, orn = p.getBasePositionAndOrientation(self.rocket_id)
        
        # Apply gravity (always present)
        mass = p.getDynamicsInfo(self.rocket_id, -1)[0]
        gravity_force = [0, 0, -9.81 * mass]
        p.applyExternalForce(self.rocket_id, -1, gravity_force, pos, p.WORLD_FRAME)
        
        # Apply thrust if fuel available
        if self.fuel_remaining > 0:
            # Consume fuel
            fuel_consumption = 0.001  # 0.1% per step
            self.fuel_remaining = max(0, self.fuel_remaining - fuel_consumption)
            
            # Calculate thrust vector in rocket frame
            thrust_magnitude = self.thrust_profile
            pitch_angle, yaw_angle = self.gimbal_angles
            
            thrust_vector_local = np.array([
                thrust_magnitude * np.sin(yaw_angle),
                thrust_magnitude * np.sin(pitch_angle),
                thrust_magnitude * np.cos(pitch_angle) * np.cos(yaw_angle)
            ])
            
            # Transform to world frame
            rotation_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
            thrust_vector_world = rotation_matrix @ thrust_vector_local
            
            # Apply thrust at rocket base
            thrust_position = np.array(pos) + rotation_matrix @ np.array([0, 0, -0.5])
            p.applyExternalForce(
                self.rocket_id, -1,
                thrust_vector_world.tolist(),
                thrust_position.tolist(),
                p.WORLD_FRAME
            )
            
        # Apply aerodynamic forces (simplified)
        self._apply_aerodynamics()
    
    def _apply_aerodynamics(self):
        """Apply realistic aerodynamic forces"""
        pos, orn = p.getBasePositionAndOrientation(self.rocket_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.rocket_id)
        
        # Air density decreases with altitude
        altitude = pos[2]
        air_density = 1.225 * np.exp(-altitude / 8400)  # Exponential atmosphere
        
        # Drag force
        velocity_mag = np.linalg.norm(linear_vel)
        if velocity_mag > 0.1:
            drag_coefficient = 0.47  # Cylinder
            frontal_area = np.pi * 0.05**2  # m^2
            drag_magnitude = 0.5 * air_density * velocity_mag**2 * drag_coefficient * frontal_area
            
            drag_direction = -np.array(linear_vel) / velocity_mag
            drag_force = drag_magnitude * drag_direction
            
            p.applyExternalForce(self.rocket_id, -1, drag_force.tolist(), pos, p.WORLD_FRAME)
        
        # Angular damping
        angular_damping = 0.02 * air_density
        damping_torque = -angular_damping * np.array(angular_vel)
        p.applyExternalTorque(self.rocket_id, -1, damping_torque.tolist(), p.WORLD_FRAME)
    
    def _get_enhanced_observation(self) -> np.ndarray:
        """Get enhanced observation vector"""
        pos, orn = p.getBasePositionAndOrientation(self.rocket_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.rocket_id)
        
        # Mission phase as normalized value
        phase_value = list(MissionPhase).index(self.current_phase) / len(MissionPhase)
        
        # Mission progress (0 to 1)
        mission_progress = min(1.0, self.current_step / self.max_episode_steps)
        
        obs = np.array([
            orn[0], orn[1], orn[2], orn[3],  # Quaternion
            angular_vel[0], angular_vel[1], angular_vel[2],  # Angular velocity
            self.fuel_remaining,  # Fuel
            phase_value,  # Mission phase
            mission_progress  # Mission progress
        ], dtype=np.float32)
        
        return obs
    
    def _get_state_dict(self) -> dict:
        """Get comprehensive state dictionary"""
        pos, orn = p.getBasePositionAndOrientation(self.rocket_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.rocket_id)
        
        # Calculate tilt angle from vertical
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        tilt_angle = np.sqrt(pitch**2 + yaw**2)
        
        return {
            'position': pos,
            'orientation': orn,
            'linear_velocity': linear_vel,
            'angular_velocity': angular_vel,
            'altitude': pos[2],
            'tilt_angle': tilt_angle,
            'angular_velocity_mag': np.linalg.norm(angular_vel),
            'horizontal_velocity': np.sqrt(linear_vel[0]**2 + linear_vel[1]**2),
            'vertical_velocity': abs(linear_vel[2]),
            'fuel_remaining': self.fuel_remaining,
            'mission_phase': self.current_phase.value,
            'mission_successful': self.mission_successful,
            'target_altitude': 3.0,  # Default target
            'crashed': pos[2] < 0.1,
        }
    
    def _update_mission_phase(self, state: dict):
        """Update mission phase based on flight conditions"""
        altitude = state['altitude']
        tilt = state['tilt_angle']
        fuel = state['fuel_remaining']
        
        # Phase transitions based on flight state
        if self.current_phase == MissionPhase.BOOST and fuel < 0.8:
            self.current_phase = MissionPhase.COAST
            self.phase_start_time = self.current_step
            
        elif self.current_phase == MissionPhase.COAST and altitude < 5.0:
            self.current_phase = MissionPhase.LANDING
            self.phase_start_time = self.current_step
            
        elif self.current_phase == MissionPhase.LANDING and altitude < 1.0:
            self.current_phase = MissionPhase.TOUCHDOWN
            self.phase_start_time = self.current_step
            
        elif self.current_phase == MissionPhase.TOUCHDOWN and altitude < 0.5:
            if tilt < 0.087 and state['angular_velocity_mag'] < 0.1:
                self.current_phase = MissionPhase.COMPLETE
                self.mission_successful = True
    
    def _check_mission_success(self, state: dict):
        """Enhanced mission success detection"""
        if self.mission_successful:
            return True
        
        # Check all success criteria
        criteria_met = {
            SuccessCriteria.ATTITUDE: state['tilt_angle'] < self.mission_success.max_tilt_angle,
            SuccessCriteria.VELOCITY: (
                state['vertical_velocity'] < self.mission_success.max_vertical_velocity and
                state['horizontal_velocity'] < self.mission_success.max_horizontal_velocity
            ),
            SuccessCriteria.POSITION: (
                self.mission_success.min_altitude <= state['altitude'] <= self.mission_success.max_altitude
            ),
            SuccessCriteria.STABILITY: state['angular_velocity_mag'] < self.mission_success.max_angular_velocity,
        }
        
        # Store criteria in history
        self.mission_success.criteria_history.append(criteria_met)
        
        # Check if all criteria have been met for required duration
        if len(self.mission_success.criteria_history) >= self.mission_success.success_duration:
            recent_criteria = list(self.mission_success.criteria_history)
            
            # All criteria must be met for the entire duration
            all_met = all(
                all(criteria[c] for c in SuccessCriteria if c != SuccessCriteria.FUEL)
                for criteria in recent_criteria
            )
            
            if all_met:
                self.mission_successful = True
                if self.debug:
                    self.logger.info(f"Mission success achieved at step {self.current_step}")
        
        return self.mission_successful
    
    def _check_termination(self, state: dict) -> Tuple[bool, bool]:
        """Enhanced termination checking"""
        terminated = False
        truncated = False
        
        # Mission success termination
        if self.mission_successful:
            terminated = True
            return terminated, truncated
        
        # Failure conditions
        if state['crashed']:
            terminated = True
        elif state['tilt_angle'] > 0.52:  # 30 degrees (safety limit)
            terminated = True
        elif state['altitude'] > 20.0:  # Too high
            terminated = True
        elif np.sqrt(state['position'][0]**2 + state['position'][1]**2) > 50.0:  # Too far
            terminated = True
        
        # Episode length limit
        if self.current_step >= self.max_episode_steps:
            truncated = True
        
        return terminated, truncated
    
    def _get_enhanced_info(self) -> dict:
        """Get enhanced info dictionary"""
        pos, orn = p.getBasePositionAndOrientation(self.rocket_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.rocket_id)
        
        euler = p.getEulerFromQuaternion(orn)
        tilt_angle = np.sqrt(euler[1]**2 + euler[2]**2)
        
        return {
            'position': pos,
            'altitude': pos[2],
            'tilt_angle_deg': np.degrees(tilt_angle),
            'angular_velocity_mag': np.linalg.norm(angular_vel),
            'fuel_remaining': self.fuel_remaining,
            'mission_phase': self.current_phase.value,
            'mission_successful': self.mission_successful,
            'step': self.current_step,
            'success_criteria_met': len(self.mission_success.criteria_history) > 0 and 
                                  all(all(c.values()) for c in list(self.mission_success.criteria_history)[-10:]) if len(self.mission_success.criteria_history) >= 10 else False
        }
    
    def render(self, mode: str = "human"):
        """Enhanced rendering with mission status"""
        # Basic PyBullet rendering is handled automatically
        pass
    
    def close(self):
        """Clean up resources"""
        if hasattr(self, 'physics_client') and self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None

# Factory function for creating the enhanced environment
def make_enhanced_tvc_env(**kwargs) -> EnhancedRocketTVCEnv:
    """Create enhanced TVC environment with state-of-the-art features"""
    return EnhancedRocketTVCEnv(**kwargs)

if __name__ == "__main__":
    # Test the enhanced environment
    env = EnhancedRocketTVCEnv(
        render_mode="human",
        debug=True,
        enable_hierarchical=True,
        enable_curiosity=True,
        enable_physics_informed=True
    )
    
    print("Enhanced TVC Environment Test")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run test episode
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 50 == 0:
            print(f"Step {step}: Reward={reward:.3f}, Phase={info['mission_phase']}, "
                  f"Success={info['mission_successful']}, Altitude={info['altitude']:.2f}m")
        
        if terminated or truncated:
            print(f"Episode finished at step {step}")
            print(f"Final success: {info['mission_successful']}")
            break
    
    env.close()
