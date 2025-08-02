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
Model Rocket TVC Environment

A high-fidelity PyBullet-based simulation environment for training Deep Reinforcement Learning
agents to control model rocket attitude using Thrust Vector Control (TVC).

Features:
- 6-DOF rocket physics simulation with PyBullet
- Domain randomization for sim-to-real transfer
- Gymnasium-compatible interface
- Integration with RocketPy for enhanced aerodynamics
- Realistic sensor noise modeling
- Customizable reward functions
"""

import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Any
import logging
import math
import random
from dataclasses import dataclass

# Import RocketPy for high-fidelity aerodynamics (optional)
try:
    from rocketpy import Rocket, Environment as RocketPyEnv, SolidMotor
    ROCKETPY_AVAILABLE = True
    logging.info("✅ RocketPy available - high-fidelity aerodynamics enabled")
except ImportError:
    ROCKETPY_AVAILABLE = False
    logging.info("ℹ️ RocketPy not available - using simplified aerodynamics model")


@dataclass
class RocketConfig:
    """Configuration parameters for the rocket simulation."""
    # Physical properties
    mass: float = 1.5  # kg, dry mass
    radius: float = 0.05  # m, rocket radius
    length: float = 0.8  # m, rocket length
    inertia_ratio: float = 10.0  # I_xx / I_zz ratio
    
    # Motor properties
    thrust_mean: float = 50.0  # N, average thrust
    thrust_std: float = 5.0  # N, thrust variation
    burn_time: float = 3.0  # s, motor burn time
    
    # TVC properties
    max_gimbal_angle: float = 25.0  # degrees, maximum gimbal deflection (increased for better control authority)
    gimbal_response_time: float = 0.05  # s, servo response time
    max_gimbal_rate: float = 100.0  # degrees/s, maximum gimbal rate to prevent chattering
    
    # Domain randomization ranges
    mass_variation: float = 0.2  # ±20% mass variation
    cg_offset_max: float = 0.05  # m, maximum CG offset
    thrust_variation: float = 0.3  # ±30% thrust variation
    
    # Physics simulation parameters
    gravity: float = -9.81  # m/s², gravitational acceleration
    physics_timestep: float = 1/240.0  # s, physics simulation timestep
    air_density: float = 1.225  # kg/m³, air density at sea level
    drag_coefficient: float = 0.5  # dimensionless, drag coefficient
    
    # Initial conditions
    initial_altitude: float = 1.0  # m, starting altitude above ground
    target_altitude: float = 5.0  # m, target altitude
    max_initial_tilt: float = 2.0  # degrees, maximum random initial tilt
    max_initial_angular_vel: float = 0.5  # rad/s, maximum random initial angular velocity
    
    # Wind and disturbances
    wind_force_max: float = 2.0  # N, maximum wind force in each direction
    
    # Sensor noise parameters
    gyro_noise_std: float = 0.1  # rad/s, gyroscope noise standard deviation
    quaternion_noise_std: float = 0.01  # quaternion noise standard deviation
    
    # Reward function parameters
    attitude_penalty_gain: float = 15.0  # attitude penalty gain (k_angle)
    angular_velocity_penalty_gain: float = 0.2  # angular velocity penalty gain (k_vel)
    control_effort_penalty_gain: float = 0.02  # control effort penalty gain (k_action)
    saturation_threshold: float = 0.8  # control saturation threshold (0-1)
    saturation_penalty: float = 2.0  # penalty for exceeding saturation threshold
    saturation_bonus: float = 0.1  # bonus for staying below saturation threshold
    stability_angle_threshold: float = 3.0  # degrees, angle threshold for stability bonus
    stability_angular_vel_threshold: float = 0.5  # rad/s, angular velocity threshold for stability bonus
    stability_bonus: float = 2.0  # stability bonus value
    tilt_improvement_threshold: float = 0.1  # degrees, threshold for tilt improvement detection
    tilt_improvement_bonus: float = 1.0  # bonus for improving tilt
    tilt_degradation_penalty: float = 2.0  # penalty for increasing tilt
    efficiency_bonus: float = 0.5  # fuel efficiency bonus multiplier
    
    # Altitude management
    min_safe_altitude: float = 0.5  # m, minimum safe altitude
    max_safe_altitude: float = 15.0  # m, maximum safe altitude
    altitude_penalty_gain: float = 5.0  # penalty gain for unsafe altitudes
    nominal_altitude_bonus: float = 0.2  # bonus for staying in safe altitude range
    
    # Termination conditions
    ground_termination_height: float = 0.1  # m, altitude below which rocket is considered crashed
    max_tilt_degrees: float = 45.0  # degrees, maximum tilt before termination
    max_horizontal_distance: float = 50.0  # m, maximum horizontal distance before termination
    max_termination_altitude: float = 20.0  # m, maximum altitude before termination
    max_angular_velocity: float = 10.0  # rad/s, maximum angular velocity before termination
    
    # Termination penalties
    crash_penalty: float = -50.0  # penalty for crashing
    tilt_penalty: float = -30.0  # penalty for excessive tilt
    altitude_penalty: float = -20.0  # penalty for too high altitude
    angular_velocity_penalty: float = -25.0  # penalty for excessive angular velocity
    
    # Aerodynamics parameters
    aerodynamic_damping_coefficient: float = 0.02  # aerodynamic damping coefficient
    minimum_thrust: float = 10.0  # N, minimum thrust when fuel is available
    
    # Physics properties
    lateral_friction: float = 0.1  # lateral friction coefficient
    spinning_friction: float = 0.01  # spinning friction coefficient
    rolling_friction: float = 0.01  # rolling friction coefficient
    restitution: float = 0.1  # restitution coefficient
    linear_damping: float = 0.01  # linear damping coefficient
    angular_damping: float = 0.01  # angular damping coefficient
    
    # Visual properties
    rocket_color: List[float] = None  # RGBA color [r, g, b, a], defaults to [0.8, 0.2, 0.2, 1.0]
    
    def __post_init__(self):
        """Post-initialization to set defaults for mutable types."""
        if self.rocket_color is None:
            self.rocket_color = [0.8, 0.2, 0.2, 1.0]


class RocketTVCEnv(gym.Env):
    """
    PyBullet-based Model Rocket TVC Environment for Deep Reinforcement Learning.
    
    The environment simulates a model rocket with thrust vector control attempting
    to maintain stable vertical flight. The agent controls gimbal angles to 
    stabilize the rocket against perturbations.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        config: Optional[RocketConfig] = None,
        max_episode_steps: int = 1000,
        render_mode: Optional[str] = None,
        domain_randomization: bool = True,
        use_rocketpy: bool = False,
        sensor_noise: bool = True,
        debug: bool = False
    ):
        """
        Initialize the Rocket TVC Environment.
        
        Args:
            config: Rocket configuration parameters
            max_episode_steps: Maximum steps per episode
            render_mode: Rendering mode ('human', 'rgb_array', or None)
            domain_randomization: Enable domain randomization
            use_rocketpy: Use RocketPy for enhanced aerodynamics
            sensor_noise: Enable sensor noise simulation
            debug: Enable debug mode with additional logging
        """
        super().__init__()
        
        self.config = config or RocketConfig()
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.domain_randomization = domain_randomization
        self.use_rocketpy = use_rocketpy and ROCKETPY_AVAILABLE
        self.sensor_noise = sensor_noise
        self.debug = debug
        
        # Episode tracking
        self.current_step = 0
        self.episode_count = 0
        
        # Physics simulation
        self.physics_client = None
        self.rocket_id = None
        self.ground_id = None
        
        # State variables
        self.initial_position = [0, 0, self.config.initial_altitude]  # Start at configured altitude
        self.target_position = [0, 0, self.config.target_altitude]   # Target altitude from config
        
        # Control variables
        self.current_thrust = 0.0
        self.gimbal_angles = np.zeros(2)  # [pitch, yaw]
        self.previous_gimbal_angles = np.zeros(2)  # For rate limiting
        self.motor_burn_remaining = 0.0
        
        # Domain randomization parameters (set each episode)
        self.current_mass = self.config.mass
        self.current_thrust_profile = None
        self.current_cg_offset = np.zeros(3)
        
        # Define action and observation spaces
        self._setup_spaces()
        
        # Initialize RocketPy if available
        if self.use_rocketpy:
            self._setup_rocketpy()
        
        # Logging setup
        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
        
        self.logger = logging.getLogger(__name__)
    
    def _setup_spaces(self):
        """Define the action and observation spaces."""
        # Action space: continuous gimbal angles [pitch, yaw] normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation space: [qx, qy, qz, qw, wx, wy, wz, fuel_remaining]
        # Quaternion (4) + Angular velocities (3) + Fuel (1) = 8 dimensions
        obs_low = np.array([
            -1.0, -1.0, -1.0, -1.0,  # Quaternion bounds
            -10.0, -10.0, -10.0,      # Angular velocity bounds (rad/s)
            0.0                       # Fuel remaining [0, 1]
        ], dtype=np.float32)
        
        obs_high = np.array([
            1.0, 1.0, 1.0, 1.0,      # Quaternion bounds
            10.0, 10.0, 10.0,        # Angular velocity bounds (rad/s)
            1.0                      # Fuel remaining [0, 1]
        ], dtype=np.float32)
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
    
    def _setup_rocketpy(self):
        """Initialize RocketPy environment for enhanced aerodynamics."""
        if not ROCKETPY_AVAILABLE:
            return
        
        try:
            # Create RocketPy environment
            self.rocketpy_env = RocketPyEnv(latitude=32.990254, longitude=-106.974998)
            
            # Define motor (simplified)
            self.rocketpy_motor = SolidMotor(
                thrust_source="data/motor_thrust.csv",  # Would need actual data
                dry_mass=0.1,
                dry_inertia=(0.01, 0.01, 0.001),
                nozzle_radius=0.02,
                grain_number=1,
                grain_density=1800,
                grain_outer_radius=0.015,
                grain_initial_inner_radius=0.005,
                grain_initial_height=0.1
            )
            
            # Create rocket
            self.rocketpy_rocket = Rocket(
                radius=self.config.radius,
                mass=self.config.mass,
                inertia=(0.1, 0.1, 0.01),
                power_off_drag="data/drag_curve.csv",  # Would need actual data
                power_on_drag="data/drag_curve.csv"
            )
            
            self.rocketpy_rocket.add_motor(self.rocketpy_motor, position=-0.4)
            
        except Exception as e:
            self.logger.warning(f"Failed to initialize RocketPy: {e}")
            self.use_rocketpy = False
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Disconnect previous physics client if exists
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        # Initialize PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Configure physics
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, self.config.gravity)
        p.setTimeStep(self.config.physics_timestep)
        p.setRealTimeSimulation(0)
        
        # Load ground plane
        self.ground_id = p.loadURDF("plane.urdf")
        
        # Apply domain randomization
        if self.domain_randomization:
            self._apply_domain_randomization()
        else:
            self._reset_to_nominal_parameters()
        
        # Create rocket
        self._create_rocket()
        
        # Reset episode variables
        self.current_step = 0
        self.episode_count += 1
        self.motor_burn_remaining = self.config.burn_time
        self.gimbal_angles = np.zeros(2)
        self.previous_gimbal_angles = np.zeros(2)  # Reset previous angles
        self.previous_tilt = 0.0  # Track tilt for reward computation
        
        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _apply_domain_randomization(self):
        """Apply domain randomization to rocket parameters."""
        # Randomize mass
        mass_factor = 1.0 + np.random.uniform(
            -self.config.mass_variation, 
            self.config.mass_variation
        )
        self.current_mass = self.config.mass * mass_factor
        
        # Randomize center of gravity offset (more conservative to prevent instability)
        self.current_cg_offset = np.random.uniform(
            -self.config.cg_offset_max * 0.5,  # Reduced from full range
            self.config.cg_offset_max * 0.5,
            size=3
        )
        
        # Randomize thrust profile
        thrust_factor = 1.0 + np.random.uniform(
            -self.config.thrust_variation,
            self.config.thrust_variation
        )
        self.current_thrust_profile = self.config.thrust_mean * thrust_factor
        
        # Randomize initial conditions for more robust training
        # Small random initial orientation (up to configured degrees)
        initial_tilt = np.random.uniform(-self.config.max_initial_tilt, self.config.max_initial_tilt, size=2)  # pitch, yaw in degrees
        initial_quat = p.getQuaternionFromEuler([0, np.radians(initial_tilt[0]), np.radians(initial_tilt[1])])
        
        # Small random initial angular velocity (up to configured rad/s)
        initial_angular_vel = np.random.uniform(-self.config.max_initial_angular_vel, self.config.max_initial_angular_vel, size=3)
        
        # Store for use in _create_rocket
        self.initial_orientation = initial_quat
        self.initial_angular_velocity = initial_angular_vel
        
        # Add slight random wind disturbance
        self.wind_force = np.random.uniform(-self.config.wind_force_max, self.config.wind_force_max, size=3)  # N
        
        if self.debug:
            self.logger.debug(f"Domain randomization - Mass: {self.current_mass:.3f}, "
                            f"CG offset: {self.current_cg_offset}, "
                            f"Thrust factor: {thrust_factor:.3f}, "
                            f"Initial tilt: {initial_tilt}, "
                            f"Wind: {self.wind_force}")
            
    def _reset_to_nominal_parameters(self):
        """Reset to nominal (non-randomized) parameters."""
        self.current_mass = self.config.mass
        self.current_cg_offset = np.zeros(3)
        self.current_thrust_profile = self.config.thrust_mean
        self.initial_orientation = [0, 0, 0, 1]  # No rotation
        self.initial_angular_velocity = [0, 0, 0]  # No initial rotation
        self.wind_force = np.zeros(3)  # No wind
    
    def _create_rocket(self):
        """Create the rocket in PyBullet."""
        # Create rocket body (cylinder)
        collision_shape = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=self.config.radius,
            height=self.config.length
        )
        
        visual_shape = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.config.radius,
            length=self.config.length,
            rgbaColor=self.config.rocket_color
        )
        
        # Calculate inertia
        I_xx = self.current_mass * (3 * self.config.radius**2 + self.config.length**2) / 12
        I_zz = self.current_mass * self.config.radius**2 / 2
        
        self.rocket_id = p.createMultiBody(
            baseMass=self.current_mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=self.initial_position,
            baseOrientation=getattr(self, 'initial_orientation', [0, 0, 0, 1]),  # Use randomized if available
            baseInertialFramePosition=self.current_cg_offset,
            baseInertialFrameOrientation=[0, 0, 0, 1]
        )
        
        # Set initial angular velocity if randomized
        if hasattr(self, 'initial_angular_velocity'):
            p.resetBaseVelocity(self.rocket_id, [0, 0, 0], self.initial_angular_velocity)
        
        # Set physics properties
        p.changeDynamics(
            self.rocket_id,
            -1,  # Base link
            lateralFriction=self.config.lateral_friction,
            spinningFriction=self.config.spinning_friction,
            rollingFriction=self.config.rolling_friction,
            restitution=self.config.restitution,
            linearDamping=self.config.linear_damping,
            angularDamping=self.config.angular_damping
        )
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step."""
        # Ensure action is in correct format
        action = np.clip(action, -1.0, 1.0)
        
        # Convert normalized action to gimbal angles with rate limiting
        dt = self.config.physics_timestep
        desired_gimbal_angles = action * np.radians(self.config.max_gimbal_angle)
        
        # Apply gimbal rate limiting to prevent chattering
        max_angle_change = np.radians(self.config.max_gimbal_rate) * dt
        angle_diff = desired_gimbal_angles - self.previous_gimbal_angles
        angle_diff = np.clip(angle_diff, -max_angle_change, max_angle_change)
        
        self.gimbal_angles = self.previous_gimbal_angles + angle_diff
        self.previous_gimbal_angles = self.gimbal_angles.copy()
        
        # Apply control forces
        self._apply_control_forces()
        
        # Apply aerodynamic forces
        self._apply_aerodynamic_forces()
        
        # Step the physics simulation
        p.stepSimulation()
        
        # Update time
        self.current_step += 1
        self.motor_burn_remaining = max(0, self.motor_burn_remaining - dt)
        
        # Get observation and compute reward
        obs = self._get_observation()
        reward = self._compute_reward()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= self.max_episode_steps
        
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info
    
    def _apply_control_forces(self):
        """Apply thrust and gimbal forces to the rocket."""
        # Always apply gravity explicitly to ensure proper physics
        gravity_force = [0, 0, self.config.gravity * self.current_mass]
        pos, orn = p.getBasePositionAndOrientation(self.rocket_id)
        p.applyExternalForce(
            self.rocket_id,
            -1,
            gravity_force,
            pos,
            p.WORLD_FRAME
        )
        
        # Only apply thrust if fuel is available
        if self.motor_burn_remaining <= 0:
            if self.debug and self.current_step % 60 == 0:
                self.logger.debug("Motor burned out - no thrust applied")
            return
        
        # Calculate thrust magnitude with noise
        thrust_noise = 0.0
        if self.sensor_noise:
            thrust_noise = np.random.normal(0, self.config.thrust_std)
        
        thrust_magnitude = max(0, self.current_thrust_profile + thrust_noise)
        
        # Ensure minimum thrust when fuel is available
        if thrust_magnitude < self.config.minimum_thrust:  # Minimum thrust from config
            thrust_magnitude = self.config.minimum_thrust
        
        # Convert gimbal angles to thrust vector in rocket frame
        pitch_angle, yaw_angle = self.gimbal_angles
        
        # Thrust vector in rocket frame (nominally pointing up in +Z direction)
        thrust_vector_local = np.array([
            thrust_magnitude * np.sin(yaw_angle),    # X component
            thrust_magnitude * np.sin(pitch_angle),  # Y component  
            thrust_magnitude * np.cos(pitch_angle) * np.cos(yaw_angle)  # Z component
        ])
        
        # Convert to world frame
        rotation_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        thrust_vector_world = rotation_matrix @ thrust_vector_local
        
        # Apply force at the rocket's base (motor location)
        force_position = np.array(pos) + rotation_matrix @ np.array([0, 0, -self.config.length/2])
        
        p.applyExternalForce(
            self.rocket_id,
            -1,  # Base link
            thrust_vector_world,
            force_position,
            p.WORLD_FRAME
        )
        
        if self.debug and self.current_step % 60 == 0:
            self.logger.debug(f"Applied thrust: {thrust_magnitude:.1f}N at angles "
                            f"[{np.degrees(pitch_angle):.1f}°, {np.degrees(yaw_angle):.1f}°]")
    
    def _apply_aerodynamic_forces(self):
        """Apply aerodynamic forces and moments."""
        # Get rocket state
        pos, orn = p.getBasePositionAndOrientation(self.rocket_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.rocket_id)
        
        # Convert to numpy arrays
        velocity = np.array(linear_vel)
        omega = np.array(angular_vel)
        
        if self.use_rocketpy:
            # Use RocketPy for detailed aerodynamics (placeholder - would need integration)
            drag_force = self._compute_rocketpy_aerodynamics(pos, orn, velocity)
        else:
            # Simplified aerodynamic model
            drag_force = self._compute_simple_aerodynamics(velocity, orn)
        
        # Apply aerodynamic drag
        p.applyExternalForce(
            self.rocket_id,
            -1,
            drag_force,
            pos,
            p.WORLD_FRAME
        )
        
        # Apply wind disturbance if domain randomization is enabled
        if hasattr(self, 'wind_force') and self.domain_randomization:
            p.applyExternalForce(
                self.rocket_id,
                -1,
                self.wind_force,
                pos,
                p.WORLD_FRAME
            )
        
        # Apply aerodynamic moments (improved damping)
        # More realistic aerodynamic damping based on rocket geometry
        reference_area = np.pi * self.config.radius**2
        length_scale = self.config.length
        damping_coefficient = self.config.aerodynamic_damping_coefficient * reference_area * length_scale
        aerodynamic_moment = -damping_coefficient * omega
        
        p.applyExternalTorque(
            self.rocket_id,
            -1,
            aerodynamic_moment,
            p.WORLD_FRAME
        )
    
    def _compute_simple_aerodynamics(self, velocity: np.ndarray, orientation: Tuple) -> np.ndarray:
        """Compute simplified aerodynamic forces."""
        # Air density from config
        rho = self.config.air_density
        
        # Reference area (cross-sectional)
        A_ref = np.pi * self.config.radius**2
        
        # Drag coefficient from config
        Cd = self.config.drag_coefficient
        
        # Velocity magnitude
        v_mag = np.linalg.norm(velocity)
        
        if v_mag < 0.1:  # Avoid division by zero
            return np.zeros(3)
        
        # Drag force (opposite to velocity direction)
        drag_magnitude = 0.5 * rho * Cd * A_ref * v_mag**2
        drag_direction = -velocity / v_mag
        
        return drag_magnitude * drag_direction
    
    def _compute_rocketpy_aerodynamics(self, position: Tuple, orientation: Tuple, velocity: np.ndarray) -> np.ndarray:
        """Compute aerodynamics using RocketPy (placeholder)."""
        # This would require deeper integration with RocketPy
        # For now, fall back to simple model
        return self._compute_simple_aerodynamics(velocity, orientation)
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        # Get rocket state
        pos, orn = p.getBasePositionAndOrientation(self.rocket_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.rocket_id)
        
        # Convert quaternion to numpy array and normalize
        quat = np.array(orn)  # [x, y, z, w]
        quat = quat / np.linalg.norm(quat)  # Ensure normalized
        
        # Angular velocities
        omega = np.array(angular_vel)
        
        # Add sensor noise if enabled
        if self.sensor_noise:
            # IMU noise characteristics from config
            gyro_noise = np.random.normal(0, self.config.gyro_noise_std, 3)  # rad/s
            quat_noise = np.random.normal(0, self.config.quaternion_noise_std, 4)  # quaternion noise
            
            omega += gyro_noise
            quat += quat_noise
            quat = quat / np.linalg.norm(quat)  # Renormalize after noise
        
        # Fuel remaining (normalized)
        fuel_remaining = max(0, self.motor_burn_remaining / self.config.burn_time)
        
        # Construct observation vector
        obs = np.concatenate([
            quat,           # Quaternion [qx, qy, qz, qw]
            omega,          # Angular velocities [wx, wy, wz]
            [fuel_remaining] # Fuel remaining [0, 1]
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self) -> float:
        """Compute the reward for the current state."""
        # Get rocket state
        pos, orn = p.getBasePositionAndOrientation(self.rocket_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.rocket_id)
        
        # Convert quaternion to euler angles for attitude calculation
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        
        # Total tilt angle from vertical
        theta_total = np.sqrt(pitch**2 + yaw**2)
        theta_deg = np.degrees(theta_total)
        
        # Track tilt improvement/degradation
        tilt_change = theta_deg - self.previous_tilt
        self.previous_tilt = theta_deg
        
        # Reward components
        # 1. Attitude reward (stronger exponential decay with tilt angle)
        k_angle = self.config.attitude_penalty_gain
        R_attitude = np.exp(-k_angle * theta_total**2)
        
        # 2. Angular velocity penalty (encourage damping)
        k_vel = self.config.angular_velocity_penalty_gain
        omega_mag = np.linalg.norm(angular_vel)
        R_angular_vel = -k_vel * omega_mag**2
        
        # 3. Control effort penalty with anti-saturation bonus
        k_action = self.config.control_effort_penalty_gain
        action_mag = np.linalg.norm(self.gimbal_angles)
        max_action = np.radians(self.config.max_gimbal_angle)
        
        # Standard control effort penalty
        R_control_effort = -k_action * action_mag**2
        
        # Anti-saturation bonus: reward for staying away from control limits
        saturation_level = action_mag / max_action
        if saturation_level > self.config.saturation_threshold:  # If using >threshold of control authority
            R_saturation_penalty = -self.config.saturation_penalty * (saturation_level - self.config.saturation_threshold)  # Additional penalty
        else:
            R_saturation_penalty = self.config.saturation_bonus * (self.config.saturation_threshold - saturation_level)  # Small bonus for margin
        
        # 4. Altitude management
        altitude = pos[2]
        if altitude < self.config.min_safe_altitude:  # Too low
            R_altitude = -self.config.altitude_penalty_gain * (self.config.min_safe_altitude - altitude)
        elif altitude > self.config.max_safe_altitude:  # Too high
            R_altitude = -0.5 * (altitude - self.config.max_safe_altitude)
        else:
            R_altitude = self.config.nominal_altitude_bonus  # Small bonus for reasonable altitude
        
        # 5. Stability bonus (low angular velocity when upright)
        if theta_total < np.radians(self.config.stability_angle_threshold):  # Within configured degrees of vertical
            if omega_mag < self.config.stability_angular_vel_threshold:  # Low angular velocity
                R_stability = self.config.stability_bonus  # Configured bonus
            else:
                R_stability = 0.5
        else:
            R_stability = 0.0
        
        # 6. Tilt improvement reward
        if tilt_change < -self.config.tilt_improvement_threshold:  # Tilt is decreasing significantly
            R_tilt_improvement = self.config.tilt_improvement_bonus
        elif tilt_change > self.config.tilt_improvement_threshold:  # Tilt is increasing significantly
            R_tilt_improvement = -self.config.tilt_degradation_penalty  # Strong penalty for getting worse
        else:
            R_tilt_improvement = 0.0
        
        # 7. Fuel efficiency reward (only when motor is burning)
        if self.motor_burn_remaining > 0:
            fuel_remaining = self.motor_burn_remaining / self.config.burn_time
            # Bonus for maintaining control with less control effort
            if theta_total < np.radians(2) and action_mag < max_action * 0.5:
                R_efficiency = self.config.efficiency_bonus * fuel_remaining
            else:
                R_efficiency = 0.0
        else:
            # When fuel is out, strong reward for maintaining attitude
            R_efficiency = 2.0 * R_attitude if theta_total < np.radians(5) else -1.0
        
        # Total reward
        total_reward = (R_attitude + R_angular_vel + R_control_effort + 
                       R_saturation_penalty + R_altitude + R_stability + 
                       R_tilt_improvement + R_efficiency)
        
        if self.debug and self.current_step % 60 == 0:  # Log every second
            self.logger.debug(f"Reward components - Attitude: {R_attitude:.3f}, "
                            f"Angular vel: {R_angular_vel:.3f}, Control: {R_control_effort:.3f}, "
                            f"Saturation: {R_saturation_penalty:.3f}, Altitude: {R_altitude:.3f}, "
                            f"Stability: {R_stability:.3f}, Tilt improvement: {R_tilt_improvement:.3f}, "
                            f"Efficiency: {R_efficiency:.3f}, Total: {total_reward:.3f}")
        
        return total_reward
    
    def _check_termination(self) -> bool:
        """Check if the episode should terminate."""
        # Get rocket state
        pos, orn = p.getBasePositionAndOrientation(self.rocket_id)
        
        # Check if rocket crashed (hit ground)
        if pos[2] < self.config.ground_termination_height:
            if self.debug:
                self.logger.debug("Episode terminated: Rocket crashed")
            return True
        
        # Check if rocket tilted too much (failure threshold)
        euler = p.getEulerFromQuaternion(orn)
        roll, pitch, yaw = euler
        theta_total = np.sqrt(pitch**2 + yaw**2)
        
        if theta_total > np.radians(self.config.max_tilt_degrees):
            if self.debug:
                self.logger.debug(f"Episode terminated: Excessive tilt ({np.degrees(theta_total):.1f}°)")
            return True
        
        # Check if rocket flew too far horizontally
        horizontal_distance = np.sqrt(pos[0]**2 + pos[1]**2)
        if horizontal_distance > self.config.max_horizontal_distance:
            if self.debug:
                self.logger.debug(f"Episode terminated: Too far horizontally ({horizontal_distance:.1f}m)")
            return True
        
        return False
    
    def _get_info(self) -> Dict:
        """Get additional information about the current state."""
        if self.rocket_id is None:
            return {}
        
        # Get rocket state
        pos, orn = p.getBasePositionAndOrientation(self.rocket_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.rocket_id)
        
        # Convert quaternion to euler
        euler = p.getEulerFromQuaternion(orn)
        
        info = {
            "position": pos,
            "orientation_euler": euler,
            "orientation_quaternion": orn,
            "linear_velocity": linear_vel,
            "angular_velocity": angular_vel,
            "altitude": pos[2],
            "tilt_angle_deg": np.degrees(np.sqrt(euler[1]**2 + euler[2]**2)),
            "fuel_remaining": max(0, self.motor_burn_remaining / self.config.burn_time),
            "gimbal_angles_deg": np.degrees(self.gimbal_angles),
            "episode_step": self.current_step,
            "episode_count": self.episode_count
        }
        
        return info
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human":
            # Camera follows the rocket
            if self.rocket_id is not None:
                pos, _ = p.getBasePositionAndOrientation(self.rocket_id)
                p.resetDebugVisualizerCamera(
                    cameraDistance=5.0,
                    cameraYaw=45,
                    cameraPitch=-30,
                    cameraTargetPosition=pos
                )
        elif self.render_mode == "rgb_array":
            # Return RGB array for video recording
            if self.rocket_id is not None:
                pos, _ = p.getBasePositionAndOrientation(self.rocket_id)
                view_matrix = p.computeViewMatrixFromYawPitchRoll(
                    cameraTargetPosition=pos,
                    distance=5.0,
                    yaw=45,
                    pitch=-30,
                    roll=0,
                    upAxisIndex=2
                )
                proj_matrix = p.computeProjectionMatrixFOV(
                    fov=60,
                    aspect=1.0,
                    nearVal=0.1,
                    farVal=100.0
                )
                
                width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
                    width=640,
                    height=480,
                    viewMatrix=view_matrix,
                    projectionMatrix=proj_matrix
                )
                
                return np.array(rgb_img).reshape(height, width, 4)[:, :, :3]
        
        return None
    
    def close(self):
        """Clean up the environment."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


# Convenience function to create the environment
def make_rocket_tvc_env(**kwargs) -> RocketTVCEnv:
    """Create and return a RocketTVCEnv instance."""
    return RocketTVCEnv(**kwargs)


if __name__ == "__main__":
    # Test the environment
    env = RocketTVCEnv(render_mode="human", debug=True)
    
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Run a short test episode
    for step in range(100):
        action = env.action_space.sample()  # Random actions
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 20 == 0:
            print(f"Step {step}: Reward={reward:.3f}, "
                  f"Altitude={info['altitude']:.2f}m, "
                  f"Tilt={info['tilt_angle_deg']:.1f}°")
        
        if terminated or truncated:
            print(f"Episode finished at step {step}")
            break
    
    env.close()
