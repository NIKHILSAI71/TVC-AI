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
Test suite for the Rocket TVC environment.
"""

import pytest
import numpy as np
import gymnasium as gym

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from env import RocketTVCEnv, RocketConfig, make_training_env, make_evaluation_env


class TestRocketTVCEnv:
    """Test suite for RocketTVCEnv."""
    
    def test_environment_creation(self):
        """Test environment can be created successfully."""
        env = RocketTVCEnv()
        assert env is not None
        env.close()
    
    def test_observation_space(self):
        """Test observation space is correctly defined."""
        env = RocketTVCEnv()
        
        obs_space = env.observation_space
        assert isinstance(obs_space, gym.spaces.Box)
        assert obs_space.shape == (8,)  # qx, qy, qz, qw, wx, wy, wz, fuel
        
        env.close()
    
    def test_action_space(self):
        """Test action space is correctly defined."""
        env = RocketTVCEnv()
        
        action_space = env.action_space
        assert isinstance(action_space, gym.spaces.Box)
        assert action_space.shape == (2,)  # pitch, yaw gimbal angles
        assert np.allclose(action_space.low, -1.0)
        assert np.allclose(action_space.high, 1.0)
        
        env.close()
    
    def test_reset_functionality(self):
        """Test environment reset works correctly."""
        env = RocketTVCEnv()
        
        obs, info = env.reset()
        
        # Check observation
        assert obs.shape == (8,)
        assert np.all(np.isfinite(obs))
        
        # Check quaternion is normalized
        quat = obs[:4]
        quat_norm = np.linalg.norm(quat)
        assert np.isclose(quat_norm, 1.0, atol=1e-3)
        
        # Check info dict
        assert isinstance(info, dict)
        assert 'position' in info
        assert 'altitude' in info
        
        env.close()
    
    def test_step_functionality(self):
        """Test environment step works correctly."""
        env = RocketTVCEnv()
        
        obs, _ = env.reset()
        action = np.array([0.0, 0.0])  # No control input
        
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Check outputs
        assert next_obs.shape == (8,)
        assert np.all(np.isfinite(next_obs))
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()
    
    def test_domain_randomization(self):
        """Test domain randomization varies parameters."""
        env1 = RocketTVCEnv(domain_randomization=True)
        env2 = RocketTVCEnv(domain_randomization=True)
        
        # Reset multiple times and check for variation
        masses1 = []
        masses2 = []
        
        for _ in range(5):
            env1.reset()
            env2.reset()
            masses1.append(env1.current_mass)
            masses2.append(env2.current_mass)
        
        # Should have some variation
        assert np.std(masses1) > 0 or np.std(masses2) > 0
        
        env1.close()
        env2.close()
    
    def test_reward_computation(self):
        """Test reward computation is reasonable."""
        env = RocketTVCEnv()
        
        obs, _ = env.reset()
        
        # Test with no control (should receive some reward for initial upright position)
        action = np.array([0.0, 0.0])
        _, reward, _, _, _ = env.step(action)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        assert not np.isinf(reward)
        
        env.close()
    
    def test_episode_termination(self):
        """Test episode termination conditions."""
        env = RocketTVCEnv(max_episode_steps=50)
        
        obs, _ = env.reset()
        
        # Run episode to completion
        for step in range(100):  # More than max_episode_steps
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        # Should terminate due to max steps or other condition
        assert terminated or truncated
        
        env.close()


class TestRocketConfig:
    """Test suite for RocketConfig."""
    
    def test_default_config(self):
        """Test default configuration is valid."""
        config = RocketConfig()
        
        assert config.mass > 0
        assert config.radius > 0
        assert config.length > 0
        assert config.thrust_mean > 0
        assert config.burn_time > 0
        assert config.max_gimbal_angle > 0
    
    def test_config_modification(self):
        """Test configuration can be modified."""
        config = RocketConfig()
        
        original_mass = config.mass
        config.mass = 2.0
        
        assert config.mass != original_mass
        assert config.mass == 2.0


class TestEnvironmentFactories:
    """Test environment factory functions."""
    
    def test_make_training_env(self):
        """Test training environment factory."""
        env = make_training_env()
        assert env is not None
        assert hasattr(env, 'domain_randomization')
        env.close()
    
    def test_make_evaluation_env(self):
        """Test evaluation environment factory."""
        env = make_evaluation_env()
        assert env is not None
        env.close()
    
    def test_factory_with_kwargs(self):
        """Test factory functions with custom arguments."""
        env = make_training_env(
            domain_randomization=False,
            sensor_noise=False,
            max_episode_steps=500
        )
        
        assert env.max_episode_steps == 500
        env.close()


if __name__ == "__main__":
    pytest.main([__file__])
