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
Environment Registration Module

This module provides registration utilities for the Rocket TVC environments
with Gymnasium, enabling easy creation and configuration of environments
for different training scenarios.
"""

import gymnasium as gym
from gymnasium.envs.registration import register
from .rocket_tvc_env import RocketTVCEnv, RocketConfig

# Register the main TVC environment
register(
    id='RocketTVC-v0',
    entry_point='env.rocket_tvc_env:RocketTVCEnv',
    max_episode_steps=1000,
    kwargs={
        'domain_randomization': True,
        'sensor_noise': True,
        'use_rocketpy': False
    }
)

# Register a simplified version for faster training
register(
    id='RocketTVC-Simple-v0', 
    entry_point='env.rocket_tvc_env:RocketTVCEnv',
    max_episode_steps=500,
    kwargs={
        'domain_randomization': False,
        'sensor_noise': False,
        'use_rocketpy': False
    }
)

# Register a high-fidelity version with RocketPy
register(
    id='RocketTVC-HighFidelity-v0',
    entry_point='env.rocket_tvc_env:RocketTVCEnv', 
    max_episode_steps=1500,
    kwargs={
        'domain_randomization': True,
        'sensor_noise': True,
        'use_rocketpy': True
    }
)

# Pre-configured environments for different scenarios
def make_training_env(**kwargs):
    """Create environment optimized for training."""
    default_kwargs = {
        'domain_randomization': True,
        'sensor_noise': True,
        'max_episode_steps': 1000,
        'debug': False
    }
    default_kwargs.update(kwargs)
    return RocketTVCEnv(**default_kwargs)

def make_evaluation_env(**kwargs):
    """Create environment for evaluation with nominal parameters."""
    default_kwargs = {
        'domain_randomization': False,
        'sensor_noise': False,
        'max_episode_steps': 2000,
        'debug': True
    }
    default_kwargs.update(kwargs)
    return RocketTVCEnv(**default_kwargs)

def make_debug_env(**kwargs):
    """Create environment for debugging with visualization."""
    default_kwargs = {
        'render_mode': 'human',
        'domain_randomization': False,
        'sensor_noise': False,
        'max_episode_steps': 1000,
        'debug': True
    }
    default_kwargs.update(kwargs)
    return RocketTVCEnv(**default_kwargs)
