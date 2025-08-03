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
Environment Registration Module - State-of-the-Art TVC

This module provides registration utilities for the Enhanced Rocket TVC environment
with Gymnasium, enabling easy creation and configuration of the state-of-the-art
environment with anti-reward-hacking features.
"""

import gymnasium as gym
from gymnasium.envs.registration import register
from .enhanced_rocket_tvc_env import EnhancedRocketTVCEnv, MissionPhase

# Register the state-of-the-art Enhanced TVC environment
register(
    id='EnhancedRocketTVC-v0',
    entry_point='env.enhanced_rocket_tvc_env:EnhancedRocketTVCEnv',
    max_episode_steps=1000,
    kwargs={
        'enable_hierarchical': True,
        'enable_curiosity': True,
        'enable_physics_informed': True,
        'debug': False
    }
)

# Register a deterministic version for evaluation
register(
    id='EnhancedRocketTVC-Eval-v0', 
    entry_point='env.enhanced_rocket_tvc_env:EnhancedRocketTVCEnv',
    max_episode_steps=1000,
    kwargs={
        'enable_hierarchical': False,
        'enable_curiosity': False,
        'enable_physics_informed': False,
        'debug': False
    }
)

# Register a debug version with visualization
register(
    id='EnhancedRocketTVC-Debug-v0',
    entry_point='env.enhanced_rocket_tvc_env:EnhancedRocketTVCEnv', 
    max_episode_steps=1000,
    kwargs={
        'enable_hierarchical': True,
        'enable_curiosity': True,
        'enable_physics_informed': True,
        'debug': True
    }
)

# Pre-configured environments for different scenarios
def make_training_env(config=None, **kwargs):
    """Create enhanced environment optimized for state-of-the-art training."""
    default_kwargs = {
        'max_episode_steps': 1000,
        'enable_hierarchical': True,
        'enable_curiosity': True,
        'enable_physics_informed': True,
        'debug': False
    }
    default_kwargs.update(kwargs)
    return EnhancedRocketTVCEnv(config=config, **default_kwargs)

def make_evaluation_env(config=None, **kwargs):
    """Create environment for evaluation with deterministic behavior."""
    default_kwargs = {
        'max_episode_steps': 1000,
        'enable_hierarchical': False,
        'enable_curiosity': False,
        'enable_physics_informed': False,
        'debug': False
    }
    default_kwargs.update(kwargs)
    return EnhancedRocketTVCEnv(config=config, **default_kwargs)

def make_debug_env(config=None, **kwargs):
    """Create environment for debugging with all features enabled."""
    default_kwargs = {
        'render_mode': 'human',
        'max_episode_steps': 1000,
        'enable_hierarchical': True,
        'enable_curiosity': True,
        'enable_physics_informed': True,
        'debug': True
    }
    default_kwargs.update(kwargs)
    return EnhancedRocketTVCEnv(config=config, **default_kwargs)

# Export main classes and functions
__all__ = [
    'EnhancedRocketTVCEnv',
    'MissionPhase',
    'make_training_env',
    'make_evaluation_env', 
    'make_debug_env'
]
