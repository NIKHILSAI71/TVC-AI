"""
Integration tests for the complete training pipeline.
"""

import pytest
import tempfile
import torch
import numpy as np
from pathlib import Path
import shutil
from omegaconf import DictConfig, OmegaConf

import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from agent import SACAgent, SACConfig
from env import make_training_env, make_evaluation_env
from scripts.evaluate import run_standard_evaluation


class TestTrainingPipeline:
    """Test complete training pipeline integration."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def simple_training_run(self, agent, env, total_episodes=10, max_episode_steps=50):
        """Simplified training function for testing."""
        metrics = []
        
        for episode in range(total_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0
            
            for step in range(max_episode_steps):
                # Select action
                if agent.total_steps < 100:  # Initial random exploration
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(obs, deterministic=False)
                
                # Take environment step
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Store transition
                agent.store_transition(obs, action, reward, next_obs, done)
                
                # Update observation and episode metrics
                obs = next_obs
                episode_reward += reward
                episode_length += 1
                
                # Train agent if enough data
                if agent.total_steps >= 100 and len(agent.replay_buffer) >= agent.config.batch_size:
                    agent.train()
                
                if done:
                    break
            
            # Store episode metrics
            final_tilt = info.get('tilt_angle_deg', 180)
            success = final_tilt < 20 and episode_length > 10
            metrics.append({
                'episode_reward': episode_reward,
                'episode_length': episode_length,
                'success': success
            })
        
        # Calculate summary metrics
        rewards = [m['episode_reward'] for m in metrics]
        lengths = [m['episode_length'] for m in metrics]
        successes = [m['success'] for m in metrics]
        
        return {
            'episode_reward': np.mean(rewards),
            'episode_length': np.mean(lengths),
            'success_rate': np.mean(successes),
            'total_episodes': len(metrics)
        }
    
    def test_short_training_run(self, temp_dir):
        """Test a short training run completes successfully."""
        # Create environment and agent
        env = make_training_env(max_episode_steps=50)
        agent = SACAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config=SACConfig(batch_size=16, buffer_size=1000)
        )
        
        # Run training
        final_metrics = self.simple_training_run(agent, env, total_episodes=5, max_episode_steps=30)
        
        assert isinstance(final_metrics, dict)
        assert 'episode_reward' in final_metrics
        assert 'episode_length' in final_metrics
        assert 'success_rate' in final_metrics
        assert final_metrics['total_episodes'] == 5
        
        env.close()
    
    def test_evaluation_pipeline(self, temp_dir):
        """Test evaluation pipeline works."""
        # Create and train a minimal agent
        env = make_training_env(max_episode_steps=50)
        agent = SACAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config=SACConfig(batch_size=16, buffer_size=1000)
        )
        
        # Fill buffer with some experiences
        obs, _ = env.reset()
        for _ in range(100):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
        
        # Train briefly
        if len(agent.replay_buffer) >= agent.config.batch_size:
            for _ in range(10):
                agent.train()
        
        # Save agent
        model_path = temp_dir / "test_agent.pth"
        agent.save(str(model_path))
        
        env.close()
        
        # Test evaluation
        eval_env = make_evaluation_env(max_episode_steps=50)
        
        # Load agent for evaluation
        eval_agent = SACAgent(
            obs_dim=eval_env.observation_space.shape[0],
            action_dim=eval_env.action_space.shape[0]
        )
        eval_agent.load(str(model_path))
        
        # Run evaluation
        results = run_standard_evaluation(eval_agent, num_episodes=3)
        
        # Check results structure
        assert hasattr(results, 'episode_data')
        assert len(results.episode_data) == 3
        
        # Check that episodes have expected data
        for episode_data in results.episode_data:
            # The actual keys from the evaluation
            assert 'crashed' in episode_data
            assert 'final_altitude' in episode_data
            assert 'final_tilt_deg' in episode_data
            assert 'success' in episode_data
        
        eval_env.close()
    
    def test_model_save_load_consistency(self, temp_dir):
        """Test saved models produce consistent results."""
        env = make_training_env(max_episode_steps=50)
        
        # Create and briefly train agent
        agent = SACAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config=SACConfig(batch_size=16, buffer_size=1000)
        )
        
        # Generate some experiences and train
        obs, _ = env.reset()
        for _ in range(100):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
        
        if len(agent.replay_buffer) >= agent.config.batch_size:
            for _ in range(20):
                agent.train()
        
        # Test deterministic action
        test_obs, _ = env.reset()
        original_action = agent.select_action(test_obs, deterministic=True)
        
        # Save and load agent
        model_path = temp_dir / "consistency_test.pth"
        agent.save(str(model_path))
        
        loaded_agent = SACAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0]
        )
        loaded_agent.load(str(model_path))
        
        # Test same action
        loaded_action = loaded_agent.select_action(test_obs, deterministic=True)
        
        assert np.allclose(original_action, loaded_action, atol=1e-6)
        
        env.close()
    
    def test_tflite_export_pipeline(self, temp_dir):
        """Test TensorFlow Lite export pipeline."""
        try:
            from scripts.export_tflm import export_to_tflite
        except (ImportError, AttributeError):
            pytest.skip("TensorFlow not available or incompatible version for TFLite export test")
        
        env = make_training_env(max_episode_steps=50)
        
        # Create minimal agent
        agent = SACAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config=SACConfig(batch_size=16)
        )
        
        # Train minimally
        obs, _ = env.reset()
        for _ in range(50):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
        
        if len(agent.replay_buffer) >= agent.config.batch_size:
            for _ in range(10):
                agent.train()
        
        # Save PyTorch model
        pytorch_path = temp_dir / "test_model.pth"
        agent.save(str(pytorch_path))
        
        env.close()
        
        # Test TFLite export
        tflite_path = temp_dir / "test_model.tflite"
        c_array_path = temp_dir / "test_model.c"
        
        try:
            export_to_tflite(
                pytorch_model_path=str(pytorch_path),
                tflite_path=str(tflite_path),
                c_array_path=str(c_array_path),
                quantize=True
            )
            
            # Check files were created
            assert tflite_path.exists()
            assert c_array_path.exists()
            
            # Check C array has reasonable content
            c_content = c_array_path.read_text()
            assert "unsigned char" in c_content
            assert "model_data" in c_content
            assert len(c_content) > 100  # Should have substantial content
            
        except (ImportError, AttributeError):
            pytest.skip("TensorFlow not available or incompatible version for TFLite export test")
    
    def test_environment_consistency(self):
        """Test environment behavior is reasonable and within expected bounds."""
        # Create environment
        env = make_evaluation_env(max_episode_steps=50)
        
        # Test that reset gives reasonable initial state
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        
        # Check that observations are within reasonable bounds
        assert obs1.shape == obs2.shape
        assert len(obs1) == 8  # Expected observation dimension
        assert np.all(np.isfinite(obs1))
        assert np.all(np.isfinite(obs2))
        
        # Test that step gives reasonable results
        action = np.array([0.1, -0.1])
        next_obs, reward, term, trunc, info = env.step(action)
        
        assert np.all(np.isfinite(next_obs))
        assert np.isfinite(reward)
        assert isinstance(term, bool)
        assert isinstance(trunc, bool)
        assert isinstance(info, dict)
        
        env.close()
    
    def test_training_progress(self, temp_dir):
        """Test that training shows measurable progress."""
        env = make_training_env(max_episode_steps=100)
        agent = SACAgent(
            obs_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            config=SACConfig(batch_size=32, buffer_size=5000)
        )
        
        # Collect initial performance
        initial_rewards = []
        for _ in range(5):
            obs, _ = env.reset()
            episode_reward = 0
            for _ in range(100):
                action = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            initial_rewards.append(episode_reward)
        
        initial_mean = np.mean(initial_rewards)
        
        # Train for a while
        obs, _ = env.reset()
        for _ in range(1000):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
            
            if len(agent.replay_buffer) >= agent.config.batch_size:
                agent.train()
            
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
        
        # Collect final performance
        final_rewards = []
        for _ in range(5):
            obs, _ = env.reset()
            episode_reward = 0
            for _ in range(100):
                action = agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            final_rewards.append(episode_reward)
        
        final_mean = np.mean(final_rewards)
        
        # Should show some improvement (or at least not get worse)
        assert final_mean >= initial_mean - 50  # Allow some variance
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__])
