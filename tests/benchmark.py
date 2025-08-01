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
Performance benchmarks for the TVC-AI system.
"""

import time
import numpy as np
import torch
from pathlib import Path
import tempfile
import statistics

import sys
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from agent import SACAgent
from env import make_training_env, make_evaluation_env


class PerformanceBenchmarks:
    """Performance benchmarking suite."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_environment_speed(self, num_steps=1000):
        """Benchmark environment step speed."""
        env = make_training_env(max_episode_steps=1000)
        
        obs, _ = env.reset()
        actions = [env.action_space.sample() for _ in range(num_steps)]
        
        start_time = time.time()
        
        for i in range(num_steps):
            obs, reward, terminated, truncated, _ = env.step(actions[i])
            if terminated or truncated:
                obs, _ = env.reset()
        
        end_time = time.time()
        
        steps_per_second = num_steps / (end_time - start_time)
        env.close()
        
        self.results['env_steps_per_second'] = steps_per_second
        return steps_per_second
    
    def benchmark_agent_inference(self, num_inferences=1000):
        """Benchmark agent inference speed."""
        env = make_training_env()
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = SACAgent(obs_dim, action_dim)
        
        obs, _ = env.reset()
        observations = [obs for _ in range(num_inferences)]
        
        # Warm up
        for _ in range(10):
            agent.select_action(obs, deterministic=True)
        
        # Benchmark deterministic inference
        start_time = time.time()
        for observation in observations:
            agent.select_action(observation, deterministic=True)
        end_time = time.time()
        
        det_inferences_per_second = num_inferences / (end_time - start_time)
        
        # Benchmark stochastic inference
        start_time = time.time()
        for observation in observations:
            agent.select_action(observation, deterministic=False)
        end_time = time.time()
        
        stoch_inferences_per_second = num_inferences / (end_time - start_time)
        
        env.close()
        
        self.results['agent_det_inferences_per_second'] = det_inferences_per_second
        self.results['agent_stoch_inferences_per_second'] = stoch_inferences_per_second
        
        return det_inferences_per_second, stoch_inferences_per_second
    
    def benchmark_training_step(self, num_steps=100):
        """Benchmark training step speed."""
        env = make_training_env()
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = SACAgent(obs_dim, action_dim)
        
        # Fill replay buffer
        obs, _ = env.reset()
        for _ in range(1000):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
        
        # Warm up
        for _ in range(10):
            agent.train()
        
        # Benchmark training steps
        start_time = time.time()
        for _ in range(num_steps):
            agent.train()
        end_time = time.time()
        
        training_steps_per_second = num_steps / (end_time - start_time)
        
        env.close()
        
        self.results['training_steps_per_second'] = training_steps_per_second
        return training_steps_per_second
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage."""
        env = make_training_env()
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = SACAgent(obs_dim, action_dim)
        
        # Estimate memory usage
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Fill buffer
        obs, _ = env.reset()
        for _ in range(10000):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
            obs = next_obs
            if terminated or truncated:
                obs, _ = env.reset()
        
        # Memory after filling buffer
        buffer_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Train for a while
        for _ in range(100):
            agent.train()
        
        # Memory after training
        training_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        env.close()
        
        self.results['baseline_memory_mb'] = baseline_memory
        self.results['buffer_memory_mb'] = buffer_memory
        self.results['training_memory_mb'] = training_memory
        self.results['buffer_overhead_mb'] = buffer_memory - baseline_memory
        self.results['training_overhead_mb'] = training_memory - buffer_memory
        
        return {
            'baseline': baseline_memory,
            'buffer': buffer_memory,
            'training': training_memory
        }
    
    def benchmark_convergence_speed(self, max_episodes=100):
        """Benchmark convergence speed on simple task."""
        env = make_training_env(max_episode_steps=200)
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        agent = SACAgent(obs_dim, action_dim)
        
        episode_rewards = []
        convergence_episode = None
        
        obs, _ = env.reset()
        episode_reward = 0
        episode_count = 0
        
        for step in range(max_episodes * 200):
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            episode_reward += reward
            agent.store_transition(obs, action, reward, next_obs, terminated or truncated)
            
            if len(agent.replay_buffer) > agent.config.batch_size:
                agent.train()
            
            obs = next_obs
            
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_count += 1
                
                # Check for convergence (stable performance over last 10 episodes)
                if len(episode_rewards) >= 20:
                    recent_mean = np.mean(episode_rewards[-10:])
                    older_mean = np.mean(episode_rewards[-20:-10])
                    
                    if recent_mean > older_mean and recent_mean > -50:  # Reasonable threshold
                        if convergence_episode is None:
                            convergence_episode = episode_count
                
                obs, _ = env.reset()
                episode_reward = 0
                
                if episode_count >= max_episodes:
                    break
        
        env.close()
        
        self.results['convergence_episode'] = convergence_episode
        self.results['final_reward_mean'] = np.mean(episode_rewards[-10:]) if episode_rewards else 0
        self.results['final_reward_std'] = np.std(episode_rewards[-10:]) if episode_rewards else 0
        
        return convergence_episode, episode_rewards
    
    def benchmark_device_performance(self):
        """Benchmark performance on different devices."""
        env = make_training_env()
        
        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        
        results = {}
        
        # CPU performance
        agent_cpu = SACAgent(obs_dim, action_dim)
        
        obs, _ = env.reset()
        
        # CPU inference speed
        start_time = time.time()
        for _ in range(100):
            agent_cpu.select_action(obs, deterministic=True)
        cpu_time = time.time() - start_time
        results['cpu_inference_time'] = cpu_time
        
        # GPU performance (if available)
        if torch.cuda.is_available():
            agent_gpu = SACAgent(obs_dim, action_dim)
            
            # GPU inference speed
            start_time = time.time()
            for _ in range(100):
                agent_gpu.select_action(obs, deterministic=True)
            torch.cuda.synchronize()  # Wait for GPU operations
            gpu_time = time.time() - start_time
            results['gpu_inference_time'] = gpu_time
            results['gpu_speedup'] = cpu_time / gpu_time
        
        env.close()
        
        self.results.update(results)
        return results
    
    def run_all_benchmarks(self):
        """Run all benchmarks and return results."""
        print("Running performance benchmarks...")
        
        print("1. Environment speed...")
        env_speed = self.benchmark_environment_speed()
        print(f"   Environment: {env_speed:.1f} steps/sec")
        
        print("2. Agent inference speed...")
        det_speed, stoch_speed = self.benchmark_agent_inference()
        print(f"   Deterministic: {det_speed:.1f} inferences/sec")
        print(f"   Stochastic: {stoch_speed:.1f} inferences/sec")
        
        print("3. Training speed...")
        train_speed = self.benchmark_training_step()
        print(f"   Training: {train_speed:.1f} steps/sec")
        
        print("4. Memory usage...")
        memory_usage = self.benchmark_memory_usage()
        print(f"   Baseline: {memory_usage['baseline']:.1f} MB")
        print(f"   With buffer: {memory_usage['buffer']:.1f} MB")
        print(f"   During training: {memory_usage['training']:.1f} MB")
        
        print("5. Device performance...")
        device_perf = self.benchmark_device_performance()
        print(f"   CPU inference time: {device_perf['cpu_inference_time']:.4f} sec")
        if 'gpu_speedup' in device_perf:
            print(f"   GPU speedup: {device_perf['gpu_speedup']:.2f}x")
        
        print("6. Convergence speed...")
        conv_episode, rewards = self.benchmark_convergence_speed()
        if conv_episode:
            print(f"   Convergence at episode: {conv_episode}")
        else:
            print("   No convergence detected in test period")
        
        return self.results
    
    def save_results(self, filepath):
        """Save benchmark results to file."""
        import json
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*50)
        
        print(f"Environment Performance:")
        print(f"  Steps per second: {self.results.get('env_steps_per_second', 'N/A'):.1f}")
        
        print(f"\nAgent Performance:")
        print(f"  Deterministic inference: {self.results.get('agent_det_inferences_per_second', 'N/A'):.1f} /sec")
        print(f"  Stochastic inference: {self.results.get('agent_stoch_inferences_per_second', 'N/A'):.1f} /sec")
        print(f"  Training steps: {self.results.get('training_steps_per_second', 'N/A'):.1f} /sec")
        
        print(f"\nMemory Usage:")
        print(f"  Baseline: {self.results.get('baseline_memory_mb', 'N/A'):.1f} MB")
        print(f"  Buffer overhead: {self.results.get('buffer_overhead_mb', 'N/A'):.1f} MB")
        print(f"  Training overhead: {self.results.get('training_overhead_mb', 'N/A'):.1f} MB")
        
        if 'convergence_episode' in self.results:
            conv = self.results['convergence_episode']
            print(f"\nConvergence:")
            print(f"  Episodes to convergence: {conv if conv else 'Not achieved'}")
            print(f"  Final performance: {self.results.get('final_reward_mean', 'N/A'):.1f} ± {self.results.get('final_reward_std', 'N/A'):.1f}")


if __name__ == "__main__":
    benchmarks = PerformanceBenchmarks()
    results = benchmarks.run_all_benchmarks()
    benchmarks.print_summary()
    
    # Save results
    benchmarks.save_results("benchmark_results.json")
    print(f"\nResults saved to benchmark_results.json")
