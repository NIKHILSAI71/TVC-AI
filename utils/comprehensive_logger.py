#!/usr/bin/env python3
"""
Comprehensive Device-Aware Logging System
Provides detailed logging for training progress across CPU, GPU, and TPU devices.
"""

import logging
import time
import psutil
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class DeviceAwareLogger:
    """Enhanced logger that tracks performance across different devices"""
    
    def __init__(self, name: str, log_dir: str = "./logs", device_manager=None):
        self.logger = logging.getLogger(name)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.device_manager = device_manager
        self.use_emojis = True  # Will be set to False if encoding issues
        
        # Setup file and console logging
        self._setup_logging()
        
        # Performance tracking
        self.start_time = time.time()
        self.device_stats = {}
        self.training_metrics = {
            'episodes': [],
            'rewards': [],
            'success_rates': [],
            'device_utilization': [],
            'memory_usage': [],
            'step_times': [],
            'algorithm_performance': {}
        }
        
        # Log device information
        self._log_device_info()
    
    def _setup_logging(self):
        """Setup comprehensive logging with file and console handlers"""
        # Setup UTF-8 encoding for Windows console
        import sys
        if sys.platform.startswith('win'):
            try:
                sys.stdout.reconfigure(encoding='utf-8')
                sys.stderr.reconfigure(encoding='utf-8')
            except (AttributeError, OSError):
                # Fallback for older Python versions or restricted environments
                import codecs
                try:
                    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
                    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
                except (AttributeError, OSError):
                    # Final fallback - disable emojis
                    self.use_emojis = False
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(device)s] - %(message)s',
            defaults={'device': 'UNKNOWN'}
        )
        
        # Console handler with colors
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(
            self.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Performance metrics handler
        perf_handler = logging.FileHandler(
            self.log_dir / f'performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.setLevel(logging.DEBUG)
        
        # Separate performance logger
        self.perf_logger = logging.getLogger(f"{self.logger.name}.performance")
        self.perf_logger.addHandler(perf_handler)
        self.perf_logger.setLevel(logging.INFO)
    
    def _log_device_info(self):
        """Log comprehensive device information"""
        tool_emoji = "🔧" if self.use_emojis else "[DEVICE]"
        gpu_emoji = "🚀" if self.use_emojis else "[GPU]"
        tpu_emoji = "⚡" if self.use_emojis else "[TPU]"
        target_emoji = "🎯" if self.use_emojis else "[ACTIVE]"
        
        self.info(f"{tool_emoji} Device Information:")
        
        # CPU Information
        cpu_count = psutil.cpu_count(logical=False)
        cpu_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        self.info(f"  CPU: {cpu_count} cores ({cpu_logical} logical), {cpu_freq.max:.0f} MHz max")
        
        # Memory Information
        memory = psutil.virtual_memory()
        self.info(f"  RAM: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
        
        # GPU Information
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            self.info(f"  {gpu_emoji} CUDA GPUs: {gpu_count}")
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                self.info(f"    GPU {i}: {props.name}, {props.total_memory / (1024**3):.1f} GB")
        else:
            self.info(f"  {gpu_emoji} CUDA: Not available")
        
        # TPU Information
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            self.info(f"  {tpu_emoji} TPU/XLA: Available, device: {xm.xla_device()}")
        except ImportError:
            self.info(f"  {tpu_emoji} TPU/XLA: Not available")
        
        # Current device
        if self.device_manager:
            self.info(f"  {target_emoji} Active Device: {self.device_manager.device} ({self.device_manager.device_type})")
    
    def log_training_start(self, config: Dict):
        """Log training start with configuration"""
        rocket_emoji = "🚀" if self.use_emojis else "[START]"
        self.info(f"{rocket_emoji} Training Started")
        self.info(f"  Configuration: {json.dumps(config, indent=2, default=str)}")
        self.start_time = time.time()
    
    def log_episode(self, episode: int, metrics: Dict[str, Any]):
        """Log episode results with device performance"""
        # Update training metrics
        self.training_metrics['episodes'].append(episode)
        self.training_metrics['rewards'].append(metrics.get('reward', 0))
        self.training_metrics['success_rates'].append(metrics.get('success', False))
        
        # Get current device stats
        device_stats = self._get_device_stats()
        self.training_metrics['device_utilization'].append(device_stats)
        
        # Log episode info
        reward = metrics.get('reward', 0)
        success = metrics.get('success', False)
        length = metrics.get('length', 0)
        algorithm = metrics.get('algorithm_used', 'unknown')
        
        success_indicator = "PASS" if not self.use_emojis else ("✅" if success else "❌")
        if not self.use_emojis:
            success_indicator = "PASS" if success else "FAIL"
        
        self.info(f"Episode {episode:6d} | Reward: {reward:8.2f} | Success: {success_indicator} | "
                 f"Length: {length:4d} | Algorithm: {algorithm:>6s} | "
                 f"Device: {self._format_device_stats(device_stats)}")
        
        # Update algorithm performance
        if algorithm not in self.training_metrics['algorithm_performance']:
            self.training_metrics['algorithm_performance'][algorithm] = []
        self.training_metrics['algorithm_performance'][algorithm].append(reward)
        
        # Log to performance file
        self.perf_logger.info(f"EPISODE,{episode},{reward},{success},{length},{algorithm},"
                             f"{device_stats.get('cpu_percent', 0)},{device_stats.get('memory_percent', 0)},"
                             f"{device_stats.get('gpu_utilization', 0)},{device_stats.get('gpu_memory', 0)}")
    
    def log_evaluation(self, step: int, eval_results: Dict[str, float]):
        """Log evaluation results"""
        success_rate = eval_results.get('success_rate', 0)
        reward_mean = eval_results.get('reward_mean', 0)
        reward_std = eval_results.get('reward_std', 0)
        
        self.info(f">>> EVAL Step {step:8d} | Success Rate: {success_rate:6.1%} | "
                 f"Reward: {reward_mean:8.2f} ± {reward_std:6.2f}")
        
        # Log detailed metrics
        for metric, value in eval_results.items():
            if metric not in ['success_rate', 'reward_mean', 'reward_std']:
                self.debug(f"  {metric}: {value}")
    
    def log_device_performance(self, step: int):
        """Log detailed device performance metrics"""
        stats = self._get_device_stats()
        
        self.debug(f"Device Performance at step {step}:")
        self.debug(f"  CPU: {stats.get('cpu_percent', 0):.1f}%")
        self.debug(f"  Memory: {stats.get('memory_percent', 0):.1f}% ({stats.get('memory_used', 0):.1f} GB)")
        
        if stats.get('gpu_available', False):
            self.debug(f"  GPU Util: {stats.get('gpu_utilization', 0):.1f}%")
            self.debug(f"  GPU Memory: {stats.get('gpu_memory', 0):.1f}% ({stats.get('gpu_memory_used', 0):.1f} GB)")
    
    def log_algorithm_switch(self, old_algorithm: str, new_algorithm: str, reason: str = ""):
        """Log algorithm switching"""
        switch_emoji = "🔄" if self.use_emojis else "[SWITCH]"
        self.info(f"{switch_emoji} Algorithm Switch: {old_algorithm} → {new_algorithm}")
        if reason:
            self.info(f"  Reason: {reason}")
    
    def log_safety_violation(self, violation_type: str, details: Dict):
        """Log safety violations"""
        warning_emoji = "⚠️" if self.use_emojis else "[WARNING]"
        self.warning(f"{warning_emoji} Safety Violation: {violation_type}")
        for key, value in details.items():
            self.warning(f"  {key}: {value}")
    
    def log_reward_hacking_detection(self, hacking_score: float, indicators: Dict):
        """Log reward hacking detection"""
        if hacking_score > 0.7:
            alarm_emoji = "🚨" if self.use_emojis else "[ALERT]"
            self.warning(f"{alarm_emoji} Potential Reward Hacking Detected! Score: {hacking_score:.3f}")
            for indicator, value in indicators.items():
                self.warning(f"  {indicator}: {value}")
        else:
            self.debug(f"Reward hacking check: {hacking_score:.3f} (safe)")
    
    def log_training_summary(self):
        """Log comprehensive training summary"""
        total_time = time.time() - self.start_time
        total_episodes = len(self.training_metrics['episodes'])
        
        if total_episodes == 0:
            self.info("No episodes completed.")
            return
        
        avg_reward = np.mean(self.training_metrics['rewards'])
        success_rate = np.mean(self.training_metrics['success_rates'])
        
        finish_emoji = "🏁" if self.use_emojis else "[COMPLETE]"
        
        self.info("="*80)
        self.info(f"{finish_emoji} TRAINING SUMMARY")
        self.info("="*80)
        self.info(f"  Total Time: {total_time/3600:.2f} hours")
        self.info(f"  Total Episodes: {total_episodes}")
        self.info(f"  Episodes/hour: {total_episodes/(total_time/3600):.1f}")
        self.info(f"  Average Reward: {avg_reward:.2f}")
        self.info(f"  Success Rate: {success_rate:.1%}")
        
        # Algorithm performance
        if self.training_metrics['algorithm_performance']:
            self.info("  Algorithm Performance:")
            for alg, rewards in self.training_metrics['algorithm_performance'].items():
                avg_alg_reward = np.mean(rewards)
                self.info(f"    {alg}: {avg_alg_reward:.2f} (over {len(rewards)} episodes)")
        
        # Device utilization summary
        if self.training_metrics['device_utilization']:
            avg_cpu = np.mean([stats.get('cpu_percent', 0) for stats in self.training_metrics['device_utilization']])
            avg_memory = np.mean([stats.get('memory_percent', 0) for stats in self.training_metrics['device_utilization']])
            self.info(f"  Average CPU Utilization: {avg_cpu:.1f}%")
            self.info(f"  Average Memory Usage: {avg_memory:.1f}%")
            
            if any(stats.get('gpu_available', False) for stats in self.training_metrics['device_utilization']):
                avg_gpu = np.mean([stats.get('gpu_utilization', 0) for stats in self.training_metrics['device_utilization'] if stats.get('gpu_available', False)])
                avg_gpu_mem = np.mean([stats.get('gpu_memory', 0) for stats in self.training_metrics['device_utilization'] if stats.get('gpu_available', False)])
                self.info(f"  Average GPU Utilization: {avg_gpu:.1f}%")
                self.info(f"  Average GPU Memory: {avg_gpu_mem:.1f}%")
    
    def _get_device_stats(self) -> Dict[str, Any]:
        """Get current device performance statistics"""
        stats = {}
        
        # CPU and memory
        stats['cpu_percent'] = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        stats['memory_percent'] = memory.percent
        stats['memory_used'] = memory.used / (1024**3)  # GB
        
        # GPU stats if available
        if torch.cuda.is_available():
            stats['gpu_available'] = True
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                stats['gpu_utilization'] = gpu_util.gpu
                stats['gpu_memory'] = (memory_info.used / memory_info.total) * 100
                stats['gpu_memory_used'] = memory_info.used / (1024**3)  # GB
                pynvml.nvmlShutdown()
            except Exception:
                # Fallback to torch methods
                stats['gpu_utilization'] = 0  # Can't get utilization easily
                stats['gpu_memory'] = (torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()) * 100 if torch.cuda.max_memory_allocated() > 0 else 0
                stats['gpu_memory_used'] = torch.cuda.memory_allocated() / (1024**3)
        else:
            stats['gpu_available'] = False
            stats['gpu_utilization'] = 0
            stats['gpu_memory'] = 0
            stats['gpu_memory_used'] = 0
        
        return stats
    
    def _format_device_stats(self, stats: Dict) -> str:
        """Format device stats for logging"""
        cpu = stats.get('cpu_percent', 0)
        mem = stats.get('memory_percent', 0)
        
        if stats.get('gpu_available', False):
            gpu_util = stats.get('gpu_utilization', 0)
            gpu_mem = stats.get('gpu_memory', 0)
            return f"CPU: {cpu:4.1f}% | RAM: {mem:4.1f}% | GPU: {gpu_util:4.1f}% | VRAM: {gpu_mem:4.1f}%"
        else:
            return f"CPU: {cpu:4.1f}% | RAM: {mem:4.1f}%"
    
    def save_training_plots(self):
        """Save training progress plots"""
        if not self.training_metrics['episodes']:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Reward progression
        axes[0, 0].plot(self.training_metrics['episodes'], self.training_metrics['rewards'])
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Success rate (rolling average)
        if len(self.training_metrics['success_rates']) > 100:
            window = 100
            success_rolling = np.convolve(self.training_metrics['success_rates'], 
                                        np.ones(window)/window, mode='valid')
            axes[0, 1].plot(self.training_metrics['episodes'][window-1:], success_rolling)
        else:
            axes[0, 1].plot(self.training_metrics['episodes'], 
                           np.cumsum(self.training_metrics['success_rates']) / np.arange(1, len(self.training_metrics['success_rates'])+1))
        axes[0, 1].set_title('Success Rate (Rolling Average)')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].grid(True)
        
        # Device utilization
        if self.training_metrics['device_utilization']:
            cpu_util = [stats.get('cpu_percent', 0) for stats in self.training_metrics['device_utilization']]
            mem_util = [stats.get('memory_percent', 0) for stats in self.training_metrics['device_utilization']]
            
            axes[1, 0].plot(self.training_metrics['episodes'], cpu_util, label='CPU')
            axes[1, 0].plot(self.training_metrics['episodes'], mem_util, label='Memory')
            
            if any(stats.get('gpu_available', False) for stats in self.training_metrics['device_utilization']):
                gpu_util = [stats.get('gpu_utilization', 0) for stats in self.training_metrics['device_utilization']]
                axes[1, 0].plot(self.training_metrics['episodes'], gpu_util, label='GPU')
            
            axes[1, 0].set_title('Device Utilization')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('Utilization (%)')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Algorithm performance comparison
        if self.training_metrics['algorithm_performance']:
            alg_names = list(self.training_metrics['algorithm_performance'].keys())
            alg_rewards = [np.mean(rewards) for rewards in self.training_metrics['algorithm_performance'].values()]
            
            axes[1, 1].bar(alg_names, alg_rewards)
            axes[1, 1].set_title('Algorithm Performance Comparison')
            axes[1, 1].set_xlabel('Algorithm')
            axes[1, 1].set_ylabel('Average Reward')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / f'training_progress_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        chart_emoji = "📊" if self.use_emojis else "[CHARTS]"
        self.info(f"{chart_emoji} Training plots saved to {self.log_dir}")
    
    # Convenience methods
    def info(self, message: str):
        device_info = {'device': self.device_manager.device_type if self.device_manager else 'CPU'}
        self.logger.info(message, extra=device_info)
    
    def debug(self, message: str):
        device_info = {'device': self.device_manager.device_type if self.device_manager else 'CPU'}
        self.logger.debug(message, extra=device_info)
    
    def warning(self, message: str):
        device_info = {'device': self.device_manager.device_type if self.device_manager else 'CPU'}
        self.logger.warning(message, extra=device_info)
    
    def error(self, message: str):
        device_info = {'device': self.device_manager.device_type if self.device_manager else 'CPU'}
        self.logger.error(message, extra=device_info)
