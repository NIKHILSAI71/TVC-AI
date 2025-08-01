#!/usr/bin/env python3
"""
Utility script for hyperparameter tuning using Optuna.

This script automates hyperparameter optimization for the SAC agent
using Bayesian optimization to find the best configuration.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any
import tempfile
import shutil

import optuna
import numpy as np
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from agent import SACAgent, SACConfig
from env import make_training_env, make_evaluation_env


def objective(trial: optuna.Trial) -> float:
    """
    Objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        Performance metric to optimize (higher is better)
    """
    # Suggest hyperparameters
    lr_actor = trial.suggest_float("lr_actor", 1e-5, 1e-2, log=True)
    lr_critic = trial.suggest_float("lr_critic", 1e-5, 1e-2, log=True)
    lr_alpha = trial.suggest_float("lr_alpha", 1e-5, 1e-2, log=True)
    
    gamma = trial.suggest_float("gamma", 0.9, 0.999)
    tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
    
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    
    # Create SAC configuration
    config = SACConfig(
        hidden_dims=[hidden_dim, hidden_dim],
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        lr_alpha=lr_alpha,
        gamma=gamma,
        tau=tau,
        batch_size=batch_size,
        buffer_size=500000,  # Smaller buffer for faster tuning
        learning_starts=1000
    )
    
    # Create environments
    train_env = make_training_env(
        domain_randomization=True,
        sensor_noise=True,
        max_episode_steps=500,  # Shorter episodes for faster tuning
        debug=False
    )
    
    eval_env = make_evaluation_env(
        max_episode_steps=1000,
        debug=False
    )
    
    # Get environment dimensions
    obs_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.shape[0]
    
    # Create agent
    agent = SACAgent(obs_dim, action_dim, config)
    
    # Training loop (reduced for tuning)
    max_steps = 50000  # Reduced training steps
    eval_freq = 10000
    
    obs, _ = train_env.reset()
    episode_reward = 0
    best_eval_reward = float('-inf')
    
    try:
        for step in range(max_steps):
            # Select action
            if agent.total_steps < config.learning_starts:
                action = train_env.action_space.sample()
            else:
                action = agent.select_action(obs, deterministic=False)
            
            # Environment step
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated
            
            # Store transition and train
            agent.store_transition(obs, action, reward, next_obs, done)
            
            if agent.total_steps >= config.learning_starts:
                agent.train()
            
            obs = next_obs
            episode_reward += reward
            
            if done:
                obs, _ = train_env.reset()
                episode_reward = 0
            
            # Evaluation
            if step % eval_freq == 0 and step > 0:
                eval_reward = evaluate_agent(agent, eval_env, num_episodes=5)
                best_eval_reward = max(best_eval_reward, eval_reward)
                
                # Report intermediate value to Optuna
                trial.report(eval_reward, step)
                
                # Pruning: stop trial if not promising
                if trial.should_prune():
                    raise optuna.TrialPruned()
    
    except Exception as e:
        logging.warning(f"Trial failed: {e}")
        return float('-inf')
    
    finally:
        train_env.close()
        eval_env.close()
    
    return best_eval_reward


def evaluate_agent(agent: SACAgent, env, num_episodes: int = 5) -> float:
    """Quick evaluation for hyperparameter tuning."""
    total_reward = 0
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        while True:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    return total_reward / num_episodes


def run_hyperparameter_tuning(n_trials: int = 100, study_name: str = "sac_tuning"):
    """
    Run hyperparameter tuning.
    
    Args:
        n_trials: Number of optimization trials
        study_name: Name of the study
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Optuna study
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        pruner=optuna.pruners.MedianPruner()
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Print results
    print("\\nHyperparameter Tuning Results:")
    print("=" * 50)
    print(f"Number of trials: {len(study.trials)}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value:.4f}")
    
    print("\\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save results
    study.trials_dataframe().to_csv("hyperparameter_tuning_results.csv")
    
    # Create visualization (if available)
    try:
        import optuna.visualization as vis
        import plotly
        
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html("optimization_history.html")
        
        # Parameter importance
        fig = vis.plot_param_importances(study)
        fig.write_html("parameter_importance.html")
        
        print("\\nVisualization files saved:")
        print("  - optimization_history.html")
        print("  - parameter_importance.html")
        
    except ImportError:
        print("\\nInstall plotly for visualization: pip install plotly")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for SAC agent")
    parser.add_argument("--n_trials", type=int, default=100,
                       help="Number of optimization trials")
    parser.add_argument("--study_name", type=str, default="sac_tuning",
                       help="Name of the Optuna study")
    
    args = parser.parse_args()
    
    run_hyperparameter_tuning(args.n_trials, args.study_name)
