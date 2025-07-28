# Deep Reinforcement Learning for Model Rocket TVC Control

> An AI-powered flight controller that learns to stabilize a model rocket in a realistic physics simulation, designed for deployment on real-world microcontrollers.

<!-- Placeholder for a cool GIF of the rocket stabilizing -->
<!-- ![Rocket Simulation GIF](assets/simulation_demo.gif) -->

This project implements a Deep Reinforcement Learning (DRL) agent to control a model rocket's attitude using Thrust Vector Control (TVC) for stable vertical ascent. The agent learns a robust control policy through interaction with a high-fidelity, domain-randomized physics simulation (digital twin).

## Table of Contents

- Key Features
- Technical Deep Dive
  - Simulation Environment
  - DRL Agent: Soft Actor-Critic
  - State, Action, and Reward
- Project Structure
- Getting Started
  - Prerequisites
  - Installation & Setup
  - 1. Training the Agent
  - 2. Evaluating the Policy
  - 3. Deploying to a Microcontroller
- Acknowledgements & References

## Key Features

- **High-Fidelity 3D Physics**: A digital twin built with **PyBullet** simulates rigid body dynamics, gravity, variable mass, and TVC motor thrust. **RocketPy** can be integrated for accurate aerodynamic force modeling.
- **Sim-to-Real via Domain Randomization**: To ensure the policy is robust, the simulation randomizes key physical parameters on every run: rocket mass, center of gravity (CG), motor thrust curves, sensor noise (IMU), and actuator delay/response.
- **Standard Gymnasium Interface**: The environment adheres to the modern **Gymnasium** API, making it compatible with a wide range of DRL libraries and algorithms.
- **State-of-the-Art DRL Agent**: Employs **Soft Actor-Critic (SAC)**, an off-policy algorithm known for its sample efficiency and stability, making it ideal for complex physics tasks.
- **Shaped Reward Function**: A carefully designed reward function guides the agent to maintain vertical stability, minimize angular velocity (reduce oscillations), and use control inputs efficiently.
- **End-to-End Workflow**: Provides a complete pipeline from training and evaluation to model quantization and deployment.
- **MCU-Ready Deployment**: The trained policy is optimized using **Post-Training Quantization (8-bit)** and converted into a C-array via **TensorFlow Lite for Microcontrollers (TFLM)** for fast, on-device inference.

## Technical Deep Dive

### Simulation Environment (Digital Twin)
The core of this project is the simulated environment. A robust policy can only be learned if the simulation it's trained in is a close-enough approximation of reality.
- **Physics Engine**: PyBullet is used for its fast and stable rigid body dynamics simulation.
- **State Representation**: The rocket's state is observed at each time step. The state vector includes:
  - **Attitude (Quaternion)**: `[qx, qy, qz, qw]` - A 4D representation to avoid gimbal lock.
  - **Angular Velocity**: `[ω_x, ω_y, ω_z]` - Critical for damping rotational motion.
- **Imperfections**: Real-world rockets are not perfect. Domain randomization introduces controlled chaos, forcing the agent to learn a policy that can handle variations in hardware and conditions, which is the key to bridging the "sim-to-real" gap.

### DRL Agent: Soft Actor-Critic
We use Soft Actor-Critic (SAC) for its balance of exploration and exploitation.
- **Entropy Maximization**: SAC's objective is to maximize not only the cumulative reward but also the entropy of its policy. This encourages broader exploration, preventing the agent from settling into a suboptimal local minimum and making it more resilient to perturbations.
- **Actor-Critic Architecture**:
  - **Actor (Policy)**: An MLP that takes the rocket's state as input and outputs the parameters (mean and standard deviation) of a Gaussian distribution for each action (gimbal pitch, gimbal yaw). The actual action is then sampled from this distribution.
  - **Critic (Value Function)**: One or more MLPs that learn to estimate the expected future reward from a given state-action pair. This estimate is used to "criticize" and improve the actor's policy.

### State, Action, and Reward
The interaction between the agent and environment is defined by these three components:

- **State Space (Observations)**: `s_t = [qx, qy, qz, qw, ω_x, ω_y, ω_z]`
- **Action Space (Control Inputs)**: `a_t = [gimbal_pitch_angle, gimbal_yaw_angle]` (continuous values, typically scaled to `[-1, 1]`).
- **Reward Function (`r_t`)**: The agent's behavior is shaped by the reward `r_t` it receives at each step. Our function is a sum of several components:
  - **Attitude Reward**: `R_attitude = exp(-k_angle * θ_total^2)` where `θ_total` is the total angle off vertical. This provides a strong, smooth incentive to stay upright.
  - **Angular Velocity Penalty**: `R_ang_vel = -k_vel * ||ω||^2`. This penalizes rapid spinning and encourages smooth, damped control.
  - **Control Effort Penalty**: `R_action = -k_action * ||a||^2`. This penalizes large, aggressive gimbal movements, promoting efficiency.
  - **Termination Penalty**: A large negative reward (e.g., -100) is given if the rocket tilts beyond a failure threshold (e.g., 20°), immediately ending the episode.

## Project Structure
```
tvc-ai/
├── agent/              # DRL agent implementation (SAC algorithm, networks)
├── env/                # Rocket physics simulation (Gymnasium environment)
├── models/             # Saved model checkpoints and exported TFLM models
├── notebooks/          # Jupyter notebooks for analysis and visualization
├── scripts/            # High-level scripts for training, evaluation, etc.
│   ├── train.py
│   ├── evaluate.py
│   └── export_tflm.py
├── .gitignore
├── README.md
└── requirements.txt
```

