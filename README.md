# TVC-AI: Deep Reinforcement Learning for Rocket Thrust Vector Control

![TVC-AI Banner](https://img.shields.io/badge/TVC--AI-Rocket%20Control-blue?style=for-the-badge&logo=rocket)

A comprehensive Deep Reinforcement Learning system for controlling model rocket attitude using Thrust Vector Control (TVC). This project implements a Soft Actor-Critic (SAC) agent trained in a realistic PyBullet physics simulation with domain randomization for robust sim-to-real transfer.

## 🚀 Features

- **SAC Algorithm**: State-of-the-art continuous control with automatic entropy tuning
- **Realistic Physics**: 6-DOF rocket dynamics simulation using PyBullet
- **Domain Randomization**: Robust training with mass, thrust, and noise variations
- **Microcontroller Deployment**: TensorFlow Lite quantization for embedded systems
- **Comprehensive Training**: Full pipeline with monitoring, evaluation, and export tools
- **Real-time Capable**: Sub-millisecond inference for real-time control

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
├── scripts/            # High-level scripts for training, evaluation, etc.
│   ├── train.py
│   ├── evaluate.py
│   └── export_tflm.py
├── config/             # Configuration files (Hydra-based)
├── tests/              # Test suite and benchmarks
├── utils/              # Utility functions and helpers
├── verify_installation.py
├── setup.py           # Automated setup script
├── requirements.txt   # Full dependencies
├── requirements-minimal.txt  # Core dependencies only
├── requirements-dev.txt      # Development dependencies
├── .gitignore
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.8-3.11** (3.10.11 recommended)
- **8GB+ RAM** (16GB recommended for training)
- **CUDA-capable GPU** (optional but recommended for faster training)
- **~2-5GB disk space** (depending on installation type)

### Quick Installation

**Option 1: Automated Setup (Recommended)**
```bash
# Clone the repository
git clone <repository-url>
cd TVC-AI

# Run automated setup
python setup.py
```

**Option 2: Manual Installation**
```bash
# Clone the repository
git clone <repository-url>
cd TVC-AI

# Choose your installation type:

# Minimal (core functionality only - ~2GB)
pip install -r requirements-minimal.txt

# Full (all features - ~4GB, recommended)
pip install -r requirements.txt

# Development (includes testing tools - ~5GB)
pip install -r requirements.txt -r requirements-dev.txt

# Verify installation
python verify_installation.py
```

### First Steps

1. **Verify Installation**:
   ```bash
   python verify_installation.py
   ```

2. **Train Your First Model**:
   ```bash
   python scripts/train.py
   ```

3. **Monitor Training**:
   ```bash
   tensorboard --logdir logs/
   ```

4. **Evaluate Trained Model**:
   ```bash
   python scripts/evaluate.py --model_path models/best_model.pth
   ```

### Troubleshooting Installation

**Common Issues**:

- **PyBullet installation fails**: Install Visual C++ redistributables on Windows
- **CUDA out of memory**: Use CPU training: `python scripts/train.py device=cpu`
- **Package conflicts**: Use a virtual environment:
  ```bash
  python -m venv tvc_env
  source tvc_env/bin/activate  # On Windows: tvc_env\Scripts\activate
  pip install -r requirements.txt
  ```

For detailed usage instructions, see [USAGE.md](USAGE.md).

