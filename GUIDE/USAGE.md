# TVC-AI System Usage Guide

## Quick Setup and Usage Examples

This guide provides step-by-step instructions for setting up and using the TVC-AI system.

## Prerequisites

Ensure you have:
- Python 3.8 or higher
- At least 8GB RAM
- CUDA-capable GPU (recommended but not required)

## Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd TVC-AI

# 2. Install dependencies (choose one option)

# Option A: Minimal installation (core functionality only)
pip install -r requirements-minimal.txt

# Option B: Full installation (all features)
pip install -r requirements.txt

# Option C: Development installation (includes testing and dev tools)
pip install -r requirements.txt -r requirements-dev.txt

# 3. (Optional) Install TensorFlow for deployment if not included above
pip install tensorflow>=2.10.0,<2.14.0
```

### Installation Options Explained

**Minimal Installation (`requirements-minimal.txt`)**:
- Core functionality only (training and basic evaluation)
- Fastest installation
- ~2GB download
- Includes: PyTorch, PyBullet, Gymnasium, basic utilities

**Full Installation (`requirements.txt`)**:
- All features including advanced visualization and monitoring
- Recommended for most users
- ~4GB download
- Includes: All minimal packages + TensorBoard, advanced plotting, TensorFlow

**Development Installation (`requirements-dev.txt`)**:
- Everything in full installation + development tools
- Required for contributing to the project
- ~5GB download
- Includes: Testing frameworks, code formatters, type checking

### Verify Installation

After installation, run the verification script:

```bash
# Verify your installation
python verify_installation.py
```

This will check:
- Python version compatibility
- All required packages and versions
- Basic functionality of core components
- Available optional features

If verification passes, you're ready to use TVC-AI!

## Basic Usage

### 1. Train Your First Model

```bash
# Start training with default settings
python scripts/train.py

# Monitor progress with TensorBoard
tensorboard --logdir logs/
```

This will:
- Create a SAC agent with default hyperparameters
- Train for 2000 episodes (~2-4 hours on modern hardware)
- Save the best model to `models/best_model.pth`
- Log training metrics to TensorBoard

### 2. Evaluate the Trained Model

```bash
# Basic evaluation
python scripts/evaluate.py --model_path models/best_model.pth
```

This will:
- Run 100 evaluation episodes
- Report success rate and performance metrics
- Generate trajectory plots (if requested)
- Save results to `evaluation_results/`

### 3. Export for Deployment

```bash
# Export to TensorFlow Lite with quantization
python scripts/export_tflm.py --model_path models/best_model.pth --quantize

# Generate C array for microcontrollers
python scripts/export_tflm.py --model_path models/best_model.pth --c-array --output rocket_model.c
```

This will:
- Convert PyTorch model to TensorFlow Lite
- Apply INT8 quantization for smaller size
- Generate C header file for embedded deployment

## Advanced Usage

### Custom Training Configuration

Create a custom config file:

```yaml
# config/my_config.yaml
agent:
  learning_rate: 1e-4
  batch_size: 512
  hidden_size: 512

env:
  max_episode_steps: 2000
  domain_randomization: true

training:
  total_episodes: 5000
  eval_interval: 200
```

Then train with:
```bash
python scripts/train.py --config-name my_config
```

### Hyperparameter Tuning

```bash
# Run automated hyperparameter optimization
python scripts/tune_hyperparameters.py --n_trials 50 --study_name tvc_study
```

### Multiple Environment Variants

Train on different rocket configurations:

```bash
# Train on lightweight rocket
python scripts/train.py env=lightweight_rocket

# Train on heavy payload rocket
python scripts/train.py env=heavy_rocket

# Train with aggressive domain randomization
python scripts/train.py env=aggressive_randomization
```

### Visualization and Analysis

```bash
# Generate training plots
python scripts/visualize.py --log-dir logs/experiment_1

# Compare multiple experiments
python scripts/visualize.py --log-dirs logs/exp1 logs/exp2 logs/exp3

# Generate HTML report
python scripts/visualize.py --log-dir logs/experiment_1 --generate-report
```

## Testing and Validation

### Run Test Suite

```bash
# Run all tests
python tests/run_tests.py

# Run only unit tests
python tests/run_tests.py --type unit

# Run with coverage report
python tests/run_tests.py --coverage
```

### Performance Benchmarking

```bash
# Run performance benchmarks
python tests/benchmark.py

# This will test:
# - Environment simulation speed
# - Agent inference speed  
# - Training step performance
# - Memory usage
# - Convergence speed
```

## Common Workflows

### Research and Experimentation

1. **Baseline Training**:
   ```bash
   python scripts/train.py experiment_name=baseline
   ```

2. **Hyperparameter Search**:
   ```bash
   python scripts/tune_hyperparameters.py --n_trials 100
   ```

3. **Ablation Studies**:
   ```bash
   # Without domain randomization
   python scripts/train.py env.domain_randomization=false experiment_name=no_dr
   
   # Different reward functions
   python scripts/train.py env.reward_type=sparse experiment_name=sparse_reward
   
   # Different network architectures
   python scripts/train.py agent.hidden_size=128 experiment_name=small_net
   ```

4. **Analysis and Comparison**:
   ```bash
   python scripts/visualize.py --log-dirs logs/baseline logs/no_dr logs/sparse_reward
   ```

### Production Deployment

1. **Train Production Model**:
   ```bash
   python scripts/train.py \
     training.total_episodes=5000 \
     training.eval_interval=100 \
     experiment_name=production_v1
   ```

2. **Comprehensive Evaluation**:
   ```bash
   python scripts/evaluate.py \
     --model_path models/production_v1_best.pth \
     --num_episodes 1000
   ```

3. **Export for Deployment**:
   ```bash
   python scripts/export_tflm.py \
     --model_path models/production_v1_best.pth \
     --quantize \
     --validate \
     --output-dir deployment/v1/
   ```

4. **Validation Testing**:
   ```bash
   python scripts/validate_tflite.py \
     --tflite-path deployment/v1/model.tflite \
     --pytorch_path models/production_v1_best.pth \
     --num-tests 1000
   ```

## Configuration Options

### Agent Configuration
```yaml
agent:
  learning_rate: 3e-4      # Learning rate for all networks
  batch_size: 256          # Training batch size
  buffer_size: 1000000     # Replay buffer size
  hidden_size: 256         # Hidden layer size
  gamma: 0.99              # Discount factor
  tau: 0.005               # Soft update coefficient
  auto_entropy_tuning: true # Automatic entropy tuning
  target_entropy: -2.0     # Target entropy (if not auto)
```

### Environment Configuration
```yaml
env:
  max_episode_steps: 1000     # Max steps per episode
  domain_randomization: true  # Enable domain randomization
  sensor_noise: true          # Add sensor noise
  wind_disturbance: false     # Enable wind effects
  initial_fuel: 1.0           # Initial fuel fraction
  reward_shaping: true        # Use shaped rewards
```

### Training Configuration
```yaml
training:
  total_episodes: 2000        # Total training episodes
  eval_interval: 100          # Episodes between evaluations
  save_interval: 500          # Episodes between saves
  learning_starts: 1000       # Steps before training starts
  use_tensorboard: true       # Enable TensorBoard logging
  use_wandb: false           # Enable Weights & Biases
  checkpoint_frequency: 1000  # Checkpoint every N episodes
```

## Troubleshooting

### Common Issues and Solutions

**Memory Issues**:
```bash
# Reduce batch size
python scripts/train.py agent.batch_size=128

# Reduce buffer size
python scripts/train.py agent.buffer_size=100000
```

**Training Not Converging**:
```bash
# Increase learning rate
python scripts/train.py agent.learning_rate=1e-3

# Extend training
python scripts/train.py training.total_episodes=5000

# Check environment reward scaling
python scripts/evaluate.py --debug-rewards
```

**PyBullet GUI Issues**:
```bash
# Use headless mode
export PYBULLET_EGL=1
python scripts/train.py
```

**CUDA Out of Memory**:
```bash
# Force CPU training
python scripts/train.py device=cpu

# Or reduce batch size
python scripts/train.py agent.batch_size=64
```

### Performance Optimization

**Speed up Training**:
- Use GPU acceleration
- Increase batch size (if memory allows)
- Reduce evaluation frequency
- Use efficient data loading

**Improve Sample Efficiency**:
- Increase replay buffer size
- Use prioritized experience replay
- Tune exploration parameters
- Improve reward function design

**Better Generalization**:
- Increase domain randomization
- Use curriculum learning
- Train on diverse scenarios
- Add regularization

## Next Steps

After completing this guide, you can:

1. **Explore Advanced Features**:
   - Custom reward functions
   - Multi-objective optimization
   - Hierarchical control
   - Transfer learning

2. **Hardware Integration**:
   - Real rocket testing
   - Hardware-in-the-loop simulation
   - Sensor fusion
   - Safety systems

3. **Research Directions**:
   - Alternative algorithms (PPO, TD3)
   - Model-based approaches
   - Meta-learning
   - Sim-to-real transfer techniques

For more detailed information, see the full documentation in `Docs/` or the API reference in the code comments.
