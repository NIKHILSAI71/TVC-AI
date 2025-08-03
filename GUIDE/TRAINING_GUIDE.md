# TVC-AI Training Guide

## Quick Start

The TVC-AI system now has a simplified structure with just **2 training modes**:

### 1. Minimal Mode (Fast Training) ⚡
```bash
# Install requirements
pip install -r requirements.txt

# Run fast training (150K steps, ~1-2 hours)
python scripts/train.py --config-name=minimal
```

### 2. Production Mode (Best Performance) 🏆
```bash
# Run production training (1M steps, ~6-8 hours)
python scripts/train.py --config-name=production
```

## Training Modes Comparison

| Feature | Minimal Mode | Production Mode |
|---------|-------------|-----------------|
| **Training Steps** | 150,000 | 1,000,000 |
| **Network Size** | [256, 256] | [512, 512, 256] |
| **Buffer Size** | 50,000 | 1,000,000 |
| **Domain Randomization** | Disabled | Enabled |
| **Sensor Noise** | Disabled | Enabled |
| **Training Time** | ~1-2 hours | ~6-8 hours |
| **Target Use** | Development/Testing | Final Deployment |
| **Expected Success Rate** | 70-80% | 90-95% |

## Configuration Details

### Minimal Mode Features:
- **Fast convergence**: Optimized hyperparameters for quick learning
- **Simple environment**: No disturbances, minimal wind
- **Small network**: Faster training, less memory
- **Early stopping**: Aggressive stopping to save time
- **Learning rates**: 0.001 (fast convergence)

### Production Mode Features:
- **Maximum robustness**: Full domain randomization and sensor noise
- **Large network**: Maximum learning capacity
- **Extended training**: 1M steps for best performance
- **Realistic conditions**: Wind, mass variations, sensor noise
- **Conservative learning**: Stable, reliable convergence

## Expected Results

### Minimal Mode Success Criteria:
- **Tilt angle** < 20 degrees
- **Episode length** > 200 steps
- **Success rate** ≥ 70%
- **Training time** ≤ 2 hours

### Production Mode Success Criteria:
- **Tilt angle** < 15 degrees
- **Episode length** > 300 steps  
- **Success rate** ≥ 90%
- **Robustness** to all disturbances

## Monitoring Training

### TensorBoard (both modes):
```bash
tensorboard --logdir=outputs
```

Key metrics to watch:
- `episode/reward` - Should increase over time
- `episode/success` - Should reach target success rate
- `eval/success_rate` - Evaluation performance
- `training/critic_loss` - Should decrease and stabilize

### Training Progress:
- **Minimal**: Expect success after 50K-100K steps
- **Production**: Expect success after 300K-500K steps

## Switching Between Modes

You can easily switch configurations:
```bash
# For development and testing
python scripts/train.py --config-name=minimal

# For final model training
python scripts/train.py --config-name=production

# Override specific parameters
python scripts/train.py --config-name=minimal training.total_steps=100000
python scripts/train.py --config-name=production agent.lr_actor=0.0001
```

## Troubleshooting

### If training fails:
1. **Check GPU/CPU usage**: Ensure sufficient resources
2. **Reduce batch size**: Edit config files if memory issues
3. **Check dependencies**: Run `pip install -r requirements.txt`

### Performance tuning:
- **Faster training**: Use minimal mode with fewer steps
- **Better performance**: Use production mode with more steps
- **Memory issues**: Reduce buffer_size in config files

The system is now optimized for both speed and performance!
