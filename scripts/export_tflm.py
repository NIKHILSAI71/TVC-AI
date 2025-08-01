#!/usr/bin/env python3
"""
TensorFlow Lite Export and Quantization Script

This script converts trained PyTorch SAC models to TensorFlow Lite format
for deployment on microcontrollers. It includes:
- PyTorch to TensorFlow conversion
- Post-training quantization (8-bit)
- C-array generation for embedded deployment
- Model validation and benchmarking
"""

import os
import sys
import logging
import argparse
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from agent import SACAgent, SACConfig
from env import make_evaluation_env


class TorchToTFConverter:
    """Converts PyTorch SAC actor to TensorFlow model."""
    
    def __init__(self, torch_actor: nn.Module, input_shape: Tuple[int]):
        """
        Initialize converter.
        
        Args:
            torch_actor: PyTorch actor network
            input_shape: Input tensor shape (without batch dimension)
        """
        self.torch_actor = torch_actor
        self.input_shape = input_shape
        self.tf_model = None
    
    def convert_to_tensorflow(self) -> tf.keras.Model:
        """
        Convert PyTorch actor to TensorFlow model.
        
        Returns:
            TensorFlow Keras model equivalent to PyTorch actor
        """
        # Extract weights and architecture from PyTorch model
        torch_params = {}
        for name, param in self.torch_actor.named_parameters():
            torch_params[name] = param.detach().cpu().numpy()
        
        # Build equivalent TensorFlow model
        tf_model = self._build_tf_model(torch_params)
        
        # Validate conversion
        self._validate_conversion(tf_model)
        
        self.tf_model = tf_model
        return tf_model
    
    def _build_tf_model(self, torch_params: Dict[str, np.ndarray]) -> tf.keras.Model:
        """Build TensorFlow model from PyTorch parameters."""
        # Input layer
        inputs = tf.keras.Input(shape=self.input_shape, name='observations')
        
        # Get architecture info from PyTorch model
        hidden_layers = []
        for name, module in self.torch_actor.named_modules():
            if isinstance(module, nn.Linear) and 'shared_net' in name:
                hidden_layers.append(module.out_features)
        
        # Build shared network
        x = inputs
        layer_idx = 0
        
        for i, hidden_dim in enumerate(hidden_layers[:-1]):  # Exclude last layer
            # Dense layer
            weight_name = f'shared_net.layers.{layer_idx}.weight'
            bias_name = f'shared_net.layers.{layer_idx}.bias'
            
            if weight_name in torch_params:
                # PyTorch uses (out_features, in_features), TF uses (in_features, out_features)
                weight = torch_params[weight_name].T
                bias = torch_params[bias_name]
                
                x = tf.keras.layers.Dense(
                    hidden_dim,
                    activation='relu',
                    weights=[weight, bias],
                    name=f'dense_{i}'
                )(x)
            else:
                x = tf.keras.layers.Dense(hidden_dim, activation='relu', name=f'dense_{i}')(x)
            
            layer_idx += 2  # Skip activation layer in PyTorch indexing
        
        # Final shared layer
        final_shared_weight = torch_params.get(f'shared_net.layers.{layer_idx}.weight')
        final_shared_bias = torch_params.get(f'shared_net.layers.{layer_idx}.bias')
        
        if final_shared_weight is not None:
            shared_output = tf.keras.layers.Dense(
                final_shared_weight.shape[0],
                activation='relu',
                weights=[final_shared_weight.T, final_shared_bias],
                name='shared_output'
            )(x)
        else:
            shared_output = tf.keras.layers.Dense(256, activation='relu', name='shared_output')(x)
        
        # Mean head
        mean_weight = torch_params.get('mean_head.weight')
        mean_bias = torch_params.get('mean_head.bias')
        
        if mean_weight is not None:
            mean_output = tf.keras.layers.Dense(
                mean_weight.shape[0],
                weights=[mean_weight.T, mean_bias],
                name='mean_output'
            )(shared_output)
        else:
            mean_output = tf.keras.layers.Dense(2, name='mean_output')(shared_output)  # 2 actions for TVC
        
        # Apply tanh activation and scaling for final actions
        action_scale = 1.0  # Assuming unit action scale
        actions = tf.keras.layers.Activation('tanh', name='tanh_activation')(mean_output)
        actions = tf.keras.layers.Lambda(lambda x: x * action_scale, name='action_scaling')(actions)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=actions, name='sac_actor_policy')
        
        return model
    
    def _validate_conversion(self, tf_model: tf.keras.Model):
        """Validate that TensorFlow model produces similar outputs to PyTorch."""
        # Create test input
        test_input = np.random.randn(1, *self.input_shape).astype(np.float32)
        
        # PyTorch forward pass (deterministic)
        self.torch_actor.eval()
        with torch.no_grad():
            torch_input = torch.FloatTensor(test_input)
            torch_mean, _ = self.torch_actor.forward(torch_input)
            torch_action = torch.tanh(torch_mean).numpy()
        
        # TensorFlow forward pass
        tf_action = tf_model(test_input).numpy()
        
        # Check similarity
        max_diff = np.max(np.abs(torch_action - tf_action))
        logging.info(f"Max difference between PyTorch and TensorFlow outputs: {max_diff:.6f}")
        
        if max_diff > 0.1:  # Threshold for acceptable difference
            logging.warning(f"Large difference detected: {max_diff:.6f}")
        else:
            logging.info("Conversion validation passed!")


class TFLiteOptimizer:
    """Handles TensorFlow Lite optimization and quantization."""
    
    def __init__(self, tf_model: tf.keras.Model):
        """Initialize optimizer with TensorFlow model."""
        self.tf_model = tf_model
        self.representative_dataset = None
    
    def create_representative_dataset(self, env, num_samples: int = 100):
        """
        Create representative dataset for quantization calibration.
        
        Args:
            env: Environment to sample observations from
            num_samples: Number of representative samples
        """
        logging.info(f"Creating representative dataset with {num_samples} samples...")
        
        samples = []
        obs, _ = env.reset()
        
        for _ in tqdm(range(num_samples), desc="Collecting samples"):
            # Collect observation
            samples.append(obs.copy())
            
            # Take random action and step
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)
            
            # Reset if episode ends
            if terminated or truncated:
                obs, _ = env.reset()
        
        # Convert to generator function
        def representative_data_gen():
            for sample in samples:
                yield [sample.reshape(1, -1).astype(np.float32)]
        
        self.representative_dataset = representative_data_gen
        logging.info("Representative dataset created successfully")
    
    def convert_to_tflite(self, optimization_level: str = 'default') -> bytes:
        """
        Convert TensorFlow model to TensorFlow Lite.
        
        Args:
            optimization_level: 'none', 'default', or 'aggressive'
            
        Returns:
            TensorFlow Lite model as bytes
        """
        converter = tf.lite.TFLiteConverter.from_keras_model(self.tf_model)
        
        # Set optimization level
        if optimization_level == 'default':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        elif optimization_level == 'aggressive':
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
        
        # Use representative dataset for quantization
        if self.representative_dataset is not None:
            converter.representative_dataset = self.representative_dataset
            
            # For INT8 quantization
            if optimization_level in ['default', 'aggressive']:
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
        
        # Convert model
        tflite_model = converter.convert()
        logging.info(f"Successfully converted to TensorFlow Lite ({optimization_level} optimization)")
        return tflite_model


class ModelValidator:
    """Validates TensorFlow Lite model performance."""
    
    def __init__(self, original_agent: SACAgent, tflite_model_path: str):
        """
        Initialize validator.
        
        Args:
            original_agent: Original PyTorch SAC agent
            tflite_model_path: Path to TensorFlow Lite model
        """
        self.original_agent = original_agent
        self.interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
    
    def predict_tflite(self, observation: np.ndarray) -> np.ndarray:
        """Make prediction using TensorFlow Lite model."""
        # Prepare input
        input_data = observation.reshape(1, -1).astype(self.input_details[0]['dtype'])
        
        # Handle quantized inputs
        if self.input_details[0]['dtype'] == np.int8:
            input_scale, input_zero_point = self.input_details[0]['quantization']
            input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
        
        # Set input and run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Handle quantized outputs
        if self.output_details[0]['dtype'] == np.int8:
            output_scale, output_zero_point = self.output_details[0]['quantization']
            output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
        
        return output_data.flatten()
    
    def validate_accuracy(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """
        Validate TensorFlow Lite model accuracy against original.
        
        Args:
            env: Environment for testing
            num_episodes: Number of episodes to test
            
        Returns:
            Validation metrics
        """
        logging.info(f"Validating TensorFlow Lite model accuracy over {num_episodes} episodes...")
        
        action_differences = []
        original_rewards = []
        tflite_rewards = []
        
        for episode in tqdm(range(num_episodes), desc="Validation"):
            # Test original agent
            obs, _ = env.reset()
            original_reward = 0
            
            while True:
                original_action = self.original_agent.select_action(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(original_action)
                original_reward += reward
                
                if terminated or truncated:
                    break
            
            original_rewards.append(original_reward)
            
            # Test TensorFlow Lite model
            obs, _ = env.reset()
            tflite_reward = 0
            episode_action_diffs = []
            
            while True:
                # Get actions from both models
                original_action = self.original_agent.select_action(obs, deterministic=True)
                tflite_action = self.predict_tflite(obs)
                
                # Record action difference
                action_diff = np.mean(np.abs(original_action - tflite_action))
                episode_action_diffs.append(action_diff)
                
                # Step with TensorFlow Lite action
                obs, reward, terminated, truncated, _ = env.step(tflite_action)
                tflite_reward += reward
                
                if terminated or truncated:
                    break
            
            tflite_rewards.append(tflite_reward)
            action_differences.extend(episode_action_diffs)
        
        # Compute metrics
        metrics = {
            'mean_action_difference': np.mean(action_differences),
            'max_action_difference': np.max(action_differences),
            'original_reward_mean': np.mean(original_rewards),
            'tflite_reward_mean': np.mean(tflite_rewards),
            'reward_difference': np.mean(original_rewards) - np.mean(tflite_rewards),
            'reward_correlation': np.corrcoef(original_rewards, tflite_rewards)[0, 1]
        }
        
        return metrics
    
    def benchmark_inference(self, num_inferences: int = 1000) -> Dict[str, float]:
        """
        Benchmark inference speed.
        
        Args:
            num_inferences: Number of inferences to time
            
        Returns:
            Benchmark metrics
        """
        logging.info(f"Benchmarking inference speed over {num_inferences} inferences...")
        
        # Create random input
        input_shape = self.input_details[0]['shape'][1:]  # Remove batch dimension
        test_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warm up
        for _ in range(10):
            self.predict_tflite(test_input)
        
        # Benchmark
        import time
        start_time = time.time()
        
        for _ in range(num_inferences):
            self.predict_tflite(test_input)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_inference_time = total_time / num_inferences * 1000  # ms
        
        metrics = {
            'total_time_s': total_time,
            'avg_inference_time_ms': avg_inference_time,
            'inferences_per_second': num_inferences / total_time
        }
        
        return metrics


def generate_c_array(tflite_model: bytes, output_path: str, array_name: str = "model_data"):
    """
    Generate C array from TensorFlow Lite model.
    
    Args:
        tflite_model: TensorFlow Lite model as bytes
        output_path: Output file path
        array_name: Name of the C array variable
    """
    logging.info(f"Generating C array: {output_path}")
    
    with open(output_path, 'w') as f:
        f.write(f"// Generated TensorFlow Lite model array\\n")
        f.write(f"// Model size: {len(tflite_model)} bytes\\n\\n")
        f.write(f"#ifndef {array_name.upper()}_H\\n")
        f.write(f"#define {array_name.upper()}_H\\n\\n")
        f.write(f"const unsigned char {array_name}[] = {{\\n")
        
        # Write bytes in rows of 16
        for i in range(0, len(tflite_model), 16):
            hex_values = [f"0x{b:02x}" for b in tflite_model[i:i+16]]
            f.write("  " + ", ".join(hex_values))
            if i + 16 < len(tflite_model):
                f.write(",")
            f.write("\\n")
        
        f.write("};\\n\\n")
        f.write(f"const unsigned int {array_name}_len = {len(tflite_model)};\\n\\n")
        f.write(f"#endif  // {array_name.upper()}_H\\n")
    
    logging.info(f"C array generated successfully: {len(tflite_model)} bytes")


def create_inference_example(output_dir: str):
    """Create example C++ code for inference."""
    cpp_code = '''// TensorFlow Lite Inference Example for Rocket TVC Control
// This example shows how to use the generated model for rocket control

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"  // Generated model array

// Global variables for TensorFlow Lite Micro
tflite::MicroErrorReporter micro_error_reporter;
tflite::AllOpsResolver resolver;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

// Tensor arena for model execution (adjust size as needed)
constexpr int kTensorArenaSize = 4 * 1024;  // 4KB
uint8_t tensor_arena[kTensorArenaSize];

void setup_model() {
    // Load model from generated array
    model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        // Handle version mismatch
        return;
    }
    
    // Create interpreter
    static tflite::MicroInterpreter static_interpreter(
        model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
    interpreter = &static_interpreter;
    
    // Allocate tensors
    TfLiteStatus allocate_status = interpreter->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        // Handle allocation failure
        return;
    }
    
    // Get input and output tensors
    input = interpreter->input(0);
    output = interpreter->output(0);
}

void predict_control_action(float* observation, float* action) {
    // Copy observation to input tensor
    for (int i = 0; i < input->dims->data[1]; i++) {
        input->data.f[i] = observation[i];
    }
    
    // Run inference
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
        // Handle inference failure
        return;
    }
    
    // Copy output to action array
    for (int i = 0; i < output->dims->data[1]; i++) {
        action[i] = output->data.f[i];
    }
}

// Example usage in main control loop
void control_loop_example() {
    // Rocket state observation (8 elements for TVC environment)
    float observation[8] = {
        0.0f, 0.0f, 0.0f, 1.0f,  // Quaternion [qx, qy, qz, qw]
        0.0f, 0.0f, 0.0f,         // Angular velocities [wx, wy, wz]
        1.0f                      // Fuel remaining [0, 1]
    };
    
    // Control action output (2 elements: pitch and yaw gimbal angles)
    float action[2];
    
    // Get control action from neural network
    predict_control_action(observation, action);
    
    // Apply action to servo controls
    float gimbal_pitch = action[0];  // Normalized [-1, 1]
    float gimbal_yaw = action[1];    // Normalized [-1, 1]
    
    // Convert to servo angles (example: ±15 degrees max)
    float max_gimbal_angle = 15.0f;
    float servo_pitch_deg = gimbal_pitch * max_gimbal_angle;
    float servo_yaw_deg = gimbal_yaw * max_gimbal_angle;
    
    // Send commands to servo controllers
    // set_servo_angle(PITCH_SERVO, servo_pitch_deg);
    // set_servo_angle(YAW_SERVO, servo_yaw_deg);
}
'''
    
    output_path = Path(output_dir) / "inference_example.cpp"
    with open(output_path, 'w') as f:
        f.write(cpp_code)
    
    logging.info(f"Inference example generated: {output_path}")


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description='Export SAC model to TensorFlow Lite')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained PyTorch model')
    parser.add_argument('--output_dir', type=str, default='./exported_models',
                       help='Output directory for exported models')
    parser.add_argument('--optimization', choices=['none', 'default', 'aggressive'],
                       default='default', help='Optimization level')
    parser.add_argument('--validate', action='store_true',
                       help='Validate exported model')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark inference speed')
    parser.add_argument('--generate_c_array', action='store_true',
                       help='Generate C array for embedded deployment')
    parser.add_argument('--num_representative_samples', type=int, default=100,
                       help='Number of samples for quantization calibration')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PyTorch model
    logging.info(f"Loading PyTorch model from {args.model_path}")
    
    # Create dummy environment to get dimensions
    temp_env = make_evaluation_env()
    obs_dim = temp_env.observation_space.shape[0]
    action_dim = temp_env.action_space.shape[0]
    
    # Load SAC agent
    agent = SACAgent(obs_dim, action_dim)
    agent.load(args.model_path)
    
    logging.info("PyTorch model loaded successfully")
    
    # Convert to TensorFlow
    logging.info("Converting PyTorch model to TensorFlow...")
    converter = TorchToTFConverter(agent.actor, (obs_dim,))
    tf_model = converter.convert_to_tensorflow()
    
    # Save TensorFlow model
    tf_model_path = output_dir / "tf_model"
    tf.saved_model.save(tf_model, str(tf_model_path))
    logging.info(f"TensorFlow model saved: {tf_model_path}")
    
    # Create TensorFlow Lite optimizer
    optimizer = TFLiteOptimizer(tf_model)
    
    # Create representative dataset for quantization
    optimizer.create_representative_dataset(temp_env, args.num_representative_samples)
    
    # Convert to TensorFlow Lite
    logging.info(f"Converting to TensorFlow Lite ({args.optimization} optimization)...")
    tflite_model = optimizer.convert_to_tflite(args.optimization)
    
    # Save TensorFlow Lite model
    tflite_path = output_dir / "model.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    logging.info(f"TensorFlow Lite model saved: {tflite_path}")
    logging.info(f"Model size: {len(tflite_model)} bytes ({len(tflite_model)/1024:.1f} KB)")
    
    # Generate C array if requested
    if args.generate_c_array:
        c_array_path = output_dir / "model_data.h"
        generate_c_array(tflite_model, str(c_array_path))
        
        # Generate inference example
        create_inference_example(output_dir)
    
    # Validation and benchmarking
    if args.validate or args.benchmark:
        logging.info("Initializing model validator...")
        validator = ModelValidator(agent, str(tflite_path))
        
        if args.validate:
            logging.info("Validating model accuracy...")
            val_metrics = validator.validate_accuracy(temp_env, num_episodes=5)
            
            logging.info("Validation Results:")
            for key, value in val_metrics.items():
                logging.info(f"  {key}: {value:.6f}")
            
            # Save validation metrics
            import json
            with open(output_dir / "validation_metrics.json", 'w') as f:
                json.dump(val_metrics, f, indent=2)
        
        if args.benchmark:
            logging.info("Benchmarking inference speed...")
            bench_metrics = validator.benchmark_inference()
            
            logging.info("Benchmark Results:")
            for key, value in bench_metrics.items():
                logging.info(f"  {key}: {value:.6f}")
            
            # Save benchmark metrics
            import json
            with open(output_dir / "benchmark_metrics.json", 'w') as f:
                json.dump(bench_metrics, f, indent=2)
    
    temp_env.close()
    
    logging.info("Export completed successfully!")
    logging.info(f"All files saved to: {output_dir}")


if __name__ == "__main__":
    main()
