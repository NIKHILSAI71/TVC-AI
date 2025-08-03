#!/usr/bin/env python3
"""
Universal Device Manager for CPU/GPU/TPU Compatibility
Handles device detection, tensor operations, and model movement across different hardware types.
"""

import torch
import numpy as np
import logging
from typing import Union, Any, Optional, Dict, List
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

class DeviceManager:
    """Universal device manager for PyTorch tensors across CPU/GPU/TPU"""
    
    def __init__(self, prefer_device: Optional[str] = None, enable_tpu: bool = True):
        """
        Initialize device manager with automatic device detection.
        
        Args:
            prefer_device: Preferred device ('cpu', 'cuda', 'xla', 'auto')
            enable_tpu: Whether to attempt TPU/XLA initialization
        """
        self.device = None
        self.device_type = None
        self.xla_available = False
        self.cuda_available = torch.cuda.is_available()
        
        # Try to import XLA for TPU support
        if enable_tpu:
            try:
                import torch_xla
                import torch_xla.core.xla_model as xm
                self.xla_available = True
                self.xm = xm
                logger.info("✅ PyTorch XLA available - TPU support enabled")
            except ImportError:
                logger.info("⚠️ PyTorch XLA not available - TPU support disabled")
                self.xla_available = False
        
        # Initialize device
        self._detect_and_set_device(prefer_device)
        
        # Log device information
        self._log_device_info()
    
    def _detect_and_set_device(self, prefer_device: Optional[str] = None):
        """Detect and set the best available device"""
        
        if prefer_device == 'cpu':
            self.device = torch.device('cpu')
            self.device_type = 'cpu'
            logger.info("🖥️ Using CPU (user preference)")
            
        elif prefer_device == 'cuda' or prefer_device == 'gpu':
            if self.cuda_available:
                self.device = torch.device('cuda')
                self.device_type = 'cuda'
                logger.info(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                self.device = torch.device('cpu')
                self.device_type = 'cpu'
                
        elif prefer_device == 'xla' or prefer_device == 'tpu':
            if self.xla_available:
                self.device = self.xm.xla_device()
                self.device_type = 'xla'
                logger.info(f"⚡ Using XLA device: {self.device}")
            else:
                logger.warning("XLA/TPU requested but not available, falling back to best available")
                self._auto_detect_device()
                
        else:  # Auto detection
            self._auto_detect_device()
    
    def _auto_detect_device(self):
        """Automatically detect the best available device"""
        if self.xla_available:
            try:
                self.device = self.xm.xla_device()
                self.device_type = 'xla'
                logger.info(f"⚡ Auto-detected XLA device: {self.device}")
            except Exception as e:
                logger.warning(f"XLA device detection failed: {e}, trying CUDA")
                self._try_cuda_or_cpu()
        else:
            self._try_cuda_or_cpu()
    
    def _try_cuda_or_cpu(self):
        """Try CUDA, fallback to CPU"""
        if self.cuda_available:
            self.device = torch.device('cuda')
            self.device_type = 'cuda'
            logger.info(f"🚀 Auto-detected CUDA GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            self.device_type = 'cpu'
            logger.info("🖥️ Using CPU (no GPU/TPU available)")
    
    def _log_device_info(self):
        """Log comprehensive device information"""
        logger.info("="*60)
        logger.info("DEVICE INFORMATION")
        logger.info("="*60)
        logger.info(f"Selected device: {self.device}")
        logger.info(f"Device type: {self.device_type}")
        
        if self.device_type == 'cuda':
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"GPU name: {torch.cuda.get_device_name()}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
        elif self.device_type == 'xla':
            logger.info(f"XLA device: {self.device}")
            if hasattr(self, 'xm'):
                logger.info(f"XLA ordinal: {self.xm.get_ordinal()}")
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info("="*60)
    
    def to_device(self, tensor_or_data: Union[torch.Tensor, np.ndarray, Any], 
                  non_blocking: bool = False) -> torch.Tensor:
        """
        Move tensor or data to the managed device with proper type conversion.
        
        Args:
            tensor_or_data: Input tensor, numpy array, or other data
            non_blocking: Whether to use non-blocking transfer
            
        Returns:
            Tensor on the target device
        """
        if isinstance(tensor_or_data, np.ndarray):
            tensor = torch.from_numpy(tensor_or_data).float()
        elif isinstance(tensor_or_data, torch.Tensor):
            tensor = tensor_or_data
        else:
            # Try to convert to tensor
            try:
                tensor = torch.tensor(tensor_or_data).float()
            except Exception as e:
                raise ValueError(f"Cannot convert {type(tensor_or_data)} to tensor: {e}")
        
        # Move to device
        if self.device_type == 'xla':
            # For XLA, we don't use non_blocking
            return tensor.to(self.device)
        else:
            return tensor.to(self.device, non_blocking=non_blocking)
    
    def to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Convert tensor to numpy array with proper device handling.
        
        Args:
            tensor: Input tensor or numpy array
            
        Returns:
            Numpy array
        """
        if isinstance(tensor, np.ndarray):
            return tensor
        
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor or np.ndarray, got {type(tensor)}")
        
        # Handle different device types
        if self.device_type == 'xla':
            # For XLA tensors, we need to use XLA-specific methods
            if hasattr(self, 'xm'):
                # Mark step to ensure computation is complete
                self.xm.mark_step()
            return tensor.cpu().detach().numpy()
        else:
            # For CPU/CUDA tensors
            return tensor.detach().cpu().numpy()
    
    def move_model_to_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Move model to the managed device with proper handling for different device types.
        
        Args:
            model: PyTorch model
            
        Returns:
            Model on target device
        """
        try:
            model = model.to(self.device)
            logger.info(f"✅ Model moved to {self.device}")
            return model
        except Exception as e:
            logger.error(f"❌ Failed to move model to {self.device}: {e}")
            # Fallback to CPU
            logger.info("🔄 Falling back to CPU")
            self.device = torch.device('cpu')
            self.device_type = 'cpu'
            return model.to(self.device)
    
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: Union[str, Path]):
        """
        Save checkpoint with device-agnostic handling.
        
        Args:
            checkpoint: Checkpoint dictionary
            path: Save path
        """
        # Convert XLA tensors to CPU before saving
        if self.device_type == 'xla' and hasattr(self, 'xm'):
            # Use XLA's save utility for better handling
            self.xm.save(checkpoint, str(path))
            logger.info(f"✅ Checkpoint saved to {path} (XLA)")
        else:
            # Standard PyTorch save
            torch.save(checkpoint, str(path))
            logger.info(f"✅ Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load checkpoint with device-agnostic handling.
        
        Args:
            path: Checkpoint path
            
        Returns:
            Loaded checkpoint
        """
        try:
            if self.device_type == 'xla':
                # Load to CPU first, then move to XLA device as needed
                checkpoint = torch.load(str(path), map_location='cpu')
            else:
                checkpoint = torch.load(str(path), map_location=self.device)
            
            logger.info(f"✅ Checkpoint loaded from {path}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint from {path}: {e}")
            raise
    
    def synchronize(self):
        """Synchronize device operations"""
        if self.device_type == 'cuda':
            torch.cuda.synchronize()
        elif self.device_type == 'xla' and hasattr(self, 'xm'):
            self.xm.mark_step()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get device memory information"""
        info = {'device': str(self.device), 'type': self.device_type}
        
        if self.device_type == 'cuda':
            info.update({
                'memory_allocated': torch.cuda.memory_allocated() / 1e9,
                'memory_reserved': torch.cuda.memory_reserved() / 1e9,
                'max_memory_allocated': torch.cuda.max_memory_allocated() / 1e9,
            })
        
        return info
    
    def cleanup(self):
        """Clean up device resources"""
        if self.device_type == 'cuda':
            torch.cuda.empty_cache()
            logger.info("🧹 CUDA cache cleared")
        elif self.device_type == 'xla' and hasattr(self, 'xm'):
            # XLA cleanup if needed
            pass
    
    def is_tensor_on_device(self, tensor: torch.Tensor) -> bool:
        """Check if tensor is on the managed device"""
        if self.device_type == 'xla':
            # For XLA, check if it's an XLA tensor
            return str(tensor.device).startswith('xla')
        else:
            return tensor.device == self.device

# Global device manager instance
_global_device_manager: Optional[DeviceManager] = None

def get_device_manager(prefer_device: Optional[str] = None, 
                      enable_tpu: bool = True) -> DeviceManager:
    """Get global device manager instance"""
    global _global_device_manager
    
    if _global_device_manager is None:
        _global_device_manager = DeviceManager(prefer_device, enable_tpu)
    
    return _global_device_manager

def to_device(tensor_or_data: Union[torch.Tensor, np.ndarray, Any]) -> torch.Tensor:
    """Convenience function to move data to managed device"""
    return get_device_manager().to_device(tensor_or_data)

def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convenience function to convert tensor to numpy"""
    return get_device_manager().to_numpy(tensor)

def get_device() -> torch.device:
    """Get the managed device"""
    return get_device_manager().device

def synchronize():
    """Synchronize device operations"""
    get_device_manager().synchronize()
