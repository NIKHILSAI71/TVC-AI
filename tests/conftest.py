"""
Configuration for pytest test suite.
"""

import pytest
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def pytest_configure(config):
    """Configure pytest with custom settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks tests as benchmark tests"
    )

@pytest.fixture(scope="session")
def project_root():
    """Project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment for each test."""
    # Disable GUI for PyBullet during tests
    import os
    os.environ['PYBULLET_EGL'] = '1'
    
    # Set random seeds for reproducibility
    import random
    import numpy as np
    import torch
    
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

@pytest.fixture
def temp_model_dir(tmp_path):
    """Create temporary directory for model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir

@pytest.fixture
def temp_log_dir(tmp_path):
    """Create temporary directory for logs."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    return log_dir
