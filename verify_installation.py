"""
Installation verification script for TVC-AI.
Run this script after installation to verify all dependencies work correctly.
"""

import sys
import importlib
from packaging import version

def check_python_version():
    """Check Python version compatibility."""
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version >= (3, 8) and python_version < (3, 12):
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python version should be between 3.8 and 3.11")
        return False

def check_package(package_name, min_version=None, import_name=None):
    """Check if package is installed and meets version requirements."""
    try:
        if import_name is None:
            import_name = package_name
            
        module = importlib.import_module(import_name)
        
        if hasattr(module, '__version__'):
            installed_version = module.__version__
        elif package_name == 'torch':
            installed_version = module.__version__
        elif package_name == 'pybullet':
            import pybullet
            installed_version = pybullet.__version__ if hasattr(pybullet, '__version__') else "unknown"
        else:
            installed_version = "unknown"
        
        if min_version and installed_version != "unknown":
            try:
                if version.parse(installed_version) >= version.parse(min_version):
                    print(f"✅ {package_name}: {installed_version} (>= {min_version})")
                    return True
                else:
                    print(f"❌ {package_name}: {installed_version} (need >= {min_version})")
                    return False
            except:
                print(f"⚠️  {package_name}: {installed_version} (version check failed)")
                return True
        else:
            print(f"✅ {package_name}: {installed_version}")
            return True
            
    except ImportError:
        print(f"❌ {package_name}: Not installed")
        return False
    except Exception as e:
        print(f"⚠️  {package_name}: Error checking - {e}")
        return False

def test_core_functionality():
    """Test basic functionality of core components."""
    print("\n🧪 Testing Core Functionality...")
    
    try:
        # Test PyTorch
        import torch
        x = torch.randn(2, 3)
        y = torch.mm(x, x.t())
        print("✅ PyTorch basic operations work")
        
        # Test PyBullet
        import pybullet as p
        physicsClient = p.connect(p.DIRECT)  # Non-GUI mode
        p.disconnect()
        print("✅ PyBullet physics simulation works")
        
        # Test Gymnasium
        import gymnasium as gym
        env = gym.make('CartPole-v1')
        obs, info = env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
        print("✅ Gymnasium environment works")
        
        # Test Hydra
        from hydra import initialize, compose
        print("✅ Hydra configuration system works")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        return False

def check_optional_packages():
    """Check optional packages for enhanced functionality."""
    print("\n📦 Checking Optional Packages...")
    
    optional_packages = [
        ("tensorflow", "2.10.0", "tensorflow"),
        ("tensorboard", "2.10.0", "tensorboard"),
        ("matplotlib", "3.6.0", "matplotlib"),
        ("pandas", "1.5.0", "pandas"),
        ("scipy", "1.9.0", "scipy"),
        ("wandb", "0.13.0", "wandb"),
        ("pytest", "7.1.0", "pytest"),
    ]
    
    available_features = []
    
    for package, min_ver, import_name in optional_packages:
        if check_package(package, min_ver, import_name):
            if package == "tensorflow":
                available_features.append("TensorFlow Lite export")
            elif package == "tensorboard":
                available_features.append("TensorBoard logging")
            elif package == "matplotlib":
                available_features.append("Plotting and visualization")
            elif package == "wandb":
                available_features.append("Weights & Biases tracking")
            elif package == "pytest":
                available_features.append("Testing framework")
    
    if available_features:
        print(f"\n🎉 Available features: {', '.join(available_features)}")
    
    return len(available_features)

def main():
    """Main verification function."""
    print("🚀 TVC-AI Installation Verification")
    print("=" * 40)
    
    all_good = True
    
    # Check Python version
    all_good &= check_python_version()
    
    print("\n📋 Checking Core Dependencies...")
    
    # Core dependencies
    core_packages = [
        ("torch", "1.13.0"),
        ("numpy", "1.21.0"),
        ("pybullet", "3.2.7"),
        ("gymnasium", "0.26.0", "gymnasium"),
        ("hydra-core", "1.2.0", "hydra"),
        ("tqdm", "4.64.0"),
        ("pyyaml", "6.0.0", "yaml"),
    ]
    
    for package_info in core_packages:
        if len(package_info) == 3:
            package, min_ver, import_name = package_info
            all_good &= check_package(package, min_ver, import_name)
        else:
            package, min_ver = package_info
            all_good &= check_package(package, min_ver)
    
    # Test functionality
    if all_good:
        all_good &= test_core_functionality()
    
    # Check optional packages
    optional_count = check_optional_packages()
    
    print("\n" + "=" * 40)
    if all_good:
        print("🎉 Installation verification PASSED!")
        print("✅ All core dependencies are working correctly")
        if optional_count > 0:
            print(f"✅ {optional_count} optional features available")
        print("\nYou can now run:")
        print("  python scripts/train.py")
    else:
        print("❌ Installation verification FAILED!")
        print("Please check the error messages above and reinstall missing packages")
        print("\nTry:")
        print("  pip install -r requirements-minimal.txt")
        
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
