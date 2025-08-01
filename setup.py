#!/usr/bin/env python3
"""
TVC-AI Setup Script
Automated installation and verification for TVC-AI system.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version >= (3, 8) and version < (3, 12):
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not supported")
        print("Please use Python 3.8-3.11")
        return False

def main():
    """Main setup function."""
    print("🚀 TVC-AI Automated Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Get installation type
    print("\n📦 Installation Types:")
    print("1. Minimal (core functionality only)")
    print("2. Full (recommended - all features)")
    print("3. Development (includes dev tools)")
    
    while True:
        choice = input("\nSelect installation type (1/2/3) [2]: ").strip()
        if choice == "" or choice == "2":
            requirements_file = "requirements.txt"
            break
        elif choice == "1":
            requirements_file = "requirements-minimal.txt"
            break
        elif choice == "3":
            requirements_file = "requirements.txt"
            dev_install = True
            break
        else:
            print("Please enter 1, 2, or 3")
    
    # Check if requirements file exists
    if not Path(requirements_file).exists():
        print(f"❌ Requirements file {requirements_file} not found!")
        return 1
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        print("⚠️  Could not upgrade pip, continuing anyway...")
    
    # Install requirements
    install_cmd = f"{sys.executable} -m pip install -r {requirements_file}"
    if not run_command(install_cmd, f"Installing packages from {requirements_file}"):
        print("❌ Package installation failed!")
        return 1
    
    # Install development requirements if selected
    if choice == "3":
        if Path("requirements-dev.txt").exists():
            dev_cmd = f"{sys.executable} -m pip install -r requirements-dev.txt"
            if not run_command(dev_cmd, "Installing development packages"):
                print("⚠️  Development packages installation failed, but core installation succeeded")
    
    # Run verification
    print("\n🧪 Running installation verification...")
    if run_command(f"{sys.executable} verify_installation.py", "Verifying installation", check=False):
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Start training: python scripts/train.py")
        print("2. Run tests: python tests/run_tests.py")
        print("3. Check documentation: README.md and USAGE.md")
        return 0
    else:
        print("\n⚠️  Setup completed but verification found some issues.")
        print("Check the verification output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
