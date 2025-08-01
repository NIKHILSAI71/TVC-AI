# Windows PyBullet Installation Fix

This guide provides multiple solutions for installing PyBullet on Windows when you encounter Visual C++ build errors.

## Quick Fix Options (Try in Order)

### Option 1: Use Pre-compiled Wheel (Fastest)
```powershell
# Update pip first
python -m pip install --upgrade pip

# Try installing from a specific wheel
pip install --upgrade wheel
pip install pybullet --only-binary=pybullet
```

### Option 2: Alternative Installation Methods
```powershell
# Method A: Use conda instead of pip
conda install -c conda-forge pybullet

# Method B: Try older compatible version
pip install pybullet==3.2.6

# Method C: Force binary installation
pip install --prefer-binary pybullet
```

### Option 3: Install Visual C++ Build Tools
If the above don't work, you need to install Microsoft C++ Build Tools:

1. **Download Build Tools**:
   Go to: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   
2. **Install Required Components**:
   - Run the installer
   - Select "C++ build tools"
   - Make sure "Windows 10 SDK" is selected
   - Install and restart your computer

3. **Then install PyBullet**:
   ```powershell
   pip install pybullet
   ```

### Option 4: Use Windows Subsystem for Linux (WSL)
If you're comfortable with Linux:
```bash
# Install WSL2 if not already installed
wsl --install

# In WSL terminal:
pip install pybullet
```

## Verification
After installation, test PyBullet:
```python
import pybullet as p
print("PyBullet version:", p.getAPIVersion())
```

## If All Else Fails
Use our Docker container or switch to a different physics engine temporarily:
```powershell
# Alternative: Use a different physics engine for initial testing
pip install pymunk  # 2D physics alternative
```
