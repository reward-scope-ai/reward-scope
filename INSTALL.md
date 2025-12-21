# Installation Guide

## Quick Setup (3 Options)

### Option 1: Using pip install (Recommended)

```bash
# Navigate to the repo
cd reward-forensics

# Create a fresh virtual environment (highly recommended)
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install the package
pip install -e .

# Verify installation
python -c "from reward_scope.core import DataCollector; print('✓ Installation successful')"
```

### Option 2: Using PYTHONPATH (No installation)

```bash
# Navigate to the repo
cd reward-forensics

# Install dependencies only
pip install numpy gymnasium pytest

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify
python -c "from reward_scope.core import DataCollector; print('✓ Setup successful')"
```

### Option 3: Using requirements.txt

```bash
cd reward-forensics

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt

# Set PYTHONPATH or install with pip install -e .
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## Troubleshooting

### Issue: `pip install -e .` fails with setuptools error

**Solution 1**: Use PYTHONPATH instead
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python examples/cartpole_basic.py
```

**Solution 2**: Try upgrading pip
```bash
pip install --upgrade pip
pip install -e .
```

**Solution 3**: Install dependencies manually
```bash
pip install numpy>=1.21.0 gymnasium>=0.28.0 pytest>=7.0.0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: `ImportError: No module named 'reward_scope'`

**Solution**: Make sure you're running from the repo root
```bash
cd /path/to/reward-forensics
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python examples/cartpole_basic.py
```

### Issue: Tests fail

**Solution**: Run from repo root
```bash
cd /path/to/reward-forensics
pytest tests/ -v
```

---

## Verify Installation

```bash
# Test imports
python -c "from reward_scope.core import DataCollector, RewardDecomposer, HackingDetectorSuite; print('✓ All imports successful')"

# Run basic example
python examples/cartpole_basic.py

# Run tests
pytest tests/ -v

# Expected: 78 passed
```

---

## Quick Start Script

Copy-paste this entire block:

```bash
# Setup script
cd /path/to/reward-forensics

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install numpy gymnasium pytest

# Option A: Install package
pip install -e . || echo "Install failed, using PYTHONPATH instead"

# Option B: Use PYTHONPATH if install failed
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Test
python -c "from reward_scope.core import DataCollector; print('✓ Ready to go!')"
python examples/cartpole_basic.py
```

---

## Development Setup

```bash
# Full development environment
cd reward-forensics
python3 -m venv venv
source venv/bin/activate

# Install all dev dependencies
pip install -r requirements-dev.txt

# Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Run tests
pytest tests/ -v

# Run examples
python examples/cartpole_basic.py
python examples/cartpole_hacking_demo.py
```

---

## Environment-Specific Notes

### Using Conda

```bash
conda create -n rewardscope python=3.10
conda activate rewardscope
cd reward-forensics
pip install -r requirements-dev.txt
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Docker (Future)

```bash
# Not yet implemented
# Coming in Phase 4 with dashboard
```

### VS Code

Add to `.vscode/settings.json`:
```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}"
    ]
}
```

---

## Testing Installation

```bash
# Run this to verify everything works:
python -c "
import sys
sys.path.insert(0, '.')
from reward_scope.core import DataCollector, RewardDecomposer, HackingDetectorSuite
print('✓ DataCollector imported')
print('✓ RewardDecomposer imported')
print('✓ HackingDetectorSuite imported')
print('✓ All core modules working!')
"

pytest tests/ -v
# Expected: 78 passed
```
