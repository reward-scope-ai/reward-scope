# RewardScope 🔬

**See what your RL agent actually learned**

Real-time reward debugging and hacking detection for reinforcement learning.

[![Tests](https://img.shields.io/badge/tests-78%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

---

## 🚀 Quick Start

### Installation

#### Option 1: Install in editable mode (recommended for development)

```bash
# Clone the repository
git clone https://github.com/jimmybentley/reward-forensics.git
cd reward-forensics

# Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in editable mode
pip install -e .

# For development (includes pytest, black, ruff)
pip install -e ".[dev]"

# For Stable-Baselines3 integration
pip install -e ".[sb3]"

# Install everything
pip install -e ".[all]"
```

#### Option 2: Install from requirements.txt

```bash
# Core dependencies only
pip install -r requirements.txt

# Development dependencies
pip install -r requirements-dev.txt
```

#### Option 3: Quick test without installation

```bash
# Install minimal dependencies
pip install numpy gymnasium pytest

# Run from repo root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python3 examples/cartpole_basic.py
```

---

## 📖 Usage

### Basic Example (Phase 1 - Core Tracking)

```python
import gymnasium as gym
from reward_scope.core import DataCollector, RewardDecomposer, StepData

# Create environment
env = gym.make("CartPole-v1")

# Set up data collection
collector = DataCollector(run_name="my_experiment")

# Set up reward decomposition
decomposer = RewardDecomposer()
decomposer.register_component(
    "survival",
    lambda obs, act, info: 1.0,
    description="Survival reward"
)

# Training loop
obs, info = env.reset()
for step in range(1000):
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)

    # Decompose reward
    components = decomposer.decompose(obs, action, reward, info)

    # Log step
    step_data = StepData(
        step=step, episode=0, timestamp=time.time(),
        observation=obs, action=action, reward=reward,
        done=terminated, truncated=truncated, info=info,
        reward_components=components
    )
    collector.log_step(step_data)

    if terminated or truncated:
        collector.end_episode()
        obs, info = env.reset()
    else:
        obs = next_obs

collector.close()
```

### Hacking Detection Example (Phase 2)

```python
from reward_scope.core import HackingDetectorSuite

# Create detector suite
detector_suite = HackingDetectorSuite(
    observation_bounds=(env.observation_space.low, env.observation_space.high),
    action_bounds=(env.action_space.low, env.action_space.high),
)

# During training
alerts = detector_suite.update(
    step=step, episode=episode,
    observation=obs, action=action, reward=reward,
    reward_components=components, done=done, info=info
)

# Check for alerts
for alert in alerts:
    print(f"⚠️  {alert.type.value}: {alert.description}")
    print(f"   Suggested fix: {alert.suggested_fix}")

# Get overall hacking score
score = detector_suite.get_hacking_score()
if score > 0.5:
    print(f"⚠️  High hacking likelihood: {score:.2f}")
```

---

## 🧪 Running Examples

### Basic CartPole Example

```bash
python3 examples/cartpole_basic.py
```

Output:
```
Episode 1/5
  Reward: 13.0
  Length: 13
  Component totals: {'survival': 13.0}
...
Component Statistics:
  survival:
    Mean: 1.0000
    Std:  0.0000
    Count: 75
```

### Hacking Detection Demo

```bash
python3 examples/cartpole_hacking_demo.py
```

Output:
```
🔍 TEST 1: Action Repetition Policy
  ⚠️  ALERT at step 52: action_repetition
      Action repetition detected: 90.0% of actions are identical
  Hacking Score: 0.85 / 1.00
  ⚠️  HIGH HACKING LIKELIHOOD DETECTED!
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_collector.py -v
pytest tests/test_decomposer.py -v
pytest tests/test_detectors.py -v

# Run with coverage
pytest tests/ --cov=reward_scope --cov-report=html

# Run specific test
pytest tests/test_detectors.py::TestActionRepetitionDetector::test_detects_repeated_actions -v
```

**Expected output:**
```
============================== test session starts ==============================
...
78 passed in 1.09s
==============================
```

---

## 🏗️ Project Structure

```
reward-forensics/
├── reward_scope/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       ├── collector.py        # Data collection & SQLite storage
│       ├── decomposer.py       # Reward component decomposition
│       └── detectors.py        # Hacking detection algorithms
│
├── tests/
│   ├── test_collector.py       # 21 tests
│   ├── test_decomposer.py      # 28 tests
│   └── test_detectors.py       # 29 tests
│
├── examples/
│   ├── cartpole_basic.py       # Phase 1 demo
│   └── cartpole_hacking_demo.py # Phase 2 demo
│
├── setup.py                    # Package installation
├── requirements.txt            # Core dependencies
└── requirements-dev.txt        # Dev dependencies
```

---

## 📊 Implemented Features

### ✅ Phase 1: Core Tracking (Complete)

- **DataCollector**: SQLite-based step & episode storage
- **RewardDecomposer**: Component tracking with online statistics
- Welford's algorithm for efficient variance computation
- Support for auto-extraction from info dicts
- Query interface for data retrieval

### ✅ Phase 2: Hacking Detection (Complete)

Six detector types:
1. **State Cycling**: Detects degenerate state loops
2. **Action Repetition**: Identifies action repetition exploits
3. **Component Imbalance**: Tracks reward component dominance
4. **Reward Spiking**: Z-score based anomaly detection
5. **Boundary Exploitation**: Detects boundary value exploits
6. **Hacking Score**: Aggregated 0-1 likelihood score

---

## 🔧 Troubleshooting

### Import errors

```bash
# Make sure you're in the repo root and run:
pip install -e .

# Or set PYTHONPATH:
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Missing dependencies

```bash
# Install all dependencies
pip install -r requirements-dev.txt
```

### Tests fail with "No module named 'reward_scope'"

```bash
# Run from repo root, not from tests/ directory
cd /path/to/reward-forensics
pytest tests/ -v
```

### Gymnasium version issues

```bash
# Upgrade gymnasium
pip install --upgrade gymnasium
```

---

## 🎯 Development Workflow

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 3. Make changes to code

# 4. Run tests
pytest tests/ -v

# 5. Format code (optional)
black reward_scope/ tests/ examples/

# 6. Lint code (optional)
ruff check reward_scope/ tests/
```

---

## 📚 Next Steps

### Phase 3: Integrations (Planned)
- Gymnasium environment wrapper
- Stable-Baselines3 callback
- Integration examples with PPO/DQN

### Phase 4: Dashboard (Planned)
- FastAPI backend
- Real-time visualization with HTMX
- WebSocket live updates
- Alert management UI

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🙏 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Ensure all tests pass (`pytest tests/ -v`)
5. Submit a pull request

---

## 📮 Questions?

Open an issue on GitHub or check the examples/ directory for usage patterns.
