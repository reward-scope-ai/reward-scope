# RewardScope 🔬

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Real-time reward debugging and hacking detection for reinforcement learning.**

RewardScope helps RL practitioners debug reward functions by tracking reward components, detecting common exploitation patterns, and visualizing training dynamics through a live web dashboard.

![Dashboard Preview](docs/dashboard-preview.png)
*Dashboard showing reward timeline, component breakdown, and hacking alerts*

## Why RewardScope?

Reward functions are notoriously difficult to get right. Even well-intentioned designs can lead to [reward hacking](https://docs.google.com/spreadsheets/d/e/2PACX-1vRPiprOaC3HsCf5Tuum8bRfzYUiKLRqJmbOoC-32JorNdfyTiRRsR7Ea5eWtvsWzuxo8bjOxCG84dAg/pubhtml)—when agents find unintended ways to maximize reward that don't align with your objectives.

Research from [Anthropic](https://www.anthropic.com/) and others shows that reward misspecification can lead to emergent misalignment, especially as models become more capable. RewardScope provides real-time visibility into what your agent is actually learning.

## Features

- 🎯 **Reward Decomposition** - Track individual reward components separately
- 🚨 **Hacking Detection** - 5 detectors for common exploitation patterns
- 📊 **Live Dashboard** - Real-time visualization with FastAPI + Chart.js
- 🔌 **Easy Integration** - Works with Gymnasium, Stable-Baselines3, and Isaac Lab
- 💾 **Persistent Storage** - SQLite backend for post-training analysis
- 🎮 **CLI Tools** - Dashboard, reports, and run management

## Quick Start

### Installation

```bash
pip install reward-scope
```

### Basic Usage

Wrap your Gymnasium environment:

```python
import gymnasium as gym
from reward_scope.integrations import RewardScopeWrapper

env = gym.make("CartPole-v1")
env = RewardScopeWrapper(env, run_name="my_experiment")

# Train as usual
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

View the dashboard:

```bash
reward-scope dashboard --run-name my_experiment
```

Open http://localhost:8050 in your browser.

### With Stable-Baselines3

```python
from stable_baselines3 import PPO
from reward_scope.integrations import RewardScopeCallback

callback = RewardScopeCallback(
    run_name="ppo_experiment",
    start_dashboard=True,  # Auto-start dashboard!
)

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=50000, callback=callback)
```

The dashboard starts automatically at http://localhost:8050.

## What RewardScope Detects

### 1. Action Repetition
Agent takes the same action repeatedly (e.g., always accelerating)

### 2. State Cycling
Agent finds degenerate loop of states (e.g., spinning in circles)

### 3. Component Imbalance
One reward component dominates others (>80%)

### 4. Reward Spiking
Unnatural reward patterns or glitch states

### 5. Boundary Exploitation
Agent exploits state/action space boundaries

Each detector provides:
- **Severity score** (0-1)
- **Evidence** (detailed metrics)
- **Suggested fix** (how to address the issue)

## Reward Component Tracking

Track individual reward terms:

```python
component_fns = {
    "distance": lambda obs, act, info: info.get("distance_reward"),
    "energy": lambda obs, act, info: -0.01 * (act ** 2).sum(),
    "stability": lambda obs, act, info: -1.0 if info.get("fallen") else 0.0,
}

env = RewardScopeWrapper(
    env,
    run_name="my_experiment",
    component_fns=component_fns,
)
```

Or auto-extract from `info` dict:

```python
env = RewardScopeWrapper(
    env,
    auto_extract_prefix="reward_",  # Extracts reward_forward, reward_ctrl, etc.
)
```

## Documentation

- [Quick Start Guide](docs/quickstart.md) - Get running in 5 minutes
- [Reward Components](docs/reward_components.md) - How to track components
- [Hacking Detection](docs/hacking_detection.md) - Understanding the detectors
- [API Reference](docs/api_reference.md) - Full API documentation

## Examples

Check out the [examples/](examples/) directory:

- `cartpole_basic.py` - Simplest example to verify installation
- `lunarlander_components.py` - Multi-component reward tracking
- `mujoco_ant.py` - Complex reward with Stable-Baselines3

## CLI Commands

```bash
# Start dashboard
reward-scope dashboard --run-name my_experiment

# List all runs
reward-scope list-runs ./reward_scope_data

# Generate static report
reward-scope report ./reward_scope_data --output report.html
```

## Dashboard Features

The live dashboard shows:
- **Reward Timeline** - Line chart of reward per step
- **Component Breakdown** - Pie chart of component contributions
- **Episode History** - Bar chart of episode rewards
- **Live Stats** - Current step, episode, hacking score
- **Alerts Panel** - Real-time hacking detection alerts

All charts update in real-time via WebSocket (10Hz).

## Requirements

- Python 3.8+
- gymnasium
- numpy
- fastapi (for dashboard)
- uvicorn (for dashboard)

Optional:
- stable-baselines3 (for SB3 integration)
- mujoco (for MuJoCo environments)

## Development

```bash
git clone https://github.com/your-org/reward-scope
cd reward-scope
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run examples
python examples/cartpole_basic.py
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Citation

If you use RewardScope in your research, please cite:

```bibtex
@software{rewardscope2024,
  title = {RewardScope: Real-time Reward Debugging for Reinforcement Learning},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-org/reward-scope}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by research on reward misspecification and specification gaming
- Built with FastAPI, Gymnasium, and Stable-Baselines3
- Dashboard powered by HTMX and Chart.js (no build step!)

## Related Work

- [Specification gaming examples](https://docs.google.com/spreadsheets/d/e/2PACX-1vRPiprOaC3HsCf5Tuum8bRfzYUiKLRqJmbOoC-32JorNdfyTiRRsR7Ea5eWtvsWzuxo8bjOxCG84dAg/pubhtml) (DeepMind)
- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565) (Amodei et al., 2016)
- [Anthropic's research on AI alignment](https://www.anthropic.com/research)

---

**Made with ❤️ for safer RL development**
