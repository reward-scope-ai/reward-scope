# Quick Start

Get started with RewardScope in 5 minutes.

## Installation

```bash
pip install reward-scope
```

Or install from source:

```bash
git clone https://github.com/your-org/reward-scope
cd reward-scope
pip install -e .
```

## Basic Usage

### 1. Wrap Your Environment

```python
import gymnasium as gym
from reward_scope.integrations import RewardScopeWrapper

env = gym.make("CartPole-v1")
env = RewardScopeWrapper(
    env,
    run_name="my_experiment",
    verbose=1
)

# Train as usual
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### 2. View the Dashboard

```bash
reward-scope dashboard --run-name my_experiment --data-dir ./reward_scope_data
```

Open http://localhost:8050 in your browser.

## With Stable-Baselines3

```python
from stable_baselines3 import PPO
from reward_scope.integrations import RewardScopeCallback

callback = RewardScopeCallback(
    run_name="ppo_experiment",
    start_dashboard=True,  # Auto-start dashboard!
    verbose=1
)

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=50000, callback=callback)
```

The dashboard starts automatically at http://localhost:8050.

## Tracking Reward Components

```python
from reward_scope.integrations import RewardScopeWrapper

# Define component functions
component_fns = {
    "distance": lambda obs, act, info: info.get("distance_reward", 0.0),
    "energy": lambda obs, act, info: info.get("energy_cost", 0.0),
}

env = RewardScopeWrapper(
    env,
    run_name="my_experiment",
    component_fns=component_fns,
    verbose=1
)
```

Or auto-extract from info dict:

```python
env = RewardScopeWrapper(
    env,
    run_name="my_experiment",
    auto_extract_prefix="reward_",  # Extracts reward_forward, reward_ctrl, etc.
    verbose=1
)
```

## Next Steps

- [Reward Components Guide](reward_components.md) - Learn about reward decomposition
- [Hacking Detection Guide](hacking_detection.md) - Understanding the detectors
- [API Reference](api_reference.md) - Detailed API documentation
- Check out the [examples](../examples/) directory

## Troubleshooting

### Dashboard not showing data

Make sure to wait a few seconds for data to be written. The dashboard polls every 2-5 seconds.

### Import errors

RewardScope has minimal dependencies by default. For full functionality:

```bash
pip install reward-scope[all]
```

This includes stable-baselines3 and development tools.

