# Quick Start

Get started with RewardScope in 5 minutes.

## Installation

```bash
pip install reward-scope
```

Or install from source:

```bash
git clone https://github.com/reward-scope-ai/reward-scope
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
reward-scope dashboard --data-dir ./reward_scope_data
# Select your run from the sidebar
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

## With WandB

Log RewardScope metrics to Weights & Biases:

```python
import wandb
from stable_baselines3 import PPO
from reward_scope.integrations import RewardScopeCallback

# Initialize WandB first
wandb.init(project="my-rl-project", name="ppo_experiment")

# Enable WandB logging in RewardScope
callback = RewardScopeCallback(
    run_name="ppo_experiment",
    wandb_logging=True,  # Log to WandB!
    verbose=1
)

model = PPO("MlpPolicy", env)
model.learn(total_timesteps=50000, callback=callback)
```

WandB integration also works with the Gymnasium wrapper:

```python
import wandb
import gymnasium as gym
from reward_scope.integrations import RewardScopeWrapper

wandb.init(project="my-rl-project", name="my_experiment")

env = gym.make("CartPole-v1")
env = RewardScopeWrapper(
    env,
    run_name="my_experiment",
    wandb_logging=True,  # Log to WandB!
    verbose=1
)

# Train as usual...
```

**Metrics logged per episode:**
- `rewardscope/hacking_score` - Overall hacking score (0-1)
- `rewardscope/episode_reward` - Total episode reward
- `rewardscope/episode_length` - Steps in episode
- `rewardscope/component/{name}` - Each reward component total
- `rewardscope/alerts_count` - Number of alerts

High severity alerts (>0.7) are logged as WandB warnings.

**Installation:**
```bash
pip install reward-scope[wandb]
```

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
- [Hacking Detection Guide](hacking_detection.md) - Understanding the detectors and adaptive baselines
- [API Reference](api_reference.md) - Detailed API documentation
- Check out the [examples](../examples/) directory

**Note:** RewardScope includes adaptive baselines by default to reduce false positives. The system learns "normal" patterns during warmup and modulates alerts accordingly. See [Hacking Detection Guide](hacking_detection.md#adaptive-baselines-two-layer-detection) for details.

## Troubleshooting

### Dashboard not showing data

Make sure to wait a few seconds for data to be written. The dashboard polls every 2-5 seconds.

### Import errors

RewardScope has minimal dependencies by default. For full functionality:

```bash
pip install reward-scope[all]
```

This includes stable-baselines3 and development tools.

