# Reward Components

Learn how to decompose and track reward components in your RL training.

## Why Track Components?

Complex reward functions are often composed of multiple terms:

```
total_reward = w1 * distance_to_goal + w2 * energy_cost + w3 * stability
```

Tracking each component separately helps you:
- Identify which components dominate (reward imbalance)
- Debug unintended reward interactions
- Detect when agents exploit one component while ignoring others

## Method 1: Component Functions

Define functions that extract each component:

```python
from reward_scope.integrations import RewardScopeWrapper

component_fns = {
    "distance": lambda obs, act, info: info.get("distance_reward", 0.0),
    "energy": lambda obs, act, info: -0.01 * (act ** 2).sum(),  # Energy penalty
    "stability": lambda obs, act, info: -0.1 if abs(obs[2]) > 0.3 else 0.0,
}

env = RewardScopeWrapper(
    env,
    run_name="my_experiment",
    component_fns=component_fns,
)
```

Each function receives `(observation, action, info)` and returns a float.

## Method 2: Auto-Extract from Info Dict

If your environment provides reward components in the `info` dict:

```python
env = RewardScopeWrapper(
    env,
    run_name="my_experiment",
    auto_extract_prefix="reward_",  # Extracts keys starting with "reward_"
)
```

For example, if `info` contains:
```python
{
    "reward_forward": 0.5,
    "reward_ctrl": -0.1,
    "reward_survive": 1.0,
    "other_stuff": 123  # Ignored
}
```

RewardScope will automatically track `forward`, `ctrl`, and `survive` components.

## Method 3: Register Components Manually

For more control:

```python
from reward_scope.core.decomposer import RewardDecomposer

decomposer = RewardDecomposer()

decomposer.register_component(
    name="distance",
    compute_fn=lambda obs, act, info: -info.get("distance", 0.0),
    description="Negative distance to goal",
    expected_range=(-10.0, 0.0),  # For anomaly detection
    weight=1.0,  # If known
)
```

## Tracking Residuals

By default, RewardScope tracks a "residual" component:

```
residual = total_reward - sum(component_rewards)
```

This helps detect:
- Missing reward components
- Unaccounted rewards (potential bugs)
- Numerical precision issues

Disable with `track_residual=False`.

## Component Statistics

Access statistics for each component:

```python
stats = env.get_component_stats()

for comp_name, comp_stats in stats.items():
    print(f"{comp_name}:")
    print(f"  Mean: {comp_stats['mean']:.4f}")
    print(f"  Std:  {comp_stats['std']:.4f}")
    print(f"  Min:  {comp_stats['min']:.4f}")
    print(f"  Max:  {comp_stats['max']:.4f}")
```

## Dominance Detection

Check if one component dominates:

```python
dominant = decomposer.check_dominance(threshold=0.8)
if dominant:
    print(f"Warning: {dominant} contributes >80% of total reward")
```

This can indicate:
- Reward imbalance (one term too large)
- Agent exploiting a single component
- Need to rebalance reward weights

## Isaac Lab Integration

For Isaac Lab environments with reward config:

```python
from reward_scope.core.decomposer import IsaacLabDecomposer

decomposer = IsaacLabDecomposer.from_reward_cfg(reward_cfg)
```

This automatically registers all reward terms with their weights.

## Best Practices

1. **Start Simple**: Begin with 2-3 major components
2. **Name Clearly**: Use descriptive names like `"distance_to_goal"` not `"comp1"`
3. **Expected Ranges**: Specify ranges to catch anomalies early
4. **Check Residuals**: Large residuals indicate missing components
5. **Monitor Dashboard**: Watch component breakdown chart in real-time

## Example: Locomotion Task

```python
component_fns = {
    "forward_velocity": lambda o, a, i: i.get("velocity_x", 0.0),
    "energy_cost": lambda o, a, i: -0.01 * np.sum(a ** 2),
    "stability": lambda o, a, i: -1.0 if i.get("fallen", False) else 0.0,
    "action_smoothness": lambda o, a, i: -0.001 * np.sum(np.diff(a) ** 2),
}
```

## Troubleshooting

**Components not showing in dashboard:**
- Check that component functions don't raise exceptions
- Verify the info dict contains expected keys
- Enable `verbose=2` to see component values printed

**Large residuals:**
- Your component functions may not sum to the total reward
- Check if the environment adds additional rewards
- Use `track_residual=False` if this is expected

