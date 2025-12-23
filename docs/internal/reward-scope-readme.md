# RewardScope MVP - Quick Reference

## What We're Building

A Python SDK + web dashboard for RL reward debugging:
- **Reward decomposition**: Break down composite rewards into trackable components
- **Hacking detection**: Automatically detect when agents exploit reward functions
- **Real-time dashboard**: Visualize training with live updates

## Target Integration

1. **Stable-Baselines3** (primary)
2. **Gymnasium** (wrapper)
3. **Isaac Lab** (future)

## Core Features (MVP)

### 1. Reward Decomposition
```python
callback = RewardScopeCallback(
    reward_components={
        "distance": lambda o, a, i: i.get("distance_reward", 0),
        "energy": lambda o, a, i: i.get("energy_reward", 0),
    },
    # OR auto-extract from info dict
    auto_extract_prefix="reward_",
)
```

### 2. Hacking Detection Types
- **State Cycling**: Agent finds degenerate loop states
- **Action Repetition**: Exploiting reward through repeated actions
- **Component Imbalance**: One reward component dominates
- **Reward Spiking**: Unnatural reward spikes
- **Boundary Exploitation**: Staying at state/action limits

### 3. Dashboard
- Real-time reward timeline
- Component breakdown (pie chart)
- Episode history (bar chart)
- Alert panel with suggested fixes
- WebSocket for live updates

## File Structure

```
reward-scope/
├── reward_scope/
│   ├── core/
│   │   ├── collector.py      # SQLite data storage
│   │   ├── decomposer.py     # Reward component tracking
│   │   └── detectors.py      # Hacking detection
│   ├── integrations/
│   │   ├── stable_baselines.py
│   │   └── gymnasium.py
│   ├── dashboard/
│   │   ├── app.py            # FastAPI server
│   │   └── templates/
│   └── cli.py
├── examples/
│   ├── cartpole_basic.py
│   ├── lunarlander_hacking.py
│   └── mujoco_ant.py
└── tests/
```

## Implementation Order

| Week | Focus | Deliverable |
|------|-------|-------------|
| 1-2 | Core SDK | collector.py, decomposer.py, tests |
| 2-3 | Detectors | All 5 detector classes, tests |
| 3 | Integrations | SB3 callback, Gymnasium wrapper |
| 4 | Dashboard | FastAPI app, WebSocket, charts |
| 5 | Polish | Examples, docs, PyPI prep |

## Usage Example

```python
from stable_baselines3 import PPO
from reward_scope import RewardScopeCallback

env = gym.make("Ant-v4")

callback = RewardScopeCallback(
    run_name="ant_experiment",
    auto_extract_prefix="reward_",  # Ant has reward_forward, reward_ctrl, etc.
    start_dashboard=True,
)

model = PPO("MlpPolicy", env)
model.learn(100000, callback=callback)

# Check results
print(f"Hacking score: {callback.get_hacking_score()}")
for alert in callback.get_alerts():
    print(f"[{alert.type}] {alert.description}")
```

## Tech Stack

- **Backend**: Python 3.9+, FastAPI, SQLite
- **Frontend**: HTMX, Chart.js (no React/npm needed)
- **Testing**: pytest, pytest-asyncio
- **Environments**: Gymnasium, Stable-Baselines3

## Success Criteria

1. Works on CartPole, LunarLander, MuJoCo Ant
2. Detects 3+ hacking patterns
3. Dashboard updates < 200ms latency
4. < 5% training overhead
5. 10-minute onboarding time

## Key Design Decisions

1. **SQLite for storage** - Simple, no external dependencies
2. **HTMX over React** - Faster to build, no JS build step
3. **Callback pattern** - Standard SB3 integration
4. **Optional observation logging** - Can be huge, off by default
5. **Configurable thresholds** - Users can tune sensitivity

---

**Full spec**: See `reward-forensics-mvp-spec.md` for complete implementation details.
