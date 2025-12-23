# API Reference

## Core Classes

### DataCollector

Collects and stores training data in SQLite.

```python
from reward_scope.core.collector import DataCollector

collector = DataCollector(
    run_name="my_experiment",
    storage_dir="./reward_scope_data",
    buffer_size=1000,  # Flush to DB every N steps
)
```

**Methods:**
- `log_step(data: StepData)` - Log a single step
- `end_episode() -> EpisodeData` - Signal episode end, compute aggregates
- `get_recent_steps(n: int) -> List[StepData]` - Get last N steps
- `get_episode_history(n: int) -> List[EpisodeData]` - Get last N episodes
- `query_steps(start_step, end_step, episode)` - Query with filters
- `close()` - Flush buffers and close database

### RewardDecomposer

Decomposes rewards into trackable components.

```python
from reward_scope.core.decomposer import RewardDecomposer

decomposer = RewardDecomposer(
    auto_extract_prefix="reward_",  # Auto-extract from info
    track_residual=True,  # Track unexplained reward
)
```

**Methods:**
- `register_component(name, compute_fn, description, expected_range, weight)` - Add component
- `decompose(observation, action, total_reward, info) -> Dict[str, float]` - Decompose reward
- `get_component_stats() -> Dict[str, Dict]` - Get running statistics
- `check_dominance(threshold=0.8) -> List[str]` - Check if component dominates

### HackingDetectorSuite

Runs all hacking detectors.

```python
from reward_scope.core.detectors import HackingDetectorSuite

suite = HackingDetectorSuite(
    enable_state_cycling=True,
    enable_action_repetition=True,
    enable_component_imbalance=True,
    enable_reward_spiking=True,
    enable_boundary_exploitation=True,
    observation_bounds=(low, high),  # Optional
    action_bounds=(low, high),  # Optional
)
```

**Methods:**
- `update(step, episode, observation, action, reward, components, done, info) -> List[HackingAlert]`
- `on_episode_end(episode_stats) -> List[HackingAlert]` - Episode-level checks
- `get_all_alerts() -> List[HackingAlert]` - Get all historical alerts
- `get_hacking_score() -> float` - Overall hacking score (0-1)
- `reset()` - Reset detector state

## Integration Classes

### RewardScopeWrapper

Gymnasium environment wrapper.

```python
from reward_scope.integrations import RewardScopeWrapper

env = RewardScopeWrapper(
    env,
    run_name="my_run",
    storage_dir="./reward_scope_data",
    # Decomposer settings
    auto_extract_prefix=None,
    component_fns=None,
    # Detector settings
    enable_state_cycling=True,
    enable_action_repetition=True,
    enable_component_imbalance=True,
    enable_reward_spiking=True,
    enable_boundary_exploitation=True,
    # Dashboard
    start_dashboard=False,
    dashboard_port=8050,
    # Other
    verbose=0,
)
```

**Methods:**
- `reset()` - Reset environment (standard Gym interface)
- `step(action)` - Take step (standard Gym interface)
- `close()` - Close environment and flush data
- `get_alerts() -> List[HackingAlert]` - Get detected alerts
- `get_hacking_score() -> float` - Get hacking score
- `get_component_stats() -> Dict` - Get component statistics
- `get_episode_history(n) -> List[EpisodeData]` - Get episode history

**Info Dict Extensions:**

The wrapper adds these keys to the `info` dict:
- `reward_components`: Dict[str, float] - Decomposed components
- `hacking_alerts`: List[HackingAlert] - Alerts from this step
- `hacking_score`: float - Current hacking score

### RewardScopeCallback

Stable-Baselines3 callback.

```python
from reward_scope.integrations import RewardScopeCallback

callback = RewardScopeCallback(
    run_name="my_run",
    storage_dir="./reward_scope_data",
    # Decomposer settings
    auto_extract_prefix=None,
    component_fns=None,
    # Detector settings
    enable_state_cycling=True,
    enable_action_repetition=True,
    enable_component_imbalance=True,
    enable_reward_spiking=True,
    enable_boundary_exploitation=True,
    observation_bounds=None,
    action_bounds=None,
    # Dashboard
    start_dashboard=False,
    dashboard_port=8050,
    # Other
    verbose=1,
)

model.learn(total_timesteps=50000, callback=callback)
```

**Methods:**
- `get_alerts() -> List[HackingAlert]` - Get detected alerts
- `get_hacking_score() -> float` - Get hacking score

**Attributes:**
- `collector`: DataCollector instance
- `decomposer`: RewardDecomposer instance
- `detector_suite`: HackingDetectorSuite instance
- `step_count`: int - Total steps
- `episode_count`: int - Total episodes

## Data Classes

### StepData

```python
from reward_scope.core.collector import StepData

step_data = StepData(
    step=0,
    episode=0,
    timestamp=time.time(),
    observation=obs,
    action=action,
    reward=reward,
    done=terminated,
    truncated=truncated,
    info=info,
    reward_components={"comp1": 1.0},
    value_estimate=None,  # Optional
    action_probs=None,  # Optional
)
```

### EpisodeData

```python
from reward_scope.core.collector import EpisodeData

episode_data = EpisodeData(
    episode=0,
    total_reward=100.0,
    length=200,
    start_time=time.time(),
    end_time=time.time() + 10,
    component_totals={"comp1": 100.0},
    hacking_score=0.0,
    hacking_flags=["action_repetition"],
)
```

### HackingAlert

```python
from reward_scope.core.detectors import HackingAlert, HackingType

alert = HackingAlert(
    type=HackingType.ACTION_REPETITION,
    severity=0.8,  # 0.0-1.0
    step=1000,
    episode=5,
    description="Action repetition detected: 95% identical",
    evidence={"repetition_rate": 0.95, "action": 1},
    suggested_fix="Add action diversity bonus or randomize actions",
)
```

## CLI Commands

### dashboard

Start the web dashboard:

```bash
reward-scope dashboard \
    --data-dir ./reward_scope_data \
    --port 8050
```

**Note:** Select training runs from the sidebar in the dashboard UI.

### list-runs

List available training runs:

```bash
reward-scope list-runs ./reward_scope_data
```

### report

Generate static HTML report:

```bash
reward-scope report ./reward_scope_data \\
    --run-name my_experiment \\
    --output report.html
```

## Dashboard API Endpoints

When running the dashboard, these endpoints are available:

- `GET /` - Main dashboard page
- `GET /api/reward-history?n=100` - Recent step rewards
- `GET /api/component-breakdown?n=100` - Component aggregation
- `GET /api/episode-history?n=50` - Episode statistics
- `GET /api/alerts` - Hacking alerts
- `WebSocket /ws/live` - Real-time updates (10Hz)

## Type Hints

RewardScope includes type hints for all public APIs. Use with mypy:

```bash
pip install mypy
mypy your_code.py
```

## Environment Variables

None currently. Configuration is done via constructor arguments or CLI flags.

