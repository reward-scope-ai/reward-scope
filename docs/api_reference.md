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
    disable_detectors=None,           # List of detector names to disable
    # Alert callbacks
    on_alert=None,                    # Callback or list of callbacks for alerts
    # Two-layer detection (adaptive baselines)
    adaptive_baseline=True,           # Enable two-layer detection
    baseline_window=50,               # Rolling window size
    baseline_warmup=20,               # Deprecated, use min/max_warmup
    baseline_sensitivity=2.0,         # Std devs for "abnormal"
    min_warmup_episodes=10,           # Min warmup before auto-calibration
    max_warmup_episodes=50,           # Max warmup (safety valve)
    stability_threshold=0.1,          # Variance threshold for stability
    # Custom detectors
    custom_detectors=None,            # List of custom detector functions
    # Dashboard
    start_dashboard=False,
    dashboard_port=8050,
    # WandB
    wandb_logging=False,
    # Other
    verbose=0,                        # 0=silent, 1=episode summaries, 2=alerts, 3=debug
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
- `export_alerts(path, format=None)` - Export alerts to JSON or CSV
- `export_episode_history(path, format=None)` - Export episode history to JSON or CSV
- `print_summary()` - Print formatted summary of the run

**Info Dict Extensions:**

The wrapper adds these keys to the `info` dict:
- `reward_components`: Dict[str, float] - Decomposed components
- `hacking_alerts`: List[dict] - Alerts from this step (includes `confidence` and `alert_severity`)
- `hacking_score`: float - Current hacking score
- `baseline_active`: bool - Whether adaptive baseline is active
- `baseline_warmup_progress`: float - Warmup progress (0.0 to 1.0)
- `suppressed_count`: int - Number of suppressed false positives
- `warning_count`: int - Number of soft warnings

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
    disable_detectors=None,           # List of detector names to disable
    # Alert callbacks
    on_alert=None,                    # Callback or list of callbacks for alerts
    # Two-layer detection (adaptive baselines)
    adaptive_baseline=True,           # Enable two-layer detection
    baseline_window=50,               # Rolling window size
    baseline_warmup=20,               # Deprecated, use min/max_warmup
    baseline_sensitivity=2.0,         # Std devs for "abnormal"
    min_warmup_episodes=10,           # Min warmup before auto-calibration
    max_warmup_episodes=50,           # Max warmup (safety valve)
    stability_threshold=0.1,          # Variance threshold for stability
    # Custom detectors
    custom_detectors=None,            # List of custom detector functions
    # Dashboard
    start_dashboard=False,
    dashboard_port=8050,
    # WandB
    wandb_logging=False,
    # Other
    verbose=1,                        # 0=silent, 1=episode summaries, 2=alerts, 3=debug
)

model.learn(total_timesteps=50000, callback=callback)
```

**Methods:**
- `get_alerts() -> List[HackingAlert]` - Get detected alerts
- `get_hacking_score() -> float` - Get hacking score
- `export_alerts(path, format=None)` - Export alerts to JSON or CSV
- `export_episode_history(path, format=None)` - Export episode history to JSON or CSV
- `print_summary()` - Print formatted summary of the run

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
from reward_scope.core.baseline import AlertSeverity

alert = HackingAlert(
    type=HackingType.ACTION_REPETITION,
    severity=0.8,  # 0.0-1.0
    step=1000,
    episode=5,
    description="Action repetition detected: 95% identical",
    evidence={"repetition_rate": 0.95, "action": 1},
    suggested_fix="Add action diversity bonus or randomize actions",
    # Two-layer detection fields
    alert_severity=AlertSeverity.ALERT,  # ALERT, WARNING, or SUPPRESSED
    baseline_z_score=2.5,                # Z-score against baseline
    confidence=0.61,                     # Confidence score (0.0-1.0)
)
```

**AlertSeverity Values:**
- `AlertSeverity.ALERT` - Confirmed by both static and baseline
- `AlertSeverity.WARNING` - Baseline abnormal but static didn't fire
- `AlertSeverity.SUPPRESSED` - Static fired but baseline says normal (likely false positive)

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

## Callback Signatures

### Alert Callback

The `on_alert` parameter accepts a function or list of functions with this signature:

```python
def my_alert_handler(alert: HackingAlert) -> None:
    """
    Called when a hacking alert is detected.

    Args:
        alert: The HackingAlert object containing:
            - type: HackingType enum
            - severity: float (0.0-1.0)
            - alert_severity: AlertSeverity (ALERT, WARNING, or SUPPRESSED)
            - description: str
            - evidence: dict
            - suggested_fix: str
            - confidence: Optional[float]
            - baseline_z_score: Optional[float]

    Note: Only called for ALERT and WARNING severity (not SUPPRESSED)
    """
    print(f"Alert: {alert.type.value} - {alert.description}")
```

**Usage:**

```python
def log_alert(alert):
    with open("alerts.log", "a") as f:
        f.write(f"{alert.type.value}: {alert.description}\n")

env = RewardScopeWrapper(
    env,
    on_alert=log_alert  # Single callback
)

# Or multiple callbacks:
env = RewardScopeWrapper(
    env,
    on_alert=[log_alert, send_to_slack, trigger_early_stop]
)
```

### Custom Detector Function

See [Hacking Detection Guide](hacking_detection.md#custom-detectors) for the full signature and examples.

## Verbose Levels

The `verbose` parameter controls output verbosity:

- **0** - Silent (no output)
- **1** - Episode summaries and final summary (default for callback)
  - Episode completion messages
  - Baseline warmup progress
  - Final summary with alert counts
- **2** - Step-level alerts as they fire
  - All verbose=1 output
  - Real-time alerts with severity and confidence
  - Episode-level alerts
- **3** - Debug output
  - All verbose=2 output
  - Alert evidence dictionaries
  - Suggested fixes
  - Baseline z-scores
  - Component breakdowns

**Example output at different levels:**

```
# verbose=1
[RewardScope] Episode 5 complete: reward=23.00, length=23, hacking_score=0.120
[RewardScope] Episode 10 started (baseline warmup: 50%)

# verbose=2 (includes verbose=1 plus)
[RewardScope] ALERT (confidence=0.81): action_repetition: 95% of actions identical

# verbose=3 (includes verbose=2 plus)
  Evidence: {'repetition_rate': 0.95, 'action': 1}
  Fix: Add action diversity bonus or randomize actions
  Baseline z-score: 3.42
```

## Type Hints

RewardScope includes type hints for all public APIs. Use with mypy:

```bash
pip install mypy
mypy your_code.py
```

## Environment Variables

None currently. Configuration is done via constructor arguments or CLI flags.

