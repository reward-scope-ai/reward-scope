# Hacking Detection

RewardScope includes 5 detectors that identify common reward hacking patterns.

## What is Reward Hacking?

Reward hacking occurs when an agent finds an unintended way to maximize reward that doesn't align with the task objective. Common examples:
- Agent spinning in circles because angular velocity is rewarded
- Exploiting physics glitches for infinite reward
- Focusing on one easy reward component while ignoring others

## Detectors

### 1. Action Repetition Detector

**Detects:** Agent taking the same action repeatedly

**Symptoms:**
- > 90% of recent actions are identical
- Low action entropy
- High reward from constant action

**Example:** Agent always accelerates because velocity is rewarded without directional control.

**Tuning:**
```python
RewardScopeWrapper(
    env,
    enable_action_repetition=True,
    # Default thresholds:
    # window_size=50
    # repetition_threshold=0.9
)
```

**Fix:** Add penalties for action changes, or include directional objectives.

### 2. State Cycling Detector

**Detects:** Agent finding a degenerate loop of states

**Symptoms:**
- Repeating observation patterns (cycles of length 3-20)
- High reward from state cycle
- Low state diversity

**Example:** Robotic arm oscillating back and forth because each movement gives reward.

**Tuning:**
```python
# Default thresholds:
# cycle_threshold=0.8 (similarity)
# min_cycle_length=3
# max_cycle_length=20
```

**Fix:** Add task completion criteria, penalize oscillations, reward progress.

### 3. Component Imbalance Detector

**Detects:** One reward component dominating others (>80%)

**Symptoms:**
- Single component contributes > 80% of total reward
- Other components consistently near zero
- Agent ignores multi-objective nature of task

**Example:** In locomotion, agent maximizes velocity while ignoring energy efficiency and stability.

**Tuning:**
```python
# Default thresholds:
# dominance_threshold=0.8 (80%)
# imbalance_episodes=5 (flag after N episodes)
```

**Fix:** Rebalance reward weights, increase penalties for neglected objectives.

### 4. Reward Spiking Detector

**Detects:** Unnatural reward patterns or glitch states

**Symptoms:**
- Sudden large reward spikes (> 5σ from mean)
- Reward variance much higher than expected
- Bimodal distribution (exploit vs non-exploit)

**Example:** Agent finds a physics glitch state that gives massive reward.

**Tuning:**
```python
# Default thresholds:
# spike_std_threshold=5.0 (# of std devs)
# variance_ratio_threshold=10.0
```

**Fix:** Debug the reward function, add reward clipping, fix physics issues.

### 5. Boundary Exploitation Detector

**Detects:** Agent exploiting state/action space boundaries

**Symptoms:**
- Observations frequently at min/max values (>50% of time)
- Actions saturated at limits
- Reward correlated with boundary proximity

**Example:** Agent pushes joint to limit where physics solver breaks down.

**Tuning:**
```python
# Default thresholds:
# boundary_threshold=0.95 (95% of max range)
# boundary_frequency_threshold=0.5 (50% of time)
```

**Fix:** Add boundary penalties, fix physics at boundaries, expand valid range.

## Adaptive Baselines (Two-Layer Detection)

Static thresholds can produce false positives, especially in the early training phases where behavior is naturally variable. Adaptive baselines add a second layer of detection that learns what's "normal" for your specific training run.

### How It Works

1. **Static detector fires + baseline abnormal** → **ALERT** (confirmed)
2. **Static detector fires + baseline normal** → **SUPPRESSED** (likely false positive)
3. **Static detector doesn't fire + baseline abnormal** → **WARNING** (unusual but not severe)
4. **Static detector doesn't fire + baseline normal** → No alert

### Parameters

```python
env = RewardScopeWrapper(
    env,
    adaptive_baseline=True,          # Enable two-layer detection (default)
    baseline_window=50,              # Rolling window for statistics
    baseline_warmup=20,              # Deprecated, use min/max_warmup
    baseline_sensitivity=2.0,        # Std devs for "abnormal" threshold
    min_warmup_episodes=10,          # Min episodes before baseline activates
    max_warmup_episodes=50,          # Max warmup (safety valve)
    stability_threshold=0.1,         # Variance threshold for auto-calibration
)
```

### Confidence Scores

Alerts include a confidence score (0.0 to 1.0) based on how far the metric deviates from baseline:

- **2σ deviation** → 0.50 confidence
- **3σ deviation** → 0.69 confidence
- **4σ deviation** → 0.80 confidence

Access confidence in alert output:

```python
for alert in alerts:
    print(f"{alert.type.value}: confidence={alert.confidence:.2f}")
```

### Auto-Calibration

By default, warmup ends automatically when baseline statistics stabilize (low variance in recent episodes) rather than waiting for a fixed number of episodes. This reduces false positives during stable environments while protecting against environments that need more warmup time.

### Known Limitation

If an agent gradually drifts into hacking over 100+ episodes, the baseline drifts with it. The static detectors are the backstop for this case.

### Example Output

```
[RewardScope] Episode 5 started (baseline warmup: 80%)
[RewardScope] Episode 6 complete: reward=23.00, length=23
[RewardScope] WARNING (confidence=0.56): baseline_deviation: Unusual reward: 43.00 is 2.3σ from baseline
```

## Hacking Score

Each detector has a 0-1 severity score. The overall hacking score is:

```
hacking_score = max(detector_severities)
```

- **0.0-0.3**: Low likelihood of hacking ✓
- **0.3-0.7**: Potential issues ⚠️
- **0.7-1.0**: High likelihood of hacking 🚨

Access the score:

```python
score = env.get_hacking_score()
alerts = env.get_alerts()
```

## Using Alerts

```python
for alert in alerts:
    print(f"[{alert.type.value}] Severity: {alert.severity:.2f}")
    print(f"  Description: {alert.description}")
    print(f"  Evidence: {alert.evidence}")
    print(f"  Suggested fix: {alert.suggested_fix}")
```

## Tuning Detectors

### Disable Specific Detectors

```python
RewardScopeWrapper(
    env,
    enable_state_cycling=False,  # Too many false positives
    enable_action_repetition=True,
    enable_component_imbalance=True,
    enable_reward_spiking=True,
    enable_boundary_exploitation=True,
)
```

### Custom Thresholds

For lower-level control:

```python
from reward_scope.core.detectors import HackingDetectorSuite

detector_suite = HackingDetectorSuite(
    enable_action_repetition=True,
    # ... other flags
)

# Access individual detectors
action_detector = detector_suite.detectors[0]
action_detector.repetition_threshold = 0.95  # More strict
```

## Custom Detectors

You can register custom detection functions to catch domain-specific hacking patterns.

### Detector Signature

Each custom detector is a function with the following signature:

```python
from typing import Optional
from reward_scope.core.detectors import HackingAlert, HackingType

def my_detector(
    step: int,
    episode: int,
    obs,
    action,
    reward: float,
    components: dict,
    info: dict
) -> Optional[HackingAlert]:
    """
    Custom detector function.

    Args:
        step: Current step number
        episode: Current episode number
        obs: Current observation
        action: Action taken
        reward: Reward received
        components: Dict of reward components
        info: Environment info dict

    Returns:
        HackingAlert if hacking detected, None otherwise
    """
    # Your detection logic here
    if some_hacking_condition:
        return HackingAlert(
            type=HackingType.CUSTOM,
            severity=0.8,  # 0.0 to 1.0
            step=step,
            episode=episode,
            description="Description of the issue",
            evidence={"key": "value"},
            suggested_fix="How to fix this issue",
        )
    return None
```

### Example: Negative Reward Detector

Detect when an agent consistently receives negative rewards:

```python
from reward_scope.core.detectors import HackingAlert, HackingType

# Track state across calls (use a class for more complex state)
negative_reward_count = 0
total_steps = 0

def negative_reward_detector(step, episode, obs, action, reward, components, info):
    global negative_reward_count, total_steps

    total_steps += 1
    if reward < 0:
        negative_reward_count += 1

    # Check after every 100 steps
    if total_steps >= 100 and total_steps % 100 == 0:
        negative_rate = negative_reward_count / total_steps
        if negative_rate > 0.8:  # 80% negative rewards
            return HackingAlert(
                type=HackingType.CUSTOM,
                severity=negative_rate,
                step=step,
                episode=episode,
                description=f"Agent receiving {negative_rate:.1%} negative rewards",
                evidence={
                    "negative_rate": negative_rate,
                    "total_steps": total_steps,
                },
                suggested_fix="Check reward function or agent exploration",
            )
    return None
```

### Example: Velocity Threshold Detector

Detect when a robot moves too fast (domain-specific safety check):

```python
import numpy as np
from reward_scope.core.detectors import HackingAlert, HackingType

MAX_SAFE_VELOCITY = 5.0

def velocity_detector(step, episode, obs, action, reward, components, info):
    # Assumes velocity is stored in info dict
    velocity = info.get("velocity", 0)
    if isinstance(velocity, (list, np.ndarray)):
        velocity = np.linalg.norm(velocity)

    if velocity > MAX_SAFE_VELOCITY:
        severity = min(1.0, velocity / (2 * MAX_SAFE_VELOCITY))
        return HackingAlert(
            type=HackingType.CUSTOM,
            severity=severity,
            step=step,
            episode=episode,
            description=f"Velocity {velocity:.2f} exceeds safe limit {MAX_SAFE_VELOCITY}",
            evidence={
                "velocity": velocity,
                "max_safe": MAX_SAFE_VELOCITY,
            },
            suggested_fix="Add velocity penalty or clipping to reward function",
        )
    return None
```

### Registering Custom Detectors

Pass a list of detector functions to `RewardScopeWrapper` or `RewardScopeCallback`:

```python
from reward_scope.integrations import RewardScopeWrapper

env = RewardScopeWrapper(
    env,
    run_name="custom_detection",
    custom_detectors=[
        negative_reward_detector,
        velocity_detector,
    ],
)
```

Or with Stable-Baselines3:

```python
from reward_scope.integrations import RewardScopeCallback

callback = RewardScopeCallback(
    run_name="custom_detection",
    custom_detectors=[
        negative_reward_detector,
        velocity_detector,
    ],
)
```

### Two-Layer Detection for Custom Detectors

Custom detectors participate in the two-layer detection system. The baseline tracker uses the alert's severity value as the metric for comparison. If the severity is within normal ranges for your training run, the alert may be suppressed as a false positive.

To ensure your custom alerts are always shown (bypass baseline suppression), set a high severity (close to 1.0) for critical issues.

### Best Practices for Custom Detectors

1. **Keep it simple**: Each detector should check one specific condition
2. **Use appropriate severity**: 0.3-0.5 for warnings, 0.7-1.0 for critical issues
3. **Include evidence**: Add relevant data to the `evidence` dict for debugging
4. **Provide actionable fixes**: The `suggested_fix` should be specific and helpful
5. **Handle errors gracefully**: Return `None` if detection fails (don't raise exceptions)
6. **Use stateful classes**: For complex state tracking, use a class with `__call__`:

```python
class StatefulDetector:
    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold
        self.history = []

    def __call__(self, step, episode, obs, action, reward, components, info):
        self.history.append(reward)
        if len(self.history) > 100:
            avg = sum(self.history[-100:]) / 100
            if avg < self.threshold:
                return HackingAlert(...)
        return None

# Use as: custom_detectors=[StatefulDetector(threshold=0.5)]
```

## False Positives

**Action Repetition:**
- Expected in discrete tasks with dominant strategies (e.g., always jump in some platformers)
- Adjust threshold or disable for these tasks

**Boundary Exploitation:**
- Some tasks legitimately use boundaries (e.g., wall following)
- Set appropriate `boundary_threshold` based on task

**Component Imbalance:**
- Expected during early training (exploration focuses on easy rewards)
- Monitor over multiple phases of training

## Best Practices

1. **Baseline First**: Run on a known-good policy to calibrate thresholds
2. **Monitor Trends**: A single alert isn't always a problem—watch for sustained patterns
3. **Context Matters**: What's hacking in one task may be optimal in another
4. **Combine with Evaluation**: Use alerts alongside task-specific metrics
5. **Iterate**: Adjust thresholds based on your environment's characteristics

## Dashboard Alerts Panel

The dashboard shows:
- Recent alerts with episode numbers
- Alert type and severity
- Auto-refreshes every 2 seconds

Click on an alert to see:
- Full evidence dictionary
- Suggested fixes
- Link to relevant episode in timeline

## Example: Debugging a Hacking Agent

```python
# Train agent
env = RewardScopeWrapper(env, run_name="debug_run", verbose=2)
# ... train ...

# Check for hacking
alerts = env.get_alerts()
if alerts:
    print(f"⚠️  Found {len(alerts)} potential issues")
    
    # Group by type
    by_type = {}
    for alert in alerts:
        by_type.setdefault(alert.type.value, []).append(alert)
    
    # Prioritize by frequency
    for alert_type, type_alerts in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"\n{alert_type}: {len(type_alerts)} occurrences")
        print(f"  Suggested fix: {type_alerts[0].suggested_fix}")
```

## Research Context

Reward hacking is a major challenge in RL safety. See:
- [Specification gaming examples](https://docs.google.com/spreadsheets/d/e/2PACX-1vRPiprOaC3HsCf5Tuum8bRfzYUiKLRqJmbOoC-32JorNdfyTiRRsR7Ea5eWtvsWzuxo8bjOxCG84dAg/pubhtml) (DeepMind)
- Anthropic's work on emergent misalignment
- Amodei et al., "Concrete Problems in AI Safety" (2016)

