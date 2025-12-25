"""
Reward Hacking Detection Module

Implements detection algorithms for common reward hacking patterns:
1. Proxy-True Divergence: High reward but poor true objective
2. State Cycling: Agent finds degenerate loop states
3. Action Repetition: Exploiting reward through repeated actions
4. Boundary Exploitation: Staying at state space boundaries
5. Reward Spiking: Unnatural reward patterns
6. Component Imbalance: One component dominates others
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque
import hashlib

from .baselines import BaselineCollector
from .baseline import BaselineTracker, AlertSeverity, classify_alert, zscore_to_confidence


class HackingType(Enum):
    """Types of reward hacking."""
    PROXY_DIVERGENCE = "proxy_divergence"
    STATE_CYCLING = "state_cycling"
    ACTION_REPETITION = "action_repetition"
    BOUNDARY_EXPLOITATION = "boundary_exploitation"
    REWARD_SPIKING = "reward_spiking"
    COMPONENT_IMBALANCE = "component_imbalance"
    BASELINE_DEVIATION = "baseline_deviation"


@dataclass
class HackingAlert:
    """A detected hacking instance."""
    type: HackingType
    severity: float  # 0.0 to 1.0
    step: int
    episode: int
    description: str
    evidence: Dict[str, Any]  # Supporting data
    suggested_fix: str
    # Two-layer detection fields
    alert_severity: AlertSeverity = field(default=AlertSeverity.ALERT)
    baseline_z_score: Optional[float] = None  # z-score against baseline if available
    confidence: Optional[float] = None  # 0.0 to 1.0, based on z-score deviation


class BaseDetector:
    """Base class for hacking detectors."""

    # Each detector should define its baseline metric name
    baseline_metric: Optional[str] = None

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.alerts: List[HackingAlert] = []

    def update(
        self,
        step: int,
        episode: int,
        observation: Any,
        action: Any,
        reward: float,
        reward_components: Dict[str, float],
        done: bool,
        info: Dict[str, Any],
        baseline_tracker: Optional[BaselineTracker] = None,
    ) -> Optional[HackingAlert]:
        """
        Process a new step and check for hacking.
        Returns alert if hacking detected, None otherwise.

        Args:
            baseline_tracker: Optional BaselineTracker for two-layer detection.
                If provided and active, detector uses two-layer logic to classify
                alerts as ALERT (confirmed), SUPPRESSED (false positive), or
                WARNING (baseline abnormal but no static alert).
        """
        raise NotImplementedError

    def _apply_two_layer_logic(
        self,
        alert: HackingAlert,
        metric_value: float,
        baseline_tracker: Optional[BaselineTracker],
    ) -> Optional[HackingAlert]:
        """
        Apply two-layer detection logic to an alert.

        Args:
            alert: The alert from static detection
            metric_value: The metric value to check against baseline
            baseline_tracker: The baseline tracker

        Returns:
            Modified alert (ALERT, SUPPRESSED), or None if should be suppressed
        """
        if baseline_tracker is None or not baseline_tracker.is_active:
            # No baseline active, return alert as-is
            return alert

        if self.baseline_metric is None:
            # Detector doesn't have a baseline metric, return as-is
            return alert

        # Check if baseline considers this abnormal
        baseline_abnormal = baseline_tracker.is_abnormal(self.baseline_metric, metric_value)
        z_score = baseline_tracker.get_z_score(self.baseline_metric, metric_value)
        severity = classify_alert(True, baseline_abnormal)

        # Update alert with two-layer info
        alert.alert_severity = severity
        alert.baseline_z_score = z_score
        alert.confidence = zscore_to_confidence(z_score)

        if severity == AlertSeverity.SUPPRESSED:
            # Track suppression but don't add to alerts list
            baseline_tracker.record_suppressed()
            return None  # Suppress the alert

        return alert

    def reset(self) -> None:
        """Reset detector state (e.g., at episode end)."""
        pass

    def get_episode_metric(self) -> Optional[float]:
        """Get the episode-level metric value for baseline tracking.

        Subclasses should override this to provide their specific metric
        (e.g., state_revisit_rate for StateCyclingDetector).
        """
        return None


class StateCyclingDetector(BaseDetector):
    """
    Detects when agent finds a degenerate cycle of states.

    Symptoms:
    - High reward from repeating same state sequence
    - Low state diversity over time
    - Periodic observation patterns

    Example: Agent learns to spin in circles because angular
    velocity reward outweighs forward progress reward.
    """

    baseline_metric = "state_revisit_rate"

    def __init__(
        self,
        window_size: int = 100,
        cycle_threshold: float = 0.8,  # Similarity threshold
        min_cycle_length: int = 3,
        max_cycle_length: int = 20,
    ):
        super().__init__(window_size)
        self.cycle_threshold = cycle_threshold
        self.min_cycle_length = min_cycle_length
        self.max_cycle_length = max_cycle_length
        self.observation_buffer: deque = deque(maxlen=window_size)
        self.last_alert_step = -1000  # Avoid spamming alerts

        # Track state revisit rate for baseline
        self._unique_states: set = set()
        self._total_states: int = 0

    def update(self, step, episode, observation, action, reward,
               reward_components, done, info,
               baseline_tracker: Optional[BaselineTracker] = None) -> Optional[HackingAlert]:
        """
        Check for state cycling.

        Algorithm:
        1. Hash/compress observation for comparison
        2. Look for repeating patterns in observation history
        3. If cycle detected and reward is high, flag as hacking
        """
        # Add observation to buffer
        obs_hash = self._compute_observation_hash(observation)
        self.observation_buffer.append(obs_hash)

        # Track state revisit rate
        self._total_states += 1
        self._unique_states.add(obs_hash)

        # Need enough data to detect cycles
        if len(self.observation_buffer) < 2 * self.min_cycle_length:
            return None

        # Avoid spamming alerts
        if step - self.last_alert_step < 50:
            return None

        # Find cycles
        cycles = self._find_cycles()

        if cycles:
            # Get the most significant cycle
            cycle_length, similarity = max(cycles, key=lambda x: x[1])

            if similarity >= self.cycle_threshold:
                severity = min(1.0, similarity)

                alert = HackingAlert(
                    type=HackingType.STATE_CYCLING,
                    severity=severity,
                    step=step,
                    episode=episode,
                    description=f"Detected state cycle of length {cycle_length} with {similarity:.1%} similarity",
                    evidence={
                        "cycle_length": cycle_length,
                        "similarity": similarity,
                        "recent_reward": reward,
                        "state_revisit_rate": self.get_episode_metric(),
                    },
                    suggested_fix="Check if reward encourages staying in a loop. Add diversity bonus or forward progress reward.",
                )

                # Apply two-layer logic if baseline is available
                alert = self._apply_two_layer_logic(
                    alert, self.get_episode_metric() or 0.0, baseline_tracker
                )
                if alert is not None:
                    self.alerts.append(alert)
                    self.last_alert_step = step
                return alert

        return None

    def get_episode_metric(self) -> Optional[float]:
        """Get state revisit rate (1 - unique/total)."""
        if self._total_states == 0:
            return None
        # Revisit rate: 0 = all unique, 1 = all same
        return 1.0 - (len(self._unique_states) / self._total_states)

    def _compute_observation_hash(self, obs: Any) -> str:
        """Create hashable representation of observation."""
        if obs is None:
            return "none"

        # Convert to numpy array if needed
        if isinstance(obs, (list, tuple)):
            obs = np.array(obs)
        elif not isinstance(obs, np.ndarray):
            # For non-array observations, use string representation
            return str(obs)

        # Discretize continuous observations for comparison
        # Round to 2 decimal places to allow for small numerical differences
        discretized = np.round(obs.flatten(), decimals=2)

        # Create hash
        return hashlib.md5(discretized.tobytes()).hexdigest()[:8]

    def _find_cycles(self) -> List[Tuple[int, float]]:
        """Find cycles in observation buffer. Returns [(length, similarity)]."""
        cycles = []
        buffer_list = list(self.observation_buffer)
        n = len(buffer_list)

        # Check for different cycle lengths
        for cycle_len in range(self.min_cycle_length, min(self.max_cycle_length + 1, n // 2)):
            # Compare recent observations with observations cycle_len steps ago
            matches = 0
            comparisons = min(cycle_len, n - cycle_len)

            for i in range(comparisons):
                if buffer_list[-(i + 1)] == buffer_list[-(i + 1 + cycle_len)]:
                    matches += 1

            if comparisons > 0:
                similarity = matches / comparisons
                if similarity > 0.5:  # At least 50% match
                    cycles.append((cycle_len, similarity))

        return cycles

    def reset(self) -> None:
        """Reset detector state at episode end."""
        self.observation_buffer.clear()
        self._unique_states.clear()
        self._total_states = 0


class ActionRepetitionDetector(BaseDetector):
    """
    Detects exploitation through repeated actions.

    Symptoms:
    - Same action taken repeatedly
    - High reward from constant action
    - Low action entropy

    Example: Agent learns to always accelerate because velocity
    reward doesn't penalize lack of directional control.
    """

    baseline_metric = "action_entropy"

    def __init__(
        self,
        window_size: int = 50,
        repetition_threshold: float = 0.9,  # % same action
        min_action_entropy: float = 0.1,
    ):
        super().__init__(window_size)
        self.repetition_threshold = repetition_threshold
        self.min_action_entropy = min_action_entropy
        self.action_buffer: deque = deque(maxlen=window_size)
        self.last_alert_step = -1000

        # Track action counts for entropy calculation
        self._episode_action_counts: Dict[str, int] = {}

    def update(self, step, episode, observation, action, reward,
               reward_components, done, info,
               baseline_tracker: Optional[BaselineTracker] = None) -> Optional[HackingAlert]:
        # Convert action to hashable type
        action_hash = self._hash_action(action)
        self.action_buffer.append(action_hash)

        # Track action for entropy calculation
        self._episode_action_counts[action_hash] = \
            self._episode_action_counts.get(action_hash, 0) + 1

        # Need enough data
        if len(self.action_buffer) < self.window_size:
            return None

        # Avoid spamming
        if step - self.last_alert_step < 50:
            return None

        # Calculate action repetition rate
        from collections import Counter
        action_counts = Counter(self.action_buffer)
        most_common_action, count = action_counts.most_common(1)[0]
        repetition_rate = count / len(self.action_buffer)

        if repetition_rate >= self.repetition_threshold:
            severity = min(1.0, repetition_rate)
            entropy = self.get_episode_metric() or 0.0

            alert = HackingAlert(
                type=HackingType.ACTION_REPETITION,
                severity=severity,
                step=step,
                episode=episode,
                description=f"Action repetition detected: {repetition_rate:.1%} of recent actions are identical",
                evidence={
                    "repetition_rate": repetition_rate,
                    "most_common_action": str(most_common_action),
                    "window_size": len(self.action_buffer),
                    "action_entropy": entropy,
                },
                suggested_fix="Add action diversity bonus or entropy regularization to policy.",
            )

            # Apply two-layer logic if baseline is available
            # Note: Low entropy is abnormal, so we check with the entropy value
            alert = self._apply_two_layer_logic(alert, entropy, baseline_tracker)
            if alert is not None:
                self.alerts.append(alert)
                self.last_alert_step = step
            return alert

        return None

    def get_episode_metric(self) -> Optional[float]:
        """Get action entropy for the episode."""
        if not self._episode_action_counts:
            return None
        total = sum(self._episode_action_counts.values())
        if total == 0:
            return None
        probs = np.array(list(self._episode_action_counts.values())) / total
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)

    def _hash_action(self, action: Any) -> str:
        """Convert action to hashable string."""
        if isinstance(action, (int, str)):
            return str(action)
        elif isinstance(action, (list, tuple, np.ndarray)):
            # Discretize continuous actions
            arr = np.array(action).flatten()
            discretized = np.round(arr, decimals=1)
            return str(discretized.tolist())
        else:
            return str(action)

    def reset(self) -> None:
        """Reset at episode end."""
        self.action_buffer.clear()
        self._episode_action_counts.clear()


class ComponentImbalanceDetector(BaseDetector):
    """
    Detects when one reward component dominates others.

    Symptoms:
    - Single component >> 80% of total reward
    - Other components consistently near zero or negative
    - Agent ignores multi-objective nature of task

    Example: In locomotion, agent maximizes velocity reward
    while ignoring energy efficiency and stability rewards.
    """

    # Component imbalance uses component-specific baselines
    baseline_metric = None  # Set dynamically per component

    def __init__(
        self,
        window_size: int = 100,
        dominance_threshold: float = 0.8,
        imbalance_episodes: int = 5,  # Flag after N episodes of imbalance
    ):
        super().__init__(window_size)
        self.dominance_threshold = dominance_threshold
        self.imbalance_episodes = imbalance_episodes
        self.episode_component_ratios: deque = deque(maxlen=imbalance_episodes)
        self.current_episode_components: Dict[str, float] = {}
        self._last_dominance_ratio: float = 0.0
        self._last_dominant_component: str = ""

    def update(self, step, episode, observation, action, reward,
               reward_components, done, info,
               baseline_tracker: Optional[BaselineTracker] = None) -> Optional[HackingAlert]:
        # Accumulate components for current episode
        for comp_name, comp_value in reward_components.items():
            if comp_name == "residual":
                continue  # Ignore residual
            if comp_name not in self.current_episode_components:
                self.current_episode_components[comp_name] = 0.0
            self.current_episode_components[comp_name] += abs(comp_value)

        return None  # Check at episode end

    def on_episode_end(
        self,
        episode_component_totals: Dict[str, float],
        baseline_tracker: Optional[BaselineTracker] = None,
    ) -> Optional[HackingAlert]:
        """Check for sustained imbalance across episodes."""
        if not episode_component_totals:
            return None

        # Filter out residual
        totals = {k: abs(v) for k, v in episode_component_totals.items() if k != "residual"}

        if not totals:
            return None

        # Calculate dominance ratio
        total_abs = sum(totals.values())
        if total_abs == 0:
            return None

        dominant_component = max(totals.items(), key=lambda x: x[1])
        dominant_name, dominant_value = dominant_component
        dominance_ratio = dominant_value / total_abs

        # Store for episode metric
        self._last_dominance_ratio = dominance_ratio
        self._last_dominant_component = dominant_name

        # Store ratio for this episode
        self.episode_component_ratios.append({
            "dominant": dominant_name,
            "ratio": dominance_ratio,
        })

        # Check if consistently imbalanced
        if len(self.episode_component_ratios) >= self.imbalance_episodes:
            # Check if same component dominates across episodes
            recent_dominants = [r["dominant"] for r in self.episode_component_ratios]
            avg_ratio = np.mean([r["ratio"] for r in self.episode_component_ratios])

            # Check if one component consistently dominates
            from collections import Counter
            dominant_counts = Counter(recent_dominants)
            most_common, count = dominant_counts.most_common(1)[0]

            if count >= self.imbalance_episodes - 1 and avg_ratio >= self.dominance_threshold:
                severity = min(1.0, avg_ratio)

                alert = HackingAlert(
                    type=HackingType.COMPONENT_IMBALANCE,
                    severity=severity,
                    step=0,  # Episode-level alert
                    episode=len(self.alerts),
                    description=f"Component '{most_common}' dominates {avg_ratio:.1%} of total reward across {count} episodes",
                    evidence={
                        "dominant_component": most_common,
                        "dominance_ratio": avg_ratio,
                        "episodes_checked": len(self.episode_component_ratios),
                        "component_totals": totals,
                    },
                    suggested_fix=f"Rebalance component weights or add constraints on '{most_common}'.",
                )

                # Apply two-layer logic using component-specific baseline
                if baseline_tracker is not None and baseline_tracker.is_active:
                    metric_name = f"component:{most_common}"
                    baseline_abnormal = baseline_tracker.is_abnormal(metric_name, avg_ratio)
                    z_score = baseline_tracker.get_z_score(metric_name, avg_ratio)
                    severity_class = classify_alert(True, baseline_abnormal)

                    alert.alert_severity = severity_class
                    alert.baseline_z_score = z_score

                    if severity_class == AlertSeverity.SUPPRESSED:
                        baseline_tracker.record_suppressed()
                        self.current_episode_components.clear()
                        return None  # Suppress

                self.alerts.append(alert)
                self.current_episode_components.clear()
                return alert

        # Reset for next episode
        self.current_episode_components.clear()
        return None

    def get_episode_metric(self) -> Optional[float]:
        """Get the dominance ratio of the most dominant component."""
        return self._last_dominance_ratio if self._last_dominance_ratio > 0 else None

    def reset(self) -> None:
        """Reset episode accumulator."""
        self.current_episode_components.clear()


class RewardSpikingDetector(BaseDetector):
    """
    Detects unnatural reward patterns.

    Symptoms:
    - Sudden large reward spikes
    - Reward variance much higher than expected
    - Bimodal reward distribution (exploit vs non-exploit)

    Example: Agent finds glitch state that gives massive reward.
    """

    baseline_metric = "reward"

    def __init__(
        self,
        window_size: int = 500,
        spike_std_threshold: float = 5.0,  # Flag if > N std from mean
        variance_ratio_threshold: float = 10.0,
    ):
        super().__init__(window_size)
        self.spike_std_threshold = spike_std_threshold
        self.variance_ratio_threshold = variance_ratio_threshold
        self.reward_buffer: deque = deque(maxlen=window_size)
        self.running_mean: float = 0.0
        self.running_var: float = 1.0
        self.count: int = 0
        self.last_alert_step = -1000

        # Track episode reward for baseline
        self._episode_reward: float = 0.0

    def update(self, step, episode, observation, action, reward,
               reward_components, done, info,
               baseline_tracker: Optional[BaselineTracker] = None) -> Optional[HackingAlert]:
        self.reward_buffer.append(reward)
        self._episode_reward += reward

        # Update running statistics (Welford's algorithm)
        self.count += 1
        delta = reward - self.running_mean
        self.running_mean += delta / self.count
        delta2 = reward - self.running_mean
        self.running_var += delta * delta2

        # Need enough data
        if len(self.reward_buffer) < 50:
            return None

        # Avoid spamming
        if step - self.last_alert_step < 100:
            return None

        # Calculate statistics
        mean = np.mean(self.reward_buffer)
        std = np.std(self.reward_buffer)

        if std == 0:
            return None

        # Check for spike
        z_score = abs((reward - mean) / std)

        if z_score >= self.spike_std_threshold:
            severity = min(1.0, z_score / (2 * self.spike_std_threshold))

            alert = HackingAlert(
                type=HackingType.REWARD_SPIKING,
                severity=severity,
                step=step,
                episode=episode,
                description=f"Reward spike detected: {reward:.2f} is {z_score:.1f}σ from mean {mean:.2f}",
                evidence={
                    "reward": reward,
                    "mean": mean,
                    "std": std,
                    "z_score": z_score,
                },
                suggested_fix="Check for reward clipping or investigate state that caused spike.",
            )

            # Apply two-layer logic - use the reward value against baseline
            alert = self._apply_two_layer_logic(alert, reward, baseline_tracker)
            if alert is not None:
                self.alerts.append(alert)
                self.last_alert_step = step
            return alert

        return None

    def get_episode_metric(self) -> Optional[float]:
        """Get the total episode reward."""
        return self._episode_reward

    def reset(self) -> None:
        """Reset at episode end (but keep buffer for cross-episode detection)."""
        self._episode_reward = 0.0
        # Don't clear buffer - we want to detect spikes across episodes


class BoundaryExploitationDetector(BaseDetector):
    """
    Detects when agent exploits state/action space boundaries.

    Symptoms:
    - Observations frequently at min/max values
    - Actions saturated at limits
    - Reward correlated with boundary proximity

    Example: Agent pushes joint to limit where physics breaks down.
    """

    baseline_metric = "boundary_hit_rate"

    def __init__(
        self,
        window_size: int = 100,
        boundary_threshold: float = 0.95,  # % of max range
        boundary_frequency_threshold: float = 0.5,  # % of time at boundary
        observation_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ):
        super().__init__(window_size)
        self.boundary_threshold = boundary_threshold
        self.boundary_frequency_threshold = boundary_frequency_threshold
        self.observation_bounds = observation_bounds
        self.action_bounds = action_bounds
        self.boundary_counts: Dict[str, int] = {"obs": 0, "action": 0}
        self.total_steps = 0
        self.last_alert_step = -1000

        # Track episode boundary hit rate
        self._episode_boundary_hits: int = 0
        self._episode_steps: int = 0

    def update(self, step, episode, observation, action, reward,
               reward_components, done, info,
               baseline_tracker: Optional[BaselineTracker] = None) -> Optional[HackingAlert]:
        self.total_steps += 1
        self._episode_steps += 1

        # Check observation boundaries
        at_obs_boundary = False
        if self.observation_bounds is not None and observation is not None:
            at_obs_boundary = self._is_at_boundary(observation, self.observation_bounds)
            if at_obs_boundary:
                self.boundary_counts["obs"] += 1

        # Check action boundaries
        at_action_boundary = False
        if self.action_bounds is not None and action is not None:
            at_action_boundary = self._is_at_boundary(action, self.action_bounds)
            if at_action_boundary:
                self.boundary_counts["action"] += 1

        # Track for episode metric
        if at_obs_boundary or at_action_boundary:
            self._episode_boundary_hits += 1

        # Need enough data
        if self.total_steps < self.window_size:
            return None

        # Avoid spamming
        if step - self.last_alert_step < 100:
            return None

        # Check boundary frequency (over sliding window)
        window_steps = min(self.total_steps, self.window_size)
        obs_freq = self.boundary_counts["obs"] / window_steps
        action_freq = self.boundary_counts["action"] / window_steps

        max_freq = max(obs_freq, action_freq)
        boundary_type = "observation" if obs_freq > action_freq else "action"

        if max_freq >= self.boundary_frequency_threshold:
            severity = min(1.0, max_freq)

            alert = HackingAlert(
                type=HackingType.BOUNDARY_EXPLOITATION,
                severity=severity,
                step=step,
                episode=episode,
                description=f"Agent at {boundary_type} boundary {max_freq:.1%} of the time",
                evidence={
                    "observation_boundary_freq": obs_freq,
                    "action_boundary_freq": action_freq,
                    "window_size": window_steps,
                    "boundary_hit_rate": self.get_episode_metric(),
                },
                suggested_fix="Add penalty for boundary proximity or check if bounds are too restrictive.",
            )

            # Apply two-layer logic
            alert = self._apply_two_layer_logic(alert, max_freq, baseline_tracker)
            if alert is not None:
                self.alerts.append(alert)
                self.last_alert_step = step
                # Reset counts to avoid repeated alerts for same behavior
                self.boundary_counts = {"obs": 0, "action": 0}
                self.total_steps = 0
            return alert

        return None

    def get_episode_metric(self) -> Optional[float]:
        """Get the boundary hit rate for the episode."""
        if self._episode_steps == 0:
            return None
        return self._episode_boundary_hits / self._episode_steps

    def _is_at_boundary(self, value: Any, bounds: Tuple) -> bool:
        """Check if value is at boundary."""
        if value is None or bounds is None:
            return False

        low, high = bounds

        # Convert to numpy array
        if isinstance(value, (list, tuple)):
            value = np.array(value)
        elif isinstance(value, (int, float)):
            value = np.array([value])
        elif not isinstance(value, np.ndarray):
            return False

        # Flatten for comparison
        value = value.flatten()

        if isinstance(low, (int, float)):
            low = np.full_like(value, low, dtype=float)
        else:
            low = np.array(low, dtype=float).flatten()

        if isinstance(high, (int, float)):
            high = np.full_like(value, high, dtype=float)
        else:
            high = np.array(high, dtype=float).flatten()

        # Ensure same shape
        if value.shape != low.shape or value.shape != high.shape:
            return False

        # Skip dimensions with infinite bounds
        finite_mask = np.isfinite(low) & np.isfinite(high)
        if not np.any(finite_mask):
            return False  # All bounds are infinite, can't check

        # Calculate how close to boundary (as fraction of range) only for finite bounds
        range_size = high - low
        range_size = np.where(range_size == 0, 1, range_size)  # Avoid division by zero

        with np.errstate(divide='ignore', invalid='ignore'):
            low_proximity = (value - low) / range_size
            high_proximity = (high - value) / range_size

        # Only check finite bounds
        at_low = np.any(finite_mask & (low_proximity < (1 - self.boundary_threshold)))
        at_high = np.any(finite_mask & (high_proximity < (1 - self.boundary_threshold)))

        return at_low or at_high

    def reset(self) -> None:
        """Reset at episode end."""
        # Reset episode tracking
        self._episode_boundary_hits = 0
        self._episode_steps = 0
        # Keep boundary counts across episodes to detect sustained exploitation


class HackingDetectorSuite:
    """
    Runs all detectors and aggregates results with optional two-layer detection.

    Usage:
        suite = HackingDetectorSuite()
        suite.update(step, episode, obs, action, reward, components, done, info)
        alerts = suite.get_alerts()

    With two-layer detection (recommended):
        suite = HackingDetectorSuite(
            adaptive_baseline=True,  # Enable two-layer detection
            baseline_window=50,       # Rolling window for baseline
            baseline_warmup=20,       # Episodes before adaptive layer activates
            baseline_sensitivity=2.0, # Std devs for "abnormal" threshold
        )

    Two-layer detection logic:
    1. Static detector fires + baseline abnormal → ALERT (confirmed)
    2. Static detector fires + baseline normal → SUPPRESSED (likely false positive)
    3. Static doesn't fire + baseline abnormal → WARNING (unusual but not severe)
    4. Static doesn't fire + baseline normal → No alert

    Legacy mode (Phase 1):
        suite = HackingDetectorSuite(
            use_adaptive_baselines=True,  # Legacy Phase 1 mode
            calibration_episodes=20,
        )
    """

    def __init__(
        self,
        enable_state_cycling: bool = True,
        enable_action_repetition: bool = True,
        enable_component_imbalance: bool = True,
        enable_reward_spiking: bool = True,
        enable_boundary_exploitation: bool = True,
        observation_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        action_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        # Two-layer detection settings (Phase 2)
        adaptive_baseline: bool = True,
        baseline_window: int = 50,
        baseline_warmup: int = 20,
        baseline_sensitivity: float = 2.0,
        # Auto-calibration settings (Phase 5)
        min_warmup_episodes: int = 10,
        max_warmup_episodes: int = 50,
        stability_threshold: float = 0.1,
        # Legacy adaptive baseline settings (Phase 1 - experimental)
        use_adaptive_baselines: bool = False,
        calibration_episodes: int = 20,
        baseline_sigma_threshold: float = 3.0,
    ):
        self.detectors: List[BaseDetector] = []

        if enable_state_cycling:
            self.detectors.append(StateCyclingDetector())
        if enable_action_repetition:
            self.detectors.append(ActionRepetitionDetector())
        if enable_component_imbalance:
            self.detectors.append(ComponentImbalanceDetector())
        if enable_reward_spiking:
            self.detectors.append(RewardSpikingDetector())
        if enable_boundary_exploitation:
            self.detectors.append(BoundaryExploitationDetector(
                observation_bounds=observation_bounds,
                action_bounds=action_bounds,
            ))

        # Two-layer detection (Phase 2)
        self.adaptive_baseline = adaptive_baseline
        self.baseline_tracker: Optional[BaselineTracker] = None

        if adaptive_baseline:
            self.baseline_tracker = BaselineTracker(
                window=baseline_window,
                warmup=baseline_warmup,
                sensitivity=baseline_sensitivity,
                min_warmup_episodes=min_warmup_episodes,
                max_warmup_episodes=max_warmup_episodes,
                stability_threshold=stability_threshold,
            )

        # Legacy adaptive baselines (Phase 1 - experimental)
        self.use_adaptive_baselines = use_adaptive_baselines
        self.calibration_episodes = calibration_episodes
        self.baseline_collector: Optional[BaselineCollector] = None

        if use_adaptive_baselines:
            self.baseline_collector = BaselineCollector(
                calibration_episodes=calibration_episodes,
                sigma_threshold=baseline_sigma_threshold,
            )

        # Episode-level tracking for baseline
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        self._current_episode_actions: Dict[str, int] = {}
        self._current_episode_components: Dict[str, float] = {}

        # Storage for suppressed/warning alerts
        self._suppressed_alerts: List[HackingAlert] = []
        self._warning_alerts: List[HackingAlert] = []

    def update(
        self,
        step: int,
        episode: int,
        observation: Any,
        action: Any,
        reward: float,
        reward_components: Dict[str, float],
        done: bool,
        info: Dict[str, Any],
    ) -> List[HackingAlert]:
        """Run all detectors and return any alerts."""
        alerts = []

        # Track episode statistics for baseline tracker (Phase 2)
        if self.baseline_tracker is not None:
            self._current_episode_reward += reward
            self._current_episode_length += 1

            # Track action distribution
            action_key = self._hash_action(action)
            self._current_episode_actions[action_key] = \
                self._current_episode_actions.get(action_key, 0) + 1

            # Track component totals
            for comp_name, comp_value in reward_components.items():
                if comp_name != "residual":
                    self._current_episode_components[comp_name] = \
                        self._current_episode_components.get(comp_name, 0.0) + comp_value

        # Record step for legacy adaptive baselines (Phase 1)
        if self.baseline_collector is not None:
            self.baseline_collector.record_step(action, reward, reward_components)

        # During legacy calibration phase, suppress standard detector alerts
        if self.use_adaptive_baselines and not self.is_calibrated:
            # Still run detectors to keep their internal state updated,
            # but don't return alerts during calibration
            for detector in self.detectors:
                detector.update(
                    step, episode, observation, action,
                    reward, reward_components, done, info
                )
            return alerts

        # Run all detectors - pass baseline_tracker for two-layer logic
        for detector in self.detectors:
            alert = detector.update(
                step, episode, observation, action,
                reward, reward_components, done, info,
                baseline_tracker=self.baseline_tracker,
            )
            if alert:
                alerts.append(alert)

        return alerts

    def _hash_action(self, action: Any) -> str:
        """Convert action to a hashable string for tracking."""
        if isinstance(action, (int, str)):
            return str(action)
        elif isinstance(action, (list, tuple, np.ndarray)):
            arr = np.array(action).flatten()
            discretized = np.round(arr, decimals=1)
            return str(discretized.tolist())
        else:
            return str(action)

    def _compute_action_entropy(self) -> float:
        """Compute action entropy for current episode."""
        if not self._current_episode_actions:
            return 0.0
        total = sum(self._current_episode_actions.values())
        if total == 0:
            return 0.0
        probs = np.array(list(self._current_episode_actions.values())) / total
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)

    def _compute_component_ratios(self) -> Dict[str, float]:
        """Compute component ratios for current episode."""
        total_abs = sum(abs(v) for v in self._current_episode_components.values())
        if total_abs == 0:
            return {}
        return {
            name: abs(value) / total_abs
            for name, value in self._current_episode_components.items()
        }

    @property
    def is_calibrated(self) -> bool:
        """Check if adaptive baseline calibration is complete."""
        if self.baseline_collector is None:
            return True  # Not using adaptive baselines
        return self.baseline_collector.is_calibrated

    @property
    def calibration_progress(self) -> float:
        """Get calibration progress (0.0 to 1.0)."""
        if self.baseline_collector is None:
            return 1.0
        return self.baseline_collector.calibration_progress

    def on_episode_end(self, episode_stats: Dict) -> List[HackingAlert]:
        """Run episode-end checks with two-layer detection."""
        alerts = []
        current_episode = episode_stats.get("episode", 0)

        # Compute current episode metrics for baseline tracker
        action_entropy = self._compute_action_entropy()
        component_ratios = self._compute_component_ratios()

        # Collect detector-specific episode metrics
        state_revisit_rate = None
        boundary_hit_rate = None
        for detector in self.detectors:
            if isinstance(detector, StateCyclingDetector):
                state_revisit_rate = detector.get_episode_metric()
            elif isinstance(detector, BoundaryExploitationDetector):
                boundary_hit_rate = detector.get_episode_metric()

        # Update baseline tracker with episode stats (Phase 2)
        if self.baseline_tracker is not None:
            baseline_update = {
                "reward": self._current_episode_reward,
                "length": self._current_episode_length,
                "action_entropy": action_entropy,
                "component_ratios": component_ratios,
            }
            # Only add if we have values
            if state_revisit_rate is not None:
                baseline_update["state_revisit_rate"] = state_revisit_rate
            if boundary_hit_rate is not None:
                baseline_update["boundary_hit_rate"] = boundary_hit_rate

            self.baseline_tracker.update(baseline_update)

        # Run standard episode-end detectors first
        static_alerts = []
        for detector in self.detectors:
            if isinstance(detector, ComponentImbalanceDetector):
                alert = detector.on_episode_end(
                    episode_stats.get("component_totals", {}),
                    baseline_tracker=self.baseline_tracker,
                )
                if alert:
                    static_alerts.append(alert)

        # Apply two-layer detection logic (Phase 2)
        if self.baseline_tracker is not None and self.baseline_tracker.is_active:
            # Process static alerts through baseline filter
            for alert in static_alerts:
                # Determine which metric this alert relates to
                metric_name = self._get_metric_for_alert(alert)
                metric_value = self._get_value_for_alert(alert)

                if metric_name and metric_value is not None:
                    baseline_abnormal = self.baseline_tracker.is_abnormal(metric_name, metric_value)
                    z_score = self.baseline_tracker.get_z_score(metric_name, metric_value)
                    severity = classify_alert(True, baseline_abnormal)

                    # Update alert with two-layer info
                    alert.alert_severity = severity
                    alert.baseline_z_score = z_score

                    if severity == AlertSeverity.ALERT:
                        alerts.append(alert)
                    elif severity == AlertSeverity.SUPPRESSED:
                        self._suppressed_alerts.append(alert)
                        self.baseline_tracker.record_suppressed()
                else:
                    # Can't determine metric, treat as regular alert
                    alerts.append(alert)

            # Check for baseline-only warnings (abnormal but no static alert)
            self._check_baseline_warnings(current_episode, alerts)
        else:
            # Baseline not active yet, pass through static alerts
            alerts.extend(static_alerts)

        # Handle legacy adaptive baseline episode end (Phase 1)
        if self.baseline_collector is not None:
            result = self.baseline_collector.end_episode()

            # Start new episode tracking
            self.baseline_collector.start_episode()

            # Fire alerts for baseline deviations (only after calibration)
            if result.get("deviations"):
                for deviation in result["deviations"]:
                    alert = HackingAlert(
                        type=HackingType.BASELINE_DEVIATION,
                        severity=min(1.0, abs(deviation["z_score"]) / 6.0),
                        step=0,
                        episode=result["episode"],
                        description=deviation["description"],
                        evidence={
                            "deviation_type": deviation["type"],
                            "value": deviation["value"],
                            "baseline_mean": deviation["baseline_mean"],
                            "baseline_std": deviation["baseline_std"],
                            "z_score": deviation["z_score"],
                            "component": deviation.get("component"),
                        },
                        suggested_fix="Check if training dynamics have changed significantly from baseline.",
                    )
                    alerts.append(alert)
                    if not hasattr(self, '_baseline_alerts'):
                        self._baseline_alerts: List[HackingAlert] = []
                    self._baseline_alerts.append(alert)

        # During legacy calibration, suppress component imbalance alerts
        if self.use_adaptive_baselines and not self.is_calibrated:
            # Filter out non-baseline alerts during legacy calibration
            alerts = [a for a in alerts if a.type == HackingType.BASELINE_DEVIATION]

        # Reset episode tracking
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        self._current_episode_actions.clear()
        self._current_episode_components.clear()

        return alerts

    def _get_metric_for_alert(self, alert: HackingAlert) -> Optional[str]:
        """Map an alert type to its corresponding baseline metric."""
        alert_to_metric = {
            HackingType.REWARD_SPIKING: "reward",
            HackingType.COMPONENT_IMBALANCE: None,  # Handled specially
            HackingType.ACTION_REPETITION: "action_entropy",
        }
        return alert_to_metric.get(alert.type)

    def _get_value_for_alert(self, alert: HackingAlert) -> Optional[float]:
        """Get the value to compare against baseline for an alert."""
        if alert.type == HackingType.REWARD_SPIKING:
            return alert.evidence.get("reward")
        elif alert.type == HackingType.ACTION_REPETITION:
            return self._compute_action_entropy()
        return None

    def _check_baseline_warnings(self, episode: int, alerts: List[HackingAlert]) -> None:
        """Check for baseline-only warnings (abnormal but no static alert)."""
        if self.baseline_tracker is None or not self.baseline_tracker.is_active:
            return

        # Check core metrics for warnings
        metrics_to_check = [
            ("reward", self._current_episode_reward),
            ("length", float(self._current_episode_length)),
            ("action_entropy", self._compute_action_entropy()),
        ]

        for metric_name, value in metrics_to_check:
            if self.baseline_tracker.is_abnormal(metric_name, value):
                # Check if there's already a static alert for this
                has_static = any(
                    self._get_metric_for_alert(a) == metric_name
                    for a in alerts
                )
                if not has_static:
                    z_score = self.baseline_tracker.get_z_score(metric_name, value)
                    confidence = zscore_to_confidence(z_score)
                    warning = HackingAlert(
                        type=HackingType.BASELINE_DEVIATION,
                        severity=min(0.5, abs(z_score) / 6.0),  # Lower severity for warnings
                        step=0,
                        episode=episode,
                        description=f"Unusual {metric_name}: {value:.2f} is {abs(z_score):.1f}σ from baseline",
                        evidence={
                            "metric": metric_name,
                            "value": value,
                            "z_score": z_score,
                            "confidence": confidence,
                            "baseline_mean": self.baseline_tracker._reward_stats.mean if metric_name == "reward"
                                else self.baseline_tracker._length_stats.mean if metric_name == "length"
                                else self.baseline_tracker._entropy_stats.mean,
                        },
                        suggested_fix="Monitor for consistent deviation from baseline.",
                        alert_severity=AlertSeverity.WARNING,
                        baseline_z_score=z_score,
                        confidence=confidence,
                    )
                    alerts.append(warning)
                    self._warning_alerts.append(warning)
                    self.baseline_tracker.record_warning()

    def get_all_alerts(self, include_suppressed: bool = False) -> List[HackingAlert]:
        """
        Get all historical alerts.

        Args:
            include_suppressed: If True, include suppressed alerts in the result
        """
        alerts = []
        for detector in self.detectors:
            alerts.extend(detector.alerts)
        # Include baseline deviation alerts (legacy Phase 1)
        if hasattr(self, '_baseline_alerts'):
            alerts.extend(self._baseline_alerts)
        # Include warning alerts from two-layer detection
        alerts.extend(self._warning_alerts)
        # Optionally include suppressed alerts
        if include_suppressed:
            alerts.extend(self._suppressed_alerts)
        return sorted(alerts, key=lambda a: (a.episode, a.step))

    def get_suppressed_alerts(self) -> List[HackingAlert]:
        """Get all alerts that were suppressed by baseline filter."""
        return list(self._suppressed_alerts)

    def get_warning_alerts(self) -> List[HackingAlert]:
        """Get all soft warnings (baseline abnormal but no static alert)."""
        return list(self._warning_alerts)

    def get_hacking_score(self) -> float:
        """
        Compute overall hacking score (0-1).
        Higher = more likely the agent is hacking.

        Note: Only counts non-suppressed alerts. Suppressed alerts are
        considered false positives by the baseline filter.
        """
        # Only count non-suppressed alerts
        all_alerts = [a for a in self.get_all_alerts()
                      if a.alert_severity != AlertSeverity.SUPPRESSED]

        if not all_alerts:
            return 0.0

        # Weight ALERT higher than WARNING
        weighted_severities = []
        for a in all_alerts[-10:]:  # Last 10 alerts
            weight = 1.0 if a.alert_severity == AlertSeverity.ALERT else 0.5
            weighted_severities.append(a.severity * weight)

        avg_severity = np.mean(weighted_severities) if weighted_severities else 0.0

        # Factor in alert frequency
        alert_frequency = min(1.0, len(all_alerts[-10:]) / 10.0)

        # Combine severity and frequency
        score = (avg_severity + alert_frequency) / 2.0

        return min(1.0, score)

    def reset(self) -> None:
        """Reset all detectors (but preserve baseline tracking)."""
        for detector in self.detectors:
            detector.reset()
        # Note: We intentionally do NOT reset the baseline_tracker or
        # baseline_collector here because baselines should persist across episodes

    def get_baseline_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of adaptive baseline statistics (if enabled)."""
        summary = {}

        # Phase 2 baseline tracker summary
        if self.baseline_tracker is not None:
            summary["tracker"] = self.baseline_tracker.get_baseline_summary()

        # Legacy Phase 1 baseline collector summary
        if self.baseline_collector is not None:
            summary["collector"] = self.baseline_collector.get_baseline_summary()

        return summary if summary else None

    def get_suppressed_count(self) -> int:
        """Get count of suppressed alerts."""
        if self.baseline_tracker is not None:
            return self.baseline_tracker.suppressed_count
        return len(self._suppressed_alerts)

    def get_warning_count(self) -> int:
        """Get count of soft warnings."""
        if self.baseline_tracker is not None:
            return self.baseline_tracker.warning_count
        return len(self._warning_alerts)

    @property
    def baseline_is_active(self) -> bool:
        """Check if two-layer baseline detection is active."""
        if self.baseline_tracker is not None:
            return self.baseline_tracker.is_active
        return False

    @property
    def baseline_warmup_progress(self) -> float:
        """Get baseline warmup progress (0.0 to 1.0)."""
        if self.baseline_tracker is not None:
            return self.baseline_tracker.warmup_progress
        return 1.0

    def reset_baselines(self) -> None:
        """Reset all baselines (start fresh tracking)."""
        # Reset Phase 2 tracker
        if self.baseline_tracker is not None:
            self.baseline_tracker.reset()
        self._suppressed_alerts.clear()
        self._warning_alerts.clear()

        # Reset legacy Phase 1 collector
        if self.baseline_collector is not None:
            self.baseline_collector.reset()
        if hasattr(self, '_baseline_alerts'):
            self._baseline_alerts.clear()
