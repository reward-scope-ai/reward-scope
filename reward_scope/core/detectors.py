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
from dataclasses import dataclass
from enum import Enum
import numpy as np
from collections import deque
import hashlib


class HackingType(Enum):
    """Types of reward hacking."""
    PROXY_DIVERGENCE = "proxy_divergence"
    STATE_CYCLING = "state_cycling"
    ACTION_REPETITION = "action_repetition"
    BOUNDARY_EXPLOITATION = "boundary_exploitation"
    REWARD_SPIKING = "reward_spiking"
    COMPONENT_IMBALANCE = "component_imbalance"


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


class BaseDetector:
    """Base class for hacking detectors."""

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
    ) -> Optional[HackingAlert]:
        """
        Process a new step and check for hacking.
        Returns alert if hacking detected, None otherwise.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset detector state (e.g., at episode end)."""
        pass


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

    def update(self, step, episode, observation, action, reward,
               reward_components, done, info) -> Optional[HackingAlert]:
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
                    },
                    suggested_fix="Check if reward encourages staying in a loop. Add diversity bonus or forward progress reward.",
                )

                self.alerts.append(alert)
                self.last_alert_step = step
                return alert

        return None

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

    def update(self, step, episode, observation, action, reward,
               reward_components, done, info) -> Optional[HackingAlert]:
        # Convert action to hashable type
        action_hash = self._hash_action(action)
        self.action_buffer.append(action_hash)

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
                },
                suggested_fix="Add action diversity bonus or entropy regularization to policy.",
            )

            self.alerts.append(alert)
            self.last_alert_step = step
            return alert

        return None

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

    def update(self, step, episode, observation, action, reward,
               reward_components, done, info) -> Optional[HackingAlert]:
        # Accumulate components for current episode
        for comp_name, comp_value in reward_components.items():
            if comp_name == "residual":
                continue  # Ignore residual
            if comp_name not in self.current_episode_components:
                self.current_episode_components[comp_name] = 0.0
            self.current_episode_components[comp_name] += abs(comp_value)

        return None  # Check at episode end

    def on_episode_end(self, episode_component_totals: Dict[str, float]) -> Optional[HackingAlert]:
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

                self.alerts.append(alert)
                return alert

        # Reset for next episode
        self.current_episode_components.clear()
        return None

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

    def update(self, step, episode, observation, action, reward,
               reward_components, done, info) -> Optional[HackingAlert]:
        self.reward_buffer.append(reward)

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

            self.alerts.append(alert)
            self.last_alert_step = step
            return alert

        return None

    def reset(self) -> None:
        """Reset at episode end (but keep buffer for cross-episode detection)."""
        pass  # Don't clear buffer - we want to detect spikes across episodes


class BoundaryExploitationDetector(BaseDetector):
    """
    Detects when agent exploits state/action space boundaries.

    Symptoms:
    - Observations frequently at min/max values
    - Actions saturated at limits
    - Reward correlated with boundary proximity

    Example: Agent pushes joint to limit where physics breaks down.
    """

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

    def update(self, step, episode, observation, action, reward,
               reward_components, done, info) -> Optional[HackingAlert]:
        self.total_steps += 1

        # Check observation boundaries
        if self.observation_bounds is not None and observation is not None:
            if self._is_at_boundary(observation, self.observation_bounds):
                self.boundary_counts["obs"] += 1

        # Check action boundaries
        if self.action_bounds is not None and action is not None:
            if self._is_at_boundary(action, self.action_bounds):
                self.boundary_counts["action"] += 1

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
                },
                suggested_fix="Add penalty for boundary proximity or check if bounds are too restrictive.",
            )

            self.alerts.append(alert)
            self.last_alert_step = step
            # Reset counts to avoid repeated alerts for same behavior
            self.boundary_counts = {"obs": 0, "action": 0}
            self.total_steps = 0
            return alert

        return None

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
        # Keep boundary counts across episodes to detect sustained exploitation
        pass


class HackingDetectorSuite:
    """
    Runs all detectors and aggregates results.

    Usage:
        suite = HackingDetectorSuite()
        suite.update(step, episode, obs, action, reward, components, done, info)
        alerts = suite.get_alerts()
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
        for detector in self.detectors:
            alert = detector.update(
                step, episode, observation, action,
                reward, reward_components, done, info
            )
            if alert:
                alerts.append(alert)
        return alerts

    def on_episode_end(self, episode_stats: Dict) -> List[HackingAlert]:
        """Run episode-end checks."""
        alerts = []
        for detector in self.detectors:
            if isinstance(detector, ComponentImbalanceDetector):
                alert = detector.on_episode_end(episode_stats.get("component_totals", {}))
                if alert:
                    alerts.append(alert)
        return alerts

    def get_all_alerts(self) -> List[HackingAlert]:
        """Get all historical alerts."""
        alerts = []
        for detector in self.detectors:
            alerts.extend(detector.alerts)
        return sorted(alerts, key=lambda a: a.step)

    def get_hacking_score(self) -> float:
        """
        Compute overall hacking score (0-1).
        Higher = more likely the agent is hacking.
        """
        all_alerts = self.get_all_alerts()

        if not all_alerts:
            return 0.0

        # Weight recent alerts more heavily
        if len(all_alerts) == 0:
            return 0.0

        # Take average severity of recent alerts
        recent_alerts = all_alerts[-10:]  # Last 10 alerts
        avg_severity = np.mean([a.severity for a in recent_alerts])

        # Factor in alert frequency
        alert_frequency = min(1.0, len(recent_alerts) / 10.0)

        # Combine severity and frequency
        score = (avg_severity + alert_frequency) / 2.0

        return min(1.0, score)

    def reset(self) -> None:
        """Reset all detectors."""
        for detector in self.detectors:
            detector.reset()
