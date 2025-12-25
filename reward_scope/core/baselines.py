"""
Adaptive Baseline Module (Experimental)

Per-run adaptive baselines that learn "normal" behavior from the first N episodes
of a training run, then flag deviations from this baseline.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import numpy as np
from collections import deque


@dataclass
class BaselineStats:
    """Statistics collected during baseline calibration."""
    mean: float = 0.0
    std: float = 0.0
    count: int = 0
    _m2: float = 0.0  # For Welford's algorithm

    def update(self, value: float) -> None:
        """Update running statistics using Welford's algorithm."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2

        if self.count > 1:
            self.std = np.sqrt(self._m2 / self.count)

    def is_deviation(self, value: float, sigma_threshold: float = 3.0) -> bool:
        """Check if value deviates more than sigma_threshold from baseline."""
        if self.count < 2 or self.std == 0:
            return False
        z_score = abs(value - self.mean) / self.std
        return z_score > sigma_threshold

    def get_z_score(self, value: float) -> float:
        """Get the z-score for a value."""
        if self.count < 2 or self.std == 0:
            return 0.0
        return (value - self.mean) / self.std


@dataclass
class EpisodeMetrics:
    """Metrics collected per episode for baseline comparison."""
    total_reward: float = 0.0
    length: int = 0
    component_totals: Dict[str, float] = field(default_factory=dict)
    action_entropy: float = 0.0

    # For computing action entropy
    _action_counts: Dict[str, int] = field(default_factory=dict)
    _total_actions: int = 0


class BaselineCollector:
    """
    Collects baseline statistics during a calibration phase.

    During calibration (first N episodes):
    - Tracks action distribution entropy
    - Tracks reward per episode (mean, std)
    - Tracks component ratios (mean, std per component)
    - Tracks episode length (mean, std)

    After calibration:
    - Provides methods to check if current values deviate from baseline

    Usage:
        collector = BaselineCollector(calibration_episodes=20)

        # During training
        for episode in range(100):
            collector.start_episode()
            for step in range(max_steps):
                collector.record_step(action, reward, reward_components)
            collector.end_episode()

            # Check for deviations after calibration
            if collector.is_calibrated:
                deviations = collector.check_deviations()
    """

    def __init__(self, calibration_episodes: int = 20, sigma_threshold: float = 3.0):
        """
        Args:
            calibration_episodes: Number of episodes to collect for baseline.
            sigma_threshold: Number of standard deviations for deviation detection.
        """
        self.calibration_episodes = calibration_episodes
        self.sigma_threshold = sigma_threshold

        # Episode-level baseline statistics
        self.reward_baseline = BaselineStats()
        self.length_baseline = BaselineStats()
        self.entropy_baseline = BaselineStats()

        # Component-level baselines (ratios to total reward)
        self.component_ratio_baselines: Dict[str, BaselineStats] = {}

        # Current episode tracking
        self._current_episode = EpisodeMetrics()
        self._episodes_completed = 0

        # Track known components
        self._known_components: set = set()

    @property
    def is_calibrated(self) -> bool:
        """Check if baseline calibration is complete."""
        return self._episodes_completed >= self.calibration_episodes

    @property
    def calibration_progress(self) -> float:
        """Get calibration progress as a fraction (0.0 to 1.0)."""
        return min(1.0, self._episodes_completed / self.calibration_episodes)

    def start_episode(self) -> None:
        """Start tracking a new episode."""
        self._current_episode = EpisodeMetrics()

    def record_step(
        self,
        action: Any,
        reward: float,
        reward_components: Dict[str, float],
    ) -> None:
        """Record a step's data for baseline collection."""
        # Accumulate reward
        self._current_episode.total_reward += reward
        self._current_episode.length += 1

        # Accumulate component totals
        for comp_name, comp_value in reward_components.items():
            if comp_name == "residual":
                continue
            if comp_name not in self._current_episode.component_totals:
                self._current_episode.component_totals[comp_name] = 0.0
            self._current_episode.component_totals[comp_name] += comp_value
            self._known_components.add(comp_name)

        # Track action for entropy calculation
        action_key = self._hash_action(action)
        if action_key not in self._current_episode._action_counts:
            self._current_episode._action_counts[action_key] = 0
        self._current_episode._action_counts[action_key] += 1
        self._current_episode._total_actions += 1

    def end_episode(self) -> Dict[str, Any]:
        """
        End the current episode and update baselines.

        Returns:
            Dict with current episode metrics and deviation info (if calibrated).
        """
        ep = self._current_episode

        # Calculate action entropy for this episode
        if ep._total_actions > 0:
            probs = np.array(list(ep._action_counts.values())) / ep._total_actions
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            ep.action_entropy = entropy

        # Calculate component ratios
        total_abs_reward = sum(abs(v) for v in ep.component_totals.values())
        component_ratios = {}
        if total_abs_reward > 0:
            for comp_name, comp_total in ep.component_totals.items():
                component_ratios[comp_name] = abs(comp_total) / total_abs_reward

        result = {
            "episode": self._episodes_completed,
            "total_reward": ep.total_reward,
            "length": ep.length,
            "action_entropy": ep.action_entropy,
            "component_ratios": component_ratios,
            "is_calibrated": self.is_calibrated,
            "calibration_progress": self.calibration_progress,
        }

        if not self.is_calibrated:
            # Still in calibration phase - update baselines
            self.reward_baseline.update(ep.total_reward)
            self.length_baseline.update(ep.length)
            self.entropy_baseline.update(ep.action_entropy)

            # Update component ratio baselines
            for comp_name, ratio in component_ratios.items():
                if comp_name not in self.component_ratio_baselines:
                    self.component_ratio_baselines[comp_name] = BaselineStats()
                self.component_ratio_baselines[comp_name].update(ratio)
        else:
            # After calibration - check for deviations
            deviations = self._check_episode_deviations(ep, component_ratios)
            result["deviations"] = deviations

        self._episodes_completed += 1
        return result

    def _check_episode_deviations(
        self,
        episode: EpisodeMetrics,
        component_ratios: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """Check if current episode deviates from baseline."""
        deviations = []

        # Check reward deviation
        if self.reward_baseline.is_deviation(episode.total_reward, self.sigma_threshold):
            z = self.reward_baseline.get_z_score(episode.total_reward)
            deviations.append({
                "type": "reward",
                "value": episode.total_reward,
                "baseline_mean": self.reward_baseline.mean,
                "baseline_std": self.reward_baseline.std,
                "z_score": z,
                "description": f"Episode reward {episode.total_reward:.2f} is {abs(z):.1f}σ from baseline mean {self.reward_baseline.mean:.2f}",
            })

        # Check length deviation
        if self.length_baseline.is_deviation(episode.length, self.sigma_threshold):
            z = self.length_baseline.get_z_score(episode.length)
            deviations.append({
                "type": "length",
                "value": episode.length,
                "baseline_mean": self.length_baseline.mean,
                "baseline_std": self.length_baseline.std,
                "z_score": z,
                "description": f"Episode length {episode.length} is {abs(z):.1f}σ from baseline mean {self.length_baseline.mean:.1f}",
            })

        # Check entropy deviation
        if self.entropy_baseline.is_deviation(episode.action_entropy, self.sigma_threshold):
            z = self.entropy_baseline.get_z_score(episode.action_entropy)
            deviations.append({
                "type": "action_entropy",
                "value": episode.action_entropy,
                "baseline_mean": self.entropy_baseline.mean,
                "baseline_std": self.entropy_baseline.std,
                "z_score": z,
                "description": f"Action entropy {episode.action_entropy:.3f} is {abs(z):.1f}σ from baseline mean {self.entropy_baseline.mean:.3f}",
            })

        # Check component ratio deviations
        for comp_name, ratio in component_ratios.items():
            if comp_name in self.component_ratio_baselines:
                baseline = self.component_ratio_baselines[comp_name]
                if baseline.is_deviation(ratio, self.sigma_threshold):
                    z = baseline.get_z_score(ratio)
                    deviations.append({
                        "type": "component_ratio",
                        "component": comp_name,
                        "value": ratio,
                        "baseline_mean": baseline.mean,
                        "baseline_std": baseline.std,
                        "z_score": z,
                        "description": f"Component '{comp_name}' ratio {ratio:.1%} is {abs(z):.1f}σ from baseline {baseline.mean:.1%}",
                    })

        return deviations

    def get_baseline_summary(self) -> Dict[str, Any]:
        """Get a summary of the collected baselines."""
        return {
            "is_calibrated": self.is_calibrated,
            "episodes_collected": self._episodes_completed,
            "calibration_episodes": self.calibration_episodes,
            "reward": {
                "mean": self.reward_baseline.mean,
                "std": self.reward_baseline.std,
            },
            "length": {
                "mean": self.length_baseline.mean,
                "std": self.length_baseline.std,
            },
            "action_entropy": {
                "mean": self.entropy_baseline.mean,
                "std": self.entropy_baseline.std,
            },
            "component_ratios": {
                name: {"mean": b.mean, "std": b.std}
                for name, b in self.component_ratio_baselines.items()
            },
        }

    def _hash_action(self, action: Any) -> str:
        """Convert action to a hashable string."""
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
        """Reset all baseline statistics (start fresh calibration)."""
        self.reward_baseline = BaselineStats()
        self.length_baseline = BaselineStats()
        self.entropy_baseline = BaselineStats()
        self.component_ratio_baselines.clear()
        self._current_episode = EpisodeMetrics()
        self._episodes_completed = 0
        self._known_components.clear()
