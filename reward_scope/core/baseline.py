"""
Adaptive Baseline Tracker for Two-Layer Detection

Provides rolling statistics tracking over the last N episodes to determine
what's "normal" for a training run. Works alongside static detectors to
reduce false positives while keeping static thresholds as a safety net.

Two-layer detection logic:
1. Static detector fires, baseline says "normal" → Suppress alert
2. Static detector fires, baseline says "abnormal" → Fire alert (confirmed)
3. Static detector doesn't fire, baseline says "abnormal" → Soft warning
4. Static detector doesn't fire, baseline says "normal" → No alert

Known limitation (documented):
If agent gradually drifts into hacking over 100+ episodes, baseline drifts
with it. Static detectors are the backstop for this case.
"""

from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np


class AlertSeverity(Enum):
    """Alert severity levels for two-layer detection."""
    ALERT = "alert"           # Static fired + baseline confirms abnormal
    WARNING = "warning"       # Baseline abnormal but static didn't fire
    SUPPRESSED = "suppressed" # Static fired but baseline says normal


@dataclass
class RollingStats:
    """Rolling statistics for a single metric using a ring buffer."""
    window_size: int = 50
    _values: deque = field(default_factory=lambda: deque(maxlen=50))

    def __post_init__(self):
        # Reinitialize deque with correct maxlen if window_size differs
        if self._values.maxlen != self.window_size:
            self._values = deque(maxlen=self.window_size)

    def update(self, value: float) -> None:
        """Add a new value to the rolling window."""
        self._values.append(value)

    @property
    def count(self) -> int:
        """Number of values in the buffer."""
        return len(self._values)

    @property
    def mean(self) -> float:
        """Rolling mean of values in the buffer."""
        if not self._values:
            return 0.0
        return float(np.mean(self._values))

    @property
    def std(self) -> float:
        """Rolling standard deviation of values in the buffer."""
        if len(self._values) < 2:
            return 0.0
        return float(np.std(self._values))

    def is_abnormal(self, value: float, sensitivity: float = 2.0) -> bool:
        """
        Check if a value is abnormal compared to the rolling baseline.

        Args:
            value: The value to check
            sensitivity: Number of standard deviations for "abnormal" threshold

        Returns:
            True if value deviates more than sensitivity * std from mean
        """
        if len(self._values) < 2:
            return False  # Not enough data

        std = self.std
        if std == 0:
            # Zero variance - fall back to exact match check
            # This handles edge case of constant values
            return abs(value - self.mean) > 1e-6

        z_score = abs(value - self.mean) / std
        return z_score > sensitivity

    def get_z_score(self, value: float) -> float:
        """Get the z-score for a value against the rolling baseline."""
        if len(self._values) < 2:
            return 0.0
        std = self.std
        if std == 0:
            return 0.0
        return (value - self.mean) / std

    def get_stats(self) -> Dict[str, float]:
        """Get current rolling statistics."""
        return {
            "mean": self.mean,
            "std": self.std,
            "count": self.count,
            "min": float(min(self._values)) if self._values else 0.0,
            "max": float(max(self._values)) if self._values else 0.0,
        }


class BaselineTracker:
    """
    Tracks rolling statistics over the last N episodes to determine
    what's "normal" for a training run.

    Usage:
        tracker = BaselineTracker(window=50, warmup=20, sensitivity=2.0)

        # After each episode
        tracker.update({
            "reward": episode_reward,
            "length": episode_length,
            "action_entropy": avg_entropy,
            "component_ratios": {"velocity": 0.6, "energy": 0.4},
        })

        # Check if a metric is abnormal
        if tracker.is_active:
            is_abnormal = tracker.is_abnormal("reward", current_reward)

    Attributes:
        window: Number of episodes for rolling statistics (default 50)
        warmup: Minimum episodes before adaptive layer activates (default 20)
        sensitivity: Number of std devs for "abnormal" threshold (default 2.0)
    """

    def __init__(
        self,
        window: int = 50,
        warmup: int = 20,
        sensitivity: float = 2.0,
    ):
        """
        Args:
            window: Number of episodes for rolling window
            warmup: Minimum episodes before adaptive layer activates
            sensitivity: Number of std devs for "abnormal" threshold
        """
        self.window = window
        self.warmup = warmup
        self.sensitivity = sensitivity

        # Core metric trackers
        self._reward_stats = RollingStats(window_size=window)
        self._length_stats = RollingStats(window_size=window)
        self._entropy_stats = RollingStats(window_size=window)

        # New metrics for per-detector baseline integration
        self._state_revisit_rate_stats = RollingStats(window_size=window)
        self._boundary_hit_rate_stats = RollingStats(window_size=window)

        # Component ratio trackers (dynamic - created as needed)
        self._component_stats: Dict[str, RollingStats] = {}

        # Detector score trackers (for per-detector raw scores)
        self._detector_stats: Dict[str, RollingStats] = {}

        # Episode counter
        self._episodes_seen = 0

        # Track suppressed alerts for reporting
        self._suppressed_count = 0
        self._warning_count = 0

    @property
    def is_active(self) -> bool:
        """Check if adaptive layer is active (warmup complete)."""
        return self._episodes_seen >= self.warmup

    @property
    def warmup_progress(self) -> float:
        """Get warmup progress as a fraction (0.0 to 1.0)."""
        return min(1.0, self._episodes_seen / self.warmup)

    @property
    def episodes_seen(self) -> int:
        """Number of episodes processed."""
        return self._episodes_seen

    @property
    def suppressed_count(self) -> int:
        """Number of alerts suppressed by baseline."""
        return self._suppressed_count

    @property
    def warning_count(self) -> int:
        """Number of soft warnings issued."""
        return self._warning_count

    def update(self, episode_stats: Dict[str, Any]) -> None:
        """
        Update baseline with episode statistics.

        Args:
            episode_stats: Dict containing:
                - reward: float - total episode reward
                - length: int - episode length in steps
                - action_entropy: float - average action entropy
                - state_revisit_rate: float - fraction of states that are revisits
                - boundary_hit_rate: float - fraction of steps at bounds
                - component_ratios: Dict[str, float] - per-component ratios
                - detector_scores: Dict[str, float] - per-detector raw scores
        """
        # Update core metrics
        if "reward" in episode_stats:
            self._reward_stats.update(episode_stats["reward"])

        if "length" in episode_stats:
            self._length_stats.update(episode_stats["length"])

        if "action_entropy" in episode_stats:
            self._entropy_stats.update(episode_stats["action_entropy"])

        # Update new per-detector metrics
        if "state_revisit_rate" in episode_stats:
            self._state_revisit_rate_stats.update(episode_stats["state_revisit_rate"])

        if "boundary_hit_rate" in episode_stats:
            self._boundary_hit_rate_stats.update(episode_stats["boundary_hit_rate"])

        # Update component ratio baselines
        if "component_ratios" in episode_stats:
            for comp_name, ratio in episode_stats["component_ratios"].items():
                if comp_name not in self._component_stats:
                    self._component_stats[comp_name] = RollingStats(window_size=self.window)
                self._component_stats[comp_name].update(ratio)

        # Update detector score baselines
        if "detector_scores" in episode_stats:
            for detector_name, score in episode_stats["detector_scores"].items():
                if detector_name not in self._detector_stats:
                    self._detector_stats[detector_name] = RollingStats(window_size=self.window)
                self._detector_stats[detector_name].update(score)

        self._episodes_seen += 1

    def is_abnormal(
        self,
        metric: str,
        value: float,
        sensitivity: Optional[float] = None,
    ) -> bool:
        """
        Check if a metric value is abnormal compared to baseline.

        Args:
            metric: Metric name ("reward", "length", "action_entropy",
                   "component:<name>", or "detector:<name>")
            value: Current value to check
            sensitivity: Override default sensitivity (std devs)

        Returns:
            True if value is abnormal, False otherwise
        """
        if not self.is_active:
            return False  # Adaptive layer not active yet

        sens = sensitivity if sensitivity is not None else self.sensitivity

        # Route to appropriate stats tracker
        if metric == "reward":
            return self._reward_stats.is_abnormal(value, sens)
        elif metric == "length":
            return self._length_stats.is_abnormal(value, sens)
        elif metric == "action_entropy":
            return self._entropy_stats.is_abnormal(value, sens)
        elif metric == "state_revisit_rate":
            return self._state_revisit_rate_stats.is_abnormal(value, sens)
        elif metric == "boundary_hit_rate":
            return self._boundary_hit_rate_stats.is_abnormal(value, sens)
        elif metric.startswith("component:"):
            comp_name = metric[10:]  # Remove "component:" prefix
            if comp_name in self._component_stats:
                return self._component_stats[comp_name].is_abnormal(value, sens)
            return False  # Unknown component, can't determine
        elif metric.startswith("detector:"):
            detector_name = metric[9:]  # Remove "detector:" prefix
            if detector_name in self._detector_stats:
                return self._detector_stats[detector_name].is_abnormal(value, sens)
            return False
        else:
            return False  # Unknown metric

    def get_z_score(self, metric: str, value: float) -> float:
        """Get z-score for a metric value against baseline."""
        if not self.is_active:
            return 0.0

        if metric == "reward":
            return self._reward_stats.get_z_score(value)
        elif metric == "length":
            return self._length_stats.get_z_score(value)
        elif metric == "action_entropy":
            return self._entropy_stats.get_z_score(value)
        elif metric == "state_revisit_rate":
            return self._state_revisit_rate_stats.get_z_score(value)
        elif metric == "boundary_hit_rate":
            return self._boundary_hit_rate_stats.get_z_score(value)
        elif metric.startswith("component:"):
            comp_name = metric[10:]
            if comp_name in self._component_stats:
                return self._component_stats[comp_name].get_z_score(value)
        elif metric.startswith("detector:"):
            detector_name = metric[9:]
            if detector_name in self._detector_stats:
                return self._detector_stats[detector_name].get_z_score(value)
        return 0.0

    def record_suppressed(self) -> None:
        """Record that an alert was suppressed."""
        self._suppressed_count += 1

    def record_warning(self) -> None:
        """Record that a soft warning was issued."""
        self._warning_count += 1

    def get_baseline_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked baselines."""
        summary = {
            "is_active": self.is_active,
            "episodes_seen": self._episodes_seen,
            "warmup": self.warmup,
            "window": self.window,
            "sensitivity": self.sensitivity,
            "suppressed_count": self._suppressed_count,
            "warning_count": self._warning_count,
            "metrics": {},
        }

        # Add core metrics
        if self._reward_stats.count > 0:
            summary["metrics"]["reward"] = self._reward_stats.get_stats()
        if self._length_stats.count > 0:
            summary["metrics"]["length"] = self._length_stats.get_stats()
        if self._entropy_stats.count > 0:
            summary["metrics"]["action_entropy"] = self._entropy_stats.get_stats()
        if self._state_revisit_rate_stats.count > 0:
            summary["metrics"]["state_revisit_rate"] = self._state_revisit_rate_stats.get_stats()
        if self._boundary_hit_rate_stats.count > 0:
            summary["metrics"]["boundary_hit_rate"] = self._boundary_hit_rate_stats.get_stats()

        # Add component metrics
        for comp_name, stats in self._component_stats.items():
            if stats.count > 0:
                summary["metrics"][f"component:{comp_name}"] = stats.get_stats()

        # Add detector metrics
        for detector_name, stats in self._detector_stats.items():
            if stats.count > 0:
                summary["metrics"][f"detector:{detector_name}"] = stats.get_stats()

        return summary

    def reset(self) -> None:
        """Reset all baseline statistics."""
        self._reward_stats = RollingStats(window_size=self.window)
        self._length_stats = RollingStats(window_size=self.window)
        self._entropy_stats = RollingStats(window_size=self.window)
        self._state_revisit_rate_stats = RollingStats(window_size=self.window)
        self._boundary_hit_rate_stats = RollingStats(window_size=self.window)
        self._component_stats.clear()
        self._detector_stats.clear()
        self._episodes_seen = 0
        self._suppressed_count = 0
        self._warning_count = 0


def classify_alert(
    static_fired: bool,
    baseline_abnormal: bool,
) -> Optional[AlertSeverity]:
    """
    Classify an alert based on two-layer detection logic.

    Args:
        static_fired: Whether the static detector fired
        baseline_abnormal: Whether baseline considers this abnormal

    Returns:
        AlertSeverity or None if no alert should be emitted
    """
    if static_fired and baseline_abnormal:
        return AlertSeverity.ALERT
    elif static_fired and not baseline_abnormal:
        return AlertSeverity.SUPPRESSED
    elif not static_fired and baseline_abnormal:
        return AlertSeverity.WARNING
    else:
        return None  # No alert
