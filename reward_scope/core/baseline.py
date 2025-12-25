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


def zscore_to_confidence(z_score: float) -> float:
    """
    Convert a z-score to a confidence value (0.0 to 1.0).

    Uses a sigmoid-like mapping:
    - 2σ from baseline = 0.5 confidence
    - 3σ = 0.75 confidence
    - 4σ+ = 0.9+ confidence

    The formula is: confidence = 1 - 1 / (1 + (|z| / 2)^2)
    This gives:
    - z=0: 0.0
    - z=2: 0.5
    - z=3: 0.69
    - z=4: 0.80
    - z=5: 0.86
    - z=6: 0.90

    Args:
        z_score: The z-score value (can be positive or negative)

    Returns:
        Confidence value between 0.0 and 1.0
    """
    z = abs(z_score)
    if z < 0.1:
        return 0.0
    # Sigmoid-like mapping centered at z=2 for 0.5 confidence
    confidence = 1.0 - 1.0 / (1.0 + (z / 2.0) ** 2)
    return min(1.0, confidence)


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

    def get_recent_variance(self, n: int = 5) -> float:
        """
        Get variance of the last n values.

        Used for stability detection - low variance means stable.
        """
        if len(self._values) < n:
            return float('inf')  # Not enough data
        recent = list(self._values)[-n:]
        return float(np.var(recent))


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

    Auto-calibration:
        Warmup ends automatically when baseline stats stabilize, rather than
        waiting for a fixed number of episodes. This reduces false positives
        from environments that stabilize quickly, while still protecting
        against environments that need more time.
    """

    def __init__(
        self,
        window: int = 50,
        warmup: int = 20,
        sensitivity: float = 2.0,
        # Auto-calibration settings
        min_warmup_episodes: int = 10,
        max_warmup_episodes: int = 50,
        stability_threshold: float = 0.1,
        stability_window: int = 5,
    ):
        """
        Args:
            window: Number of episodes for rolling window
            warmup: DEPRECATED - use min/max_warmup_episodes instead.
                    Still accepted for backwards compatibility.
            sensitivity: Number of std devs for "abnormal" threshold
            min_warmup_episodes: Minimum episodes before warmup can end (default 10)
            max_warmup_episodes: Maximum warmup - activate anyway after this (default 50)
            stability_threshold: Normalized variance threshold for stability (default 0.1)
            stability_window: Number of recent episodes to check for stability (default 5)
        """
        self.window = window
        self.warmup = warmup  # Keep for backwards compat
        self.sensitivity = sensitivity

        # Auto-calibration settings
        self.min_warmup_episodes = min_warmup_episodes
        self.max_warmup_episodes = max_warmup_episodes
        self.stability_threshold = stability_threshold
        self.stability_window = stability_window

        # Track when/why warmup ended
        self._warmup_ended = False
        self._warmup_ended_reason: Optional[str] = None
        self._warmup_ended_episode: Optional[int] = None

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
        """
        Check if adaptive layer is active (warmup complete).

        Uses auto-calibration: warmup ends when either:
        1. Max warmup episodes reached (safety valve)
        2. Min warmup reached AND variance has stabilized
        """
        if self._warmup_ended:
            return True

        # Check if warmup should end
        if self._check_warmup_complete():
            self._warmup_ended = True
            return True

        return False

    def _check_warmup_complete(self) -> bool:
        """
        Check if warmup should complete based on auto-calibration.

        Returns True if:
        - Max warmup episodes reached, OR
        - Min warmup reached AND variance is stable
        """
        # Max warmup reached - activate anyway
        if self._episodes_seen >= self.max_warmup_episodes:
            self._warmup_ended_reason = f"max warmup ({self.max_warmup_episodes} episodes) reached"
            self._warmup_ended_episode = self._episodes_seen
            return True

        # Haven't hit minimum yet
        if self._episodes_seen < self.min_warmup_episodes:
            return False

        # Check if variance is stable
        if self._variance_is_stable():
            self._warmup_ended_reason = f"stabilized at episode {self._episodes_seen}"
            self._warmup_ended_episode = self._episodes_seen
            return True

        return False

    def _variance_is_stable(self) -> bool:
        """
        Check if the variance of key metrics has stabilized.

        We check the normalized variance (coefficient of variation squared)
        of the last N values. If it's below the threshold for key metrics,
        we consider the baseline stable.
        """
        # Key metrics to check for stability
        stats_to_check = [
            ("reward", self._reward_stats),
            ("length", self._length_stats),
        ]

        for name, stats in stats_to_check:
            if stats.count < self.stability_window:
                return False  # Not enough data

            # Get recent variance
            recent_var = stats.get_recent_variance(self.stability_window)

            # Normalize by mean to get coefficient of variation squared
            # This makes the threshold comparable across different scales
            mean = stats.mean
            if mean == 0:
                # If mean is 0, use absolute variance check
                if recent_var > self.stability_threshold:
                    return False
            else:
                # Coefficient of variation squared
                cv_squared = recent_var / (mean ** 2)
                if cv_squared > self.stability_threshold:
                    return False

        return True

    @property
    def warmup_ended_reason(self) -> Optional[str]:
        """Get the reason warmup ended (for debug output)."""
        return self._warmup_ended_reason

    @property
    def warmup_ended_episode(self) -> Optional[int]:
        """Get the episode when warmup ended."""
        return self._warmup_ended_episode

    @property
    def warmup_progress(self) -> float:
        """
        Get warmup progress as a fraction (0.0 to 1.0).

        With auto-calibration, progress is based on min_warmup_episodes
        but can complete early if variance stabilizes.
        """
        if self._warmup_ended:
            return 1.0
        return min(1.0, self._episodes_seen / self.min_warmup_episodes)

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
            # Auto-calibration info
            "min_warmup_episodes": self.min_warmup_episodes,
            "max_warmup_episodes": self.max_warmup_episodes,
            "stability_threshold": self.stability_threshold,
            "warmup_ended_reason": self._warmup_ended_reason,
            "warmup_ended_episode": self._warmup_ended_episode,
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
        # Reset auto-calibration state
        self._warmup_ended = False
        self._warmup_ended_reason = None
        self._warmup_ended_episode = None


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
