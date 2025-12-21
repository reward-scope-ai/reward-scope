"""
Reward Decomposition Module

Allows users to define reward components and tracks them separately.
Supports both explicit decomposition (user provides components) and
inferred decomposition (when reward function returns dict).
"""

from typing import Callable, Dict, List, Optional, Union, Any
from dataclasses import dataclass
import numpy as np


@dataclass
class RewardComponent:
    """Definition of a reward component."""
    name: str
    description: str = ""
    expected_range: tuple = (-np.inf, np.inf)  # For anomaly detection
    is_sparse: bool = False  # Sparse rewards need different analysis
    weight: float = 1.0  # If known, the weight in composite reward


class RewardDecomposer:
    """
    Decomposes rewards into trackable components.

    Usage (Explicit):
        decomposer = RewardDecomposer()
        decomposer.register_component("distance", lambda obs, action, info: -info["distance"])
        decomposer.register_component("energy", lambda obs, action, info: -info["energy_cost"])

        components = decomposer.decompose(obs, action, reward, info)
        # Returns: {"distance": -0.5, "energy": -0.1, "residual": 0.0}

    Usage (From Info Dict):
        # If env.step() returns info={"reward_distance": -0.5, "reward_energy": -0.1}
        decomposer = RewardDecomposer(auto_extract_prefix="reward_")
        components = decomposer.decompose(obs, action, reward, info)
    """

    def __init__(
        self,
        auto_extract_prefix: Optional[str] = None,
        track_residual: bool = True,
    ):
        """
        Args:
            auto_extract_prefix: If set, auto-extract components from info dict
                                 matching this prefix (e.g., "reward_")
            track_residual: Whether to track unexplained reward as "residual"
        """
        self.components: Dict[str, RewardComponent] = {}
        self.component_fns: Dict[str, Callable] = {}
        self.auto_extract_prefix = auto_extract_prefix
        self.track_residual = track_residual

        # Statistics tracking (Welford's algorithm)
        self._component_stats: Dict[str, Dict[str, float]] = {}
        self._component_counts: Dict[str, int] = {}
        self._component_means: Dict[str, float] = {}
        self._component_m2s: Dict[str, float] = {}  # For variance calculation
        self._component_mins: Dict[str, float] = {}
        self._component_maxs: Dict[str, float] = {}

    def register_component(
        self,
        name: str,
        compute_fn: Optional[Callable[[Any, Any, Dict], float]] = None,
        description: str = "",
        expected_range: tuple = (-np.inf, np.inf),
        is_sparse: bool = False,
        weight: float = 1.0,
    ) -> None:
        """
        Register a reward component.

        Args:
            name: Unique name for this component
            compute_fn: Function(obs, action, info) -> float
                        If None, will try to extract from info dict
            description: Human-readable description
            expected_range: (min, max) for anomaly detection
            is_sparse: Whether this component is sparse (mostly zero)
            weight: Weight in the composite reward (if known)
        """
        component = RewardComponent(
            name=name,
            description=description,
            expected_range=expected_range,
            is_sparse=is_sparse,
            weight=weight,
        )

        self.components[name] = component

        if compute_fn is not None:
            self.component_fns[name] = compute_fn

        # Initialize statistics
        self._component_counts[name] = 0
        self._component_means[name] = 0.0
        self._component_m2s[name] = 0.0
        self._component_mins[name] = np.inf
        self._component_maxs[name] = -np.inf

    def decompose(
        self,
        observation: Any,
        action: Any,
        total_reward: float,
        info: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Decompose a reward into components.

        Returns:
            Dict mapping component names to their values.
            Includes "residual" if track_residual=True and components
            don't sum to total_reward.
        """
        components_dict: Dict[str, float] = {}

        # Extract from registered component functions
        for name, fn in self.component_fns.items():
            try:
                value = float(fn(observation, action, info))
                components_dict[name] = value
            except Exception as e:
                # If component function fails, set to 0
                components_dict[name] = 0.0

        # Auto-extract from info dict if prefix is set
        if self.auto_extract_prefix:
            for key, value in info.items():
                if key.startswith(self.auto_extract_prefix):
                    # Extract component name by removing prefix
                    comp_name = key[len(self.auto_extract_prefix):]
                    if comp_name not in components_dict:
                        try:
                            components_dict[comp_name] = float(value)
                        except (TypeError, ValueError):
                            # Skip if value can't be converted to float
                            pass

        # Calculate residual if enabled
        if self.track_residual:
            component_sum = sum(components_dict.values())
            residual = total_reward - component_sum
            if abs(residual) > 1e-6:  # Only add if non-negligible
                components_dict["residual"] = residual

        # Update statistics for each component
        for name, value in components_dict.items():
            self._update_stats(name, value)

        return components_dict

    def _update_stats(self, name: str, value: float) -> None:
        """Update running statistics using Welford's algorithm."""
        if name not in self._component_counts:
            self._component_counts[name] = 0
            self._component_means[name] = 0.0
            self._component_m2s[name] = 0.0
            self._component_mins[name] = np.inf
            self._component_maxs[name] = -np.inf

        # Update count
        count = self._component_counts[name] + 1
        self._component_counts[name] = count

        # Update mean and M2 (for variance) using Welford's algorithm
        delta = value - self._component_means[name]
        self._component_means[name] += delta / count
        delta2 = value - self._component_means[name]
        self._component_m2s[name] += delta * delta2

        # Update min and max
        self._component_mins[name] = min(self._component_mins[name], value)
        self._component_maxs[name] = max(self._component_maxs[name], value)

    def get_component_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get running statistics for each component.

        Returns:
            {component_name: {"mean": x, "std": y, "min": z, "max": w, "count": n}}
        """
        stats = {}

        for name in self._component_counts:
            count = self._component_counts[name]

            if count == 0:
                stats[name] = {
                    "mean": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                    "count": 0,
                }
            else:
                mean = self._component_means[name]
                variance = self._component_m2s[name] / count if count > 0 else 0.0
                std = np.sqrt(variance)

                stats[name] = {
                    "mean": mean,
                    "std": std,
                    "min": self._component_mins[name],
                    "max": self._component_maxs[name],
                    "count": count,
                }

        return stats

    def check_dominance(self, threshold: float = 0.8) -> List[str]:
        """
        Check if any single component dominates the total reward.

        Returns list of component names that contribute > threshold of total.
        """
        stats = self.get_component_stats()
        dominant_components = []

        if not stats:
            return dominant_components

        # Calculate total absolute contribution across all components
        total_abs_contribution = sum(
            abs(s["mean"]) * s["count"]
            for s in stats.values()
        )

        if total_abs_contribution == 0:
            return dominant_components

        # Check each component's contribution ratio
        for name, s in stats.items():
            component_contribution = abs(s["mean"]) * s["count"]
            ratio = component_contribution / total_abs_contribution

            if ratio > threshold:
                dominant_components.append(name)

        return dominant_components


class IsaacLabDecomposer(RewardDecomposer):
    """
    Specialized decomposer for Isaac Lab reward terms.

    Isaac Lab rewards are typically defined as:
        reward_cfg = {
            "track_lin_vel_xy_exp": {"weight": 1.0},
            "track_ang_vel_z_exp": {"weight": 0.5},
            "lin_vel_z_l2": {"weight": -2.0},
            ...
        }

    This decomposer can parse the reward config and auto-register components.
    """

    @classmethod
    def from_reward_cfg(cls, reward_cfg: Dict) -> "IsaacLabDecomposer":
        """Create decomposer from Isaac Lab reward configuration."""
        decomposer = cls()

        for term_name, term_config in reward_cfg.items():
            weight = term_config.get("weight", 1.0)

            decomposer.register_component(
                name=term_name,
                compute_fn=lambda obs, act, info, tn=term_name: info.get(tn, 0.0),
                description=f"Isaac Lab reward term: {term_name}",
                weight=weight,
            )

        return decomposer
