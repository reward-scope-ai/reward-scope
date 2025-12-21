"""
Tests for reward_scope.core.decomposer
"""

import pytest
import numpy as np

from reward_scope.core.decomposer import (
    RewardDecomposer,
    RewardComponent,
    IsaacLabDecomposer,
)


class TestRewardComponent:
    """Test RewardComponent dataclass."""

    def test_create_minimal(self):
        """Test creating RewardComponent with minimal fields."""
        comp = RewardComponent(name="test")
        assert comp.name == "test"
        assert comp.description == ""
        assert comp.expected_range == (-np.inf, np.inf)
        assert comp.is_sparse is False
        assert comp.weight == 1.0

    def test_create_with_all_fields(self):
        """Test creating RewardComponent with all fields."""
        comp = RewardComponent(
            name="distance",
            description="Distance to goal",
            expected_range=(-10.0, 0.0),
            is_sparse=False,
            weight=2.0,
        )
        assert comp.name == "distance"
        assert comp.description == "Distance to goal"
        assert comp.expected_range == (-10.0, 0.0)
        assert comp.weight == 2.0


class TestRewardDecomposer:
    """Test RewardDecomposer class."""

    def test_init_defaults(self):
        """Test default initialization."""
        decomposer = RewardDecomposer()
        assert decomposer.auto_extract_prefix is None
        assert decomposer.track_residual is True
        assert len(decomposer.components) == 0
        assert len(decomposer.component_fns) == 0

    def test_init_with_prefix(self):
        """Test initialization with auto-extract prefix."""
        decomposer = RewardDecomposer(auto_extract_prefix="reward_")
        assert decomposer.auto_extract_prefix == "reward_"

    def test_register_component(self):
        """Test registering a component."""
        decomposer = RewardDecomposer()
        decomposer.register_component(
            "distance",
            lambda o, a, i: i.get("distance", 0),
            description="Distance to goal",
        )

        assert "distance" in decomposer.components
        assert "distance" in decomposer.component_fns
        assert decomposer.components["distance"].name == "distance"

    def test_register_component_without_function(self):
        """Test registering a component without compute function."""
        decomposer = RewardDecomposer()
        decomposer.register_component("test", compute_fn=None)

        assert "test" in decomposer.components
        assert "test" not in decomposer.component_fns

    def test_decompose_with_registered_components(self):
        """Test decomposing with registered component functions."""
        decomposer = RewardDecomposer()
        decomposer.register_component("a", lambda o, a, i: i.get("a", 0))
        decomposer.register_component("b", lambda o, a, i: i.get("b", 0))

        result = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=10.0,
            info={"a": 3.0, "b": 7.0},
        )

        assert result["a"] == 3.0
        assert result["b"] == 7.0
        assert result.get("residual", 0) == 0.0

    def test_decompose_with_residual(self):
        """Test that residual is calculated correctly."""
        decomposer = RewardDecomposer(track_residual=True)
        decomposer.register_component("known", lambda o, a, i: 5.0)

        result = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=10.0,
            info={},
        )

        assert result["known"] == 5.0
        assert result["residual"] == 5.0

    def test_decompose_without_residual(self):
        """Test decomposing without residual tracking."""
        decomposer = RewardDecomposer(track_residual=False)
        decomposer.register_component("known", lambda o, a, i: 5.0)

        result = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=10.0,
            info={},
        )

        assert result["known"] == 5.0
        assert "residual" not in result

    def test_auto_extract_prefix(self):
        """Test auto-extraction from info dict with prefix."""
        decomposer = RewardDecomposer(auto_extract_prefix="reward_")

        result = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=10.0,
            info={
                "reward_distance": 3.0,
                "reward_energy": -1.0,
                "other_stuff": 999,
            },
        )

        assert "distance" in result
        assert "energy" in result
        assert "other_stuff" not in result
        assert result["distance"] == 3.0
        assert result["energy"] == -1.0

    def test_auto_extract_with_registered_components(self):
        """Test that registered components and auto-extract work together."""
        decomposer = RewardDecomposer(auto_extract_prefix="reward_")
        decomposer.register_component("manual", lambda o, a, i: 5.0)

        result = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=10.0,
            info={"reward_auto": 3.0},
        )

        assert result["manual"] == 5.0
        assert result["auto"] == 3.0

    def test_component_function_exception(self):
        """Test that exceptions in component functions are handled."""
        decomposer = RewardDecomposer()

        def buggy_fn(o, a, i):
            raise ValueError("Intentional error")

        decomposer.register_component("buggy", buggy_fn)

        result = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=10.0,
            info={},
        )

        # Should set to 0 on error
        assert result["buggy"] == 0.0

    def test_get_component_stats_empty(self):
        """Test getting stats when no data has been processed."""
        decomposer = RewardDecomposer()
        stats = decomposer.get_component_stats()
        assert stats == {}

    def test_get_component_stats(self):
        """Test getting component statistics."""
        decomposer = RewardDecomposer()
        decomposer.register_component("comp", lambda o, a, i: i.get("comp", 0))

        # Process some data
        for i in range(10):
            decomposer.decompose(
                observation=None,
                action=None,
                total_reward=float(i),
                info={"comp": float(i)},
            )

        stats = decomposer.get_component_stats()

        assert "comp" in stats
        assert stats["comp"]["count"] == 10
        assert stats["comp"]["mean"] == pytest.approx(4.5)
        assert stats["comp"]["min"] == 0.0
        assert stats["comp"]["max"] == 9.0
        assert stats["comp"]["std"] > 0

    def test_get_component_stats_with_residual(self):
        """Test that residual is included in stats."""
        decomposer = RewardDecomposer(track_residual=True)
        decomposer.register_component("known", lambda o, a, i: 5.0)

        for i in range(5):
            decomposer.decompose(
                observation=None,
                action=None,
                total_reward=10.0,
                info={},
            )

        stats = decomposer.get_component_stats()

        assert "known" in stats
        assert "residual" in stats
        assert stats["known"]["mean"] == 5.0
        assert stats["residual"]["mean"] == 5.0

    def test_check_dominance_no_components(self):
        """Test dominance check with no components."""
        decomposer = RewardDecomposer()
        dominant = decomposer.check_dominance()
        assert dominant == []

    def test_check_dominance_balanced(self):
        """Test dominance check with balanced components."""
        decomposer = RewardDecomposer()
        decomposer.register_component("a", lambda o, a, i: 5.0)
        decomposer.register_component("b", lambda o, a, i: 5.0)

        for i in range(10):
            decomposer.decompose(
                observation=None,
                action=None,
                total_reward=10.0,
                info={},
            )

        dominant = decomposer.check_dominance(threshold=0.8)
        # Neither component should be dominant (each is 50%)
        assert dominant == []

    def test_check_dominance_imbalanced(self):
        """Test dominance check with imbalanced components."""
        decomposer = RewardDecomposer()
        decomposer.register_component("dominant", lambda o, a, i: 9.0)
        decomposer.register_component("minor", lambda o, a, i: 1.0)

        for i in range(10):
            decomposer.decompose(
                observation=None,
                action=None,
                total_reward=10.0,
                info={},
            )

        dominant = decomposer.check_dominance(threshold=0.8)
        # Dominant component should be detected (90% of total)
        assert "dominant" in dominant

    def test_welford_algorithm(self):
        """Test that Welford's algorithm correctly computes variance."""
        decomposer = RewardDecomposer()
        decomposer.register_component("test", lambda o, a, i: i["value"])

        # Known values with known statistics
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        expected_mean = 5.0
        expected_variance = 4.0
        expected_std = 2.0

        for val in values:
            decomposer.decompose(
                observation=None,
                action=None,
                total_reward=val,
                info={"value": val},
            )

        stats = decomposer.get_component_stats()

        assert stats["test"]["mean"] == pytest.approx(expected_mean)
        assert stats["test"]["std"] == pytest.approx(expected_std)

    def test_handles_negative_rewards(self):
        """Test that negative rewards are handled correctly."""
        decomposer = RewardDecomposer()
        decomposer.register_component("negative", lambda o, a, i: -5.0)

        result = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=-5.0,
            info={},
        )

        assert result["negative"] == -5.0

    def test_multiple_decompose_calls(self):
        """Test that multiple decompose calls update statistics correctly."""
        decomposer = RewardDecomposer()
        decomposer.register_component("comp", lambda o, a, i: i.get("val", 0))

        # First call
        result1 = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=5.0,
            info={"val": 5.0},
        )

        # Second call
        result2 = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=10.0,
            info={"val": 10.0},
        )

        stats = decomposer.get_component_stats()
        assert stats["comp"]["count"] == 2
        assert stats["comp"]["mean"] == pytest.approx(7.5)


class TestIsaacLabDecomposer:
    """Test IsaacLabDecomposer class."""

    def test_from_reward_cfg(self):
        """Test creating IsaacLabDecomposer from reward config."""
        reward_cfg = {
            "track_lin_vel_xy_exp": {"weight": 1.0},
            "track_ang_vel_z_exp": {"weight": 0.5},
            "lin_vel_z_l2": {"weight": -2.0},
        }

        decomposer = IsaacLabDecomposer.from_reward_cfg(reward_cfg)

        assert "track_lin_vel_xy_exp" in decomposer.components
        assert "track_ang_vel_z_exp" in decomposer.components
        assert "lin_vel_z_l2" in decomposer.components

        assert decomposer.components["track_lin_vel_xy_exp"].weight == 1.0
        assert decomposer.components["track_ang_vel_z_exp"].weight == 0.5
        assert decomposer.components["lin_vel_z_l2"].weight == -2.0

    def test_decompose_with_isaac_lab_info(self):
        """Test decomposing Isaac Lab style info dict."""
        reward_cfg = {
            "term1": {"weight": 1.0},
            "term2": {"weight": 0.5},
        }

        decomposer = IsaacLabDecomposer.from_reward_cfg(reward_cfg)

        result = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=7.5,
            info={"term1": 5.0, "term2": 2.5},
        )

        assert result["term1"] == 5.0
        assert result["term2"] == 2.5

    def test_empty_reward_cfg(self):
        """Test with empty reward config."""
        decomposer = IsaacLabDecomposer.from_reward_cfg({})
        assert len(decomposer.components) == 0

        result = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=0.0,
            info={},
        )

        # Should only have residual if tracking is enabled
        assert "residual" not in result or abs(result.get("residual", 0.0)) < 1e-6


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_reward(self):
        """Test handling of zero reward."""
        decomposer = RewardDecomposer()
        decomposer.register_component("comp", lambda o, a, i: 0.0)

        result = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=0.0,
            info={},
        )

        assert result["comp"] == 0.0
        assert abs(result.get("residual", 0.0)) < 1e-6

    def test_very_large_reward(self):
        """Test handling of very large rewards."""
        decomposer = RewardDecomposer()
        decomposer.register_component("large", lambda o, a, i: 1e6)

        result = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=1e6,
            info={},
        )

        assert result["large"] == 1e6

    def test_very_small_residual(self):
        """Test that very small residuals are not added."""
        decomposer = RewardDecomposer(track_residual=True)
        decomposer.register_component("comp", lambda o, a, i: 10.0)

        result = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=10.0,
            info={},
        )

        # Residual should not be added if negligible
        assert "residual" not in result or abs(result.get("residual", 0.0)) < 1e-6

    def test_non_numeric_info_value(self):
        """Test that non-numeric values in info are skipped during auto-extract."""
        decomposer = RewardDecomposer(auto_extract_prefix="reward_")

        result = decomposer.decompose(
            observation=None,
            action=None,
            total_reward=5.0,
            info={
                "reward_valid": 5.0,
                "reward_invalid": "not a number",
                "reward_none": None,
            },
        )

        assert "valid" in result
        assert result["valid"] == 5.0
        # Invalid values should be skipped
        assert "invalid" not in result
        assert "none" not in result
