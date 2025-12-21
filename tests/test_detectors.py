"""
Tests for reward_scope.core.detectors
"""

import pytest
import numpy as np

from reward_scope.core.detectors import (
    HackingDetectorSuite,
    HackingAlert,
    HackingType,
    StateCyclingDetector,
    ActionRepetitionDetector,
    ComponentImbalanceDetector,
    RewardSpikingDetector,
    BoundaryExploitationDetector,
)


class TestHackingAlert:
    """Test HackingAlert dataclass."""

    def test_create_alert(self):
        """Test creating a hacking alert."""
        alert = HackingAlert(
            type=HackingType.ACTION_REPETITION,
            severity=0.9,
            step=100,
            episode=5,
            description="Test alert",
            evidence={"key": "value"},
            suggested_fix="Fix it",
        )

        assert alert.type == HackingType.ACTION_REPETITION
        assert alert.severity == 0.9
        assert alert.step == 100
        assert alert.episode == 5


class TestActionRepetitionDetector:
    """Test ActionRepetitionDetector."""

    def test_detects_repeated_actions(self):
        """Test detection of repeated actions."""
        detector = ActionRepetitionDetector(
            window_size=10,
            repetition_threshold=0.8,
        )

        # Feed same action multiple times
        alert = None
        for i in range(15):
            alert = detector.update(
                step=i,
                episode=0,
                observation=np.array([0.0]),
                action=0,  # Same action
                reward=1.0,
                reward_components={},
                done=False,
                info={},
            )

        # Should have detected repetition
        assert alert is not None or len(detector.alerts) > 0
        if alert:
            assert alert.type == HackingType.ACTION_REPETITION
            assert alert.severity > 0.7

    def test_no_detection_with_varied_actions(self):
        """Test that varied actions don't trigger detection."""
        detector = ActionRepetitionDetector(
            window_size=10,
            repetition_threshold=0.8,
        )

        # Feed varied actions
        for i in range(20):
            alert = detector.update(
                step=i,
                episode=0,
                observation=np.array([0.0]),
                action=i % 5,  # Varied actions
                reward=1.0,
                reward_components={},
                done=False,
                info={},
            )

        # Should not detect repetition
        assert len(detector.alerts) == 0

    def test_handles_continuous_actions(self):
        """Test handling of continuous action spaces."""
        detector = ActionRepetitionDetector(
            window_size=10,
            repetition_threshold=0.8,
        )

        # Feed similar continuous actions
        for i in range(15):
            alert = detector.update(
                step=i,
                episode=0,
                observation=np.array([0.0]),
                action=np.array([0.5, 0.5]),  # Same continuous action
                reward=1.0,
                reward_components={},
                done=False,
                info={},
            )

        # Should detect repetition
        assert len(detector.alerts) > 0

    def test_reset(self):
        """Test reset clears action buffer."""
        detector = ActionRepetitionDetector(window_size=10)

        for i in range(5):
            detector.update(
                step=i, episode=0, observation=None, action=0,
                reward=1.0, reward_components={}, done=False, info={}
            )

        assert len(detector.action_buffer) == 5

        detector.reset()
        assert len(detector.action_buffer) == 0


class TestComponentImbalanceDetector:
    """Test ComponentImbalanceDetector."""

    def test_detects_dominant_component(self):
        """Test detection of dominant reward component."""
        detector = ComponentImbalanceDetector(
            dominance_threshold=0.8,
            imbalance_episodes=3,
        )

        # Simulate 3 episodes with one dominant component
        for ep in range(3):
            for i in range(10):
                detector.update(
                    step=ep * 10 + i,
                    episode=ep,
                    observation=None,
                    action=0,
                    reward=10.0,
                    reward_components={
                        "dominant": 9.5,  # 95%
                        "minor": 0.5,     # 5%
                    },
                    done=False,
                    info={},
                )

            # Check at episode end
            alert = detector.on_episode_end({
                "dominant": 95.0,
                "minor": 5.0,
            })

            if ep >= 2:  # Should trigger on 3rd episode
                assert alert is not None
                assert alert.type == HackingType.COMPONENT_IMBALANCE

    def test_no_detection_with_balanced_components(self):
        """Test that balanced components don't trigger detection."""
        detector = ComponentImbalanceDetector(
            dominance_threshold=0.8,
            imbalance_episodes=3,
        )

        # Simulate episodes with balanced components
        for ep in range(3):
            for i in range(10):
                detector.update(
                    step=ep * 10 + i,
                    episode=ep,
                    observation=None,
                    action=0,
                    reward=10.0,
                    reward_components={
                        "comp1": 5.0,  # 50%
                        "comp2": 5.0,  # 50%
                    },
                    done=False,
                    info={},
                )

            alert = detector.on_episode_end({
                "comp1": 50.0,
                "comp2": 50.0,
            })

        # Should not detect imbalance
        assert len(detector.alerts) == 0

    def test_ignores_residual(self):
        """Test that residual component is ignored."""
        detector = ComponentImbalanceDetector()

        detector.update(
            step=0, episode=0, observation=None, action=0,
            reward=10.0,
            reward_components={"comp": 5.0, "residual": 5.0},
            done=False, info={}
        )

        # Residual should not be counted
        assert "residual" not in detector.current_episode_components


class TestRewardSpikingDetector:
    """Test RewardSpikingDetector."""

    def test_detects_reward_spike(self):
        """Test detection of reward spikes."""
        detector = RewardSpikingDetector(
            window_size=100,
            spike_std_threshold=5.0,
        )

        # Feed normal rewards
        for i in range(60):
            detector.update(
                step=i,
                episode=0,
                observation=None,
                action=0,
                reward=1.0,  # Normal reward
                reward_components={},
                done=False,
                info={},
            )

        # Feed spike
        alert = detector.update(
            step=60,
            episode=0,
            observation=None,
            action=0,
            reward=50.0,  # Huge spike
            reward_components={},
            done=False,
            info={},
        )

        # Should detect spike
        assert alert is not None or len(detector.alerts) > 0
        if alert:
            assert alert.type == HackingType.REWARD_SPIKING

    def test_no_detection_with_stable_rewards(self):
        """Test that stable rewards don't trigger detection."""
        detector = RewardSpikingDetector(
            spike_std_threshold=5.0,
        )

        # Feed stable rewards
        for i in range(100):
            alert = detector.update(
                step=i, episode=0, observation=None, action=0,
                reward=1.0 + 0.1 * np.sin(i * 0.1),  # Small variation
                reward_components={}, done=False, info={}
            )

        # Should not detect spikes
        assert len(detector.alerts) == 0

    def test_handles_negative_rewards(self):
        """Test handling of negative rewards."""
        detector = RewardSpikingDetector()

        for i in range(60):
            detector.update(
                step=i, episode=0, observation=None, action=0,
                reward=-1.0,
                reward_components={}, done=False, info={}
            )

        # Large negative spike
        alert = detector.update(
            step=60, episode=0, observation=None, action=0,
            reward=-50.0,
            reward_components={}, done=False, info={}
        )

        # Should detect spike (absolute value)
        assert len(detector.reward_buffer) > 0


class TestStateCyclingDetector:
    """Test StateCyclingDetector."""

    def test_detects_state_cycle(self):
        """Test detection of repeating state patterns."""
        detector = StateCyclingDetector(
            window_size=50,
            cycle_threshold=0.8,
            min_cycle_length=3,
            max_cycle_length=10,
        )

        # Create a repeating pattern: [1, 2, 3, 1, 2, 3, ...]
        cycle = [np.array([1.0, 0.0]), np.array([2.0, 0.0]), np.array([3.0, 0.0])]

        alert = None
        for i in range(30):
            obs = cycle[i % len(cycle)]
            alert = detector.update(
                step=i,
                episode=0,
                observation=obs,
                action=0,
                reward=1.0,
                reward_components={},
                done=False,
                info={},
            )

        # Should detect cycle
        assert alert is not None or len(detector.alerts) > 0
        if alert:
            assert alert.type == HackingType.STATE_CYCLING

    def test_no_detection_with_varied_states(self):
        """Test that varied states don't trigger detection."""
        detector = StateCyclingDetector(
            cycle_threshold=0.8,
            min_cycle_length=3,
        )

        # Feed varied observations
        for i in range(30):
            obs = np.array([float(i), float(i * 2)])
            detector.update(
                step=i, episode=0, observation=obs, action=0,
                reward=1.0, reward_components={}, done=False, info={}
            )

        # Should not detect cycles
        assert len(detector.alerts) == 0

    def test_reset(self):
        """Test reset clears observation buffer."""
        detector = StateCyclingDetector()

        for i in range(5):
            detector.update(
                step=i, episode=0, observation=np.array([float(i)]),
                action=0, reward=1.0, reward_components={},
                done=False, info={}
            )

        assert len(detector.observation_buffer) == 5

        detector.reset()
        assert len(detector.observation_buffer) == 0


class TestBoundaryExploitationDetector:
    """Test BoundaryExploitationDetector."""

    def test_detects_observation_boundary_exploitation(self):
        """Test detection of staying at observation boundaries."""
        obs_bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))

        detector = BoundaryExploitationDetector(
            window_size=20,
            boundary_threshold=0.95,
            boundary_frequency_threshold=0.5,
            observation_bounds=obs_bounds,
        )

        # Feed observations at boundary
        for i in range(25):
            alert = detector.update(
                step=i,
                episode=0,
                observation=np.array([0.99, 0.99]),  # At high boundary
                action=np.array([0.0]),
                reward=1.0,
                reward_components={},
                done=False,
                info={},
            )

        # Should detect boundary exploitation
        assert len(detector.alerts) > 0 or alert is not None

    def test_detects_action_boundary_exploitation(self):
        """Test detection of saturated actions."""
        action_bounds = (np.array([-1.0]), np.array([1.0]))

        detector = BoundaryExploitationDetector(
            window_size=20,
            boundary_threshold=0.95,
            boundary_frequency_threshold=0.5,
            action_bounds=action_bounds,
        )

        # Feed actions at boundary
        for i in range(25):
            alert = detector.update(
                step=i,
                episode=0,
                observation=np.array([0.0, 0.0]),
                action=np.array([0.99]),  # At boundary
                reward=1.0,
                reward_components={},
                done=False,
                info={},
            )

        # Should detect boundary exploitation
        assert len(detector.alerts) > 0 or alert is not None

    def test_no_detection_with_center_values(self):
        """Test that centered values don't trigger detection."""
        obs_bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))

        detector = BoundaryExploitationDetector(
            boundary_frequency_threshold=0.5,
            observation_bounds=obs_bounds,
        )

        # Feed observations near center
        for i in range(30):
            detector.update(
                step=i, episode=0,
                observation=np.array([0.0, 0.0]),  # Center
                action=0, reward=1.0,
                reward_components={}, done=False, info={}
            )

        # Should not detect boundary exploitation
        assert len(detector.alerts) == 0

    def test_handles_scalar_bounds(self):
        """Test handling of scalar bounds."""
        detector = BoundaryExploitationDetector(
            observation_bounds=(np.array([-1.0]), np.array([1.0])),
        )

        # Should handle scalar observations
        detector.update(
            step=0, episode=0, observation=0.99,
            action=0, reward=1.0, reward_components={},
            done=False, info={}
        )

        # No errors should occur
        assert True


class TestHackingDetectorSuite:
    """Test HackingDetectorSuite."""

    def test_init_all_detectors(self):
        """Test initialization with all detectors enabled."""
        suite = HackingDetectorSuite()

        # Should have all 5 detectors
        assert len(suite.detectors) == 5

    def test_init_selective_detectors(self):
        """Test initialization with selective detectors."""
        suite = HackingDetectorSuite(
            enable_state_cycling=True,
            enable_action_repetition=True,
            enable_component_imbalance=False,
            enable_reward_spiking=False,
            enable_boundary_exploitation=False,
        )

        # Should have only 2 detectors
        assert len(suite.detectors) == 2

    def test_update_runs_all_detectors(self):
        """Test that update runs all enabled detectors."""
        suite = HackingDetectorSuite()

        alerts = suite.update(
            step=0,
            episode=0,
            observation=np.array([0.0]),
            action=0,
            reward=1.0,
            reward_components={"comp": 1.0},
            done=False,
            info={},
        )

        # Should return a list (may be empty)
        assert isinstance(alerts, list)

    def test_get_all_alerts(self):
        """Test getting all alerts from all detectors."""
        suite = HackingDetectorSuite()

        # Generate some alerts by feeding repetitive data
        for i in range(60):
            suite.update(
                step=i, episode=0,
                observation=np.array([1.0]),
                action=0,  # Same action
                reward=1.0,
                reward_components={"comp": 1.0},
                done=False, info={}
            )

        alerts = suite.get_all_alerts()

        # Should return sorted list
        assert isinstance(alerts, list)
        # Check if sorted by step
        if len(alerts) > 1:
            assert alerts[0].step <= alerts[-1].step

    def test_get_hacking_score_no_alerts(self):
        """Test hacking score with no alerts."""
        suite = HackingDetectorSuite()

        score = suite.get_hacking_score()
        assert score == 0.0

    def test_get_hacking_score_with_alerts(self):
        """Test hacking score calculation with alerts."""
        suite = HackingDetectorSuite()

        # Manually add an alert to a detector
        alert = HackingAlert(
            type=HackingType.ACTION_REPETITION,
            severity=0.9,
            step=10,
            episode=0,
            description="Test",
            evidence={},
            suggested_fix="Fix",
        )

        suite.detectors[0].alerts.append(alert)

        score = suite.get_hacking_score()
        assert 0.0 <= score <= 1.0
        assert score > 0.0

    def test_on_episode_end(self):
        """Test episode-end processing."""
        suite = HackingDetectorSuite()

        # Should not crash with empty stats
        alerts = suite.on_episode_end({})
        assert isinstance(alerts, list)

        # Should process component totals
        alerts = suite.on_episode_end({
            "component_totals": {"comp1": 10.0, "comp2": 1.0}
        })
        assert isinstance(alerts, list)

    def test_reset(self):
        """Test reset resets all detectors."""
        suite = HackingDetectorSuite()

        # Feed some data
        for i in range(10):
            suite.update(
                step=i, episode=0, observation=np.array([0.0]),
                action=0, reward=1.0, reward_components={},
                done=False, info={}
            )

        suite.reset()

        # Detectors should be reset (buffers cleared)
        for detector in suite.detectors:
            if isinstance(detector, ActionRepetitionDetector):
                assert len(detector.action_buffer) == 0
            elif isinstance(detector, StateCyclingDetector):
                assert len(detector.observation_buffer) == 0

    def test_with_bounds(self):
        """Test initialization with observation/action bounds."""
        obs_bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
        action_bounds = (np.array([-1.0]), np.array([1.0]))

        suite = HackingDetectorSuite(
            observation_bounds=obs_bounds,
            action_bounds=action_bounds,
        )

        # Should pass bounds to boundary detector
        for detector in suite.detectors:
            if isinstance(detector, BoundaryExploitationDetector):
                assert detector.observation_bounds == obs_bounds
                assert detector.action_bounds == action_bounds


class TestIntegration:
    """Integration tests combining multiple detectors."""

    def test_multiple_hacking_types(self):
        """Test detecting multiple types of hacking simultaneously."""
        suite = HackingDetectorSuite()

        # Feed data that triggers multiple detectors
        for i in range(60):
            alerts = suite.update(
                step=i,
                episode=0,
                observation=np.array([1.0, 1.0]),  # Same state
                action=0,  # Same action
                reward=1.0,
                reward_components={"dominant": 0.95, "minor": 0.05},
                done=False,
                info={},
            )

        # Should potentially detect action repetition and state cycling
        all_alerts = suite.get_all_alerts()

        # May have detected multiple types
        alert_types = set(a.type for a in all_alerts)

        # At least should have processed without errors
        assert isinstance(all_alerts, list)

    def test_episode_lifecycle(self):
        """Test full episode lifecycle with detectors."""
        suite = HackingDetectorSuite()

        # Simulate full episode
        for i in range(50):
            alerts = suite.update(
                step=i,
                episode=0,
                observation=np.random.randn(4),
                action=np.random.randint(0, 2),
                reward=np.random.randn(),
                reward_components={"comp1": np.random.randn()},
                done=False,
                info={},
            )

        # End episode
        episode_alerts = suite.on_episode_end({
            "component_totals": {"comp1": 10.0}
        })

        # Reset for next episode
        suite.reset()

        # Should complete without errors
        assert isinstance(episode_alerts, list)
