"""
Tests for reward_scope.core.collector
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import time
import numpy as np

from reward_scope.core.collector import (
    DataCollector,
    StepData,
    EpisodeData,
    _serialize_to_json,
    _deserialize_from_json,
)


class TestSerialization:
    """Test JSON serialization helpers."""

    def test_serialize_none(self):
        assert _serialize_to_json(None) is None

    def test_serialize_simple_types(self):
        result = _serialize_to_json({"a": 1, "b": 2.5, "c": "test"})
        assert '"a": 1' in result
        assert '"b": 2.5' in result
        assert '"c": "test"' in result

    def test_serialize_numpy_array(self):
        arr = np.array([1, 2, 3])
        result = _serialize_to_json(arr)
        assert "[1, 2, 3]" in result

    def test_serialize_numpy_scalar(self):
        val = np.float32(3.14)
        result = _serialize_to_json(val)
        assert "3.14" in result

    def test_deserialize_none(self):
        assert _deserialize_from_json(None) is None

    def test_deserialize_dict(self):
        json_str = '{"a": 1, "b": 2}'
        result = _deserialize_from_json(json_str)
        assert result == {"a": 1, "b": 2}


class TestDataCollector:
    """Test DataCollector class."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def collector(self, temp_dir):
        """Create a DataCollector instance for testing."""
        collector = DataCollector(
            run_name="test_run",
            storage_dir=temp_dir,
            buffer_size=10,
        )
        yield collector
        collector.close()

    def test_init_creates_database(self, temp_dir):
        """Test that initialization creates database file."""
        collector = DataCollector(run_name="test_init", storage_dir=temp_dir)
        db_path = Path(temp_dir) / "test_init.db"
        assert db_path.exists()
        collector.close()

    def test_log_step(self, collector):
        """Test logging a single step."""
        step_data = StepData(
            step=0,
            episode=0,
            timestamp=time.time(),
            observation=[1.0, 2.0, 3.0],
            action=1,
            reward=1.0,
            done=False,
            truncated=False,
            info={"test": "value"},
            reward_components={"comp1": 0.5, "comp2": 0.5},
        )

        collector.log_step(step_data)
        assert len(collector.step_buffer) == 1
        assert len(collector.current_episode_steps) == 1

    def test_buffer_flush(self, collector):
        """Test that buffer flushes after reaching buffer_size."""
        # Log buffer_size steps
        for i in range(collector.buffer_size):
            step_data = StepData(
                step=i,
                episode=0,
                timestamp=time.time(),
                observation=[float(i)],
                action=0,
                reward=1.0,
                done=False,
                truncated=False,
                info={},
            )
            collector.log_step(step_data)

        # Buffer should be flushed
        assert len(collector.step_buffer) == 0

    def test_end_episode(self, collector):
        """Test episode ending and aggregation."""
        # Log some steps
        for i in range(5):
            step_data = StepData(
                step=i,
                episode=0,
                timestamp=time.time(),
                observation=[float(i)],
                action=0,
                reward=1.0,
                done=False,
                truncated=False,
                info={},
                reward_components={"comp1": 0.6, "comp2": 0.4},
            )
            collector.log_step(step_data)

        # End episode
        episode_data = collector.end_episode()

        assert episode_data.episode == 0
        assert episode_data.total_reward == 5.0
        assert episode_data.length == 5
        assert episode_data.component_totals["comp1"] == pytest.approx(3.0)
        assert episode_data.component_totals["comp2"] == pytest.approx(2.0)

        # Episode tracking should be reset
        assert len(collector.current_episode_steps) == 0

    def test_get_recent_steps(self, collector):
        """Test retrieving recent steps."""
        # Log some steps and flush
        for i in range(15):
            step_data = StepData(
                step=i,
                episode=0,
                timestamp=time.time(),
                observation=[float(i)],
                action=i % 3,
                reward=float(i),
                done=False,
                truncated=False,
                info={},
            )
            collector.log_step(step_data)

        collector._flush_step_buffer()

        # Get recent steps
        recent = collector.get_recent_steps(n=10)

        assert len(recent) == 10
        # Should be in chronological order (most recent last)
        assert recent[-1].step == 14
        assert recent[0].step == 5

    def test_get_episode_history(self, collector):
        """Test retrieving episode history."""
        # Create multiple episodes
        for ep in range(3):
            for i in range(5):
                step_data = StepData(
                    step=ep * 5 + i,
                    episode=ep,
                    timestamp=time.time(),
                    observation=[float(i)],
                    action=0,
                    reward=1.0,
                    done=False,
                    truncated=False,
                    info={},
                )
                collector.log_step(step_data)

            collector.end_episode()

        # Get episode history
        episodes = collector.get_episode_history(n=2)

        assert len(episodes) == 2
        # Should be in chronological order
        assert episodes[-1].episode == 2
        assert episodes[0].episode == 1

    def test_query_steps_by_episode(self, collector):
        """Test querying steps by episode number."""
        # Create two episodes
        for ep in range(2):
            for i in range(5):
                step_data = StepData(
                    step=ep * 5 + i,
                    episode=ep,
                    timestamp=time.time(),
                    observation=[float(i)],
                    action=0,
                    reward=1.0,
                    done=False,
                    truncated=False,
                    info={},
                )
                collector.log_step(step_data)

        collector._flush_step_buffer()

        # Query episode 1
        steps = collector.query_steps(episode=1)

        assert len(steps) == 5
        assert all(s.episode == 1 for s in steps)
        assert steps[0].step == 5
        assert steps[-1].step == 9

    def test_query_steps_by_range(self, collector):
        """Test querying steps by step range."""
        # Log steps
        for i in range(20):
            step_data = StepData(
                step=i,
                episode=0,
                timestamp=time.time(),
                observation=[float(i)],
                action=0,
                reward=1.0,
                done=False,
                truncated=False,
                info={},
            )
            collector.log_step(step_data)

        collector._flush_step_buffer()

        # Query range
        steps = collector.query_steps(start_step=5, end_step=10)

        assert len(steps) == 6  # Inclusive
        assert steps[0].step == 5
        assert steps[-1].step == 10

    def test_handles_numpy_observations(self, collector):
        """Test that numpy arrays in observations are handled correctly."""
        step_data = StepData(
            step=0,
            episode=0,
            timestamp=time.time(),
            observation=np.array([1.0, 2.0, 3.0]),
            action=np.array([0.5]),
            reward=1.0,
            done=False,
            truncated=False,
            info={"array": np.array([4.0, 5.0])},
        )

        collector.log_step(step_data)
        collector._flush_step_buffer()

        # Retrieve and verify
        steps = collector.get_recent_steps(n=1)
        assert len(steps) == 1
        assert steps[0].observation == [1.0, 2.0, 3.0]
        assert steps[0].action == [0.5]

    def test_empty_episode(self, collector):
        """Test ending an episode with no steps."""
        episode_data = collector.end_episode()

        assert episode_data.total_reward == 0.0
        assert episode_data.length == 0

    def test_multiple_episodes(self, collector):
        """Test tracking multiple episodes."""
        total_episodes = 5

        for ep in range(total_episodes):
            for i in range(10):
                step_data = StepData(
                    step=ep * 10 + i,
                    episode=ep,
                    timestamp=time.time(),
                    observation=[float(i)],
                    action=0,
                    reward=1.0,
                    done=False,
                    truncated=False,
                    info={},
                    reward_components={"comp": 1.0},
                )
                collector.log_step(step_data)

            collector.end_episode()

        # Verify episode history
        episodes = collector.get_episode_history(n=total_episodes)
        assert len(episodes) == total_episodes
        assert all(ep.length == 10 for ep in episodes)
        assert all(ep.total_reward == 10.0 for ep in episodes)


class TestStepData:
    """Test StepData dataclass."""

    def test_create_minimal(self):
        """Test creating StepData with minimal required fields."""
        step = StepData(
            step=0,
            episode=0,
            timestamp=time.time(),
            observation=None,
            action=0,
            reward=1.0,
            done=False,
            truncated=False,
            info={},
        )
        assert step.reward_components == {}
        assert step.value_estimate is None
        assert step.action_probs is None

    def test_create_with_all_fields(self):
        """Test creating StepData with all fields."""
        step = StepData(
            step=0,
            episode=0,
            timestamp=time.time(),
            observation=[1, 2, 3],
            action=1,
            reward=1.0,
            done=True,
            truncated=False,
            info={"key": "value"},
            reward_components={"comp1": 0.5, "comp2": 0.5},
            value_estimate=0.8,
            action_probs=[0.1, 0.9],
        )
        assert step.reward_components["comp1"] == 0.5
        assert step.value_estimate == 0.8
        assert len(step.action_probs) == 2


class TestEpisodeData:
    """Test EpisodeData dataclass."""

    def test_create_minimal(self):
        """Test creating EpisodeData with minimal required fields."""
        episode = EpisodeData(
            episode=0,
            total_reward=100.0,
            length=50,
            start_time=0.0,
            end_time=10.0,
        )
        assert episode.component_totals == {}
        assert episode.hacking_score == 0.0
        assert episode.hacking_flags == []

    def test_create_with_all_fields(self):
        """Test creating EpisodeData with all fields."""
        episode = EpisodeData(
            episode=0,
            total_reward=100.0,
            length=50,
            start_time=0.0,
            end_time=10.0,
            component_totals={"comp1": 60.0, "comp2": 40.0},
            hacking_score=0.75,
            hacking_flags=["action_repetition", "state_cycling"],
        )
        assert episode.component_totals["comp1"] == 60.0
        assert episode.hacking_score == 0.75
        assert len(episode.hacking_flags) == 2
