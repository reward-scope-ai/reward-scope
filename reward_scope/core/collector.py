"""
Data Collector Module

Stores training data in SQLite for persistence and fast querying.
Supports real-time streaming to dashboard via WebSocket.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import sqlite3
import json
import time
from pathlib import Path
import numpy as np


@dataclass
class StepData:
    """Data collected at each environment step."""
    step: int
    episode: int
    timestamp: float

    # Core RL data
    observation: Any  # Will be serialized to JSON
    action: Any
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any]

    # Reward decomposition (if available)
    reward_components: Dict[str, float] = field(default_factory=dict)

    # Optional: value estimates, action probabilities
    value_estimate: Optional[float] = None
    action_probs: Optional[List[float]] = None


@dataclass
class EpisodeData:
    """Aggregated episode-level statistics."""
    episode: int
    total_reward: float
    length: int
    start_time: float
    end_time: float

    # Component-wise totals
    component_totals: Dict[str, float] = field(default_factory=dict)

    # Hacking indicators (computed post-episode)
    hacking_score: float = 0.0
    hacking_flags: List[str] = field(default_factory=list)


# Database schema (SQLite)
SCHEMA = """
CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    step INTEGER NOT NULL,
    episode INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    reward REAL NOT NULL,
    done INTEGER NOT NULL,
    truncated INTEGER NOT NULL,
    reward_components TEXT,  -- JSON
    value_estimate REAL,
    observation TEXT,  -- JSON (optional, can be disabled for space)
    action TEXT,  -- JSON
    info TEXT  -- JSON
);

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode INTEGER UNIQUE NOT NULL,
    total_reward REAL NOT NULL,
    length INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    component_totals TEXT,  -- JSON
    hacking_score REAL DEFAULT 0.0,
    hacking_flags TEXT  -- JSON
);

CREATE TABLE IF NOT EXISTS live_state (
    key TEXT PRIMARY KEY,
    value TEXT  -- JSON
);

CREATE INDEX IF NOT EXISTS idx_steps_episode ON steps(episode);
CREATE INDEX IF NOT EXISTS idx_steps_step ON steps(step);
CREATE INDEX IF NOT EXISTS idx_episodes_episode ON episodes(episode);
"""


def _serialize_to_json(obj: Any) -> str:
    """Convert object to JSON, handling numpy arrays."""
    if obj is None:
        return None

    def convert(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, (list, tuple)):
            return [convert(item) for item in o]
        elif isinstance(o, dict):
            return {k: convert(v) for k, v in o.items()}
        return o

    return json.dumps(convert(obj))


def _deserialize_from_json(json_str: Optional[str]) -> Any:
    """Convert JSON string back to object."""
    if json_str is None:
        return None
    return json.loads(json_str)


class DataCollector:
    """
    Collects and stores RL training data.

    Usage:
        collector = DataCollector(run_name="my_experiment")
        collector.log_step(step_data)
        collector.end_episode()
    """

    def __init__(
        self,
        run_name: str,
        storage_dir: str = "./reward_scope_data",
        buffer_size: int = 1000,  # Flush to DB every N steps
        enable_streaming: bool = True,
    ):
        """
        Args:
            run_name: Unique identifier for this training run
            storage_dir: Directory to store SQLite database
            buffer_size: Number of steps to buffer before DB write
            enable_streaming: Whether to stream to WebSocket for live dashboard
        """
        self.run_name = run_name
        self.storage_dir = Path(storage_dir)
        self.buffer_size = buffer_size
        self.enable_streaming = enable_streaming

        # Create storage directory if needed
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Database path
        self.db_path = self.storage_dir / f"{run_name}.db"

        # Initialize database
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

        # Buffers for batch writing
        self.step_buffer: List[StepData] = []

        # Episode tracking
        self.current_episode_steps: List[StepData] = []
        self.current_episode_start_time: Optional[float] = None

    def log_step(self, data: StepData) -> None:
        """Log a single environment step."""
        # Add to buffer
        self.step_buffer.append(data)
        self.current_episode_steps.append(data)

        # Track episode start time
        if self.current_episode_start_time is None:
            self.current_episode_start_time = data.timestamp

        # Flush if buffer is full
        if len(self.step_buffer) >= self.buffer_size:
            self._flush_step_buffer()

    def _flush_step_buffer(self) -> None:
        """Write buffered steps to database."""
        if not self.step_buffer:
            return

        cursor = self.conn.cursor()

        for step in self.step_buffer:
            cursor.execute(
                """
                INSERT INTO steps (
                    step, episode, timestamp, reward, done, truncated,
                    reward_components, value_estimate, observation, action, info
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    step.step,
                    step.episode,
                    step.timestamp,
                    step.reward,
                    1 if step.done else 0,
                    1 if step.truncated else 0,
                    _serialize_to_json(step.reward_components),
                    step.value_estimate,
                    _serialize_to_json(step.observation),
                    _serialize_to_json(step.action),
                    _serialize_to_json(step.info),
                )
            )

        self.conn.commit()
        self.step_buffer.clear()

    def end_episode(self) -> EpisodeData:
        """
        Signal end of episode, compute aggregates.
        Returns episode summary.
        """
        # Flush any remaining steps
        self._flush_step_buffer()

        if not self.current_episode_steps:
            # Return empty episode data
            return EpisodeData(
                episode=0,
                total_reward=0.0,
                length=0,
                start_time=time.time(),
                end_time=time.time(),
            )

        # Compute episode statistics
        episode_num = self.current_episode_steps[0].episode
        total_reward = sum(step.reward for step in self.current_episode_steps)
        length = len(self.current_episode_steps)
        start_time = self.current_episode_start_time or self.current_episode_steps[0].timestamp
        end_time = self.current_episode_steps[-1].timestamp

        # Aggregate component totals
        component_totals: Dict[str, float] = {}
        for step in self.current_episode_steps:
            for comp_name, comp_value in step.reward_components.items():
                if comp_name not in component_totals:
                    component_totals[comp_name] = 0.0
                component_totals[comp_name] += comp_value

        episode_data = EpisodeData(
            episode=episode_num,
            total_reward=total_reward,
            length=length,
            start_time=start_time,
            end_time=end_time,
            component_totals=component_totals,
            hacking_score=0.0,  # Will be updated by detectors
            hacking_flags=[],
        )

        # Store episode data (use REPLACE to handle re-runs with same run_name)
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO episodes (
                episode, total_reward, length, start_time, end_time,
                component_totals, hacking_score, hacking_flags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                episode_data.episode,
                episode_data.total_reward,
                episode_data.length,
                episode_data.start_time,
                episode_data.end_time,
                _serialize_to_json(episode_data.component_totals),
                episode_data.hacking_score,
                _serialize_to_json(episode_data.hacking_flags),
            )
        )
        self.conn.commit()

        # Reset episode tracking
        self.current_episode_steps.clear()
        self.current_episode_start_time = None

        return episode_data

    def update_episode_hacking_data(
        self,
        episode: int,
        hacking_score: float,
        hacking_flags: List[str],
    ) -> None:
        """
        Update hacking score and flags for a specific episode.

        This is called after detectors have analyzed the episode data.

        Args:
            episode: Episode number to update
            hacking_score: Overall hacking score (0-1)
            hacking_flags: List of detected hacking patterns
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE episodes
            SET hacking_score = ?, hacking_flags = ?
            WHERE episode = ?
            """,
            (
                hacking_score,
                _serialize_to_json(hacking_flags),
                episode,
            )
        )
        self.conn.commit()

    def update_live_hacking_state(
        self,
        episode: int,
        hacking_score: float,
        alert_count: int,
    ) -> None:
        """
        Update live hacking state for in-progress episode.

        Args:
            episode: Current episode number
            hacking_score: Current hacking score (0-1)
            alert_count: Number of alerts detected so far
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO live_state (key, value)
            VALUES ('current_hacking', ?)
            """,
            (
                _serialize_to_json({
                    "episode": episode,
                    "hacking_score": hacking_score,
                    "alert_count": alert_count,
                    "timestamp": time.time(),
                }),
            )
        )
        self.conn.commit()

    def get_live_hacking_state(self) -> Optional[Dict[str, Any]]:
        """
        Get live hacking state for in-progress episode.

        Returns:
            Dict with episode, hacking_score, alert_count, timestamp, or None if not available
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT value FROM live_state WHERE key = 'current_hacking'
            """
        )
        row = cursor.fetchone()
        if row:
            return _deserialize_from_json(row[0])
        return None

    def clear_live_hacking_state(self) -> None:
        """Clear live hacking state (call when episode completes)."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            DELETE FROM live_state WHERE key = 'current_hacking'
            """
        )
        self.conn.commit()

    def get_recent_steps(self, n: int = 100) -> List[StepData]:
        """Get most recent N steps for dashboard."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT step, episode, timestamp, reward, done, truncated,
                   reward_components, value_estimate, observation, action, info
            FROM steps
            ORDER BY id DESC
            LIMIT ?
            """,
            (n,)
        )

        steps = []
        for row in cursor.fetchall():
            steps.append(StepData(
                step=row[0],
                episode=row[1],
                timestamp=row[2],
                observation=_deserialize_from_json(row[8]),
                action=_deserialize_from_json(row[9]),
                reward=row[3],
                done=bool(row[4]),
                truncated=bool(row[5]),
                info=_deserialize_from_json(row[10]) or {},
                reward_components=_deserialize_from_json(row[6]) or {},
                value_estimate=row[7],
            ))

        return list(reversed(steps))  # Return in chronological order

    def get_episode_history(self, n: int = 50) -> List[EpisodeData]:
        """Get most recent N episodes."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT episode, total_reward, length, start_time, end_time,
                   component_totals, hacking_score, hacking_flags
            FROM episodes
            ORDER BY id DESC
            LIMIT ?
            """,
            (n,)
        )

        episodes = []
        for row in cursor.fetchall():
            episodes.append(EpisodeData(
                episode=row[0],
                total_reward=row[1],
                length=row[2],
                start_time=row[3],
                end_time=row[4],
                component_totals=_deserialize_from_json(row[5]) or {},
                hacking_score=row[6],
                hacking_flags=_deserialize_from_json(row[7]) or [],
            ))

        return list(reversed(episodes))  # Return in chronological order

    def query_steps(
        self,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        episode: Optional[int] = None,
    ) -> List[StepData]:
        """Query steps with filters."""
        cursor = self.conn.cursor()

        query = """
            SELECT step, episode, timestamp, reward, done, truncated,
                   reward_components, value_estimate, observation, action, info
            FROM steps
            WHERE 1=1
        """
        params = []

        if start_step is not None:
            query += " AND step >= ?"
            params.append(start_step)

        if end_step is not None:
            query += " AND step <= ?"
            params.append(end_step)

        if episode is not None:
            query += " AND episode = ?"
            params.append(episode)

        query += " ORDER BY step ASC"

        cursor.execute(query, params)

        steps = []
        for row in cursor.fetchall():
            steps.append(StepData(
                step=row[0],
                episode=row[1],
                timestamp=row[2],
                observation=_deserialize_from_json(row[8]),
                action=_deserialize_from_json(row[9]),
                reward=row[3],
                done=bool(row[4]),
                truncated=bool(row[5]),
                info=_deserialize_from_json(row[10]) or {},
                reward_components=_deserialize_from_json(row[6]) or {},
                value_estimate=row[7],
            ))

        return steps

    def close(self) -> None:
        """Flush buffers and close database connection."""
        self._flush_step_buffer()
        self.conn.close()
