# Reward Forensics MVP - Technical Specification

## Project Overview

**Name:** RewardScope (working title)  
**Tagline:** "See what your RL agent actually learned"  
**Purpose:** Real-time reward debugging and hacking detection for reinforcement learning

### Problem Statement
RL practitioners spend excessive time debugging reward functions. Symptoms include:
- Agents exploiting unintended shortcuts (reward hacking)
- One reward component dominating others
- Policies that look good on metrics but behave incorrectly
- Errors that manifest globally ("loss explodes, KL collapses, rewards oscillate")

### MVP Scope
A Python SDK + web dashboard that:
1. Hooks into standard RL training loops
2. Decomposes and tracks reward components in real-time
3. Detects common reward hacking patterns
4. Visualizes policy behavior anomalies

### Target Users (V1)
- Robotics RL engineers using Isaac Lab, Gymnasium, or Stable-Baselines3
- Solo developers and small teams (< 10 people)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     User's Training Script                       │
│                                                                   │
│   from reward_scope import RewardScopeCallback                   │
│   callback = RewardScopeCallback(reward_components=[...])        │
│   model.learn(callback=callback)                                 │
│                                                                   │
└──────────────────────────┬────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RewardScope SDK (Python)                      │
│                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Callback/Hook   │  │ Reward Decomp   │  │ Hacking         │  │
│  │ Layer           │  │ Engine          │  │ Detector        │  │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                     │           │
│           └────────────────────┼─────────────────────┘           │
│                                │                                  │
│                    ┌───────────▼───────────┐                     │
│                    │   Data Collector      │                     │
│                    │   (SQLite + JSON)     │                     │
│                    └───────────┬───────────┘                     │
│                                │                                  │
└────────────────────────────────┼─────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Web Dashboard (FastAPI + HTMX)                │
│                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Reward Timeline │  │ Component       │  │ Alert           │  │
│  │ Charts          │  │ Breakdown       │  │ Panel           │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐                       │
│  │ Trajectory      │  │ Hacking         │                       │
│  │ Viewer          │  │ Diagnosis       │                       │
│  └─────────────────┘  └─────────────────┘                       │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Directory Structure

```
reward-scope/
├── README.md
├── pyproject.toml
├── setup.py
│
├── reward_scope/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── collector.py        # Data collection and storage
│   │   ├── decomposer.py       # Reward component decomposition
│   │   └── detectors.py        # Hacking detection algorithms
│   │
│   ├── integrations/
│   │   ├── __init__.py
│   │   ├── gymnasium.py        # Gymnasium wrapper
│   │   ├── stable_baselines.py # SB3 callback
│   │   └── isaac_lab.py        # Isaac Lab callback (future)
│   │
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI application
│   │   ├── routes.py           # API endpoints
│   │   └── templates/
│   │       ├── base.html
│   │       ├── index.html
│   │       ├── components/
│   │       │   ├── reward_chart.html
│   │       │   ├── component_breakdown.html
│   │       │   ├── alerts.html
│   │       │   └── trajectory_viewer.html
│   │       └── static/
│   │           ├── styles.css
│   │           └── charts.js
│   │
│   └── cli.py                  # Command-line interface
│
├── tests/
│   ├── __init__.py
│   ├── test_collector.py
│   ├── test_decomposer.py
│   ├── test_detectors.py
│   ├── test_gymnasium_integration.py
│   └── test_sb3_integration.py
│
├── examples/
│   ├── cartpole_basic.py       # Simplest example
│   ├── lunarlander_hacking.py  # Demonstrates hacking detection
│   └── mujoco_ant.py           # MuJoCo example with complex rewards
│
└── docs/
    ├── quickstart.md
    ├── reward_components.md
    └── hacking_detection.md
```

---

## Component Specifications

### 1. Core: Data Collector (`reward_scope/core/collector.py`)

**Purpose:** Collect and store training data for analysis

```python
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
        pass  # Implementation details below
    
    def log_step(self, data: StepData) -> None:
        """Log a single environment step."""
        pass
    
    def end_episode(self) -> EpisodeData:
        """
        Signal end of episode, compute aggregates.
        Returns episode summary.
        """
        pass
    
    def get_recent_steps(self, n: int = 100) -> List[StepData]:
        """Get most recent N steps for dashboard."""
        pass
    
    def get_episode_history(self, n: int = 50) -> List[EpisodeData]:
        """Get most recent N episodes."""
        pass
    
    def query_steps(
        self,
        start_step: Optional[int] = None,
        end_step: Optional[int] = None,
        episode: Optional[int] = None,
    ) -> List[StepData]:
        """Query steps with filters."""
        pass
    
    def close(self) -> None:
        """Flush buffers and close database connection."""
        pass


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

CREATE INDEX IF NOT EXISTS idx_steps_episode ON steps(episode);
CREATE INDEX IF NOT EXISTS idx_steps_step ON steps(step);
CREATE INDEX IF NOT EXISTS idx_episodes_episode ON episodes(episode);
"""
```

**Implementation Notes:**
- Use connection pooling for concurrent access
- Implement write buffering to avoid DB bottleneck during training
- Support optional observation storage (can be disabled for large obs spaces)
- JSON serialize numpy arrays properly

---

### 2. Core: Reward Decomposer (`reward_scope/core/decomposer.py`)

**Purpose:** Break down composite rewards into trackable components

```python
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
        pass
    
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
        pass
    
    def get_component_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get running statistics for each component.
        
        Returns:
            {component_name: {"mean": x, "std": y, "min": z, "max": w, "count": n}}
        """
        pass
    
    def check_dominance(self, threshold: float = 0.8) -> List[str]:
        """
        Check if any single component dominates the total reward.
        
        Returns list of component names that contribute > threshold of total.
        """
        pass


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
        pass
```

**Implementation Notes:**
- Handle numpy arrays vs Python floats gracefully
- Compute running statistics efficiently (Welford's algorithm)
- Support vectorized environments (batch decomposition)

---

### 3. Core: Hacking Detectors (`reward_scope/core/detectors.py`)

**Purpose:** Detect common reward hacking patterns

```python
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
    
    def update(self, step, episode, observation, action, reward, 
               reward_components, done, info) -> Optional[HackingAlert]:
        """
        Check for state cycling.
        
        Algorithm:
        1. Hash/compress observation for comparison
        2. Look for repeating patterns in observation history
        3. If cycle detected and reward is high, flag as hacking
        """
        pass
    
    def _compute_observation_hash(self, obs: Any) -> str:
        """Create hashable representation of observation."""
        pass
    
    def _find_cycles(self) -> List[Tuple[int, float]]:
        """Find cycles in observation buffer. Returns [(length, similarity)]."""
        pass


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
    
    def update(self, step, episode, observation, action, reward,
               reward_components, done, info) -> Optional[HackingAlert]:
        pass


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
    
    def update(self, step, episode, observation, action, reward,
               reward_components, done, info) -> Optional[HackingAlert]:
        pass
    
    def on_episode_end(self, episode_component_totals: Dict[str, float]) -> Optional[HackingAlert]:
        """Check for sustained imbalance across episodes."""
        pass


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
    
    def update(self, step, episode, observation, action, reward,
               reward_components, done, info) -> Optional[HackingAlert]:
        pass


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
        self.boundary_counts: Dict[str, int] = {}
    
    def update(self, step, episode, observation, action, reward,
               reward_components, done, info) -> Optional[HackingAlert]:
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
        pass
    
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
        pass
    
    def reset(self) -> None:
        """Reset all detectors."""
        for detector in self.detectors:
            detector.reset()
```

**Implementation Notes:**
- Use efficient data structures (numpy for observation comparisons)
- Make detection thresholds configurable
- Include suggested fixes in alerts based on hacking type

---

### 4. Integration: Stable-Baselines3 (`reward_scope/integrations/stable_baselines.py`)

**Purpose:** Seamless integration with Stable-Baselines3

```python
"""
Stable-Baselines3 Integration

Provides a callback that hooks into SB3 training loop.
"""

from typing import Dict, List, Optional, Callable, Any, Union
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from reward_scope.core.collector import DataCollector, StepData
from reward_scope.core.decomposer import RewardDecomposer
from reward_scope.core.detectors import HackingDetectorSuite, HackingAlert


class RewardScopeCallback(BaseCallback):
    """
    Stable-Baselines3 callback for reward monitoring and hacking detection.
    
    Usage:
        from reward_scope import RewardScopeCallback
        
        callback = RewardScopeCallback(
            run_name="my_experiment",
            reward_components={
                "distance": lambda obs, act, info: info.get("distance_reward", 0),
                "energy": lambda obs, act, info: info.get("energy_reward", 0),
            },
            start_dashboard=True,
        )
        
        model = PPO("MlpPolicy", env)
        model.learn(total_timesteps=100000, callback=callback)
    """
    
    def __init__(
        self,
        run_name: str = "default_run",
        reward_components: Optional[Dict[str, Callable]] = None,
        auto_extract_prefix: Optional[str] = "reward_",
        storage_dir: str = "./reward_scope_data",
        start_dashboard: bool = True,
        dashboard_port: int = 8050,
        enable_hacking_detection: bool = True,
        log_observations: bool = False,  # Can be large
        verbose: int = 1,
    ):
        """
        Args:
            run_name: Unique name for this training run
            reward_components: Dict of {name: compute_fn} for decomposition
            auto_extract_prefix: Auto-extract components from info with this prefix
            storage_dir: Directory for storing data
            start_dashboard: Whether to start web dashboard automatically
            dashboard_port: Port for dashboard server
            enable_hacking_detection: Whether to run hacking detectors
            log_observations: Whether to store observations (can be large)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        
        self.run_name = run_name
        self.storage_dir = storage_dir
        self.start_dashboard = start_dashboard
        self.dashboard_port = dashboard_port
        self.log_observations = log_observations
        
        # Initialize components
        self.collector = DataCollector(run_name, storage_dir)
        
        self.decomposer = RewardDecomposer(auto_extract_prefix=auto_extract_prefix)
        if reward_components:
            for name, fn in reward_components.items():
                self.decomposer.register_component(name, fn)
        
        self.detector_suite = HackingDetectorSuite() if enable_hacking_detection else None
        
        # State tracking
        self.current_episode = 0
        self.episode_step = 0
        self.total_steps = 0
        self.pending_alerts: List[HackingAlert] = []
        
        # Dashboard process
        self._dashboard_process = None
    
    def _on_training_start(self) -> None:
        """Called at the beginning of training."""
        if self.start_dashboard:
            self._start_dashboard()
        
        # Get observation/action bounds for boundary detector
        if self.detector_suite and hasattr(self.training_env, 'observation_space'):
            obs_space = self.training_env.observation_space
            if hasattr(obs_space, 'low') and hasattr(obs_space, 'high'):
                # Update boundary detector with actual bounds
                pass
    
    def _on_step(self) -> bool:
        """
        Called at each environment step.
        
        Returns:
            True to continue training, False to stop.
        """
        # Get current step data from the model/environment
        # Note: SB3 stores info in self.locals
        
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])
        observations = self.locals.get("new_obs", None)
        actions = self.locals.get("actions", None)
        
        # Handle vectorized environments
        for env_idx in range(len(rewards)):
            reward = rewards[env_idx]
            info = infos[env_idx] if env_idx < len(infos) else {}
            done = dones[env_idx]
            obs = observations[env_idx] if observations is not None else None
            action = actions[env_idx] if actions is not None else None
            
            # Decompose reward
            components = self.decomposer.decompose(obs, action, reward, info)
            
            # Create step data
            step_data = StepData(
                step=self.total_steps,
                episode=self.current_episode,
                timestamp=time.time(),
                observation=obs if self.log_observations else None,
                action=action,
                reward=reward,
                done=done,
                truncated=info.get("TimeLimit.truncated", False),
                info=info,
                reward_components=components,
                value_estimate=self.locals.get("values", [None])[env_idx] 
                    if "values" in self.locals else None,
            )
            
            # Log step
            self.collector.log_step(step_data)
            
            # Run hacking detection
            if self.detector_suite:
                alerts = self.detector_suite.update(
                    step=self.total_steps,
                    episode=self.current_episode,
                    observation=obs,
                    action=action,
                    reward=reward,
                    reward_components=components,
                    done=done,
                    info=info,
                )
                self.pending_alerts.extend(alerts)
                
                # Print alerts if verbose
                for alert in alerts:
                    if self.verbose >= 1:
                        print(f"[RewardScope] ALERT: {alert.type.value} - {alert.description}")
            
            # Handle episode end
            if done:
                episode_data = self.collector.end_episode()
                if self.detector_suite:
                    self.detector_suite.on_episode_end(episode_data.component_totals)
                    self.detector_suite.reset()
                self.current_episode += 1
                self.episode_step = 0
            else:
                self.episode_step += 1
            
            self.total_steps += 1
        
        return True  # Continue training
    
    def _on_training_end(self) -> None:
        """Called at the end of training."""
        self.collector.close()
        self._stop_dashboard()
        
        # Print summary
        if self.verbose >= 1:
            print(f"\n[RewardScope] Training complete.")
            print(f"  Total steps: {self.total_steps}")
            print(f"  Total episodes: {self.current_episode}")
            print(f"  Hacking alerts: {len(self.pending_alerts)}")
            
            if self.pending_alerts:
                print("\n  Alert summary:")
                from collections import Counter
                alert_counts = Counter(a.type.value for a in self.pending_alerts)
                for alert_type, count in alert_counts.items():
                    print(f"    {alert_type}: {count}")
    
    def _start_dashboard(self) -> None:
        """Start the dashboard server in a subprocess."""
        import subprocess
        import sys
        
        self._dashboard_process = subprocess.Popen(
            [sys.executable, "-m", "reward_scope.dashboard", 
             "--port", str(self.dashboard_port),
             "--data-dir", self.storage_dir,
             "--run-name", self.run_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        if self.verbose >= 1:
            print(f"[RewardScope] Dashboard started at http://localhost:{self.dashboard_port}")
    
    def _stop_dashboard(self) -> None:
        """Stop the dashboard server."""
        if self._dashboard_process:
            self._dashboard_process.terminate()
            self._dashboard_process = None
    
    def get_alerts(self) -> List[HackingAlert]:
        """Get all detected hacking alerts."""
        return self.pending_alerts.copy()
    
    def get_hacking_score(self) -> float:
        """Get current hacking likelihood score."""
        if self.detector_suite:
            return self.detector_suite.get_hacking_score()
        return 0.0


# Convenience function
def wrap_env_with_decomposition(
    env,
    reward_components: Dict[str, Callable],
) -> VecEnv:
    """
    Wrap an environment to automatically add reward components to info.
    
    Useful when you can't modify the environment but want decomposition.
    """
    pass  # Implementation: create wrapper that intercepts step()
```

**Implementation Notes:**
- Handle both single and vectorized environments
- Support async dashboard updates
- Graceful handling of missing info keys

---

### 5. Integration: Gymnasium (`reward_scope/integrations/gymnasium.py`)

**Purpose:** Environment wrapper for Gymnasium

```python
"""
Gymnasium Integration

Provides an environment wrapper that tracks rewards and detects hacking.
Useful when not using Stable-Baselines3.
"""

from typing import Dict, Optional, Callable, Any, Tuple, SupportsFloat
import gymnasium as gym
from gymnasium import Wrapper
import numpy as np

from reward_scope.core.collector import DataCollector, StepData
from reward_scope.core.decomposer import RewardDecomposer
from reward_scope.core.detectors import HackingDetectorSuite


class RewardScopeWrapper(Wrapper):
    """
    Gymnasium wrapper for reward tracking and hacking detection.
    
    Usage:
        import gymnasium as gym
        from reward_scope import RewardScopeWrapper
        
        env = gym.make("LunarLander-v2")
        env = RewardScopeWrapper(
            env,
            run_name="lunar_lander_exp",
            reward_components={
                "shaping": lambda o, a, i: i.get("shaping_reward", 0),
            },
        )
        
        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Access reward scope data
            print(f"Components: {info['reward_components']}")
            print(f"Hacking alerts: {info['hacking_alerts']}")
    """
    
    def __init__(
        self,
        env: gym.Env,
        run_name: str = "default_run",
        reward_components: Optional[Dict[str, Callable]] = None,
        auto_extract_prefix: Optional[str] = "reward_",
        storage_dir: str = "./reward_scope_data",
        enable_hacking_detection: bool = True,
        inject_into_info: bool = True,
    ):
        """
        Args:
            env: Gymnasium environment to wrap
            run_name: Unique name for this run
            reward_components: Dict of {name: compute_fn}
            auto_extract_prefix: Auto-extract from info with this prefix
            storage_dir: Storage directory
            enable_hacking_detection: Whether to run detectors
            inject_into_info: Whether to add reward_scope data to info dict
        """
        super().__init__(env)
        
        self.run_name = run_name
        self.inject_into_info = inject_into_info
        
        # Initialize components
        self.collector = DataCollector(run_name, storage_dir)
        
        self.decomposer = RewardDecomposer(auto_extract_prefix=auto_extract_prefix)
        if reward_components:
            for name, fn in reward_components.items():
                self.decomposer.register_component(name, fn)
        
        self.detector_suite = HackingDetectorSuite(
            observation_bounds=(
                env.observation_space.low if hasattr(env.observation_space, 'low') else None,
                env.observation_space.high if hasattr(env.observation_space, 'high') else None,
            ),
            action_bounds=(
                env.action_space.low if hasattr(env.action_space, 'low') else None,
                env.action_space.high if hasattr(env.action_space, 'high') else None,
            ),
        ) if enable_hacking_detection else None
        
        # State
        self.current_episode = 0
        self.current_step = 0
        self.total_steps = 0
        self._last_obs = None
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset environment and tracking state."""
        obs, info = self.env.reset(**kwargs)
        
        # End previous episode if any
        if self.current_step > 0:
            self.collector.end_episode()
            if self.detector_suite:
                self.detector_suite.reset()
            self.current_episode += 1
        
        self.current_step = 0
        self._last_obs = obs
        
        return obs, info
    
    def step(self, action) -> Tuple[Any, SupportsFloat, bool, bool, Dict]:
        """Execute step with reward tracking."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Decompose reward
        components = self.decomposer.decompose(self._last_obs, action, float(reward), info)
        
        # Create step data
        step_data = StepData(
            step=self.total_steps,
            episode=self.current_episode,
            timestamp=time.time(),
            observation=self._last_obs,
            action=action,
            reward=float(reward),
            done=terminated,
            truncated=truncated,
            info=info,
            reward_components=components,
        )
        self.collector.log_step(step_data)
        
        # Run hacking detection
        alerts = []
        if self.detector_suite:
            alerts = self.detector_suite.update(
                step=self.total_steps,
                episode=self.current_episode,
                observation=self._last_obs,
                action=action,
                reward=float(reward),
                reward_components=components,
                done=terminated,
                info=info,
            )
        
        # Inject into info
        if self.inject_into_info:
            info["reward_components"] = components
            info["hacking_alerts"] = [
                {"type": a.type.value, "severity": a.severity, "description": a.description}
                for a in alerts
            ]
            info["hacking_score"] = self.detector_suite.get_hacking_score() if self.detector_suite else 0.0
        
        # Update state
        self._last_obs = obs
        self.current_step += 1
        self.total_steps += 1
        
        # Handle episode end
        if terminated or truncated:
            episode_data = self.collector.end_episode()
            if self.detector_suite:
                self.detector_suite.on_episode_end(episode_data.component_totals)
                self.detector_suite.reset()
            self.current_episode += 1
            self.current_step = 0
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Clean up resources."""
        self.collector.close()
        super().close()
    
    def get_component_stats(self) -> Dict[str, Dict[str, float]]:
        """Get reward component statistics."""
        return self.decomposer.get_component_stats()
    
    def get_alerts(self) -> List:
        """Get all hacking alerts."""
        if self.detector_suite:
            return self.detector_suite.get_all_alerts()
        return []
```

---

### 6. Dashboard (`reward_scope/dashboard/app.py`)

**Purpose:** Real-time visualization of training

```python
"""
Web Dashboard

FastAPI application with HTMX for real-time updates.
No JavaScript framework needed - uses server-sent events.
"""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
from pathlib import Path
from typing import List, Optional
import json

from reward_scope.core.collector import DataCollector


app = FastAPI(title="RewardScope Dashboard")

# Templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Static files
static_dir = templates_dir / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global state (will be set by CLI)
collector: Optional[DataCollector] = None
run_name: str = "unknown"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "run_name": run_name,
        }
    )


@app.get("/api/reward-history")
async def get_reward_history(n: int = 100):
    """Get recent reward history."""
    if not collector:
        return {"error": "No data collector"}
    
    steps = collector.get_recent_steps(n)
    return {
        "steps": [s.step for s in steps],
        "rewards": [s.reward for s in steps],
        "episodes": [s.episode for s in steps],
    }


@app.get("/api/component-breakdown")
async def get_component_breakdown(n: int = 100):
    """Get reward component breakdown."""
    if not collector:
        return {"error": "No data collector"}
    
    steps = collector.get_recent_steps(n)
    
    # Aggregate components
    component_sums = {}
    for step in steps:
        for name, value in step.reward_components.items():
            if name not in component_sums:
                component_sums[name] = 0.0
            component_sums[name] += value
    
    return {
        "components": list(component_sums.keys()),
        "values": list(component_sums.values()),
    }


@app.get("/api/episode-history")
async def get_episode_history(n: int = 50):
    """Get episode-level statistics."""
    if not collector:
        return {"error": "No data collector"}
    
    episodes = collector.get_episode_history(n)
    return {
        "episodes": [e.episode for e in episodes],
        "total_rewards": [e.total_reward for e in episodes],
        "lengths": [e.length for e in episodes],
        "hacking_scores": [e.hacking_score for e in episodes],
    }


@app.get("/api/alerts")
async def get_alerts():
    """Get recent hacking alerts."""
    # This would come from the detector suite
    # For now, return from stored data
    if not collector:
        return {"error": "No data collector"}
    
    episodes = collector.get_episode_history(10)
    alerts = []
    for ep in episodes:
        for flag in ep.hacking_flags:
            alerts.append({
                "episode": ep.episode,
                "type": flag,
                "severity": ep.hacking_score,
            })
    
    return {"alerts": alerts}


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live updates."""
    await websocket.accept()
    
    try:
        last_step = 0
        while True:
            # Check for new data
            if collector:
                steps = collector.get_recent_steps(10)
                if steps and steps[-1].step > last_step:
                    last_step = steps[-1].step
                    
                    # Send update
                    await websocket.send_json({
                        "type": "step_update",
                        "step": last_step,
                        "reward": steps[-1].reward,
                        "components": steps[-1].reward_components,
                        "episode": steps[-1].episode,
                    })
            
            await asyncio.sleep(0.1)  # 10 Hz updates
    
    except WebSocketDisconnect:
        pass


def run_dashboard(
    data_dir: str,
    run_name_: str,
    port: int = 8050,
):
    """Start the dashboard server."""
    global collector, run_name
    
    run_name = run_name_
    collector = DataCollector(run_name, data_dir)
    
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
```

---

### 7. Dashboard Templates

**`reward_scope/dashboard/templates/index.html`:**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RewardScope - {{ run_name }}</title>
    <script src="https://unpkg.com/htmx.org@1.9.10"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <header>
        <h1>🔬 RewardScope</h1>
        <span class="run-name">{{ run_name }}</span>
        <span class="status" id="connection-status">Connecting...</span>
    </header>
    
    <main>
        <div class="grid">
            <!-- Reward Timeline -->
            <section class="card" id="reward-timeline">
                <h2>Reward Timeline</h2>
                <canvas id="reward-chart"></canvas>
            </section>
            
            <!-- Component Breakdown -->
            <section class="card" id="component-breakdown">
                <h2>Component Breakdown</h2>
                <canvas id="component-chart"></canvas>
            </section>
            
            <!-- Episode History -->
            <section class="card" id="episode-history">
                <h2>Episode History</h2>
                <canvas id="episode-chart"></canvas>
            </section>
            
            <!-- Alerts Panel -->
            <section class="card alerts" id="alerts-panel">
                <h2>⚠️ Hacking Alerts</h2>
                <div id="alerts-list" 
                     hx-get="/api/alerts" 
                     hx-trigger="every 2s"
                     hx-swap="innerHTML">
                    <p class="no-alerts">No alerts detected</p>
                </div>
            </section>
            
            <!-- Live Stats -->
            <section class="card stats" id="live-stats">
                <h2>Live Stats</h2>
                <div class="stat">
                    <span class="label">Current Step</span>
                    <span class="value" id="current-step">0</span>
                </div>
                <div class="stat">
                    <span class="label">Current Episode</span>
                    <span class="value" id="current-episode">0</span>
                </div>
                <div class="stat">
                    <span class="label">Hacking Score</span>
                    <span class="value" id="hacking-score">0.00</span>
                </div>
            </section>
        </div>
    </main>
    
    <script src="/static/charts.js"></script>
</body>
</html>
```

**`reward_scope/dashboard/templates/static/styles.css`:**

```css
* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: #0f0f0f;
    color: #e0e0e0;
    min-height: 100vh;
}

header {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 2rem;
    background: #1a1a1a;
    border-bottom: 1px solid #333;
}

header h1 {
    font-size: 1.5rem;
    font-weight: 600;
}

.run-name {
    font-family: monospace;
    background: #2a2a2a;
    padding: 0.25rem 0.75rem;
    border-radius: 4px;
    font-size: 0.875rem;
}

.status {
    margin-left: auto;
    font-size: 0.75rem;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    background: #333;
}

.status.connected {
    background: #1a4d1a;
    color: #4ade80;
}

main {
    padding: 1.5rem;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 1.5rem;
}

.card {
    background: #1a1a1a;
    border-radius: 8px;
    padding: 1.25rem;
    border: 1px solid #333;
}

.card h2 {
    font-size: 1rem;
    font-weight: 500;
    margin-bottom: 1rem;
    color: #999;
}

.card canvas {
    width: 100% !important;
    height: 200px !important;
}

.alerts {
    max-height: 300px;
    overflow-y: auto;
}

.alert-item {
    background: #2a1a1a;
    border-left: 3px solid #ef4444;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    border-radius: 0 4px 4px 0;
}

.alert-item.warning {
    border-color: #f59e0b;
    background: #2a2a1a;
}

.alert-item .type {
    font-weight: 600;
    font-size: 0.875rem;
}

.alert-item .description {
    font-size: 0.75rem;
    color: #999;
    margin-top: 0.25rem;
}

.no-alerts {
    color: #666;
    font-style: italic;
    text-align: center;
    padding: 2rem;
}

.stats .stat {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid #333;
}

.stats .stat:last-child {
    border-bottom: none;
}

.stats .label {
    color: #999;
    font-size: 0.875rem;
}

.stats .value {
    font-family: monospace;
    font-size: 1.125rem;
    font-weight: 600;
}
```

**`reward_scope/dashboard/templates/static/charts.js`:**

```javascript
// Chart.js configuration
Chart.defaults.color = '#999';
Chart.defaults.borderColor = '#333';

// Reward timeline chart
const rewardCtx = document.getElementById('reward-chart').getContext('2d');
const rewardChart = new Chart(rewardCtx, {
    type: 'line',
    data: {
        labels: [],
        datasets: [{
            label: 'Reward',
            data: [],
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            fill: true,
            tension: 0.4,
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { display: false },
            y: { beginAtZero: false }
        },
        plugins: {
            legend: { display: false }
        }
    }
});

// Component breakdown chart
const componentCtx = document.getElementById('component-chart').getContext('2d');
const componentChart = new Chart(componentCtx, {
    type: 'doughnut',
    data: {
        labels: [],
        datasets: [{
            data: [],
            backgroundColor: [
                '#3b82f6', '#10b981', '#f59e0b', 
                '#ef4444', '#8b5cf6', '#ec4899'
            ],
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { position: 'right' }
        }
    }
});

// Episode history chart
const episodeCtx = document.getElementById('episode-chart').getContext('2d');
const episodeChart = new Chart(episodeCtx, {
    type: 'bar',
    data: {
        labels: [],
        datasets: [{
            label: 'Episode Reward',
            data: [],
            backgroundColor: '#10b981',
        }]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: { display: false },
            y: { beginAtZero: false }
        },
        plugins: {
            legend: { display: false }
        }
    }
});

// WebSocket connection for live updates
const ws = new WebSocket(`ws://${window.location.host}/ws/live`);

ws.onopen = () => {
    document.getElementById('connection-status').textContent = 'Connected';
    document.getElementById('connection-status').classList.add('connected');
};

ws.onclose = () => {
    document.getElementById('connection-status').textContent = 'Disconnected';
    document.getElementById('connection-status').classList.remove('connected');
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    
    if (data.type === 'step_update') {
        // Update live stats
        document.getElementById('current-step').textContent = data.step;
        document.getElementById('current-episode').textContent = data.episode;
        
        // Update reward chart
        rewardChart.data.labels.push(data.step);
        rewardChart.data.datasets[0].data.push(data.reward);
        
        // Keep last 100 points
        if (rewardChart.data.labels.length > 100) {
            rewardChart.data.labels.shift();
            rewardChart.data.datasets[0].data.shift();
        }
        
        rewardChart.update('none');
    }
};

// Initial data fetch
async function fetchInitialData() {
    // Reward history
    const rewardRes = await fetch('/api/reward-history?n=100');
    const rewardData = await rewardRes.json();
    rewardChart.data.labels = rewardData.steps;
    rewardChart.data.datasets[0].data = rewardData.rewards;
    rewardChart.update();
    
    // Component breakdown
    const componentRes = await fetch('/api/component-breakdown?n=100');
    const componentData = await componentRes.json();
    componentChart.data.labels = componentData.components;
    componentChart.data.datasets[0].data = componentData.values.map(Math.abs);
    componentChart.update();
    
    // Episode history
    const episodeRes = await fetch('/api/episode-history?n=50');
    const episodeData = await episodeRes.json();
    episodeChart.data.labels = episodeData.episodes;
    episodeChart.data.datasets[0].data = episodeData.total_rewards;
    episodeChart.update();
}

fetchInitialData();

// Refresh data periodically
setInterval(async () => {
    const componentRes = await fetch('/api/component-breakdown?n=100');
    const componentData = await componentRes.json();
    componentChart.data.labels = componentData.components;
    componentChart.data.datasets[0].data = componentData.values.map(Math.abs);
    componentChart.update();
}, 5000);
```

---

### 8. CLI (`reward_scope/cli.py`)

```python
"""
Command-line interface for RewardScope.
"""

import click
from pathlib import Path


@click.group()
def cli():
    """RewardScope - RL Reward Debugging Tools"""
    pass


@cli.command()
@click.option('--port', default=8050, help='Dashboard port')
@click.option('--data-dir', default='./reward_scope_data', help='Data directory')
@click.option('--run-name', default='latest', help='Run name to display')
def dashboard(port: int, data_dir: str, run_name: str):
    """Start the web dashboard."""
    from reward_scope.dashboard.app import run_dashboard
    click.echo(f"Starting dashboard at http://localhost:{port}")
    run_dashboard(data_dir, run_name, port)


@cli.command()
@click.argument('data_dir')
@click.option('--run-name', default=None, help='Specific run to analyze')
@click.option('--output', default='report.html', help='Output file')
def report(data_dir: str, run_name: str, output: str):
    """Generate a static HTML report."""
    from reward_scope.core.collector import DataCollector
    
    collector = DataCollector(run_name or "latest", data_dir)
    
    # Generate report
    episodes = collector.get_episode_history(n=1000)
    
    click.echo(f"Analyzed {len(episodes)} episodes")
    click.echo(f"Report saved to {output}")


@cli.command()
@click.argument('data_dir')
def list_runs(data_dir: str):
    """List available training runs."""
    data_path = Path(data_dir)
    if not data_path.exists():
        click.echo("Data directory not found")
        return
    
    for db_file in data_path.glob("*.db"):
        click.echo(f"  - {db_file.stem}")


if __name__ == '__main__':
    cli()
```

---

### 9. Package Configuration (`pyproject.toml`)

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "reward-scope"
version = "0.1.0"
description = "Reward debugging and hacking detection for reinforcement learning"
readme = "README.md"
license = {text = "MIT"}
authors = [
    {name = "James", email = "james@example.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
keywords = ["reinforcement-learning", "debugging", "reward-shaping", "robotics"]
requires-python = ">=3.9"

dependencies = [
    "numpy>=1.21.0",
    "gymnasium>=0.28.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.23.0",
    "jinja2>=3.1.0",
    "python-multipart>=0.0.6",
    "websockets>=11.0",
    "click>=8.0.0",
]

[project.optional-dependencies]
sb3 = [
    "stable-baselines3>=2.0.0",
]
mujoco = [
    "mujoco>=2.3.0",
]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "black>=23.0.0",
    "ruff>=0.0.280",
    "mypy>=1.4.0",
]
all = [
    "reward-scope[sb3,mujoco,dev]",
]

[project.scripts]
reward-scope = "reward_scope.cli:cli"

[project.urls]
Homepage = "https://github.com/username/reward-scope"
Documentation = "https://github.com/username/reward-scope#readme"
Repository = "https://github.com/username/reward-scope"

[tool.setuptools.packages.find]
where = ["."]
include = ["reward_scope*"]

[tool.black]
line-length = 100
target-version = ["py39", "py310", "py311"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_ignores = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

---

## Example Scripts

### `examples/cartpole_basic.py`

```python
"""
Basic example: CartPole with reward tracking.

This is the simplest possible example to verify the SDK works.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from reward_scope import RewardScopeCallback


def main():
    # Create environment
    env = gym.make("CartPole-v1")
    
    # Create callback with manual reward decomposition
    # CartPole has a simple +1 reward per step
    callback = RewardScopeCallback(
        run_name="cartpole_basic",
        reward_components={
            "survival": lambda obs, act, info: 1.0,  # Always 1 for CartPole
        },
        start_dashboard=True,
        dashboard_port=8050,
        verbose=1,
    )
    
    # Train
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000, callback=callback)
    
    # Print summary
    print("\n--- Training Complete ---")
    print(f"Total alerts: {len(callback.get_alerts())}")
    print(f"Final hacking score: {callback.get_hacking_score():.2f}")


if __name__ == "__main__":
    main()
```

### `examples/lunarlander_hacking.py`

```python
"""
LunarLander example: Demonstrates hacking detection.

LunarLander has a multi-component reward:
- Shaping reward based on distance and velocity
- Leg contact bonus
- Crash penalty
- Landing bonus

This example intentionally uses a bad reward that encourages hacking.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from reward_scope import RewardScopeCallback


def main():
    env = gym.make("LunarLander-v2")
    
    # RewardScope will auto-extract components from info dict
    # if the env provides them (LunarLander doesn't by default,
    # but we can still track the composite reward)
    callback = RewardScopeCallback(
        run_name="lunarlander_hacking_demo",
        start_dashboard=True,
        dashboard_port=8050,
        verbose=1,
    )
    
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000, callback=callback)
    
    # Check for hacking
    alerts = callback.get_alerts()
    if alerts:
        print("\n--- Hacking Detected! ---")
        for alert in alerts[:5]:  # Show first 5
            print(f"  [{alert.type.value}] {alert.description}")
            print(f"    Severity: {alert.severity:.2f}")
            print(f"    Suggested fix: {alert.suggested_fix}")
    else:
        print("\nNo hacking detected.")


if __name__ == "__main__":
    main()
```

### `examples/mujoco_ant.py`

```python
"""
MuJoCo Ant example: Complex multi-component reward.

The Ant environment has multiple reward components that can
lead to interesting hacking behaviors.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from reward_scope import RewardScopeCallback


def main():
    env = gym.make("Ant-v4")
    
    # Ant-v4 has these reward components in info:
    # - reward_forward: velocity in x direction
    # - reward_ctrl: control cost (negative)
    # - reward_survive: +1 per step if not terminated
    # - reward_contact: contact cost (negative)
    
    callback = RewardScopeCallback(
        run_name="ant_multicomponent",
        auto_extract_prefix="reward_",  # Auto-extract reward_* from info
        start_dashboard=True,
        dashboard_port=8050,
        verbose=1,
    )
    
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500000, callback=callback)
    
    # Analyze component balance
    from reward_scope.core.decomposer import RewardDecomposer
    stats = callback.decomposer.get_component_stats()
    
    print("\n--- Component Statistics ---")
    for name, s in stats.items():
        print(f"  {name}:")
        print(f"    Mean: {s['mean']:.4f}")
        print(f"    Std:  {s['std']:.4f}")


if __name__ == "__main__":
    main()
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_decomposer.py

import pytest
import numpy as np
from reward_scope.core.decomposer import RewardDecomposer, RewardComponent


class TestRewardDecomposer:
    def test_register_component(self):
        decomposer = RewardDecomposer()
        decomposer.register_component(
            "distance",
            lambda o, a, i: i.get("distance", 0),
            description="Distance to goal",
        )
        assert "distance" in decomposer.components
    
    def test_decompose_with_registered_components(self):
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
    
    def test_residual_tracking(self):
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
    
    def test_auto_extract_prefix(self):
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


# tests/test_detectors.py

import pytest
from reward_scope.core.detectors import (
    ActionRepetitionDetector,
    ComponentImbalanceDetector,
    HackingType,
)


class TestActionRepetitionDetector:
    def test_detects_repeated_actions(self):
        detector = ActionRepetitionDetector(
            window_size=10,
            repetition_threshold=0.8,
        )
        
        # Feed same action 10 times
        for i in range(10):
            alert = detector.update(
                step=i,
                episode=0,
                observation=None,
                action=0,  # Same action
                reward=1.0,
                reward_components={},
                done=False,
                info={},
            )
        
        # Should have detected repetition
        assert len(detector.alerts) > 0
        assert detector.alerts[0].type == HackingType.ACTION_REPETITION


class TestComponentImbalanceDetector:
    def test_detects_dominant_component(self):
        detector = ComponentImbalanceDetector(
            dominance_threshold=0.8,
        )
        
        # Feed data where one component dominates
        for i in range(100):
            detector.update(
                step=i,
                episode=0,
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
        
        # Trigger episode end check
        alert = detector.on_episode_end({
            "dominant": 950.0,
            "minor": 50.0,
        })
        
        assert alert is not None
        assert alert.type == HackingType.COMPONENT_IMBALANCE
```

---

## Implementation Order

### Phase 1: Core SDK (Week 1-2)
1. `reward_scope/core/collector.py` - Data storage
2. `reward_scope/core/decomposer.py` - Reward decomposition
3. Basic tests for above

### Phase 2: Detectors (Week 2-3)
1. `reward_scope/core/detectors.py` - All detector classes
2. Tests for each detector type
3. Tune default thresholds on standard envs

### Phase 3: Integrations (Week 3)
1. `reward_scope/integrations/stable_baselines.py`
2. `reward_scope/integrations/gymnasium.py`
3. Integration tests with real envs

### Phase 4: Dashboard (Week 4)
1. `reward_scope/dashboard/app.py`
2. Templates and static files
3. WebSocket live updates
4. CLI commands

### Phase 5: Polish (Week 5)
1. Examples and documentation
2. Package configuration
3. README with GIFs
4. PyPI release preparation

---

## Success Metrics for MVP

1. **Works on 3 standard envs:** CartPole, LunarLander, MuJoCo Ant
2. **Detects 3+ hacking types:** Action repetition, component imbalance, reward spiking
3. **Dashboard updates in real-time:** < 200ms latency
4. **< 5% training overhead:** Minimal impact on training speed
5. **10-minute onboarding:** From pip install to working dashboard

---

## Notes for Claude Code

1. **Start with collector.py and decomposer.py** - Everything else depends on these
2. **Use pytest-asyncio for dashboard tests**
3. **Test with real SB3 training loops** - Mock tests miss integration issues
4. **Keep dependencies minimal** - Don't add heavy libs unless necessary
5. **SQLite is sufficient for MVP** - No need for Redis/Postgres yet

---

## Questions to Resolve During Implementation

1. Should observations be stored by default? (Space vs debugging utility)
2. Best way to handle vectorized envs in SB3?
3. Threshold tuning for detectors - need empirical testing
4. Dashboard: polling vs WebSocket for different use cases?
