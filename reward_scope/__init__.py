"""
RewardScope: Real-time reward debugging for reinforcement learning.
"""

__version__ = "0.2.2"

from reward_scope.core.collector import DataCollector, StepData, EpisodeData
from reward_scope.core.decomposer import RewardDecomposer, RewardComponent, IsaacLabDecomposer
from reward_scope.core.detectors import (
    HackingDetectorSuite,
    HackingAlert,
    HackingType,
    AlertSeverity,
    ActionRepetitionDetector,
    StateCyclingDetector,
    ComponentImbalanceDetector,
    RewardSpikingDetector,
    BoundaryExploitationDetector,
)
# Phase 1: Adaptive baselines (experimental)
from reward_scope.core.baselines import BaselineCollector, BaselineStats
# Phase 2: Two-layer detection
from reward_scope.core.baseline import BaselineTracker, RollingStats

__all__ = [
    "__version__",
    # Core
    "DataCollector",
    "StepData",
    "EpisodeData",
    "RewardDecomposer",
    "RewardComponent",
    "IsaacLabDecomposer",
    "HackingDetectorSuite",
    "HackingAlert",
    "HackingType",
    "AlertSeverity",
    "ActionRepetitionDetector",
    "StateCyclingDetector",
    "ComponentImbalanceDetector",
    "RewardSpikingDetector",
    "BoundaryExploitationDetector",
    # Phase 1: Adaptive baselines (experimental)
    "BaselineCollector",
    "BaselineStats",
    # Phase 2: Two-layer detection
    "BaselineTracker",
    "RollingStats",
]
