"""
RewardScope: Real-time reward debugging for reinforcement learning.
"""

__version__ = "0.1.0"

from reward_scope.core.collector import DataCollector, StepData, EpisodeData
from reward_scope.core.decomposer import RewardDecomposer, RewardComponent, IsaacLabDecomposer
from reward_scope.core.detectors import (
    HackingDetectorSuite,
    HackingAlert,
    HackingType,
    ActionRepetitionDetector,
    StateCyclingDetector,
    ComponentImbalanceDetector,
    RewardSpikingDetector,
    BoundaryExploitationDetector,
)

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
    "ActionRepetitionDetector",
    "StateCyclingDetector",
    "ComponentImbalanceDetector",
    "RewardSpikingDetector",
    "BoundaryExploitationDetector",
]
