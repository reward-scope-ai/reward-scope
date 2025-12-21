"""
Core components for reward tracking and analysis.
"""

from reward_scope.core.collector import DataCollector, StepData, EpisodeData
from reward_scope.core.decomposer import RewardDecomposer, RewardComponent
from reward_scope.core.detectors import (
    HackingDetectorSuite,
    HackingAlert,
    HackingType,
    BaseDetector,
    StateCyclingDetector,
    ActionRepetitionDetector,
    ComponentImbalanceDetector,
    RewardSpikingDetector,
    BoundaryExploitationDetector,
)

__all__ = [
    "DataCollector",
    "StepData",
    "EpisodeData",
    "RewardDecomposer",
    "RewardComponent",
    "HackingDetectorSuite",
    "HackingAlert",
    "HackingType",
    "BaseDetector",
    "StateCyclingDetector",
    "ActionRepetitionDetector",
    "ComponentImbalanceDetector",
    "RewardSpikingDetector",
    "BoundaryExploitationDetector",
]
