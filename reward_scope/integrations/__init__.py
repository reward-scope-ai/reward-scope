"""
RewardScope Integrations

Integration modules for popular RL frameworks:
- stable_baselines.py: Stable-Baselines3 callback
- gymnasium.py: Gymnasium environment wrapper
"""

from .gymnasium import RewardScopeWrapper

try:
    from .stable_baselines import RewardScopeCallback
    __all__ = ["RewardScopeWrapper", "RewardScopeCallback"]
except ImportError:
    # SB3 not installed
    __all__ = ["RewardScopeWrapper"]
