"""
Gymnasium Integration

Provides a wrapper that collects reward debugging data from Gymnasium
environments.
"""

from typing import Optional, Dict, Any, Callable, Tuple
import time
import numpy as np
import gymnasium as gym

from ..core.collector import DataCollector, StepData
from ..core.decomposer import RewardDecomposer
from ..core.detectors import HackingDetectorSuite, HackingAlert


class RewardScopeWrapper(gym.Wrapper):
    """
    Gymnasium wrapper for reward debugging.

    Wraps a Gymnasium environment to automatically collect step data,
    decompose rewards, and detect reward hacking patterns.

    The wrapper injects the following keys into the info dict:
    - 'reward_components': Dict[str, float] - Decomposed reward components
    - 'hacking_alerts': List[HackingAlert] - Any alerts detected this step
    - 'hacking_score': float - Overall hacking score (0-1)

    Usage:
        import gymnasium as gym
        from reward_scope.integrations import RewardScopeWrapper

        env = gym.make("LunarLander-v2")
        env = RewardScopeWrapper(
            env,
            run_name="lunar_lander_debug",
            auto_extract_prefix="reward_"  # if env provides components
        )

        obs, info = env.reset()
        for _ in range(1000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Access debugging info
            print(f"Components: {info['reward_components']}")
            if info['hacking_alerts']:
                print(f"Alerts: {info['hacking_alerts']}")
    """

    def __init__(
        self,
        env: gym.Env,
        run_name: str = "gym_run",
        storage_dir: str = "./reward_scope_data",
        # Decomposer settings
        auto_extract_prefix: Optional[str] = None,
        component_fns: Optional[Dict[str, Callable]] = None,
        # Detector settings
        enable_state_cycling: bool = True,
        enable_action_repetition: bool = True,
        enable_component_imbalance: bool = True,
        enable_reward_spiking: bool = True,
        enable_boundary_exploitation: bool = True,
        # Other settings
        verbose: int = 0,
    ):
        """
        Args:
            env: Gymnasium environment to wrap
            run_name: Unique name for this run
            storage_dir: Directory to store data
            auto_extract_prefix: Prefix for auto-extracting reward components from info dict
            component_fns: Dict of component_name -> function(obs, action, info) -> float
            enable_*: Enable/disable specific detectors
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(env)

        self.run_name = run_name
        self.storage_dir = storage_dir
        self.verbose = verbose

        # Initialize collector
        self.collector = DataCollector(
            run_name=run_name,
            storage_dir=storage_dir,
        )

        # Initialize decomposer
        self.decomposer = RewardDecomposer(
            auto_extract_prefix=auto_extract_prefix,
            track_residual=True,
        )

        # Register custom component functions if provided
        if component_fns:
            for name, fn in component_fns.items():
                self.decomposer.register_component(name, fn)

        # Get observation and action bounds from env if available
        observation_bounds = None
        action_bounds = None

        try:
            if hasattr(env.observation_space, 'low') and hasattr(env.observation_space, 'high'):
                observation_bounds = (
                    np.array(env.observation_space.low),
                    np.array(env.observation_space.high)
                )
        except Exception:
            pass

        try:
            if hasattr(env.action_space, 'low') and hasattr(env.action_space, 'high'):
                action_bounds = (
                    np.array(env.action_space.low),
                    np.array(env.action_space.high)
                )
        except Exception:
            pass

        # Initialize detector suite
        self.detector_suite = HackingDetectorSuite(
            enable_state_cycling=enable_state_cycling,
            enable_action_repetition=enable_action_repetition,
            enable_component_imbalance=enable_component_imbalance,
            enable_reward_spiking=enable_reward_spiking,
            enable_boundary_exploitation=enable_boundary_exploitation,
            observation_bounds=observation_bounds,
            action_bounds=action_bounds,
        )

        # Tracking
        self.episode_count = 0
        self.step_count = 0
        self.current_episode_step = 0
        self._last_observation = None
        self._last_action = None

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Reset the environment and tracking.

        Returns:
            observation, info dict (with RewardScope keys added)
        """
        # End previous episode if needed
        if self.current_episode_step > 0:
            self._end_episode()

        # Reset environment
        obs, info = self.env.reset(seed=seed, options=options)

        # Store observation
        self._last_observation = obs
        self._last_action = None

        # Reset episode tracking
        self.current_episode_step = 0

        # Add RewardScope info
        info['reward_components'] = {}
        info['hacking_alerts'] = []
        info['hacking_score'] = self.detector_suite.get_hacking_score()

        if self.verbose >= 1:
            print(f"[RewardScope] Episode {self.episode_count} started")

        return obs, info

    def step(
        self,
        action: Any,
    ) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        """
        Execute action and collect debugging data.

        Returns:
            observation, reward, terminated, truncated, info dict (with RewardScope keys)
        """
        # Execute action in environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Store for next step
        self._last_observation = obs
        self._last_action = action

        # Decompose reward into components
        reward_components = self.decomposer.decompose(
            observation=obs,
            action=action,
            total_reward=reward,
            info=info,
        )

        # Create step data
        step_data = StepData(
            step=self.step_count,
            episode=self.episode_count,
            timestamp=time.time(),
            observation=obs,
            action=action,
            reward=reward,
            done=terminated,
            truncated=truncated,
            info=info,
            reward_components=reward_components,
        )

        # Log to collector
        self.collector.log_step(step_data)

        # Update detectors
        alerts = self.detector_suite.update(
            step=self.step_count,
            episode=self.episode_count,
            observation=obs,
            action=action,
            reward=reward,
            reward_components=reward_components,
            done=terminated,
            info=info,
        )

        # Print alerts if any
        if alerts and self.verbose >= 1:
            for alert in alerts:
                print(f"[RewardScope] 🚨 {alert.type.value}: {alert.description}")
                if self.verbose >= 2:
                    print(f"  Evidence: {alert.evidence}")
                    print(f"  Fix: {alert.suggested_fix}")

        # Inject RewardScope data into info dict
        info['reward_components'] = reward_components
        info['hacking_alerts'] = alerts
        info['hacking_score'] = self.detector_suite.get_hacking_score()

        # Increment counters
        self.step_count += 1
        self.current_episode_step += 1

        # Handle episode end
        if terminated or truncated:
            self._end_episode()

        return obs, reward, terminated, truncated, info

    def _end_episode(self) -> None:
        """End the current episode and run episode-level checks."""
        # Get episode data from collector
        episode_data = self.collector.end_episode()

        # Run episode-level detectors
        episode_alerts = self.detector_suite.on_episode_end({
            "component_totals": episode_data.component_totals,
        })

        if episode_alerts and self.verbose >= 1:
            for alert in episode_alerts:
                print(f"[RewardScope] 🚨 Episode {self.episode_count}: {alert.type.value}: {alert.description}")

        # Reset detectors for next episode
        self.detector_suite.reset()

        if self.verbose >= 1:
            print(f"[RewardScope] Episode {episode_data.episode} complete: "
                  f"reward={episode_data.total_reward:.2f}, "
                  f"length={episode_data.length}, "
                  f"hacking_score={self.detector_suite.get_hacking_score():.3f}")

            # Print component breakdown if available
            if episode_data.component_totals and self.verbose >= 2:
                print("  Component breakdown:")
                for comp_name, comp_total in episode_data.component_totals.items():
                    print(f"    - {comp_name}: {comp_total:.2f}")

        # Increment episode counter
        self.episode_count += 1
        self.current_episode_step = 0

    def close(self) -> None:
        """Close the environment and flush data."""
        # End current episode if in progress
        if self.current_episode_step > 0:
            self._end_episode()

        # Close collector
        self.collector.close()

        if self.verbose >= 1:
            print(f"[RewardScope] Run complete!")
            print(f"  Total steps: {self.step_count}")
            print(f"  Total episodes: {self.episode_count}")
            print(f"  Final hacking score: {self.detector_suite.get_hacking_score():.3f}")

            all_alerts = self.detector_suite.get_all_alerts()
            if all_alerts:
                print(f"  Total alerts: {len(all_alerts)}")
                alert_types = {}
                for alert in all_alerts:
                    alert_type = alert.type.value
                    alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

                print("  Alert breakdown:")
                for alert_type, count in alert_types.items():
                    print(f"    - {alert_type}: {count}")

        # Close wrapped environment
        super().close()

    def get_alerts(self):
        """Get all hacking alerts detected so far."""
        return self.detector_suite.get_all_alerts()

    def get_hacking_score(self) -> float:
        """Get overall hacking score (0-1)."""
        return self.detector_suite.get_hacking_score()

    def get_component_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each reward component."""
        return self.decomposer.get_component_stats()

    def get_episode_history(self, n: int = 50):
        """Get recent episode history from collector."""
        return self.collector.get_episode_history(n=n)
