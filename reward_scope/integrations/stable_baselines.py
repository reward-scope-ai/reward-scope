"""
Stable-Baselines3 Integration

Provides a callback that hooks into SB3 training loops to collect
reward debugging data.
"""

from typing import Optional, Dict, Any, Callable
import time
import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import VecEnv
except ImportError:
    raise ImportError(
        "stable-baselines3 is required for SB3 integration. "
        "Install it with: pip install stable-baselines3"
    )

from ..core.collector import DataCollector, StepData
from ..core.decomposer import RewardDecomposer
from ..core.detectors import HackingDetectorSuite


class RewardScopeCallback(BaseCallback):
    """
    Stable-Baselines3 callback for reward debugging.

    Collects step data, decomposes rewards, and runs hacking detectors
    during training.

    Usage:
        from reward_scope.integrations import RewardScopeCallback

        callback = RewardScopeCallback(
            run_name="ppo_cartpole",
            auto_extract_prefix="reward_"  # if env provides reward components
        )

        model = PPO("MlpPolicy", env)
        model.learn(total_timesteps=10000, callback=callback)
    """

    def __init__(
        self,
        run_name: str = "sb3_run",
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
        observation_bounds: Optional[tuple] = None,
        action_bounds: Optional[tuple] = None,
        # Dashboard settings
        start_dashboard: bool = False,
        dashboard_port: int = 8050,
        # Other settings
        verbose: int = 1,
    ):
        """
        Args:
            run_name: Unique name for this training run
            storage_dir: Directory to store data
            auto_extract_prefix: Prefix for auto-extracting reward components from info dict
            component_fns: Dict of component_name -> function(obs, action, info) -> float
            enable_*: Enable/disable specific detectors
            observation_bounds: (low, high) for boundary exploitation detector
            action_bounds: (low, high) for boundary exploitation detector
            start_dashboard: Whether to auto-start the web dashboard
            dashboard_port: Port for the dashboard server
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(verbose=verbose)

        self.run_name = run_name
        self.storage_dir = storage_dir
        self.start_dashboard = start_dashboard
        self.dashboard_port = dashboard_port

        # Dashboard process
        self._dashboard_process = None

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
        self._episode_start_step = 0

    def _on_training_start(self) -> None:
        """Called before training starts."""
        if self.verbose >= 1:
            print(f"[RewardScope] Starting data collection for run: {self.run_name}")
            print(f"[RewardScope] Storage directory: {self.storage_dir}")

        # Start dashboard if requested
        if self.start_dashboard:
            self._start_dashboard()

    def _start_dashboard(self) -> None:
        """Start the dashboard server in a subprocess."""
        import subprocess
        import sys

        self._dashboard_process = subprocess.Popen(
            [sys.executable, "-m", "reward_scope.cli", "dashboard",
             "--run-name", self.run_name,
             "--data-dir", self.storage_dir,
             "--port", str(self.dashboard_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if self.verbose >= 1:
            print(f"[RewardScope] 🌐 Dashboard started at http://localhost:{self.dashboard_port}")

    def _stop_dashboard(self) -> None:
        """Stop the dashboard server."""
        if self._dashboard_process:
            self._dashboard_process.terminate()
            try:
                self._dashboard_process.wait(timeout=5)
            except:
                self._dashboard_process.kill()
            self._dashboard_process = None

    def _on_step(self) -> bool:
        """
        Called after each environment step (or vectorized step).

        Returns:
            True to continue training, False to stop.
        """
        # Get environment info
        # In SB3, self.training_env is a VecEnv that may contain multiple envs
        is_vec_env = isinstance(self.training_env, VecEnv)
        n_envs = self.training_env.num_envs if is_vec_env else 1

        # Get data from the rollout buffer or locals
        # SB3 stores: obs, actions, rewards, dones, infos in self.locals
        if 'infos' not in self.locals:
            # No info available yet, skip
            return True

        infos = self.locals.get('infos', [])
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])

        # Handle both vectorized and non-vectorized envs
        if not isinstance(infos, list):
            infos = [infos]
            rewards = [rewards]
            dones = [dones]

        # Get observations and actions
        # Note: In SB3, 'new_obs' is the observation after the step
        new_obs = self.locals.get('new_obs', None)
        actions = self.locals.get('actions', None)

        # Iterate over each environment in the vectorized env
        for env_idx in range(min(len(infos), n_envs)):
            info = infos[env_idx] if env_idx < len(infos) else {}
            reward = float(rewards[env_idx]) if env_idx < len(rewards) else 0.0
            done = bool(dones[env_idx]) if env_idx < len(dones) else False

            # Extract observation and action for this env
            if new_obs is not None:
                obs = new_obs[env_idx] if is_vec_env else new_obs
            else:
                obs = None

            if actions is not None:
                action = actions[env_idx] if is_vec_env else actions
            else:
                action = None

            # Check for truncated flag
            truncated = info.get('TimeLimit.truncated', False)

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
                done=done,
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
                done=done,
                info=info,
            )

            # Print alerts if any
            if alerts and self.verbose >= 1:
                for alert in alerts:
                    print(f"[RewardScope] ALERT: {alert.type.value}: {alert.description}")
                    if self.verbose >= 2:
                        print(f"  Evidence: {alert.evidence}")
                        print(f"  Fix: {alert.suggested_fix}")

            # Handle episode end
            if done or truncated:
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

                # Increment episode counter
                self.episode_count += 1
                self._episode_start_step = self.step_count + 1

                if self.verbose >= 1:
                    print(f"[RewardScope] Episode {episode_data.episode} complete: "
                          f"reward={episode_data.total_reward:.2f}, "
                          f"length={episode_data.length}, "
                          f"hacking_score={self.detector_suite.get_hacking_score():.3f}")

            # Increment step counter
            self.step_count += 1

        return True  # Continue training

    def _on_training_end(self) -> None:
        """Called when training ends."""
        if self.verbose >= 1:
            print(f"[RewardScope] Training complete!")
            print(f"  Total steps: {self.step_count}")
            print(f"  Total episodes: {self.episode_count}")
            print(f"  Hacking score: {self.detector_suite.get_hacking_score():.3f}")

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

        # Stop dashboard
        self._stop_dashboard()

        # Close collector (flush remaining data)
        self.collector.close()

    def get_alerts(self):
        """Get all hacking alerts detected so far."""
        return self.detector_suite.get_all_alerts()

    def get_hacking_score(self) -> float:
        """Get overall hacking score (0-1)."""
        return self.detector_suite.get_hacking_score()

    def get_component_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for each reward component."""
        return self.decomposer.get_component_stats()
