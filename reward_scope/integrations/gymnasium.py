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
from ..core.detectors import HackingDetectorSuite, HackingAlert, AlertSeverity


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
        # Two-layer detection settings (Phase 2)
        adaptive_baseline: bool = True,
        baseline_window: int = 50,
        baseline_warmup: int = 20,
        baseline_sensitivity: float = 2.0,
        # Auto-calibration settings (Phase 5)
        min_warmup_episodes: int = 10,
        max_warmup_episodes: int = 50,
        stability_threshold: float = 0.1,
        # Legacy adaptive baseline settings (Phase 1 - experimental)
        use_adaptive_baselines: bool = False,
        calibration_episodes: int = 20,
        baseline_sigma_threshold: float = 3.0,
        # Dashboard settings
        start_dashboard: bool = False,
        dashboard_port: int = 8050,
        # WandB settings
        wandb_logging: bool = False,
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
            adaptive_baseline: Enable two-layer detection (on by default). When enabled,
                static detectors are modulated by a rolling baseline that learns what's
                "normal" for this training run. This reduces false positives.
            baseline_window: Rolling window size for baseline statistics (default 50 episodes)
            baseline_warmup: Episodes before baseline layer activates (default 20)
            baseline_sensitivity: Std devs threshold for "abnormal" (default 2.0)
            min_warmup_episodes: Minimum warmup before auto-calibration can end (default 10)
            max_warmup_episodes: Maximum warmup - activate anyway after this (default 50)
            stability_threshold: Normalized variance threshold for stability (default 0.1)
            use_adaptive_baselines: Legacy Phase 1 adaptive baselines (experimental)
            calibration_episodes: Legacy calibration episodes (default 20)
            baseline_sigma_threshold: Legacy deviation threshold (default 3.0)
            start_dashboard: Whether to auto-start the web dashboard
            dashboard_port: Port for the dashboard server
            wandb_logging: Whether to log metrics to WandB (requires wandb.init() to be called first)
            verbose: Verbosity level (0=silent, 1=info, 2=debug)
        """
        super().__init__(env)

        self.run_name = run_name
        self.storage_dir = storage_dir
        self.verbose = verbose
        self.dashboard_port = dashboard_port
        self.wandb_logging = wandb_logging

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
            # Two-layer detection (Phase 2)
            adaptive_baseline=adaptive_baseline,
            baseline_window=baseline_window,
            baseline_warmup=baseline_warmup,
            baseline_sensitivity=baseline_sensitivity,
            # Auto-calibration (Phase 5)
            min_warmup_episodes=min_warmup_episodes,
            max_warmup_episodes=max_warmup_episodes,
            stability_threshold=stability_threshold,
            # Legacy Phase 1
            use_adaptive_baselines=use_adaptive_baselines,
            calibration_episodes=calibration_episodes,
            baseline_sigma_threshold=baseline_sigma_threshold,
        )

        # Track baseline settings for logging
        self.adaptive_baseline = adaptive_baseline
        self.use_adaptive_baselines = use_adaptive_baselines

        # Tracking
        self.episode_count = 0
        self.step_count = 0
        self.current_episode_step = 0
        self._last_observation = None
        self._last_action = None
        self._live_update_interval = 50  # Update DB every N steps for live dashboard

        # Start dashboard if requested
        if start_dashboard:
            self._start_dashboard()

    def _start_dashboard(self) -> None:
        """Start the dashboard server in a subprocess."""
        import subprocess
        import sys

        self._dashboard_process = subprocess.Popen(
            [sys.executable, "-m", "reward_scope.dashboard.app",
             "--run-name", self.run_name,
             "--data-dir", self.storage_dir,
             "--port", str(self.dashboard_port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if self.verbose >= 1:
            print(f"[RewardScope] Dashboard started at http://localhost:{self.dashboard_port}")

    def _stop_dashboard(self) -> None:
        """Stop the dashboard server."""
        if self._dashboard_process:
            self._dashboard_process.terminate()
            try:
                self._dashboard_process.wait(timeout=5)
            except:
                self._dashboard_process.kill()
            self._dashboard_process = None

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

        # Add two-layer detection info (Phase 2)
        if self.adaptive_baseline:
            info['baseline_active'] = self.detector_suite.baseline_is_active
            info['baseline_warmup_progress'] = self.detector_suite.baseline_warmup_progress
            info['suppressed_count'] = self.detector_suite.get_suppressed_count()
            info['warning_count'] = self.detector_suite.get_warning_count()

        # Add legacy adaptive baseline info if enabled (Phase 1)
        if self.use_adaptive_baselines:
            info['baseline_calibrated'] = self.detector_suite.is_calibrated
            info['calibration_progress'] = self.detector_suite.calibration_progress

        if self.verbose >= 1:
            # Show warmup progress for two-layer detection
            if self.adaptive_baseline and not self.detector_suite.baseline_is_active:
                progress = self.detector_suite.baseline_warmup_progress
                print(f"[RewardScope] Episode {self.episode_count} started (baseline warmup: {progress:.0%})")
            elif self.use_adaptive_baselines and not self.detector_suite.is_calibrated:
                progress = self.detector_suite.calibration_progress
                print(f"[RewardScope] Episode {self.episode_count} started (calibrating: {progress:.0%})")
            else:
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

        # Print alerts if any (with alert severity and confidence)
        if alerts and self.verbose >= 1:
            for alert in alerts:
                severity_label = alert.alert_severity.value.upper()
                confidence_str = f" (confidence={alert.confidence:.2f})" if alert.confidence is not None else ""
                print(f"[RewardScope] {severity_label}{confidence_str}: {alert.type.value}: {alert.description}")
                if self.verbose >= 2:
                    print(f"  Evidence: {alert.evidence}")
                    print(f"  Fix: {alert.suggested_fix}")
                    if alert.baseline_z_score is not None:
                        print(f"  Baseline z-score: {alert.baseline_z_score:.2f}")

        # Inject RewardScope data into info dict
        info['reward_components'] = reward_components
        # Convert alerts to serializable dicts (include alert_severity and confidence)
        info['hacking_alerts'] = [
            {
                "type": a.type.value,
                "severity": a.severity,
                "alert_severity": a.alert_severity.value,
                "description": a.description,
                "step": a.step,
                "episode": a.episode,
                "baseline_z_score": a.baseline_z_score,
                "confidence": a.confidence,
            }
            for a in alerts
        ]
        info['hacking_score'] = self.detector_suite.get_hacking_score()

        # Add two-layer detection info (Phase 2)
        if self.adaptive_baseline:
            info['baseline_active'] = self.detector_suite.baseline_is_active
            info['baseline_warmup_progress'] = self.detector_suite.baseline_warmup_progress
            info['suppressed_count'] = self.detector_suite.get_suppressed_count()
            info['warning_count'] = self.detector_suite.get_warning_count()

        # Add legacy adaptive baseline info if enabled (Phase 1)
        if self.use_adaptive_baselines:
            info['baseline_calibrated'] = self.detector_suite.is_calibrated
            info['calibration_progress'] = self.detector_suite.calibration_progress

        # Increment counters
        self.step_count += 1
        self.current_episode_step += 1

        # Update live hacking score periodically (for dashboard)
        if self.current_episode_step % self._live_update_interval == 0:
            self._update_live_hacking_score()

        # Handle episode end
        if terminated or truncated:
            self._end_episode()

        return obs, reward, terminated, truncated, info

    def _update_live_hacking_score(self) -> None:
        """Update the database with current in-progress hacking score."""
        # Get current hacking score and alert count from detector suite
        hacking_score = self.detector_suite.get_hacking_score()
        alert_count = len(self.detector_suite.get_all_alerts())

        # Update live state in database (for dashboard to read)
        if self.current_episode_step > 0:
            self.collector.update_live_hacking_state(
                episode=self.episode_count,
                hacking_score=hacking_score,
                alert_count=alert_count,
            )

    def _end_episode(self) -> None:
        """End the current episode and run episode-level checks."""
        # Get episode data from collector
        episode_data = self.collector.end_episode()

        # Run episode-level detectors (pass episode number for two-layer detection)
        episode_alerts = self.detector_suite.on_episode_end({
            "component_totals": episode_data.component_totals,
            "episode": self.episode_count,
        })

        if episode_alerts and self.verbose >= 1:
            for alert in episode_alerts:
                severity_label = alert.alert_severity.value.upper()
                confidence_str = f" (confidence={alert.confidence:.2f})" if alert.confidence is not None else ""
                print(f"[RewardScope] {severity_label}{confidence_str} Episode {self.episode_count}: {alert.type.value}: {alert.description}")

        # Get hacking score and flags from detector suite
        hacking_score = self.detector_suite.get_hacking_score()
        hacking_flags = [alert.type.value for alert in self.detector_suite.get_all_alerts()]
        all_alerts = self.detector_suite.get_all_alerts()

        # Update database with computed hacking data
        self.collector.update_episode_hacking_data(
            episode=episode_data.episode,
            hacking_score=hacking_score,
            hacking_flags=hacking_flags,
        )

        # Log to WandB if enabled
        if self.wandb_logging:
            self._log_to_wandb(episode_data, hacking_score, all_alerts)

        # Clear live state now that episode is complete
        self.collector.clear_live_hacking_state()

        # Reset detectors for next episode
        self.detector_suite.reset()

        if self.verbose >= 1:
            # Build episode summary with baseline info
            summary = f"[RewardScope] Episode {episode_data.episode} complete: " \
                      f"reward={episode_data.total_reward:.2f}, " \
                      f"length={episode_data.length}, " \
                      f"hacking_score={hacking_score:.3f}"

            # Add suppressed count if two-layer detection is active
            if self.adaptive_baseline and self.detector_suite.baseline_is_active:
                suppressed = self.detector_suite.get_suppressed_count()
                if suppressed > 0:
                    summary += f" (suppressed: {suppressed})"

            print(summary)

            # Print component breakdown if available
            if episode_data.component_totals and self.verbose >= 2:
                print("  Component breakdown:")
                for comp_name, comp_total in episode_data.component_totals.items():
                    print(f"    - {comp_name}: {comp_total:.2f}")

        # Increment episode counter
        self.episode_count += 1
        self.current_episode_step = 0

    def _log_to_wandb(self, episode_data, hacking_score: float, alerts: list) -> None:
        """Log episode metrics to WandB if available."""
        try:
            import wandb

            # Check if wandb is initialized
            if wandb.run is None:
                if self.verbose >= 1:
                    print("[RewardScope] Warning: wandb_logging enabled but wandb.init() not called")
                return

            # Prepare metrics dict
            metrics = {
                "rewardscope/hacking_score": hacking_score,
                "rewardscope/episode_reward": episode_data.total_reward,
                "rewardscope/episode_length": episode_data.length,
                "rewardscope/alerts_count": len(alerts),
            }

            # Log adaptive baseline metrics
            if self.adaptive_baseline:
                metrics["rewardscope/alerts_suppressed"] = self.detector_suite.get_suppressed_count()
                metrics["rewardscope/alerts_warnings"] = self.detector_suite.get_warning_count()
                metrics["rewardscope/baseline_active"] = self.detector_suite.baseline_is_active

                # Compute max confidence from alerts
                max_confidence = 0.0
                for alert in alerts:
                    if alert.confidence is not None and alert.confidence > max_confidence:
                        max_confidence = alert.confidence
                metrics["rewardscope/alert_max_confidence"] = max_confidence

            # Log component totals
            if episode_data.component_totals:
                for comp_name, comp_total in episode_data.component_totals.items():
                    metrics[f"rewardscope/component/{comp_name}"] = comp_total

            # Log metrics
            wandb.log(metrics)

            # Log high severity alerts as wandb warnings
            for alert in alerts:
                if alert.severity > 0.7:
                    wandb.alert(
                        title=f"RewardScope Alert: {alert.type.value}",
                        text=f"{alert.description}\nSeverity: {alert.severity:.2f}\nSuggested fix: {alert.suggested_fix}",
                        level=wandb.AlertLevel.WARN,
                    )

        except ImportError:
            if self.verbose >= 1:
                print("[RewardScope] Warning: wandb_logging enabled but wandb not installed")
        except Exception as e:
            if self.verbose >= 1:
                print(f"[RewardScope] Warning: Failed to log to wandb: {e}")

    def close(self) -> None:
        """Close the environment and flush data."""
        # End current episode if in progress
        if self.current_episode_step > 0:
            self._end_episode()

        # Close collector
        self.collector.close()

        # Stop dashboard
        self._stop_dashboard()

        if self.verbose >= 1:
            print(f"[RewardScope] Run complete!")
            print(f"  Total steps: {self.step_count}")
            print(f"  Total episodes: {self.episode_count}")
            print(f"  Final hacking score: {self.detector_suite.get_hacking_score():.3f}")

            all_alerts = self.detector_suite.get_all_alerts()
            if all_alerts:
                print(f"  Total alerts: {len(all_alerts)}")

                # Break down by alert severity (ALERT vs WARNING)
                alert_count = sum(1 for a in all_alerts if a.alert_severity == AlertSeverity.ALERT)
                warning_count = sum(1 for a in all_alerts if a.alert_severity == AlertSeverity.WARNING)
                if alert_count > 0 or warning_count > 0:
                    print(f"    - Confirmed alerts: {alert_count}")
                    print(f"    - Soft warnings: {warning_count}")

                # Break down by type
                alert_types = {}
                for alert in all_alerts:
                    alert_type = alert.type.value
                    alert_types[alert_type] = alert_types.get(alert_type, 0) + 1

                print("  Alert type breakdown:")
                for alert_type, count in alert_types.items():
                    print(f"    - {alert_type}: {count}")

            # Show suppressed count if two-layer detection was active
            if self.adaptive_baseline:
                suppressed = self.detector_suite.get_suppressed_count()
                if suppressed > 0:
                    print(f"  Suppressed false positives: {suppressed}")

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

    def get_baseline_summary(self) -> Optional[Dict[str, Any]]:
        """Get summary of adaptive baseline statistics (if enabled)."""
        return self.detector_suite.get_baseline_summary()

    def is_calibrated(self) -> bool:
        """Check if adaptive baseline calibration is complete."""
        return self.detector_suite.is_calibrated
