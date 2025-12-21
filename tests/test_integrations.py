"""
Tests for reward_scope.integrations

Tests the Stable-Baselines3 callback and Gymnasium wrapper integrations.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import gymnasium as gym

from reward_scope.integrations.gymnasium import RewardScopeWrapper


# Test Gymnasium Wrapper
class TestRewardScopeWrapper:
    """Test Gymnasium wrapper integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def wrapped_env(self, temp_dir):
        """Create a wrapped CartPole environment."""
        env = gym.make("CartPole-v1")
        wrapped = RewardScopeWrapper(
            env,
            run_name="test_cartpole",
            storage_dir=temp_dir,
            verbose=0,
        )
        yield wrapped
        wrapped.close()

    def test_wrapper_initialization(self, temp_dir):
        """Test wrapper initializes correctly."""
        env = gym.make("CartPole-v1")
        wrapped = RewardScopeWrapper(
            env,
            run_name="test_init",
            storage_dir=temp_dir,
        )

        assert wrapped.run_name == "test_init"
        assert wrapped.episode_count == 0
        assert wrapped.step_count == 0

        wrapped.close()

    def test_reset_adds_rewardscope_info(self, wrapped_env):
        """Test that reset adds RewardScope keys to info dict."""
        obs, info = wrapped_env.reset()

        assert 'reward_components' in info
        assert 'hacking_alerts' in info
        assert 'hacking_score' in info
        assert isinstance(info['reward_components'], dict)
        assert isinstance(info['hacking_alerts'], list)
        assert isinstance(info['hacking_score'], float)

    def test_step_adds_rewardscope_info(self, wrapped_env):
        """Test that step adds RewardScope keys to info dict."""
        obs, info = wrapped_env.reset()
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)

        assert 'reward_components' in info
        assert 'hacking_alerts' in info
        assert 'hacking_score' in info
        assert isinstance(info['reward_components'], dict)
        assert isinstance(info['hacking_alerts'], list)

    def test_step_data_collection(self, wrapped_env):
        """Test that step data is collected."""
        obs, info = wrapped_env.reset()

        # Run a few steps
        for _ in range(10):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            if terminated or truncated:
                obs, info = wrapped_env.reset()

        # Verify data was collected
        assert wrapped_env.step_count >= 10
        assert wrapped_env.collector.conn is not None

        # Flush buffer to ensure data is in database
        wrapped_env.collector._flush_step_buffer()

        # Check that steps were logged
        recent_steps = wrapped_env.collector.get_recent_steps(n=5)
        assert len(recent_steps) > 0

    def test_episode_tracking(self, wrapped_env):
        """Test that episodes are tracked correctly."""
        initial_episode_count = wrapped_env.episode_count

        obs, info = wrapped_env.reset()

        # Run until episode ends
        done = False
        steps = 0
        max_steps = 500

        while not done and steps < max_steps:
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            done = terminated or truncated
            steps += 1

        # Episode should have ended
        assert wrapped_env.episode_count > initial_episode_count

        # Check episode data was saved
        episodes = wrapped_env.collector.get_episode_history(n=1)
        assert len(episodes) > 0

    def test_reward_decomposition(self, wrapped_env):
        """Test that rewards are decomposed."""
        obs, info = wrapped_env.reset()
        action = wrapped_env.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped_env.step(action)

        # Should have reward components (at least residual since no components defined)
        assert len(info['reward_components']) > 0

    def test_custom_component_functions(self, temp_dir):
        """Test wrapper with custom component functions."""
        env = gym.make("CartPole-v1")

        # Define custom component functions
        def distance_component(obs, action, info):
            # Simple example: distance from center
            if obs is not None:
                return -abs(obs[0])  # Cart position
            return 0.0

        wrapped = RewardScopeWrapper(
            env,
            run_name="test_custom_components",
            storage_dir=temp_dir,
            component_fns={"distance": distance_component},
            verbose=0,
        )

        obs, info = wrapped.reset()
        action = wrapped.action_space.sample()
        obs, reward, terminated, truncated, info = wrapped.step(action)

        # Should have custom component
        assert 'distance' in info['reward_components']

        wrapped.close()

    def test_auto_extract_prefix(self, temp_dir):
        """Test auto-extracting components from info dict."""
        # Create a custom env that provides reward components
        class ComponentEnv(gym.Env):
            def __init__(self):
                self.observation_space = gym.spaces.Box(-1, 1, (4,))
                self.action_space = gym.spaces.Discrete(2)
                self.steps = 0

            def reset(self, seed=None, options=None):
                self.steps = 0
                return np.zeros(4), {}

            def step(self, action):
                self.steps += 1
                obs = np.random.randn(4)
                # Provide reward components with prefix
                info = {
                    "reward_distance": -0.5,
                    "reward_velocity": 0.3,
                    "other_info": "test",
                }
                reward = -0.2
                terminated = self.steps >= 10
                return obs, reward, terminated, False, info

        env = ComponentEnv()
        wrapped = RewardScopeWrapper(
            env,
            run_name="test_auto_extract",
            storage_dir=temp_dir,
            auto_extract_prefix="reward_",
            verbose=0,
        )

        obs, info = wrapped.reset()
        obs, reward, terminated, truncated, info = wrapped.step(0)

        # Should have extracted components
        assert 'distance' in info['reward_components']
        assert 'velocity' in info['reward_components']
        assert info['reward_components']['distance'] == -0.5
        assert info['reward_components']['velocity'] == 0.3

        wrapped.close()

    def test_detector_integration(self, temp_dir):
        """Test that detectors run during training."""
        env = gym.make("CartPole-v1")
        wrapped = RewardScopeWrapper(
            env,
            run_name="test_detectors",
            storage_dir=temp_dir,
            verbose=0,
        )

        obs, info = wrapped.reset()

        # Run enough steps to potentially trigger detectors
        for _ in range(100):
            action = 0  # Always take same action (should trigger action repetition)
            obs, reward, terminated, truncated, info = wrapped.step(action)
            if terminated or truncated:
                obs, info = wrapped.reset()

        # Check if any alerts were generated
        # (May or may not have alerts depending on the run, but should not crash)
        all_alerts = wrapped.get_alerts()
        assert isinstance(all_alerts, list)

        wrapped.close()

    def test_hacking_score(self, wrapped_env):
        """Test hacking score calculation."""
        obs, info = wrapped_env.reset()

        # Initial score should be 0
        assert wrapped_env.get_hacking_score() == 0.0

        # Run some steps
        for _ in range(20):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            if terminated or truncated:
                break

        # Score should be a valid float between 0 and 1
        score = wrapped_env.get_hacking_score()
        assert 0.0 <= score <= 1.0

    def test_component_stats(self, wrapped_env):
        """Test component statistics retrieval."""
        obs, info = wrapped_env.reset()

        # Run some steps
        for _ in range(20):
            action = wrapped_env.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            if terminated or truncated:
                break

        # Get component stats
        stats = wrapped_env.get_component_stats()
        assert isinstance(stats, dict)

    def test_multiple_episodes(self, wrapped_env):
        """Test running multiple complete episodes."""
        num_episodes = 3

        for ep in range(num_episodes):
            obs, info = wrapped_env.reset()
            done = False
            steps = 0
            max_steps = 500

            while not done and steps < max_steps:
                action = wrapped_env.action_space.sample()
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                done = terminated or truncated
                steps += 1

        # Should have completed multiple episodes
        assert wrapped_env.episode_count >= num_episodes

        # Check episode history
        episodes = wrapped_env.get_episode_history(n=num_episodes)
        assert len(episodes) >= num_episodes


# Test Stable-Baselines3 Callback
class TestRewardScopeCallback:
    """Test Stable-Baselines3 callback integration."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    def test_callback_requires_sb3(self):
        """Test that importing callback works when SB3 is installed."""
        try:
            from reward_scope.integrations.stable_baselines import RewardScopeCallback
            assert RewardScopeCallback is not None
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

    def test_callback_initialization(self, temp_dir):
        """Test callback initializes correctly."""
        try:
            from reward_scope.integrations.stable_baselines import RewardScopeCallback
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        callback = RewardScopeCallback(
            run_name="test_callback",
            storage_dir=temp_dir,
            verbose=0,
        )

        assert callback.run_name == "test_callback"
        assert callback.episode_count == 0
        assert callback.step_count == 0

    def test_callback_with_ppo(self, temp_dir):
        """Test callback with PPO on CartPole."""
        try:
            from stable_baselines3 import PPO
            from reward_scope.integrations.stable_baselines import RewardScopeCallback
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        # Create environment
        env = gym.make("CartPole-v1")

        # Create callback
        callback = RewardScopeCallback(
            run_name="test_ppo_cartpole",
            storage_dir=temp_dir,
            verbose=0,
        )

        # Train PPO for a few steps
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=100, callback=callback)

        # Verify data was collected
        assert callback.step_count > 0

        # Check that data was saved
        db_path = Path(temp_dir) / "test_ppo_cartpole.db"
        assert db_path.exists()

        # Get alerts and stats
        alerts = callback.get_alerts()
        assert isinstance(alerts, list)

        score = callback.get_hacking_score()
        assert 0.0 <= score <= 1.0

    def test_callback_with_vectorized_env(self, temp_dir):
        """Test callback with vectorized environment."""
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            from reward_scope.integrations.stable_baselines import RewardScopeCallback
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        # Create vectorized environment
        def make_env():
            return gym.make("CartPole-v1")

        env = DummyVecEnv([make_env for _ in range(2)])

        # Create callback
        callback = RewardScopeCallback(
            run_name="test_vec_env",
            storage_dir=temp_dir,
            verbose=0,
        )

        # Train PPO
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=100, callback=callback)

        # Verify data was collected
        assert callback.step_count > 0

    def test_callback_custom_components(self, temp_dir):
        """Test callback with custom reward components."""
        try:
            from stable_baselines3 import PPO
            from reward_scope.integrations.stable_baselines import RewardScopeCallback
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        # Define custom component function
        def position_component(obs, action, info):
            if obs is not None:
                return -abs(obs[0])  # Cart position
            return 0.0

        # Create callback with custom components
        callback = RewardScopeCallback(
            run_name="test_custom_sb3",
            storage_dir=temp_dir,
            component_fns={"position": position_component},
            verbose=0,
        )

        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=100, callback=callback)

        # Check component stats
        stats = callback.get_component_stats()
        assert 'position' in stats

    def test_callback_detector_settings(self, temp_dir):
        """Test callback with custom detector settings."""
        try:
            from stable_baselines3 import PPO
            from reward_scope.integrations.stable_baselines import RewardScopeCallback
        except ImportError:
            pytest.skip("stable-baselines3 not installed")

        # Create callback with selective detectors
        callback = RewardScopeCallback(
            run_name="test_detectors_sb3",
            storage_dir=temp_dir,
            enable_state_cycling=True,
            enable_action_repetition=True,
            enable_component_imbalance=False,
            enable_reward_spiking=False,
            enable_boundary_exploitation=False,
            verbose=0,
        )

        # Should have only 2 detectors enabled
        assert len(callback.detector_suite.detectors) == 2

        env = gym.make("CartPole-v1")
        model = PPO("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=50, callback=callback)


# Test with LunarLander
class TestLunarLanderIntegration:
    """Test integration with LunarLander environment."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    def test_lunarlander_wrapper(self, temp_dir):
        """Test Gymnasium wrapper with LunarLander."""
        try:
            env = gym.make("LunarLander-v2")
        except Exception:
            pytest.skip("LunarLander-v2 not available")

        wrapped = RewardScopeWrapper(
            env,
            run_name="test_lunarlander",
            storage_dir=temp_dir,
            verbose=0,
        )

        obs, info = wrapped.reset()

        # Run a few steps
        for _ in range(50):
            action = wrapped.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped.step(action)

            # Verify info dict has RewardScope keys
            assert 'reward_components' in info
            assert 'hacking_alerts' in info

            if terminated or truncated:
                obs, info = wrapped.reset()

        wrapped.close()

        # Verify data was saved
        db_path = Path(temp_dir) / "test_lunarlander.db"
        assert db_path.exists()


# Integration test combining both
class TestFullIntegration:
    """Test full integration scenarios."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    def test_wrapper_detects_action_repetition(self, temp_dir):
        """Test that wrapper detects action repetition."""
        env = gym.make("CartPole-v1")
        wrapped = RewardScopeWrapper(
            env,
            run_name="test_action_rep",
            storage_dir=temp_dir,
            verbose=0,
        )

        obs, info = wrapped.reset()

        # Take same action repeatedly
        for _ in range(100):
            obs, reward, terminated, truncated, info = wrapped.step(0)
            if terminated or truncated:
                obs, info = wrapped.reset()

        # Should potentially detect action repetition
        alerts = wrapped.get_alerts()
        # May or may not have alerts depending on threshold, but should not crash

        wrapped.close()

    def test_data_persistence(self, temp_dir):
        """Test that data is persisted to database."""
        env = gym.make("CartPole-v1")
        wrapped = RewardScopeWrapper(
            env,
            run_name="test_persistence",
            storage_dir=temp_dir,
            verbose=0,
        )

        obs, info = wrapped.reset()

        # Run steps
        for _ in range(50):
            action = wrapped.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped.step(action)
            if terminated or truncated:
                obs, info = wrapped.reset()

        wrapped.close()

        # Verify database file exists and has data
        db_path = Path(temp_dir) / "test_persistence.db"
        assert db_path.exists()
        assert db_path.stat().st_size > 0

        # Reopen and verify data can be read
        from reward_scope.core.collector import DataCollector

        collector = DataCollector(
            run_name="test_persistence",
            storage_dir=temp_dir,
        )

        steps = collector.get_recent_steps(n=10)
        assert len(steps) > 0

        collector.close()
