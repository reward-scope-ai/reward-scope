"""
CartPole Hacking Detection Demo

This example demonstrates how the hacking detection system works
by using policies that exhibit different hacking behaviors.
"""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np

from reward_scope.core.collector import DataCollector, StepData
from reward_scope.core.decomposer import RewardDecomposer
from reward_scope.core.detectors import HackingDetectorSuite


class HackingPolicy:
    """Base class for hacking policies."""

    def __init__(self, env):
        self.env = env

    def select_action(self, observation):
        raise NotImplementedError


class ActionRepetitionPolicy(HackingPolicy):
    """
    Always selects the same action - demonstrates action repetition hacking.
    """

    def __init__(self, env, action=1):
        super().__init__(env)
        self.fixed_action = action

    def select_action(self, observation):
        return self.fixed_action


class StateCyclingPolicy(HackingPolicy):
    """
    Oscillates between two actions - demonstrates state cycling.
    """

    def __init__(self, env):
        super().__init__(env)
        self.counter = 0

    def select_action(self, observation):
        # Alternate between actions to create a cycle
        self.counter += 1
        return self.counter % 2


class RandomPolicy(HackingPolicy):
    """Normal random policy for comparison."""

    def select_action(self, observation):
        return self.env.action_space.sample()


def run_experiment(policy_name, policy, num_episodes=10):
    """Run an experiment with a given policy."""
    print(f"\n{'='*60}")
    print(f"Experiment: {policy_name}")
    print('='*60)

    # Create environment
    env = gym.make("CartPole-v1")

    # Create components
    collector = DataCollector(
        run_name=f"cartpole_{policy_name.lower().replace(' ', '_')}",
        storage_dir="./reward_scope_data",
    )

    decomposer = RewardDecomposer(track_residual=False)
    decomposer.register_component(
        "survival",
        lambda obs, act, info: 1.0,
        description="Survival reward",
    )

    # Create detector suite with bounds
    detector_suite = HackingDetectorSuite(
        observation_bounds=(env.observation_space.low, env.observation_space.high),
        action_bounds=(np.array([0]), np.array([1])),  # Discrete actions 0, 1
    )

    # Run episodes
    total_steps = 0
    episode_rewards = []

    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Select action using policy
            action = policy.select_action(observation)

            # Take step
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Decompose reward
            components = decomposer.decompose(observation, action, reward, info)

            # Create step data
            step_data = StepData(
                step=total_steps,
                episode=episode,
                timestamp=time.time(),
                observation=observation.tolist(),
                action=int(action),
                reward=float(reward),
                done=terminated,
                truncated=truncated,
                info=info,
                reward_components=components,
            )

            # Log step
            collector.log_step(step_data)

            # Run detectors
            alerts = detector_suite.update(
                step=total_steps,
                episode=episode,
                observation=observation,
                action=action,
                reward=float(reward),
                reward_components=components,
                done=done,
                info=info,
            )

            # Print alerts in real-time
            for alert in alerts:
                print(f"  ⚠️  ALERT at step {alert.step}: {alert.type.value}")
                print(f"      {alert.description}")

            # Update
            observation = next_observation
            episode_reward += reward
            episode_length += 1
            total_steps += 1

        # End episode
        episode_data = collector.end_episode()
        episode_rewards.append(episode_reward)

        # Check for episode-level alerts
        episode_alerts = detector_suite.on_episode_end(episode_data.component_totals)
        for alert in episode_alerts:
            print(f"  ⚠️  EPISODE ALERT: {alert.type.value}")
            print(f"      {alert.description}")

        # Reset detectors
        detector_suite.reset()

        if episode % 2 == 0:
            print(f"  Episode {episode + 1}: Reward={episode_reward:.0f}, Length={episode_length}")

    # Print summary
    print(f"\n{'-'*60}")
    print("Summary:")
    print(f"  Total steps: {total_steps}")
    print(f"  Avg episode reward: {np.mean(episode_rewards):.1f} ± {np.std(episode_rewards):.1f}")
    print(f"  Avg episode length: {np.mean([r for r in episode_rewards]):.1f}")

    # Print all alerts
    all_alerts = detector_suite.get_all_alerts()
    print(f"\n  Total alerts detected: {len(all_alerts)}")

    if all_alerts:
        from collections import Counter
        alert_counts = Counter(a.type.value for a in all_alerts)
        print("  Alert breakdown:")
        for alert_type, count in alert_counts.items():
            print(f"    {alert_type}: {count}")

    # Print hacking score
    hacking_score = detector_suite.get_hacking_score()
    print(f"\n  Hacking Score: {hacking_score:.2f} / 1.00")

    if hacking_score > 0.5:
        print("  ⚠️  HIGH HACKING LIKELIHOOD DETECTED!")
    elif hacking_score > 0.2:
        print("  ⚠️  Moderate hacking likelihood")
    else:
        print("  ✓  Low hacking likelihood")

    # Cleanup
    env.close()
    collector.close()


def main():
    print("="*60)
    print("CartPole Hacking Detection Demo")
    print("="*60)
    print("\nThis demo compares different policies to show how")
    print("the hacking detection system identifies problematic behavior.")

    # Create environment for policies
    env = gym.make("CartPole-v1")

    # Test 1: Action Repetition
    print("\n\n🔍 TEST 1: Action Repetition Policy")
    print("   (Always selects action=1)")
    policy1 = ActionRepetitionPolicy(env, action=1)
    run_experiment("Action Repetition", policy1, num_episodes=5)

    # Test 2: State Cycling
    print("\n\n🔍 TEST 2: State Cycling Policy")
    print("   (Alternates between actions 0 and 1)")
    policy2 = StateCyclingPolicy(env)
    run_experiment("State Cycling", policy2, num_episodes=5)

    # Test 3: Random (baseline)
    print("\n\n🔍 TEST 3: Random Policy (Baseline)")
    print("   (Random actions - should have low hacking score)")
    policy3 = RandomPolicy(env)
    run_experiment("Random Baseline", policy3, num_episodes=5)

    env.close()

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nKey Findings:")
    print("  • Action Repetition policy triggers ACTION_REPETITION alerts")
    print("  • State Cycling policy may trigger STATE_CYCLING alerts")
    print("  • Random policy should have few/no alerts")
    print("\nNext Steps:")
    print("  • Check reward_scope_data/ for persisted data")
    print("  • Try tuning detector thresholds for your environment")
    print("  • Integrate with real RL training runs")


if __name__ == "__main__":
    main()
