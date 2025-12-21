"""
Basic example: CartPole with reward tracking.

This is a simple example to verify Phase 1 implementation works.
Uses only the core components (collector and decomposer) without
requiring Stable-Baselines3.
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
import numpy as np

from reward_scope.core.collector import DataCollector, StepData
from reward_scope.core.decomposer import RewardDecomposer


def random_policy(observation, action_space):
    """Simple random policy for testing."""
    return action_space.sample()


def main():
    print("=" * 60)
    print("CartPole Basic Example - Phase 1 Test")
    print("=" * 60)

    # Create environment
    env = gym.make("CartPole-v1")

    # Create data collector
    collector = DataCollector(
        run_name="cartpole_basic_test",
        storage_dir="./reward_scope_data",
        buffer_size=100,
    )

    # Create reward decomposer
    # CartPole has a simple +1 reward per step
    decomposer = RewardDecomposer(track_residual=True)
    decomposer.register_component(
        "survival",
        lambda obs, act, info: 1.0,  # CartPole always gives +1 reward
        description="Survival reward (constant +1)",
    )

    # Run multiple episodes
    num_episodes = 5
    total_steps = 0

    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"\nEpisode {episode + 1}/{num_episodes}")

        while not done:
            # Select action (random policy)
            action = random_policy(observation, env.action_space)

            # Take step in environment
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

            # Update tracking
            observation = next_observation
            episode_reward += reward
            episode_length += 1
            total_steps += 1

        # End episode
        episode_data = collector.end_episode()

        # Print episode summary
        print(f"  Reward: {episode_reward:.1f}")
        print(f"  Length: {episode_length}")
        print(f"  Component totals: {episode_data.component_totals}")

    # Close environment and collector
    env.close()
    collector.close()

    print("\n" + "=" * 60)
    print("Testing Data Retrieval")
    print("=" * 60)

    # Reopen collector to test data retrieval
    collector = DataCollector(
        run_name="cartpole_basic_test",
        storage_dir="./reward_scope_data",
    )

    # Test get_episode_history
    print("\nEpisode History:")
    episodes = collector.get_episode_history(n=num_episodes)
    for ep in episodes:
        print(f"  Episode {ep.episode}: Reward={ep.total_reward:.1f}, Length={ep.length}")

    # Test get_recent_steps
    print("\nRecent Steps (last 10):")
    recent_steps = collector.get_recent_steps(n=10)
    for step in recent_steps[-5:]:  # Show last 5
        print(f"  Step {step.step}: Episode={step.episode}, Reward={step.reward}")

    # Test component statistics
    print("\nComponent Statistics:")
    stats = decomposer.get_component_stats()
    for comp_name, comp_stats in stats.items():
        print(f"  {comp_name}:")
        print(f"    Mean: {comp_stats['mean']:.4f}")
        print(f"    Std:  {comp_stats['std']:.4f}")
        print(f"    Count: {comp_stats['count']}")

    # Test dominance check
    print("\nDominance Check (threshold=0.8):")
    dominant = decomposer.check_dominance(threshold=0.8)
    if dominant:
        print(f"  Dominant components: {dominant}")
    else:
        print("  No dominant components detected")

    collector.close()

    print("\n" + "=" * 60)
    print("Phase 1 Test Complete!")
    print("=" * 60)
    print(f"\nTotal steps collected: {total_steps}")
    print(f"Total episodes: {num_episodes}")
    print(f"Database saved to: ./reward_scope_data/cartpole_basic_test.db")


if __name__ == "__main__":
    main()
