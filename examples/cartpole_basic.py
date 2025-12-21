"""
CartPole Basic Example

The simplest RewardScope example. Demonstrates:
- Using the Gymnasium wrapper
- Automatic data collection
- Viewing results in the dashboard

This example trains a random agent on CartPole-v1 and collects debugging data.
Run this to verify RewardScope is installed correctly.
"""

import gymnasium as gym
from reward_scope.integrations import RewardScopeWrapper


def main():
    print("\n" + "="*60)
    print("🔬 RewardScope - CartPole Basic Example")
    print("="*60)
    print("\nThis example demonstrates the simplest RewardScope usage.")
    print("A random agent plays CartPole while we collect data.\n")

    # Create environment and wrap it with RewardScope
    env = gym.make("CartPole-v1")
    env = RewardScopeWrapper(
        env,
        run_name="cartpole_basic",
        verbose=1,
    )

    # Run episodes with random policy
    num_episodes = 5
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = env.action_space.sample()  # Random policy
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        print(f"Episode {episode + 1}: Reward = {episode_reward:.0f}")
    
    # Get summary statistics
    alerts = env.get_alerts()
    hacking_score = env.get_hacking_score()
    
    env.close()
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Episodes completed: {num_episodes}")
    print(f"Hacking score: {hacking_score:.2f}")
    print(f"Alerts detected: {len(alerts)}")
    
    if alerts:
        print("\nAlert breakdown:")
        alert_counts = {}
        for alert in alerts:
            alert_type = alert.type.value
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        for alert_type, count in alert_counts.items():
            print(f"  - {alert_type}: {count}")
    
    print(f"\nData saved to: ./reward_scope_data/cartpole_basic.db")
    print("\nTo view the dashboard:")
    print("  reward-scope dashboard --run-name cartpole_basic --data-dir ./reward_scope_data")


if __name__ == "__main__":
    main()
