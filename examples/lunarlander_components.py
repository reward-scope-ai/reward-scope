"""
LunarLander with Component Tracking

Demonstrates:
- Multi-component reward decomposition
- Custom component functions
- Hacking detection on a more complex environment

LunarLander-v2 has multiple reward components:
- Distance/velocity shaping
- Leg contact bonus
- Crash penalty
- Main engine / side engine usage

This example shows how to track each component separately.
"""

import gymnasium as gym
from reward_scope.integrations import RewardScopeWrapper


def main():
    print("\n" + "="*60)
    print("🔬 RewardScope - LunarLander Component Tracking")
    print("="*60)
    print("\nThis example demonstrates multi-component reward tracking.")
    print("We'll track different reward sources in LunarLander-v2.\n")

    # Create environment
    env = gym.make("LunarLander-v2")
    
    # Define component extraction functions
    # LunarLander provides some reward info in the info dict
    component_fns = {
        "shaping": lambda obs, act, info: info.get("shaping_reward", 0.0),
        "main_engine": lambda obs, act, info: -0.3 if act == 2 else 0.0,
        "side_engine": lambda obs, act, info: -0.03 if act in [1, 3] else 0.0,
    }
    
    # Wrap with RewardScope
    env = RewardScopeWrapper(
        env,
        run_name="lunarlander_components",
        component_fns=component_fns,
        verbose=1,
    )

    # Run episodes with random policy
    num_episodes = 10
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        
        if (episode + 1) % 5 == 0:
            print(f"Episode {episode + 1}: Reward = {episode_reward:.1f}")
    
    # Get statistics
    component_stats = env.get_component_stats()
    alerts = env.get_alerts()
    hacking_score = env.get_hacking_score()
    
    env.close()
    
    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Episodes completed: {num_episodes}")
    print(f"Hacking score: {hacking_score:.2f}")
    
    print("\nComponent Statistics:")
    for comp_name, stats in component_stats.items():
        if stats['count'] > 0:
            print(f"  {comp_name}:")
            print(f"    Mean: {stats['mean']:.3f}")
            print(f"    Total contribution: {stats['mean'] * stats['count']:.1f}")
    
    if alerts:
        print(f"\nAlerts detected: {len(alerts)}")
        alert_counts = {}
        for alert in alerts:
            alert_type = alert.type.value
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        print("Alert breakdown:")
        for alert_type, count in alert_counts.items():
            print(f"  - {alert_type}: {count}")
    else:
        print("\n✓ No hacking detected!")
    
    print(f"\nData saved to: ./reward_scope_data/lunarlander_components.db")
    print("\nTo view the dashboard:")
    print("  reward-scope dashboard --run-name lunarlander_components --data-dir ./reward_scope_data")


if __name__ == "__main__":
    main()

