"""
MuJoCo Ant with Auto-Extract Prefix

Demonstrates:
- Auto-extraction of reward components from info dict
- Complex multi-component reward (6+ components)
- Training with Stable-Baselines3
- Auto-starting the dashboard

The Ant-v4 environment provides reward components in the info dict with a "reward_" prefix.
RewardScope can automatically extract and track these.

Note: Requires mujoco to be installed: pip install mujoco
"""

import gymnasium as gym

try:
    from stable_baselines3 import PPO
    from reward_scope.integrations import RewardScopeCallback
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False
    print("Warning: stable-baselines3 not found. Install with: pip install stable-baselines3")


def main():
    if not HAS_SB3:
        print("This example requires stable-baselines3. Exiting.")
        return
    
    print("\n" + "="*60)
    print("🔬 RewardScope - MuJoCo Ant Training")
    print("="*60)
    print("\nThis example demonstrates:")
    print("- Auto-extraction of reward components (reward_* from info)")
    print("- Training with Stable-Baselines3 PPO")
    print("- Auto-starting the web dashboard")
    print("\nThe dashboard will open automatically at http://localhost:8050")
    print("Training will run for 50,000 steps (~5 minutes)\n")

    try:
        # Create environment
        env = gym.make("Ant-v4")
        print("✓ Ant-v4 environment created")
    except Exception as e:
        print(f"Error creating Ant environment: {e}")
        print("Install MuJoCo with: pip install mujoco")
        return

    # Create RewardScope callback with auto-extract
    callback = RewardScopeCallback(
        run_name="ant_training",
        auto_extract_prefix="reward_",  # Auto-extract reward_forward, reward_ctrl, etc.
        start_dashboard=True,  # Auto-start dashboard
        dashboard_port=8050,
        verbose=1,
    )

    # Create PPO model
    print("Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
    )

    print("\n" + "="*60)
    print("Starting Training")
    print("="*60)
    print("Dashboard: http://localhost:8050")
    print("Press Ctrl+C to stop early\n")

    try:
        # Train
        model.learn(
            total_timesteps=50000,
            callback=callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")

    # Get summary
    alerts = callback.get_alerts()
    hacking_score = callback.get_hacking_score()
    component_stats = callback.decomposer.get_component_stats()

    # Print summary
    print("\n" + "="*60)
    print("Training Summary")
    print("="*60)
    print(f"Total steps: {callback.step_count}")
    print(f"Total episodes: {callback.episode_count}")
    print(f"Hacking score: {hacking_score:.2f}")

    print("\nReward Components Detected:")
    for comp_name, stats in component_stats.items():
        if stats['count'] > 0:
            print(f"  {comp_name}:")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Std:  {stats['std']:.4f}")

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

    print(f"\nData saved to: ./reward_scope_data/ant_training.db")
    print("\nDashboard will remain open for 60 seconds...")
    print("Visit http://localhost:8050 to explore the data")
    
    import time
    time.sleep(60)


if __name__ == "__main__":
    main()

