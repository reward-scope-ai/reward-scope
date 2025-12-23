"""
Test dashboard with CartPole training.

This script trains a PPO agent on CartPole and launches the dashboard automatically.
"""

import gymnasium as gym
from stable_baselines3 import PPO
from reward_scope.integrations import RewardScopeCallback


def main():
    print("\n" + "=" * 60)
    print("RewardScope Dashboard Test")
    print("=" * 60)
    print("\nTraining PPO on CartPole-v1 with live dashboard...")
    print("Dashboard will be available at: http://localhost:8050\n")

    # Create environment
    env = gym.make("CartPole-v1")

    # Create callback with dashboard enabled
    callback = RewardScopeCallback(
        run_name="cartpole_dashboard_test",
        start_dashboard=True,
        dashboard_port=8050,
        verbose=1,
    )

    # Create and train model
    model = PPO("MlpPolicy", env, verbose=1)
    
    print("\nStarting training (press Ctrl+C to stop)...\n")
    
    try:
        model.learn(total_timesteps=50000, callback=callback, progress_bar=True)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal stats:")
    print(f"  Total steps: {callback.step_count}")
    print(f"  Total episodes: {callback.episode_count}")
    print(f"  Hacking score: {callback.detector_suite.get_hacking_score():.3f}")
    
    alerts = callback.detector_suite.get_all_alerts()
    if alerts:
        print(f"  Alerts detected: {len(alerts)}")
        alert_counts = {}
        for alert in alerts:
            alert_type = alert.type.value
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        print("  Alert breakdown:")
        for alert_type, count in alert_counts.items():
            print(f"    - {alert_type}: {count}")
    else:
        print("  No alerts detected ✓")
    
    print("\nDashboard will remain available for 30 seconds...")
    print("Open http://localhost:8050 in your browser to view results.")
    
    import time
    time.sleep(30)
    
    print("\nShutting down...")


if __name__ == "__main__":
    main()

