"""
CartPole with WandB Integration Example

Demonstrates:
- Using RewardScopeWrapper with WandB logging
- Logging metrics to Weights & Biases
- Tracking hacking alerts in WandB

This example shows how to integrate RewardScope metrics into your existing
WandB workflow. High severity alerts are automatically logged as WandB warnings.

Requirements:
    pip install reward-scope[wandb]
"""

import gymnasium as gym
from reward_scope.integrations import RewardScopeWrapper

# Optional: Only import wandb if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WandB not installed. Install with: pip install reward-scope[wandb]")
    exit(1)


def main():
    print("\n" + "="*60)
    print("🔬 RewardScope + WandB Integration Example")
    print("="*60)
    print("\nThis example demonstrates RewardScope's WandB integration.")
    print("Metrics will be logged to your WandB project.\n")

    # Initialize WandB
    # Set mode='disabled' to run without uploading to WandB servers
    wandb.init(
        project="rewardscope-demo",
        name="cartpole_wandb_example",
        config={
            "env": "CartPole-v1",
            "episodes": 10,
            "policy": "random",
        },
        # mode="disabled",  # Uncomment to test without uploading
    )

    # Create environment and wrap it with RewardScope
    env = gym.make("CartPole-v1")
    env = RewardScopeWrapper(
        env,
        run_name="cartpole_wandb",
        wandb_logging=True,  # Enable WandB logging!
        verbose=1,
    )

    print(f"\n📊 WandB run: {wandb.run.url}\n")

    # Run episodes with random policy
    num_episodes = 10

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            action = env.action_space.sample()  # Random policy
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Reward = {episode_reward:.0f}, "
              f"Steps = {steps}, "
              f"Hacking Score = {info['hacking_score']:.3f}")

    # Get summary statistics
    alerts = env.get_alerts()
    hacking_score = env.get_hacking_score()

    env.close()

    # Log final summary to WandB
    wandb.summary["final_hacking_score"] = hacking_score
    wandb.summary["total_alerts"] = len(alerts)

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print(f"Episodes completed: {num_episodes}")
    print(f"Final hacking score: {hacking_score:.3f}")
    print(f"Total alerts detected: {len(alerts)}")

    if alerts:
        print("\nAlert breakdown:")
        alert_counts = {}
        for alert in alerts:
            alert_type = alert.type.value
            alert_counts[alert_type] = alert_counts.get(alert_type, 0) + 1
        for alert_type, count in alert_counts.items():
            print(f"  - {alert_type}: {count}")

    print(f"\n📊 View your WandB run at: {wandb.run.url}")
    print(f"📁 Local data saved to: ./reward_scope_data/cartpole_wandb.db")

    # Finish WandB run
    wandb.finish()


if __name__ == "__main__":
    main()
