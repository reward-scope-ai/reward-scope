"""
Simple dashboard test using existing CartPole data.
"""

import sys
from pathlib import Path

# Check if we have existing data
data_dir = Path("./reward_scope_data")
if not data_dir.exists():
    print("❌ No reward_scope_data directory found.")
    print("Run one of the examples first to generate data:")
    print("  python examples/cartpole_hacking_demo.py")
    sys.exit(1)

# Find any database file
db_files = list(data_dir.glob("*.db"))
if not db_files:
    print("❌ No database files found in reward_scope_data/")
    print("Run one of the examples first to generate data:")
    print("  python examples/cartpole_hacking_demo.py")
    sys.exit(1)

# Use the most recent database
db_file = sorted(db_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
run_name = db_file.stem

print("\n" + "="*60)
print("🔬 RewardScope Dashboard")
print("="*60)
print(f"\nStarting dashboard for run: {run_name}")
print(f"Dashboard URL: http://localhost:8050")
print("\nPress Ctrl+C to stop the server...\n")

# Start dashboard
from reward_scope.dashboard.app import run_dashboard

try:
    run_dashboard(
        data_dir=str(data_dir),
        run_name_param=run_name,
        port=8050,
        host="0.0.0.0"
    )
except KeyboardInterrupt:
    print("\n\nShutting down dashboard...")

