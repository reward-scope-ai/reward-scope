"""
Command-line interface for RewardScope.
"""

import click
from pathlib import Path
import sys


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """RewardScope - RL Reward Debugging Tools"""
    pass


@cli.command()
@click.option('--port', default=8050, help='Dashboard port')
@click.option('--data-dir', default='./reward_scope_data', help='Data directory')
@click.option('--run-name', default=None, help='Run name to display (optional - can be selected in UI)')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
def dashboard(port: int, data_dir: str, run_name: str, host: str):
    """Start the web dashboard."""
    from reward_scope.dashboard.app import run_dashboard

    # Check if data directory exists
    data_path = Path(data_dir)
    if not data_path.exists():
        click.echo(f"❌ Error: Data directory not found: {data_dir}", err=True)
        sys.exit(1)

    # If run_name is provided, check if database exists
    if run_name:
        db_path = data_path / f"{run_name}.db"
        if not db_path.exists():
            click.echo(f"❌ Error: Database not found: {db_path}", err=True)
            click.echo(f"\nAvailable runs in {data_dir}:")
            for db_file in data_path.glob("*.db"):
                click.echo(f"  - {db_file.stem}")
            sys.exit(1)

    run_dashboard(data_dir, run_name, port, host)


@cli.command()
@click.argument('data_dir')
def list_runs(data_dir: str):
    """List available training runs."""
    data_path = Path(data_dir)
    if not data_path.exists():
        click.echo(f"❌ Error: Data directory not found: {data_dir}", err=True)
        sys.exit(1)
    
    db_files = list(data_path.glob("*.db"))
    if not db_files:
        click.echo(f"No training runs found in {data_dir}")
        return
    
    click.echo(f"\n📊 Training runs in {data_dir}:\n")
    for db_file in sorted(db_files, key=lambda x: x.stat().st_mtime, reverse=True):
        # Get file size and modification time
        size_mb = db_file.stat().st_size / (1024 * 1024)
        mtime = db_file.stat().st_mtime
        
        import datetime
        modified = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        click.echo(f"  • {db_file.stem}")
        click.echo(f"    Size: {size_mb:.2f} MB | Modified: {modified}")
    
    click.echo(f"\nTo view a run, use:")
    click.echo(f"  reward-scope dashboard --run-name <name> --data-dir {data_dir}")


@cli.command()
@click.argument('data_dir')
@click.option('--run-name', default=None, help='Specific run to analyze')
@click.option('--output', default='report.html', help='Output file')
def report(data_dir: str, run_name: str, output: str):
    """Generate a static HTML report."""
    from reward_scope.core.collector import DataCollector
    
    data_path = Path(data_dir)
    if not data_path.exists():
        click.echo(f"❌ Error: Data directory not found: {data_dir}", err=True)
        sys.exit(1)
    
    # If no run name specified, use the most recent one
    if not run_name:
        db_files = sorted(data_path.glob("*.db"), key=lambda x: x.stat().st_mtime, reverse=True)
        if not db_files:
            click.echo(f"❌ Error: No database files found in {data_dir}", err=True)
            sys.exit(1)
        run_name = db_files[0].stem
        click.echo(f"Using most recent run: {run_name}")
    
    # Load data
    collector = DataCollector(run_name, data_dir)
    episodes = collector.get_episode_history(n=10000)
    
    if not episodes:
        click.echo(f"❌ Error: No episode data found for run: {run_name}", err=True)
        sys.exit(1)
    
    # Generate simple HTML report
    total_steps = sum(ep.length for ep in episodes)
    avg_reward = sum(ep.total_reward for ep in episodes) / len(episodes)
    avg_length = sum(ep.length for ep in episodes) / len(episodes)
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RewardScope Report - {run_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1000px; margin: 0 auto; background: white; padding: 40px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; }}
            .stat {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #eee; }}
            .stat-label {{ font-weight: bold; }}
            .stat-value {{ color: #0066cc; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #eee; }}
            th {{ background: #f8f8f8; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔬 RewardScope Report</h1>
            <h2>{run_name}</h2>
            
            <h3>Summary</h3>
            <div class="stat">
                <span class="stat-label">Total Episodes:</span>
                <span class="stat-value">{len(episodes)}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Total Steps:</span>
                <span class="stat-value">{total_steps}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Average Reward:</span>
                <span class="stat-value">{avg_reward:.2f}</span>
            </div>
            <div class="stat">
                <span class="stat-label">Average Length:</span>
                <span class="stat-value">{avg_length:.2f}</span>
            </div>
            
            <h3>Recent Episodes</h3>
            <table>
                <tr>
                    <th>Episode</th>
                    <th>Total Reward</th>
                    <th>Length</th>
                    <th>Hacking Score</th>
                </tr>
    """
    
    # Add last 20 episodes
    for ep in episodes[-20:]:
        html += f"""
                <tr>
                    <td>{ep.episode}</td>
                    <td>{ep.total_reward:.2f}</td>
                    <td>{ep.length}</td>
                    <td>{ep.hacking_score:.2f}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
    </body>
    </html>
    """
    
    # Write report
    Path(output).write_text(html)
    click.echo(f"\n✓ Report saved to: {output}")
    click.echo(f"  Episodes analyzed: {len(episodes)}")
    click.echo(f"  Average reward: {avg_reward:.2f}")
    
    collector.close()


if __name__ == '__main__':
    cli()

