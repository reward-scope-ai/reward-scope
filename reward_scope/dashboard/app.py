"""
FastAPI Dashboard Application

Real-time web dashboard for RL training visualization.
"""

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
from pathlib import Path
from typing import Optional
import json

from ..core.collector import DataCollector


app = FastAPI(title="RewardScope Dashboard", version="0.1.0")

# Global state (set by run_dashboard function)
collector: Optional[DataCollector] = None
run_name: str = "unknown"
data_dir: str = "./reward_scope_data"

# Templates directory
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Static files directory
static_dir = templates_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard page."""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "run_name": run_name,
        }
    )


@app.get("/api/reward-history")
async def get_reward_history(n: int = 100):
    """
    Get recent reward history.
    
    Returns:
        {
            "steps": [int],
            "rewards": [float],
            "episodes": [int]
        }
    """
    if not collector:
        return {"error": "No data collector initialized"}
    
    try:
        # Flush buffer to ensure latest data is available
        collector._flush_step_buffer()
        
        steps = collector.get_recent_steps(n)
        return {
            "steps": [s.step for s in steps],
            "rewards": [s.reward for s in steps],
            "episodes": [s.episode for s in steps],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/component-breakdown")
async def get_component_breakdown(n: int = 100):
    """
    Get reward component breakdown.

    Aggregates component totals from all completed episodes to show
    the true distribution across the entire training run.

    Returns:
        {
            "components": [str],
            "values": [float]
        }
    """
    if not collector:
        return {"error": "No data collector initialized"}

    try:
        # Flush buffer to ensure latest data is available
        collector._flush_step_buffer()

        # Get all episodes (or recent episodes if there are many)
        episodes = collector.get_episode_history(n=1000)  # Get up to 1000 episodes

        # Aggregate component totals from all episodes
        component_sums = {}
        for episode in episodes:
            for name, value in episode.component_totals.items():
                if name not in component_sums:
                    component_sums[name] = 0.0
                component_sums[name] += abs(value)  # Use absolute values for pie chart

        # If no episode data yet, fall back to recent steps
        if not component_sums:
            steps = collector.get_recent_steps(n)
            for step in steps:
                for name, value in step.reward_components.items():
                    if name not in component_sums:
                        component_sums[name] = 0.0
                    component_sums[name] += abs(value)

        return {
            "components": list(component_sums.keys()),
            "values": list(component_sums.values()),
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/episode-history")
async def get_episode_history(n: int = 50):
    """
    Get episode-level statistics.
    
    Returns:
        {
            "episodes": [int],
            "total_rewards": [float],
            "lengths": [int],
            "hacking_scores": [float]
        }
    """
    if not collector:
        return {"error": "No data collector initialized"}
    
    try:
        episodes = collector.get_episode_history(n)
        return {
            "episodes": [e.episode for e in episodes],
            "total_rewards": [e.total_reward for e in episodes],
            "lengths": [e.length for e in episodes],
            "hacking_scores": [e.hacking_score for e in episodes],
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/alerts")
async def get_alerts():
    """
    Get recent hacking alerts grouped by (episode, alert_type).

    Returns:
        {
            "alert_groups": [
                {
                    "episode": int,
                    "alert_type": str,
                    "count": int,
                    "max_severity": float,
                    "description": str
                }
            ]
        }
    """
    if not collector:
        return {"error": "No data collector initialized"}

    try:
        episodes = collector.get_episode_history(50)  # Get more episodes for better grouping

        # Group alerts by (episode, type)
        from collections import defaultdict
        alert_groups = defaultdict(lambda: {"count": 0, "severity": 0.0})

        for ep in episodes:
            # Count occurrences of each alert type in this episode
            from collections import Counter
            flag_counts = Counter(ep.hacking_flags)

            for flag, count in flag_counts.items():
                key = (ep.episode, flag)
                alert_groups[key] = {
                    "count": count,
                    "severity": ep.hacking_score,
                }

        # Convert to list format for response
        grouped_alerts = []
        for (episode, alert_type), data in alert_groups.items():
            grouped_alerts.append({
                "episode": episode,
                "alert_type": alert_type,
                "count": data["count"],
                "max_severity": data["severity"],
                "description": alert_type.replace("_", " ").title(),
            })

        # Sort by episode (descending) and then by severity (descending)
        grouped_alerts.sort(key=lambda x: (x["episode"], x["max_severity"]), reverse=True)

        return {"alert_groups": grouped_alerts}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/live-hacking")
async def get_live_hacking():
    """
    Get live hacking score for in-progress episode.

    Returns:
        {
            "episode": int,
            "current_score": float,
            "alert_count": int,
            "in_progress": bool,
            "timestamp": float  # Unix timestamp of last update
        }
    """
    if not collector:
        return {"error": "No data collector initialized"}

    try:
        live_state = collector.get_live_hacking_state()

        if live_state:
            # Check if state is recent (within last 10 seconds)
            import time
            is_recent = (time.time() - live_state.get("timestamp", 0)) < 10

            return {
                "episode": live_state.get("episode", 0),
                "current_score": live_state.get("hacking_score", 0.0),
                "alert_count": live_state.get("alert_count", 0),
                "in_progress": is_recent,
                "timestamp": live_state.get("timestamp", 0),
            }
        else:
            # No live state available
            return {
                "episode": 0,
                "current_score": 0.0,
                "alert_count": 0,
                "in_progress": False,
                "timestamp": 0,
            }
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/runs")
async def get_runs():
    """
    Get list of available training runs.

    Scans data_dir for .db files and returns metadata for each run.

    Returns:
        {
            "runs": [
                {
                    "name": str,
                    "episode_count": int,
                    "max_hacking_score": float,
                    "created_timestamp": float,
                    "created_date": str
                }
            ],
            "current_run": str
        }
    """
    import sqlite3
    from datetime import datetime

    runs = []
    data_path = Path(data_dir)

    if not data_path.exists():
        return {"runs": [], "current_run": run_name}

    # Scan for all .db files
    for db_file in data_path.glob("*.db"):
        try:
            # Connect to database
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()

            # Get episode count
            cursor.execute("SELECT COUNT(*) FROM episodes")
            episode_count = cursor.fetchone()[0]

            # Get max hacking score
            cursor.execute("SELECT MAX(hacking_score) FROM episodes")
            max_hacking_result = cursor.fetchone()[0]
            max_hacking_score = max_hacking_result if max_hacking_result is not None else 0.0

            # Get created timestamp (from first episode or file creation time)
            cursor.execute("SELECT MIN(start_time) FROM episodes")
            created_timestamp_result = cursor.fetchone()[0]
            if created_timestamp_result:
                created_timestamp = created_timestamp_result
            else:
                # Fallback to file creation time
                created_timestamp = db_file.stat().st_ctime

            conn.close()

            runs.append({
                "name": db_file.stem,
                "episode_count": episode_count,
                "max_hacking_score": max_hacking_score,
                "created_timestamp": created_timestamp,
                "created_date": datetime.fromtimestamp(created_timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            })
        except Exception as e:
            # Skip databases that can't be read
            print(f"Error reading {db_file}: {e}")
            continue

    # Sort by timestamp descending (most recent first)
    runs.sort(key=lambda x: x["created_timestamp"], reverse=True)

    return {"runs": runs, "current_run": run_name}


@app.post("/api/select-run")
async def select_run(request: Request):
    """
    Select a different run to view.

    Body:
        {
            "run_name": str
        }

    Returns:
        {
            "success": bool,
            "run_name": str
        }
    """
    global collector, run_name

    try:
        body = await request.json()
        new_run_name = body.get("run_name")

        if not new_run_name:
            return {"success": False, "error": "No run_name provided"}

        # Close existing collector if any
        if collector:
            collector.close()

        # Create new collector
        run_name = new_run_name
        collector = DataCollector(run_name, data_dir)

        return {"success": True, "run_name": run_name}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket for live updates (10Hz).
    
    Sends JSON messages with step updates:
    {
        "type": "step_update",
        "step": int,
        "reward": float,
        "components": {str: float},
        "episode": int
    }
    """
    await websocket.accept()
    
    try:
        last_step = 0
        while True:
            # Check for new data
            if collector:
                try:
                    # Flush buffer to get latest data
                    collector._flush_step_buffer()
                    
                    steps = collector.get_recent_steps(10)
                    if steps and steps[-1].step > last_step:
                        last_step = steps[-1].step
                        
                        # Send update
                        await websocket.send_json({
                            "type": "step_update",
                            "step": last_step,
                            "reward": steps[-1].reward,
                            "components": steps[-1].reward_components,
                            "episode": steps[-1].episode,
                        })
                except Exception as e:
                    # Don't crash on errors, just continue
                    print(f"WebSocket error: {e}")
            
            await asyncio.sleep(0.1)  # 10 Hz updates
    
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket connection error: {e}")


def run_dashboard(
    data_dir_param: str,
    run_name_param: Optional[str] = None,
    port: int = 8050,
    host: str = "0.0.0.0",
):
    """
    Start the dashboard server.

    Args:
        data_dir_param: Directory containing the SQLite database
        run_name_param: Name of the run to display (optional - can be selected in UI)
        port: Port to run the server on
        host: Host to bind to
    """
    global collector, run_name, data_dir

    data_dir = data_dir_param

    if run_name_param:
        run_name = run_name_param
        collector = DataCollector(run_name, data_dir)
        print(f"\n🔬 RewardScope Dashboard Starting...")
        print(f"   Run: {run_name}")
        print(f"   URL: http://localhost:{port}")
        print(f"   Data: {data_dir}\n")
    else:
        # Browser mode - no run selected yet
        run_name = ""
        collector = None
        print(f"\n🔬 RewardScope Dashboard Starting (Browser Mode)...")
        print(f"   URL: http://localhost:{port}")
        print(f"   Data: {data_dir}")
        print(f"   Select a run from the dropdown in the UI\n")

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="warning")

