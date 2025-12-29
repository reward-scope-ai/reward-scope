"""
Export utilities for RewardScope data.

Provides functions to export alerts and episode history to JSON or CSV formats.
"""

from typing import List, Dict, Any, Optional
import json
import csv
from pathlib import Path


def export_alerts_to_file(alerts: List[Any], path: str, format: Optional[str] = None) -> None:
    """
    Export alerts to JSON or CSV.

    Args:
        alerts: List of HackingAlert objects to export
        path: Output file path
        format: Export format ("json" or "csv"). If None, auto-detect from file extension.

    Raises:
        ValueError: If format is unsupported or cannot be auto-detected
    """
    # Auto-detect format from extension if not specified
    if format is None:
        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        if ext == '.json':
            format = 'json'
        elif ext == '.csv':
            format = 'csv'
        else:
            raise ValueError(
                f"Cannot auto-detect format from extension '{ext}'. "
                "Please specify format explicitly ('json' or 'csv')."
            )

    format = format.lower()

    if format == 'json':
        _export_alerts_json(alerts, path)
    elif format == 'csv':
        _export_alerts_csv(alerts, path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")


def _export_alerts_json(alerts: List[Any], path: str) -> None:
    """Export alerts to JSON format."""
    alert_dicts = []
    for alert in alerts:
        alert_dict = {
            "type": alert.type.value,
            "severity": alert.severity,
            "confidence": alert.confidence,
            "step": alert.step,
            "episode": alert.episode,
            "description": alert.description,
            "evidence": alert.evidence,
            "suggested_fix": alert.suggested_fix,
        }
        # Add optional fields if they exist
        if hasattr(alert, 'alert_severity'):
            alert_dict["alert_severity"] = alert.alert_severity.value
        if hasattr(alert, 'baseline_z_score') and alert.baseline_z_score is not None:
            alert_dict["baseline_z_score"] = alert.baseline_z_score

        alert_dicts.append(alert_dict)

    with open(path, 'w') as f:
        json.dump(alert_dicts, f, indent=2)


def _export_alerts_csv(alerts: List[Any], path: str) -> None:
    """Export alerts to CSV format (evidence as JSON string)."""
    if not alerts:
        # Create empty CSV with headers
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'type', 'severity', 'confidence', 'step', 'episode',
                'description', 'evidence', 'suggested_fix', 'alert_severity',
                'baseline_z_score'
            ])
            writer.writeheader()
        return

    with open(path, 'w', newline='') as f:
        # Determine fieldnames based on first alert
        fieldnames = ['type', 'severity', 'confidence', 'step', 'episode',
                     'description', 'evidence', 'suggested_fix']

        # Add optional fields if they exist
        if hasattr(alerts[0], 'alert_severity'):
            fieldnames.append('alert_severity')
        if hasattr(alerts[0], 'baseline_z_score'):
            fieldnames.append('baseline_z_score')

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for alert in alerts:
            row = {
                'type': alert.type.value,
                'severity': alert.severity,
                'confidence': alert.confidence,
                'step': alert.step,
                'episode': alert.episode,
                'description': alert.description,
                'evidence': json.dumps(alert.evidence),  # Serialize evidence as JSON string
                'suggested_fix': alert.suggested_fix,
            }

            # Add optional fields if they exist
            if hasattr(alert, 'alert_severity'):
                row['alert_severity'] = alert.alert_severity.value
            if hasattr(alert, 'baseline_z_score'):
                row['baseline_z_score'] = alert.baseline_z_score if alert.baseline_z_score is not None else ''

            writer.writerow(row)


def export_episodes_to_file(episodes: List[Any], path: str, format: Optional[str] = None) -> None:
    """
    Export episode history to JSON or CSV.

    Args:
        episodes: List of EpisodeData objects to export
        path: Output file path
        format: Export format ("json" or "csv"). If None, auto-detect from file extension.

    Raises:
        ValueError: If format is unsupported or cannot be auto-detected
    """
    # Auto-detect format from extension if not specified
    if format is None:
        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        if ext == '.json':
            format = 'json'
        elif ext == '.csv':
            format = 'csv'
        else:
            raise ValueError(
                f"Cannot auto-detect format from extension '{ext}'. "
                "Please specify format explicitly ('json' or 'csv')."
            )

    format = format.lower()

    if format == 'json':
        _export_episodes_json(episodes, path)
    elif format == 'csv':
        _export_episodes_csv(episodes, path)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'.")


def _export_episodes_json(episodes: List[Any], path: str) -> None:
    """Export episodes to JSON format."""
    episode_dicts = []
    for episode in episodes:
        episode_dict = {
            "episode": episode.episode,
            "total_reward": episode.total_reward,
            "length": episode.length,
            "hacking_score": episode.hacking_score,
            "component_totals": episode.component_totals,
            "alert_count": len(episode.hacking_flags) if hasattr(episode, 'hacking_flags') else 0,
        }
        episode_dicts.append(episode_dict)

    with open(path, 'w') as f:
        json.dump(episode_dicts, f, indent=2)


def _export_episodes_csv(episodes: List[Any], path: str) -> None:
    """Export episodes to CSV format (component_totals as JSON string)."""
    if not episodes:
        # Create empty CSV with headers
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'episode', 'total_reward', 'length', 'hacking_score',
                'component_totals', 'alert_count'
            ])
            writer.writeheader()
        return

    with open(path, 'w', newline='') as f:
        fieldnames = ['episode', 'total_reward', 'length', 'hacking_score',
                     'component_totals', 'alert_count']

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for episode in episodes:
            row = {
                'episode': episode.episode,
                'total_reward': episode.total_reward,
                'length': episode.length,
                'hacking_score': episode.hacking_score,
                'component_totals': json.dumps(episode.component_totals),  # Serialize as JSON string
                'alert_count': len(episode.hacking_flags) if hasattr(episode, 'hacking_flags') else 0,
            }
            writer.writerow(row)
