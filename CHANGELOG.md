# Changelog

All notable changes to RewardScope will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-21

### Added
- Core data collection with SQLite backend
- Reward decomposition engine with auto-extraction
- 5 hacking detectors (action repetition, state cycling, component imbalance, reward spiking, boundary exploitation)
- Gymnasium environment wrapper integration
- Stable-Baselines3 callback integration
- Real-time web dashboard with FastAPI + HTMX + Chart.js
- WebSocket live updates (10Hz)
- CLI tool with dashboard, list-runs, and report commands
- Comprehensive documentation (quickstart, components, detectors, API)
- Example scripts (CartPole, LunarLander, MuJoCo Ant)
- Type hints (PEP 561 compliant)
- Full test suite (98+ tests)

### Documentation
- Quick start guide
- Reward components guide
- Hacking detection guide
- API reference
- README with badges and examples

### Examples
- `cartpole_basic.py` - Basic usage verification
- `lunarlander_components.py` - Multi-component tracking
- `mujoco_ant.py` - SB3 training with auto-start dashboard

## [Unreleased]

### Planned
- Isaac Lab integration
- Additional detectors (proxy-true divergence)
- Export to W&B / TensorBoard
- Multi-run comparison dashboard
- Reward function suggestion system

