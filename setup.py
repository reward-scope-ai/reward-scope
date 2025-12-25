"""
Setup script for RewardScope.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "reward-scope-readme.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="reward-scope",
    version="0.1.0",
    description="Reward debugging and hacking detection for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RewardScope Contributors",
    python_requires=">=3.8",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    package_data={
        "reward_scope": [
            "dashboard/templates/*.html",
            "dashboard/templates/static/*.css",
            "dashboard/templates/static/*.js",
        ],
    },
    install_requires=[
        "numpy>=1.21.0",
        "gymnasium>=0.28.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "jinja2>=3.1.0",
        "websockets>=11.0",
        "click>=8.0.0",
    ],
    extras_require={
        "sb3": [
            "stable-baselines3>=2.0.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
        ],
        "all": [
            "stable-baselines3>=2.0.0",
            "wandb>=0.15.0",
            "pytest>=7.0.0",
            "black>=23.0.0",
            "ruff>=0.0.280",
        ],
    },
    entry_points={
        "console_scripts": [
            "reward-scope=reward_scope.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="reinforcement-learning debugging reward-shaping robotics",
)
