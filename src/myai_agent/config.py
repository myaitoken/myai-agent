"""
Platform-appropriate config and log directory paths. Zero deps.
"""

from __future__ import annotations

import os
import platform


def get_config_dir() -> str:
    """Return the platform-appropriate config directory for myai-agent."""
    system = platform.system()
    if system == "Darwin":
        return os.path.expanduser("~/Library/Application Support/myai-agent")
    elif system == "Windows":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        return os.path.join(base, "myai-agent")
    else:  # Linux + everything else
        base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
        return os.path.join(base, "myai-agent")


def get_log_dir() -> str:
    """Return the platform-appropriate log directory."""
    system = platform.system()
    if system == "Darwin":
        return os.path.expanduser("~/Library/Logs/myai-agent")
    elif system == "Windows":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        return os.path.join(base, "myai-agent", "logs")
    else:
        base = os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state"))
        return os.path.join(base, "myai-agent")


def get_env_file() -> str:
    """Path to the agent's environment config file."""
    return os.path.join(get_config_dir(), "config.env")
