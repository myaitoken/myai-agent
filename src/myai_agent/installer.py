"""
Platform-aware service installer/uninstaller.

macOS   → launchd plist at ~/Library/LaunchAgents/io.myaitoken.agent.plist
Linux   → systemd user service at ~/.config/systemd/user/myai-agent.service
Windows → batch file + Task Scheduler via schtasks
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import textwrap
from typing import Optional

from .config import get_config_dir, get_log_dir

LABEL       = "io.myaitoken.agent"
SERVICE_NAME = "myai-agent"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _agent_exe() -> str:
    """Return the best path to the myai-agent executable."""
    exe = shutil.which("myai-agent")
    if exe:
        return exe
    # Fallback: run as module
    return f"{sys.executable} -m myai_agent"


def _run(cmd: list, check: bool = False) -> int:
    try:
        result = subprocess.run(cmd, check=check,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return result.returncode
    except FileNotFoundError:
        return 1
    except subprocess.CalledProcessError as e:
        return e.returncode


# ── macOS launchd ─────────────────────────────────────────────────────────────

def _launchd_path() -> str:
    return os.path.expanduser(f"~/Library/LaunchAgents/{LABEL}.plist")


def _build_plist(exe: str, env: dict, log_dir: str) -> str:
    env_entries = ""
    for k, v in env.items():
        if v:
            env_entries += f"        <key>{k}</key>\n        <string>{v}</string>\n"

    env_block = ""
    if env_entries:
        env_block = f"""    <key>EnvironmentVariables</key>
    <dict>
{env_entries}    </dict>
"""

    # Split exe into program + args for ProgramArguments
    parts = exe.split()
    prog_args = "\n".join(f"        <string>{p}</string>" for p in parts)
    prog_args += "\n        <string>start</string>"

    return textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
          "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
        <plist version="1.0">
        <dict>
            <key>Label</key>
            <string>{LABEL}</string>
            <key>ProgramArguments</key>
            <array>
        {prog_args}
            </array>
            <key>RunAtLoad</key>
            <true/>
            <key>KeepAlive</key>
            <true/>
            <key>StandardOutPath</key>
            <string>{log_dir}/agent.log</string>
            <key>StandardErrorPath</key>
            <string>{log_dir}/agent.error.log</string>
        {env_block}</dict>
        </plist>
    """)


def install_mac(coordinator_url: str, ollama_url: str, name: str,
                wallet: str, model: str) -> bool:
    log_dir = get_log_dir()
    os.makedirs(log_dir, exist_ok=True)

    exe = _agent_exe()
    env = {
        "COORDINATOR_URL":    coordinator_url,
        "OLLAMA_URL":         ollama_url,
        "AGENT_NAME":         name,
        "AGENT_WALLET":       wallet,
        "OLLAMA_DEFAULT_MODEL": model,
        "PATH": "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin",
    }

    plist_path = _launchd_path()
    os.makedirs(os.path.dirname(plist_path), exist_ok=True)

    # Unload existing if present
    if os.path.exists(plist_path):
        _run(["launchctl", "unload", plist_path])

    with open(plist_path, "w") as f:
        f.write(_build_plist(exe, env, log_dir))

    rc = _run(["launchctl", "load", plist_path])
    if rc != 0:
        print(f"  [!] launchctl load failed (rc={rc}). Try manually:")
        print(f"      launchctl load {plist_path}")
        return False

    print(f"  ✓ Installed as launchd service: {LABEL}")
    print(f"  ✓ Logs: {log_dir}/agent.log")
    return True


def uninstall_mac() -> bool:
    plist_path = _launchd_path()
    if os.path.exists(plist_path):
        _run(["launchctl", "unload", plist_path])
        os.remove(plist_path)
        print("  ✓ Removed launchd service")
        return True
    print("  Service not found")
    return False


def status_mac() -> dict:
    result = subprocess.run(
        ["launchctl", "list", LABEL],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    installed = os.path.exists(_launchd_path())
    running   = result.returncode == 0
    return {"installed": installed, "running": running,
            "details": result.stdout.decode().strip()}


# ── Linux systemd user ────────────────────────────────────────────────────────

def _systemd_path() -> str:
    config = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return os.path.join(config, "systemd", "user", f"{SERVICE_NAME}.service")


def _build_systemd(exe: str, env: dict) -> str:
    env_lines = "\n".join(
        f'Environment="{k}={v}"' for k, v in env.items() if v
    )
    return textwrap.dedent(f"""\
        [Unit]
        Description=MyAI GPU Agent
        After=network-online.target
        Wants=network-online.target

        [Service]
        ExecStart={exe} start
        Restart=always
        RestartSec=10
        {env_lines}

        [Install]
        WantedBy=default.target
    """)


def install_linux(coordinator_url: str, ollama_url: str, name: str,
                  wallet: str, model: str) -> bool:
    exe = _agent_exe()
    env = {
        "COORDINATOR_URL":    coordinator_url,
        "OLLAMA_URL":         ollama_url,
        "AGENT_NAME":         name,
        "AGENT_WALLET":       wallet,
        "OLLAMA_DEFAULT_MODEL": model,
    }

    service_path = _systemd_path()
    os.makedirs(os.path.dirname(service_path), exist_ok=True)

    with open(service_path, "w") as f:
        f.write(_build_systemd(exe, env))

    _run(["systemctl", "--user", "daemon-reload"])
    _run(["systemctl", "--user", "enable", "--now", SERVICE_NAME])

    print(f"  ✓ Installed systemd user service: {SERVICE_NAME}")
    print(f"  ✓ Logs: journalctl --user -u {SERVICE_NAME} -f")
    return True


def uninstall_linux() -> bool:
    service_path = _systemd_path()
    _run(["systemctl", "--user", "stop", SERVICE_NAME])
    _run(["systemctl", "--user", "disable", SERVICE_NAME])
    if os.path.exists(service_path):
        os.remove(service_path)
    _run(["systemctl", "--user", "daemon-reload"])
    print("  ✓ Removed systemd service")
    return True


def status_linux() -> dict:
    result = subprocess.run(
        ["systemctl", "--user", "is-active", SERVICE_NAME],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    active = result.stdout.decode().strip() == "active"
    return {"installed": os.path.exists(_systemd_path()), "running": active}


# ── Windows Task Scheduler ────────────────────────────────────────────────────

def _windows_bat_path() -> str:
    return os.path.join(get_config_dir(), "myai-agent.bat")


def _build_bat(exe: str, env: dict) -> str:
    set_lines = "\n".join(f"SET {k}={v}" for k, v in env.items() if v)
    return f"@ECHO OFF\n{set_lines}\n{exe} start\n"


def install_windows(coordinator_url: str, ollama_url: str, name: str,
                    wallet: str, model: str) -> bool:
    exe = _agent_exe()
    env = {
        "COORDINATOR_URL":    coordinator_url,
        "OLLAMA_URL":         ollama_url,
        "AGENT_NAME":         name,
        "AGENT_WALLET":       wallet,
        "OLLAMA_DEFAULT_MODEL": model,
    }

    config_dir = get_config_dir()
    os.makedirs(config_dir, exist_ok=True)
    bat_path = _windows_bat_path()

    with open(bat_path, "w") as f:
        f.write(_build_bat(exe, env))

    # Delete existing task if present
    _run(["schtasks", "/DELETE", "/TN", SERVICE_NAME, "/F"])

    rc = _run([
        "schtasks", "/CREATE",
        "/TN", SERVICE_NAME,
        "/TR", f'"{bat_path}"',
        "/SC", "ONLOGON",
        "/RL", "HIGHEST",
        "/F",
    ])

    if rc != 0:
        print(f"  [!] schtasks failed. Try running as Administrator.")
        print(f"  Bat file written to: {bat_path}")
        return False

    print(f"  ✓ Task Scheduler task created: {SERVICE_NAME}")
    print(f"  ✓ Agent will start on next login.")
    return True


def uninstall_windows() -> bool:
    _run(["schtasks", "/DELETE", "/TN", SERVICE_NAME, "/F"])
    bat_path = _windows_bat_path()
    if os.path.exists(bat_path):
        os.remove(bat_path)
    print("  ✓ Removed Task Scheduler task")
    return True


def status_windows() -> dict:
    result = subprocess.run(
        ["schtasks", "/QUERY", "/TN", SERVICE_NAME],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    installed = result.returncode == 0
    return {"installed": installed, "running": None}  # can't easily tell if running


# ── Dispatch ──────────────────────────────────────────────────────────────────

def install(coordinator_url: str, ollama_url: str, name: str,
            wallet: str = "", model: str = "llama3.2") -> bool:
    system = platform.system()
    if system == "Darwin":
        return install_mac(coordinator_url, ollama_url, name, wallet, model)
    elif system == "Linux":
        return install_linux(coordinator_url, ollama_url, name, wallet, model)
    elif system == "Windows":
        return install_windows(coordinator_url, ollama_url, name, wallet, model)
    else:
        print(f"  [!] Unsupported platform: {system}")
        return False


def uninstall() -> bool:
    system = platform.system()
    if system == "Darwin":
        return uninstall_mac()
    elif system == "Linux":
        return uninstall_linux()
    elif system == "Windows":
        return uninstall_windows()
    return False


def status() -> dict:
    system = platform.system()
    if system == "Darwin":
        return status_mac()
    elif system == "Linux":
        return status_linux()
    elif system == "Windows":
        return status_windows()
    return {}
