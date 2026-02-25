"""
myai-agent CLI

Usage:
  myai-agent start                        Run the agent (foreground)
  myai-agent install [options]            Install as OS service (auto-start on login)
  myai-agent uninstall                    Remove OS service
  myai-agent status                       Show agent/service status
  myai-agent logs [-n N]                  Tail agent logs
  myai-agent models                       List models available in Ollama
  myai-agent run-job <prompt> [--model M] Submit a test job locally
"""

from __future__ import annotations

import argparse
import logging
import os
import platform
import socket
import subprocess
import sys

from . import __version__


def _setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


# ── Subcommands ───────────────────────────────────────────────────────────────

def cmd_start(args):
    """Run the agent in the foreground."""
    _setup_logging(args.verbose)
    from .agent import MyAIAgent

    agent = MyAIAgent(
        coordinator_url    = args.coordinator or os.environ.get("COORDINATOR_URL", "https://api.myaitoken.io"),
        ollama_url         = args.ollama      or os.environ.get("OLLAMA_URL",      "http://localhost:11434"),
        name               = args.name        or os.environ.get("AGENT_NAME",      socket.gethostname()),
        wallet             = args.wallet      or os.environ.get("AGENT_WALLET",    ""),
        poll_interval      = int(os.environ.get("POLL_INTERVAL",      "5")),
        heartbeat_interval = int(os.environ.get("HEARTBEAT_INTERVAL", "30")),
    )
    agent.start()


def cmd_install(args):
    """Install the agent as an OS service."""
    _setup_logging()
    from . import installer

    coordinator = args.coordinator or "https://api.myaitoken.io"
    ollama      = args.ollama      or "http://localhost:11434"
    name        = args.name        or socket.gethostname()
    wallet      = args.wallet      or ""
    model       = args.model       or "llama3.2"

    print(f"\nMyAI GPU Agent v{__version__} — Install")
    print(f"  Platform    : {platform.system()}")
    print(f"  Name        : {name}")
    print(f"  Coordinator : {coordinator}")
    print(f"  Ollama      : {ollama}")
    print(f"  Model       : {model}")
    if wallet:
        print(f"  Wallet      : {wallet}")
    print()

    ok = installer.install(coordinator, ollama, name, wallet, model)
    if ok:
        print("\n✅ Done! The agent will start automatically on login.")
        print(f"   Run 'myai-agent status' to check the service.")
    else:
        print("\n❌ Installation failed. See messages above.")
        sys.exit(1)


def cmd_uninstall(args):
    """Remove the OS service."""
    _setup_logging()
    from . import installer

    print(f"\nMyAI GPU Agent — Uninstall")
    ok = installer.uninstall()
    sys.exit(0 if ok else 1)


def cmd_status(args):
    """Show agent and service status."""
    from . import installer
    from .config import get_config_dir
    from .agent import load_agent_id

    svc = installer.status()

    print(f"\nMyAI GPU Agent v{__version__}")
    print(f"  Platform  : {platform.system()}")

    try:
        agent_id = load_agent_id()
        print(f"  Agent ID  : {agent_id}")
    except Exception:
        print(f"  Agent ID  : (not configured yet)")

    print(f"  Config    : {get_config_dir()}")

    if svc:
        installed = "✓" if svc.get("installed") else "✗"
        running   = "✓" if svc.get("running") else "✗"
        print(f"  Installed : {installed}")
        print(f"  Running   : {running}")
        if svc.get("details"):
            print(f"\n  {svc['details']}")
    print()


def cmd_logs(args):
    """Tail agent logs."""
    from .config import get_log_dir

    n       = args.n or 50
    log_dir = get_log_dir()
    log_file = os.path.join(log_dir, "agent.log")

    system = platform.system()

    if system == "Darwin":
        if not os.path.exists(log_file):
            # Try launchctl stderr
            alt = os.path.join(log_dir, "agent.error.log")
            print(f"Log: {log_file} (or {alt})")
        if os.path.exists(log_file):
            os.execvp("tail", ["tail", f"-{n}", "-f", log_file])
        else:
            print(f"No log file found at {log_file}")
            print("  Is the agent installed? Run: myai-agent install")

    elif system == "Linux":
        # Try journalctl first
        result = subprocess.run(
            ["journalctl", "--user", "-u", "myai-agent", f"-n{n}", "--no-pager"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        if result.returncode == 0 and result.stdout.strip():
            print(result.stdout.decode())
        elif os.path.exists(log_file):
            os.execvp("tail", ["tail", f"-{n}", "-f", log_file])
        else:
            print("No logs found. Is the agent installed and running?")

    elif system == "Windows":
        if os.path.exists(log_file):
            # Windows: use Get-Content -Tail
            subprocess.run(
                ["powershell", "-Command", f"Get-Content -Path '{log_file}' -Tail {n} -Wait"]
            )
        else:
            print(f"No log file found at {log_file}")
    else:
        print(f"No log viewer for platform: {system}")


def cmd_models(args):
    """List available Ollama models."""
    from .agent import get_ollama_models, http

    ollama = args.ollama or os.environ.get("OLLAMA_URL", "http://localhost:11434")

    resp = http("GET", f"{ollama}/api/tags", timeout=5)
    if not resp:
        print(f"  ✗ Can't reach Ollama at {ollama}")
        print("    Is Ollama running? Try: ollama serve")
        sys.exit(1)

    models = resp.get("models", [])
    if not models:
        print(f"  No models found in Ollama at {ollama}")
        print("  Pull one: ollama pull llama3.2")
        return

    print(f"\nModels in Ollama at {ollama}:\n")
    for m in models:
        size_gb = m.get("size", 0) / 1e9
        print(f"  {m['name']:<30}  {size_gb:.1f} GB")
    print()


def cmd_run_job(args):
    """Run a single inference job locally and print the result."""
    _setup_logging(args.verbose)
    from .agent import run_ollama

    ollama = args.ollama or os.environ.get("OLLAMA_URL", "http://localhost:11434")
    model  = args.model  or "llama3.2"
    prompt = " ".join(args.prompt)

    print(f"\nRunning: model={model}, ollama={ollama}")
    print(f"Prompt : {prompt[:120]}\n")

    result = run_ollama(model, prompt, ollama_url=ollama, timeout=120)
    if result:
        print(result)
    else:
        print("  ✗ No result returned. Is Ollama running with the right model?")
        sys.exit(1)


# ── Parser ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="myai-agent",
        description="MyAI GPU compute agent — earn MYAI tokens by sharing compute",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"myai-agent {__version__}")

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # ── start ──────────────────────────────────────────────────────────────────
    p_start = sub.add_parser("start", help="Run the agent (foreground)")
    p_start.add_argument("--coordinator", metavar="URL", help="Coordinator URL")
    p_start.add_argument("--ollama",      metavar="URL", help="Ollama URL (default: http://localhost:11434)")
    p_start.add_argument("--name",        metavar="NAME", help="Agent display name")
    p_start.add_argument("--wallet",      metavar="0x...", help="Base EVM wallet address for earnings")
    p_start.add_argument("-v", "--verbose", action="store_true")
    p_start.set_defaults(func=cmd_start)

    # ── install ────────────────────────────────────────────────────────────────
    p_install = sub.add_parser("install", help="Install as OS service (auto-start on login)")
    p_install.add_argument("--coordinator", metavar="URL",   default="https://api.myaitoken.io")
    p_install.add_argument("--ollama",      metavar="URL",   default="http://localhost:11434")
    p_install.add_argument("--name",        metavar="NAME",  help="Agent display name")
    p_install.add_argument("--wallet",      metavar="0x...", help="Your Base wallet to receive MYAI rewards")
    p_install.add_argument("--model",       metavar="MODEL", default="llama3.2",
                           help="Default Ollama model (default: llama3.2)")
    p_install.set_defaults(func=cmd_install)

    # ── uninstall ──────────────────────────────────────────────────────────────
    p_un = sub.add_parser("uninstall", help="Remove OS service")
    p_un.set_defaults(func=cmd_uninstall)

    # ── status ─────────────────────────────────────────────────────────────────
    p_st = sub.add_parser("status", help="Show agent and service status")
    p_st.set_defaults(func=cmd_status)

    # ── logs ───────────────────────────────────────────────────────────────────
    p_logs = sub.add_parser("logs", help="Tail agent logs")
    p_logs.add_argument("-n", type=int, default=50, help="Number of lines (default: 50)")
    p_logs.add_argument("--ollama", metavar="URL", help="Ollama URL")
    p_logs.set_defaults(func=cmd_logs)

    # ── models ─────────────────────────────────────────────────────────────────
    p_models = sub.add_parser("models", help="List available Ollama models")
    p_models.add_argument("--ollama", metavar="URL", help="Ollama URL")
    p_models.set_defaults(func=cmd_models)

    # ── run-job ────────────────────────────────────────────────────────────────
    p_run = sub.add_parser("run-job", help="Run a single inference job locally")
    p_run.add_argument("prompt", nargs="+", help="Prompt text")
    p_run.add_argument("--model",  metavar="MODEL", default="llama3.2")
    p_run.add_argument("--ollama", metavar="URL")
    p_run.add_argument("-v", "--verbose", action="store_true")
    p_run.set_defaults(func=cmd_run_job)

    return parser


def main():
    parser = build_parser()
    args   = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
