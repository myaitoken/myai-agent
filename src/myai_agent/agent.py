"""
MyAI GPU Agent — core logic.

Registers with the coordinator, polls for jobs, runs inference via Ollama,
reports results back. Zero external dependencies — pure stdlib.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import List, Dict, Any, Optional

from . import gpu as gpu_mod
from .config import get_config_dir

log = logging.getLogger("myai_agent")

VERSION = "2.1.0"


# ── Config ─────────────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


COORDINATOR_URL    = _env("COORDINATOR_URL",    "https://api.myaitoken.io")
OLLAMA_URL         = _env("OLLAMA_URL",         "http://localhost:11434")
AGENT_NAME         = _env("AGENT_NAME",         socket.gethostname())
AGENT_WALLET       = _env("AGENT_WALLET",       "")
POLL_INTERVAL      = int(_env("POLL_INTERVAL",  "5"))
HEARTBEAT_INTERVAL = int(_env("HEARTBEAT_INTERVAL", "30"))

# Comma-separated list of models to auto-pull on startup (if not already present).
# e.g. REQUIRED_MODELS=bonsai-8b:latest,deepseek-r1:7b
# Default includes bonsai-8b as the lightweight required model.
REQUIRED_MODELS_RAW = _env("REQUIRED_MODELS", "bonsai-8b:latest")
REQUIRED_MODELS: List[str] = [
    m.strip() for m in REQUIRED_MODELS_RAW.split(",") if m.strip()
]


# ── HTTP ───────────────────────────────────────────────────────────────────────

def http(method: str, url: str, body: dict = None, timeout: int = 30) -> dict:
    """Minimal HTTP client — no external deps."""
    data = json.dumps(body).encode() if body else None
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": f"myai-agent/{VERSION}",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        log.warning(f"HTTP {e.code} {method} {url}: {e.read().decode()[:200]}")
        return {}
    except Exception as e:
        log.debug(f"Request failed {method} {url}: {e}")
        return {}


# ── Agent ID ───────────────────────────────────────────────────────────────────

def _agent_id_path() -> str:
    config_dir = get_config_dir()
    os.makedirs(config_dir, exist_ok=True)
    return os.path.join(config_dir, "agent_id")


def load_agent_id() -> str:
    path = _agent_id_path()
    if os.path.exists(path):
        return open(path).read().strip()
    agent_id = str(uuid.uuid4())
    open(path, "w").write(agent_id)
    return agent_id


# ── Ollama ─────────────────────────────────────────────────────────────────────

def get_ollama_models(ollama_url: str = OLLAMA_URL) -> List[str]:
    try:
        resp = http("GET", f"{ollama_url}/api/tags", timeout=5)
        return [m["name"] for m in resp.get("models", [])]
    except Exception:
        return []


def pull_model(model: str, ollama_url: str = OLLAMA_URL) -> bool:
    """
    Pull a model via Ollama streaming pull API.
    Streams progress lines and returns True when 'success' status received.
    Falls back gracefully if Ollama is unreachable.
    """
    log.info(f"Pulling model: {model} ...")
    url = f"{ollama_url}/api/pull"
    data = json.dumps({"name": model, "stream": True}).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            last_status = ""
            while True:
                line = resp.readline()
                if not line:
                    break
                try:
                    obj = json.loads(line.decode().strip())
                    status = obj.get("status", "")
                    if status != last_status:
                        log.info(f"  [{model}] {status}")
                        last_status = status
                    if status == "success":
                        log.info(f"  ✓ {model} pulled successfully")
                        return True
                    # Error in stream
                    if "error" in obj:
                        log.error(f"  ✗ Pull error for {model}: {obj['error']}")
                        return False
                except json.JSONDecodeError:
                    continue
        # If we got here without 'success', treat as complete (older Ollama versions)
        log.info(f"  ✓ {model} pull stream ended")
        return True
    except urllib.error.HTTPError as e:
        log.error(f"  ✗ Pull HTTP {e.code} for {model}: {e.read().decode()[:200]}")
        return False
    except Exception as e:
        log.error(f"  ✗ Pull failed for {model}: {e}")
        return False


def ensure_models(required: List[str], ollama_url: str = OLLAMA_URL) -> None:
    """
    Check which required models are missing from Ollama and pull them.
    Skips models that are already present. Non-blocking on failure.
    """
    if not required:
        return

    existing = set(get_ollama_models(ollama_url))
    if not existing and not _ollama_reachable(ollama_url):
        log.warning("Ollama not reachable — skipping model pre-pull")
        return

    # Normalize: treat "model:latest" == "model" as equal
    def norm(m: str) -> str:
        return m if ":" in m else f"{m}:latest"

    existing_norm = {norm(m) for m in existing}

    for model in required:
        if norm(model) in existing_norm:
            log.info(f"  ✓ {model} already present — skip")
        else:
            log.info(f"  ↓ {model} not found — pulling now")
            pull_model(model, ollama_url)


def _ollama_reachable(ollama_url: str = OLLAMA_URL) -> bool:
    try:
        urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5)
        return True
    except Exception:
        return False


def run_ollama(model: str, prompt: str, ollama_url: str = OLLAMA_URL, timeout: int = 120) -> str:
    """Run inference. Detects JSON messages array → /api/chat, else → /api/generate."""
    if prompt.strip().startswith("["):
        try:
            messages = json.loads(prompt)
            resp = http("POST", f"{ollama_url}/api/chat",
                        {"model": model, "messages": messages, "stream": False}, timeout=timeout)
            return resp.get("message", {}).get("content", "").strip()
        except Exception:
            pass
    resp = http("POST", f"{ollama_url}/api/generate",
                {"model": model, "prompt": prompt, "stream": False}, timeout=timeout)
    return resp.get("response", "").strip()


# ── Agent class ────────────────────────────────────────────────────────────────

class MyAIAgent:
    def __init__(
        self,
        coordinator_url: str = COORDINATOR_URL,
        ollama_url: str = OLLAMA_URL,
        name: str = AGENT_NAME,
        wallet: str = AGENT_WALLET,
        poll_interval: int = POLL_INTERVAL,
        heartbeat_interval: int = HEARTBEAT_INTERVAL,
        required_models: List[str] = None,
    ):
        self.coordinator_url    = coordinator_url.rstrip("/")
        self.ollama_url         = ollama_url.rstrip("/")
        self.name               = name
        self.wallet             = wallet
        self.poll_interval      = poll_interval
        self.heartbeat_interval = heartbeat_interval
        self.agent_id           = load_agent_id()
        self.required_models    = required_models if required_models is not None else REQUIRED_MODELS
        self._running           = False

    # ── Model Pre-pull ─────────────────────────────────────────────────────────

    def ensure_models(self) -> None:
        """Pull any required models not already present in Ollama."""
        if self.required_models:
            log.info(f"Checking required models: {', '.join(self.required_models)}")
            ensure_models(self.required_models, self.ollama_url)

    # ── Registration ───────────────────────────────────────────────────────────

    def register(self) -> bool:
        gpus   = gpu_mod.detect()
        models = get_ollama_models(self.ollama_url)

        payload = {
            "agent_id":           self.agent_id,
            "agent_name":         self.name,
            "version":            VERSION,
            "platform":           platform.system(),
            "ollama_url":         self.ollama_url,
            "gpus":               gpus,
            "models":             models,
            "wallet_address":     self.wallet,
            "price_per_hour_myai": 1.0,
        }

        resp = http("POST", f"{self.coordinator_url}/api/v1/agents/register", payload)
        if resp.get("success"):
            log.info(f"Registered as '{self.name}' (id={self.agent_id})")
            if gpus:
                log.info(f"  GPUs   : {', '.join(g['name'] for g in gpus)}")
            if models:
                log.info(f"  Models : {', '.join(models)}")
            return True
        log.error(f"Registration failed: {resp}")
        return False

    # ── Heartbeat ──────────────────────────────────────────────────────────────

    def _heartbeat_loop(self):
        while self._running:
            time.sleep(self.heartbeat_interval)
            if not self._running:
                break
            resp = http("POST",
                        f"{self.coordinator_url}/api/v1/agents/{self.agent_id}/heartbeat",
                        {"status": "online"})
            if resp.get("success"):
                log.debug("Heartbeat ok")
            else:
                log.warning("Heartbeat failed — re-registering")
                self.register()

    # ── Job handling ───────────────────────────────────────────────────────────

    def _complete_job(self, job_id: str, result: str, success: bool = True):
        http("POST",
             f"{self.coordinator_url}/api/v1/agents/{self.agent_id}/jobs/{job_id}/complete",
             {"success": success, "result": result})

    def _process_job(self, job: dict):
        job_id = job.get("job_id", "unknown")
        model  = job.get("model", "llama3.2")
        prompt = job.get("prompt", "")

        if not prompt:
            log.warning(f"Job {job_id} has empty prompt — skipping")
            self._complete_job(job_id, "", success=False)
            return

        log.info(f"Job {job_id} | model={model} | {prompt[:60]}...")
        result = run_ollama(model, prompt, self.ollama_url)

        if result:
            log.info(f"Job {job_id} done ({len(result)} chars)")
            self._complete_job(job_id, result, success=True)
        else:
            log.warning(f"Job {job_id} returned empty result")
            self._complete_job(job_id, "No response from model", success=False)

    # ── Main loop ──────────────────────────────────────────────────────────────

    def _poll_loop(self):
        log.info(f"Polling {self.coordinator_url} every {self.poll_interval}s...")
        while self._running:
            try:
                resp = http("GET",
                            f"{self.coordinator_url}/api/v1/agents/{self.agent_id}/jobs/pending",
                            timeout=10)
                for job in resp.get("data", {}).get("jobs", []):
                    self._process_job(job)
            except Exception as e:
                log.debug(f"Poll error: {e}")
            time.sleep(self.poll_interval)

    def start(self):
        """Start the agent — blocks until stopped."""
        log.info(f"MyAI GPU Agent v{VERSION} — {self.name} ({platform.system()})")
        log.info(f"  Coordinator : {self.coordinator_url}")
        log.info(f"  Ollama      : {self.ollama_url}")
        log.info(f"  Agent ID    : {self.agent_id}")

        # Pre-pull required models before registering
        self.ensure_models()

        # Register with retries
        for attempt in range(5):
            if self.register():
                break
            log.warning(f"Registration attempt {attempt + 1}/5 failed — retrying in 10s")
            time.sleep(10)
        else:
            log.error("Could not register after 5 attempts. Exiting.")
            sys.exit(1)

        self._running = True

        hb = threading.Thread(target=self._heartbeat_loop, daemon=True, name="heartbeat")
        hb.start()

        try:
            self._poll_loop()
        except KeyboardInterrupt:
            log.info("Shutting down...")
            self._running = False
