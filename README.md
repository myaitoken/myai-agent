# MyAI GPU Agent

[![PyPI version](https://badge.fury.io/py/myai-agent.svg)](https://pypi.org/project/myai-agent/)
[![CI](https://github.com/myaitoken/myai-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/myaitoken/myai-agent/actions/workflows/ci.yml)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Earn **MYAI tokens** by sharing your GPU compute power with the MyAI distributed inference network.

Your GPU runs local AI inference via [Ollama](https://ollama.com). Jobs are dispatched from the [MyAI coordinator](https://api.myaitoken.io) and you earn MYAI on Base for every completed job.

---

## Quick Install

**macOS / Linux:**
```bash
curl -fsSL https://myaitoken.io/install | bash
```

With your wallet address (to start earning immediately):
```bash
WALLET=0xYourWalletAddress curl -fsSL https://myaitoken.io/install | bash
```

**pip (any platform):**
```bash
pip install myai-agent
myai-agent install --wallet 0xYourWalletAddress
```

---

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.8+ | Zero external deps — pure stdlib |
| [Ollama](https://ollama.com) | Any | Local inference engine |
| GPU | Optional | CPU works too, just slower |

**Supported GPUs:** NVIDIA (CUDA), Apple Silicon (Metal), AMD (ROCm)

---

## Manual Setup

### 1. Install Ollama

```bash
# macOS
brew install --cask ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

### 2. Pull a model

```bash
ollama pull llama3.2        # recommended (2GB, fast)
ollama pull llama3.1:8b     # larger, better quality
ollama pull mistral         # alternative
```

### 3. Install the agent

```bash
pip install myai-agent
```

### 4. Run or install as service

**One-off run:**
```bash
myai-agent start --wallet 0xYourWallet
```

**Install as service (auto-starts on login):**
```bash
myai-agent install \
  --wallet 0xYourWallet \
  --name "My GPU Rig" \
  --model llama3.2
```

---

## Commands

```
myai-agent start              Run the agent (foreground)
myai-agent install            Install as OS service
myai-agent uninstall          Remove OS service
myai-agent status             Check service status + agent ID
myai-agent logs               Tail agent logs
myai-agent models             List available Ollama models
myai-agent run-job <prompt>   Test local inference
myai-agent --version          Print version
```

### Options (start / install)

| Flag | Default | Description |
|---|---|---|
| `--coordinator URL` | `https://api.myaitoken.io` | Coordinator endpoint |
| `--ollama URL` | `http://localhost:11434` | Ollama endpoint |
| `--wallet 0x...` | — | Base EVM wallet for MYAI rewards |
| `--name NAME` | hostname | Display name for this agent |
| `--model MODEL` | `llama3.2` | Default Ollama model |
| `--required-models M1,M2` | `bonsai-8b:latest` | Comma-separated models to auto-pull on startup |

---

## Auto Model Pull

The agent automatically pulls required models on startup if they aren't already present in Ollama.

**Default:** `bonsai-8b:latest` is pulled on every agent start if missing (~1.1 GB, 1-bit quantized).

**Customize via env var:**
```bash
REQUIRED_MODELS=bonsai-8b:latest,deepseek-r1:7b myai-agent start
```

**Disable auto-pull:**
```bash
REQUIRED_MODELS="" myai-agent start
```

The pull streams progress to the log and blocks registration until complete — so the coordinator always sees accurate model lists.

---

## How It Works

1. Agent registers with the coordinator and reports available GPUs + models
2. Coordinator receives inference jobs from users
3. Agent polls for pending jobs (`GET /api/v1/agents/{id}/jobs/pending`)
4. Agent runs inference locally via Ollama (`POST /api/generate`)
5. Results reported back to coordinator (`POST /api/v1/agents/{id}/jobs/{job_id}/complete`)
6. MYAI tokens distributed to your wallet on Base *(Phase 3)*

**Parallel inference:** Large jobs are split across multiple agents simultaneously for faster results.

---

## Platform Notes

### macOS
- Installed as a **launchd** service (`~/Library/LaunchAgents/io.myaitoken.agent.plist`)
- Logs: `~/Library/Logs/myai-agent/agent.log`
- Ollama recommended: `brew install --cask ollama`

### Linux
- Installed as a **systemd user service** (`~/.config/systemd/user/myai-agent.service`)
- Logs: `journalctl --user -u myai-agent -f`

### Windows
- Installed as a **Task Scheduler** task (runs on login)
- Download Ollama from [ollama.com/download](https://ollama.com/download)

---

## Configuration

Environment variables (set in OS service config, or export before `myai-agent start`):

| Variable | Default | Description |
|---|---|---|
| `COORDINATOR_URL` | `https://api.myaitoken.io` | Coordinator API URL |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `AGENT_NAME` | hostname | Display name |
| `AGENT_WALLET` | — | Base wallet for MYAI rewards |
| `POLL_INTERVAL` | `5` | Seconds between job polls |
| `HEARTBEAT_INTERVAL` | `30` | Seconds between heartbeats |

---

## MYAI Token

- **Contract:** `0xaff22cc20434ce43b3ea10efe10e9360390d327c`
- **Chain:** Base (Chain ID: 8453)
- **Symbol:** MYAI | Supply: 1,000,000,000

---

## License

MIT © [MyAI Token](https://myaitoken.io)
