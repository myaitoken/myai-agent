#!/usr/bin/env bash
# MyAI GPU Agent Installer
# Usage: curl -fsSL https://myaitoken.io/install | bash
# Or with wallet: WALLET=0x... bash <(curl -fsSL https://myaitoken.io/install)
#
# Supports: macOS (Intel + Apple Silicon), Linux (Ubuntu/Debian/Arch)

set -euo pipefail

COORDINATOR="${COORDINATOR_URL:-https://api.myaitoken.io}"
WALLET="${WALLET:-}"
MODEL="${MODEL:-llama3.2}"
AGENT_NAME="${AGENT_NAME:-}"
REPO="https://github.com/myaitoken/myai-agent"

# ── Colors ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${BLUE}▶${NC} $*"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC}  $*"; }
error()   { echo -e "${RED}✗${NC} $*" >&2; exit 1; }
header()  { echo -e "\n${BOLD}$*${NC}"; }

# ── Platform detection ─────────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Darwin) PLATFORM="macOS" ;;
  Linux)  PLATFORM="Linux"  ;;
  *)      error "Unsupported OS: $OS. Supported: macOS, Linux" ;;
esac

header "MyAI GPU Agent Installer"
echo -e "  Platform    : ${BOLD}$PLATFORM ($ARCH)${NC}"
echo -e "  Coordinator : ${BOLD}$COORDINATOR${NC}"
echo -e "  Model       : ${BOLD}$MODEL${NC}"
echo ""

# ── Python check ──────────────────────────────────────────────────────────────
header "1. Checking Python"
if command -v python3 &>/dev/null; then
  PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
  PY_MAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
  PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
  if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 8 ]; }; then
    error "Python 3.8+ required (found $PY_VER). Install from https://www.python.org"
  fi
  success "Python $PY_VER"
else
  error "Python 3 not found. Install from https://www.python.org or via your package manager."
fi

# ── Ollama install ─────────────────────────────────────────────────────────────
header "2. Checking Ollama"
if command -v ollama &>/dev/null; then
  OLLAMA_VER=$(ollama --version 2>/dev/null | head -1 || echo "unknown")
  success "Ollama already installed ($OLLAMA_VER)"
else
  info "Installing Ollama..."

  if [ "$PLATFORM" = "macOS" ]; then
    if command -v brew &>/dev/null; then
      info "Installing via Homebrew..."
      brew install --cask ollama
    else
      info "Downloading Ollama for macOS..."
      curl -fsSL "https://ollama.com/download/Ollama-darwin.zip" -o /tmp/Ollama.zip
      unzip -q /tmp/Ollama.zip -d /tmp/Ollama-install/
      if [ -d "/tmp/Ollama-install/Ollama.app" ]; then
        mv /tmp/Ollama-install/Ollama.app /Applications/
        # Symlink CLI
        ln -sf /Applications/Ollama.app/Contents/Resources/ollama /usr/local/bin/ollama 2>/dev/null || true
      fi
      rm -rf /tmp/Ollama.zip /tmp/Ollama-install/
    fi

  elif [ "$PLATFORM" = "Linux" ]; then
    curl -fsSL https://ollama.com/install.sh | sh
  fi

  if command -v ollama &>/dev/null; then
    success "Ollama installed"
  else
    warn "Ollama install may have succeeded but 'ollama' not on PATH yet."
    warn "You may need to open Ollama.app manually on macOS, or restart your shell."
  fi
fi

# ── Start Ollama ───────────────────────────────────────────────────────────────
header "3. Starting Ollama"
# Check if Ollama is already serving
if curl -sf http://localhost:11434/api/tags &>/dev/null; then
  success "Ollama is already running"
else
  info "Starting Ollama server..."
  if [ "$PLATFORM" = "macOS" ]; then
    # On Mac, try the app first, then CLI
    if [ -d "/Applications/Ollama.app" ]; then
      open -a Ollama 2>/dev/null || true
    fi
    # Give it a moment, then try CLI if app didn't start
    sleep 3
    if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
      ollama serve &>/dev/null &
      OLLAMA_PID=$!
      sleep 3
    fi
  else
    # Linux: ollama serve or systemd
    if systemctl is-active --quiet ollama 2>/dev/null; then
      success "Ollama systemd service is active"
    else
      ollama serve &>/dev/null &
      OLLAMA_PID=$!
      sleep 3
    fi
  fi

  if curl -sf http://localhost:11434/api/tags &>/dev/null; then
    success "Ollama is running"
  else
    warn "Could not confirm Ollama is running. The agent will retry on startup."
  fi
fi

# ── Pull model ─────────────────────────────────────────────────────────────────
header "4. Pulling model: $MODEL"
if ollama list 2>/dev/null | grep -q "^$MODEL"; then
  success "Model '$MODEL' already available"
else
  info "Pulling $MODEL (this may take a few minutes)..."
  ollama pull "$MODEL" || warn "Model pull failed — you can run 'ollama pull $MODEL' manually"
fi

# ── Install agent ─────────────────────────────────────────────────────────────
header "5. Installing MyAI Agent"
PIP_CMD="pip3"
if ! command -v pip3 &>/dev/null; then
  PIP_CMD="python3 -m pip"
fi

info "Installing myai-agent package..."
$PIP_CMD install --quiet --upgrade "git+${REPO}.git" 2>/dev/null || \
  $PIP_CMD install --quiet --upgrade myai-agent 2>/dev/null || \
  error "pip install failed. Try manually: pip3 install myai-agent"

# ── Resolve agent command — check common install locations ────────────────────
AGENT_CMD=""
for candidate in \
    "myai-agent" \
    "$HOME/.local/bin/myai-agent" \
    "/opt/homebrew/bin/myai-agent" \
    "/usr/local/bin/myai-agent" \
    "$(python3 -m site --user-base 2>/dev/null)/bin/myai-agent"
do
  if command -v "$candidate" &>/dev/null || [ -x "$candidate" ]; then
    AGENT_CMD="$candidate"
    break
  fi
done

# Fallback: run as Python module
if [ -z "$AGENT_CMD" ]; then
  if python3 -m myai_agent --version &>/dev/null; then
    AGENT_CMD="python3 -m myai_agent"
    warn "myai-agent not found on PATH — using 'python3 -m myai_agent'"
    warn "Add $(python3 -m site --user-base 2>/dev/null)/bin to your PATH to use 'myai-agent' directly."
    # Permanently fix PATH in shell config
    SHELL_RC="$HOME/.zshrc"
    [ -f "$HOME/.bashrc" ] && SHELL_RC="$HOME/.bashrc"
    USER_BIN="$(python3 -m site --user-base 2>/dev/null)/bin"
    if ! grep -q "$USER_BIN" "$SHELL_RC" 2>/dev/null; then
      echo "export PATH=\"$USER_BIN:\$PATH\"" >> "$SHELL_RC"
      success "Added $USER_BIN to PATH in $SHELL_RC (takes effect in new terminals)"
    fi
    export PATH="$(python3 -m site --user-base 2>/dev/null)/bin:$PATH"
    # Re-check after PATH fix
    command -v myai-agent &>/dev/null && AGENT_CMD="myai-agent"
  else
    error "Installation failed — myai-agent not found. Try: pip3 install --user myai-agent"
  fi
fi

success "myai-agent installed ($($AGENT_CMD --version 2>/dev/null || echo 'v2.0.0'))"

# ── Wallet prompt ─────────────────────────────────────────────────────────────
# Read from /dev/tty so prompts work even when piped via curl | bash
header "6. Configuration"
if [ -z "$WALLET" ] && [ -e /dev/tty ]; then
  echo ""
  echo "  Enter your Base wallet address to receive MYAI token rewards."
  echo "  (Press Enter to skip — you can set this later with: myai-agent install --wallet 0x...)"
  echo ""
  read -r WALLET < /dev/tty || true
  # Trim whitespace
  WALLET="${WALLET#"${WALLET%%[![:space:]]*}"}"
  WALLET="${WALLET%"${WALLET##*[![:space:]]}"}"
  [ -n "$WALLET" ] && echo "  Wallet: $WALLET"
fi

if [ -z "$AGENT_NAME" ] && [ -e /dev/tty ]; then
  DEFAULT_NAME="$(hostname -s 2>/dev/null || hostname)"
  printf "  Agent display name [%s]: " "$DEFAULT_NAME"
  read -r INPUT_NAME < /dev/tty || true
  AGENT_NAME="${INPUT_NAME:-$DEFAULT_NAME}"
fi

# ── Register as service ────────────────────────────────────────────────────────
header "7. Installing as service"

INSTALL_ARGS=(
  "--coordinator" "$COORDINATOR"
  "--ollama"      "http://localhost:11434"
  "--model"       "$MODEL"
)
[ -n "$WALLET"     ] && INSTALL_ARGS+=("--wallet" "$WALLET")
[ -n "$AGENT_NAME" ] && INSTALL_ARGS+=("--name"   "$AGENT_NAME")

$AGENT_CMD install "${INSTALL_ARGS[@]}"

# ── Start agent now ────────────────────────────────────────────────────────────
header "8. Starting agent"
if [ "$PLATFORM" = "macOS" ]; then
  launchctl start "io.myaitoken.agent" 2>/dev/null || true
  sleep 2
  $AGENT_CMD status
elif [ "$PLATFORM" = "Linux" ]; then
  systemctl --user start myai-agent 2>/dev/null || true
  sleep 2
  $AGENT_CMD status
fi

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}🎉 MyAI GPU Agent is installed and running!${NC}"
echo ""
echo -e "  ${BOLD}Useful commands:${NC}"
echo "    $AGENT_CMD status          Check if agent is running"
echo "    $AGENT_CMD logs            View live logs"
echo "    $AGENT_CMD models          List available models"
echo "    $AGENT_CMD run-job <text>  Test local inference"
echo "    $AGENT_CMD uninstall       Remove the service"
echo ""
echo -e "  ${BOLD}Dashboard:${NC} https://app.myaitoken.io"
echo ""
