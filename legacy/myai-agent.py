#!/usr/bin/env python3
"""
MyAI GPU Agent — Ollama-native + vLLM, sharding-aware.

v1.3.0 — adds:
  - Sharding capability advertisement (tensor_parallel_supported, max_tp_size,
    pipeline_supported, backend_kind) in register + heartbeat.
  - Per-GPU memory_free_mb in the gpus[] payload of every register/heartbeat.
  - Shard metadata acceptance on incoming jobs (shard_role/group/world/rank,
    tensor_parallel_size, pipeline_parallel_size). When present, the agent
    prefers vLLM if VLLM_URL is set.
  - PREFER_VLLM honored for non-sharded jobs as well (existing behavior).

Config via environment variables:
  COORDINATOR_URL      — default https://api.myaitoken.io
  OLLAMA_URL           — default http://localhost:11434
  VLLM_URL             — optional, e.g. http://localhost:8100
  PREFER_VLLM          — true/false (default true)
  AGENT_NAME           — default hostname
  AGENT_WALLET         — optional EVM wallet
  POLL_INTERVAL        — seconds between HTTP polls (default 5)
  HEARTBEAT_INTERVAL   — seconds (default 30)
  USE_WEBSOCKET        — true/false (default true)
  TENSOR_PARALLEL      — auto/on/off (default auto: ON only when vLLM healthy
                         AND ≥2 GPUs AND GPUs symmetric within ~15%)
  MAX_TP_SIZE          — override (default = #symmetric GPUs)
  PIPELINE_PARALLEL    — auto/on/off (default auto: ON whenever ≥2 GPUs;
                         works for asymmetric GPUs since layers are atomic)
  BACKEND_KIND         — auto/ollama/vllm/mixed (default auto)
"""

import os
import sys
import json
import time
import uuid
import socket
import logging
import platform
import subprocess
import threading
import urllib.request
import urllib.error
import hashlib
import tempfile
import shutil

# ── Config ────────────────────────────────────────────────────────────────────

AGENT_VERSION = "1.6.0"

COORDINATOR_URL    = os.environ.get("COORDINATOR_URL",    "https://api.myaitoken.io")
OLLAMA_URL         = os.environ.get("OLLAMA_URL",         "http://localhost:11434")
VLLM_URL           = os.environ.get("VLLM_URL",           "")
PREFER_VLLM        = os.environ.get("PREFER_VLLM", "true").lower() in ("true", "1", "yes")
AGENT_NAME         = os.environ.get("AGENT_NAME",         socket.gethostname())
AGENT_WALLET       = os.environ.get("AGENT_WALLET",       "")
# AGENT_WALLET_KEY — 0x-prefixed ETH private key for wallet-based auth (preferred)
# MYAI_API_KEY     — legacy static key (deprecated, triggers warning)
AGENT_WALLET_KEY   = (
    os.environ.get("MYAI_WALLET_PRIVATE_KEY")
    or os.environ.get("AGENT_WALLET_KEY")
    or ""
)
MYAI_API_KEY       = os.environ.get("MYAI_API_KEY", "")
POLL_INTERVAL      = int(os.environ.get("POLL_INTERVAL",  "5"))
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "30"))
USE_WEBSOCKET      = os.environ.get("USE_WEBSOCKET", "true").lower() in ("true", "1", "yes")

TENSOR_PARALLEL    = os.environ.get("TENSOR_PARALLEL", "auto").lower()
MAX_TP_SIZE_ENV    = os.environ.get("MAX_TP_SIZE", "").strip()
# PIPELINE_PARALLEL is now read inside detect_sharding_caps with "auto" default
BACKEND_KIND_ENV   = os.environ.get("BACKEND_KIND", "auto").lower()

# Deprecation deadline for API key auth
API_KEY_AUTH_DEADLINE = "2026-06-01"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
AGENT_ID_FILE = os.path.join(SCRIPT_DIR, "agent_id.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("myai-agent")

# ── Wallet Auth (inline, stdlib-only) ─────────────────────────────────────────
#
# Implements the challenge/verify handshake against /v1/auth/challenge + /v1/auth/verify.
# Falls back to legacy MYAI_API_KEY bearer if no private key is configured.
# Pure stdlib — no httpx/web3 dependency on the agent host required.
#

_wallet_token: dict = {"access": "", "expires_at": 0.0, "refresh": ""}
_wallet_lock  = threading.Lock()


def _derive_wallet_address(private_key_hex: str) -> str:
    """Derive checksummed EVM address from a 0x-prefixed private key."""
    try:
        from eth_account import Account
        return Account.from_key(private_key_hex).address
    except Exception as exc:
        log.error(f"Cannot derive wallet address: {exc}")
        sys.exit(1)


def _sign_challenge(challenge: str, private_key_hex: str) -> str:
    """EIP-191 personal_sign of a challenge string."""
    from eth_account import Account
    from eth_account.messages import encode_defunct
    signable = encode_defunct(text=challenge)
    signed   = Account.sign_message(signable, private_key=private_key_hex)
    return signed.signature.hex()


def _http_raw(method: str, url: str, body: dict = None, token: str = "",
               timeout: int = 30) -> dict:
    """Low-level HTTP call used by wallet auth (no auth header injection)."""
    data    = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode()[:300]
        log.warning(f"HTTP {e.code} {method} {url}: {body_txt}")
        return {"_status": e.code}
    except Exception as e:
        log.debug(f"Request failed {method} {url}: {e}")
        return {}


def _wallet_authenticate() -> bool:
    """
    Full challenge → sign → verify handshake.
    Stores access + refresh tokens in _wallet_token.
    Returns True on success.
    """
    if not AGENT_WALLET_KEY:
        return False
    address = _derive_wallet_address(AGENT_WALLET_KEY)
    # Step 1: get challenge
    resp = _http_raw("POST", f"{COORDINATOR_URL}/v1/auth/challenge",
                     {"address": address})
    challenge = (resp.get("data") or resp).get("challenge", "")
    if not challenge:
        log.warning("Wallet auth: no challenge returned")
        return False
    # Step 2: sign
    try:
        signature = _sign_challenge(challenge, AGENT_WALLET_KEY)
    except Exception as exc:
        log.error(f"Wallet auth: signing failed: {exc}")
        return False
    # Step 3: verify
    resp2 = _http_raw("POST", f"{COORDINATOR_URL}/v1/auth/verify",
                      {"address": address,
                       "challenge":      challenge,
                       "signature":      signature})
    token_data = resp2.get("data") or resp2
    access  = token_data.get("token") or token_data.get("access_token", "")
    refresh = token_data.get("refresh_token", "")
    expires = token_data.get("expires_in", 3600)
    if not access:
        log.warning(f"Wallet auth: verify returned no token: {resp2}")
        return False
    with _wallet_lock:
        _wallet_token["access"]     = access
        _wallet_token["refresh"]    = refresh
        _wallet_token["expires_at"] = time.time() + expires
    log.info(f"Wallet auth: authenticated as {address}")
    return True


def _wallet_refresh() -> bool:
    """Exchange refresh token for a new access token."""
    refresh = _wallet_token.get("refresh", "")
    if not refresh:
        return _wallet_authenticate()
    resp = _http_raw("POST", f"{COORDINATOR_URL}/v1/auth/refresh",
                     {"refresh_token": refresh})
    token_data = resp.get("data") or resp
    access  = token_data.get("token") or token_data.get("access_token", "")
    expires = token_data.get("expires_in", 3600)
    if not access:
        # Refresh token expired — full re-auth
        return _wallet_authenticate()
    with _wallet_lock:
        _wallet_token["access"]     = access
        _wallet_token["expires_at"] = time.time() + expires
        if token_data.get("refresh_token"):
            _wallet_token["refresh"] = token_data["refresh_token"]
    return True


def _get_auth_header() -> str:
    """
    Return Bearer token string for outgoing requests.
    Uses wallet JWT if AGENT_WALLET_KEY is set, else falls back to MYAI_API_KEY.
    Refreshes transparently 5 minutes before expiry.
    """
    if AGENT_WALLET_KEY:
        with _wallet_lock:
            needs_refresh = time.time() >= (_wallet_token["expires_at"] - 300)
            current = _wallet_token["access"]
        if needs_refresh or not current:
            _wallet_refresh()
        return _wallet_token["access"]

    # Legacy API key path — emit deprecation warning
    if MYAI_API_KEY:
        log.warning(
            f"[DEPRECATED] API key auth in use — migrate to AGENT_WALLET_KEY before "
            f"{API_KEY_AUTH_DEADLINE}. Set AGENT_WALLET_KEY=0x<private-key> in agent.env."
        )
        return MYAI_API_KEY

    return ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def http(method: str, url: str, body: dict = None, timeout: int = 30) -> dict:
    data    = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    auth_token = _get_auth_header()
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_txt = e.read().decode()[:200]
        log.warning(f"HTTP {e.code} {method} {url}: {body_txt}")
        # 401 → token may be stale; force re-auth next call
        if e.code == 401 and AGENT_WALLET_KEY:
            with _wallet_lock:
                _wallet_token["expires_at"] = 0.0
        return {}
    except Exception as e:
        log.debug(f"Request failed {method} {url}: {e}")
        return {}


def get_gpu_info() -> list:
    """Detect GPUs and report per-GPU memory_free_mb plus the standard fields."""
    gpus = []

    # ── NVIDIA ────────────────────────────────────────────────────────────
    try:
        fields = "index,name,memory.total,memory.used,memory.free,driver_version,utilization.gpu,temperature.gpu,power.draw,power.limit"
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5
        ).decode().strip()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            def _int(v):
                try: return int(float(v))
                except: return None
            def _float(v):
                try: return round(float(v), 1)
                except: return None
            gpus.append({
                "gpu_id":          _int(parts[0]) or 0,
                "name":            parts[1],
                "vram_total_mb":   _int(parts[2]) or 0,
                "vram_used_mb":    _int(parts[3]),
                "memory_free_mb":  _int(parts[4]),
                "driver_version":  parts[5],
                "utilization_gpu": _int(parts[6]) if len(parts) > 6 else 0,
                "temperature_c":   _int(parts[7]) if len(parts) > 7 else None,
                "power_draw_w":    _float(parts[8]) if len(parts) > 8 else None,
                "power_limit_w":   _float(parts[9]) if len(parts) > 9 else None,
            })
    except Exception:
        pass

    # ── AMD (sysfs) ──────────────────────────────────────────────────────
    if not gpus:
        import glob as _glob
        gpu_idx = 0
        for card_path in sorted(_glob.glob("/sys/class/drm/card*/device")):
            try:
                vendor = open(os.path.join(card_path, "vendor")).read().strip()
                if vendor != "0x1002":
                    continue

                name = "AMD GPU"
                for name_file in ("product_name", "label"):
                    p = os.path.join(card_path, name_file)
                    if os.path.exists(p):
                        name = open(p).read().strip(); break

                def _read_int(path):
                    try: return int(open(path).read().strip())
                    except: return 0

                vram_total_mb = _read_int(os.path.join(card_path, "mem_info_vram_total")) // (1024*1024)
                vram_used_b   = _read_int(os.path.join(card_path, "mem_info_vram_used"))
                vram_used_mb  = vram_used_b // (1024*1024) if vram_used_b else None
                memory_free_mb = (vram_total_mb - vram_used_mb) if (vram_total_mb and vram_used_mb is not None) else None
                util          = _read_int(os.path.join(card_path, "gpu_busy_percent"))

                temp_c = None
                power_w = None
                for hwmon in sorted(_glob.glob(os.path.join(card_path, "hwmon/hwmon*"))):
                    if temp_c is None:
                        tf = os.path.join(hwmon, "temp1_input")
                        if os.path.exists(tf):
                            try: temp_c = int(open(tf).read().strip()) // 1000
                            except: pass
                    if power_w is None:
                        for pf_name in ("power1_average", "power1_input"):
                            pf = os.path.join(hwmon, pf_name)
                            if os.path.exists(pf):
                                try: power_w = round(int(open(pf).read().strip()) / 1_000_000, 1); break
                                except: pass

                gpus.append({
                    "gpu_id":          gpu_idx,
                    "name":            name,
                    "vram_total_mb":   vram_total_mb,
                    "vram_used_mb":    vram_used_mb,
                    "memory_free_mb":  memory_free_mb,
                    "driver_version":  "amdgpu",
                    "utilization_gpu": util,
                    "temperature_c":   temp_c,
                    "power_draw_w":    power_w,
                })
                gpu_idx += 1
            except Exception:
                continue

    # ── Apple Silicon ─────────────────────────────────────────────────────
    if not gpus and sys.platform == "darwin":
        try:
            import re
            chip = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                           stderr=subprocess.DEVNULL, timeout=3).decode().strip()
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"],
                                                     stderr=subprocess.DEVNULL, timeout=3).decode().strip())
            vram_mb = mem_bytes // (1024 * 1024)
            try:
                ioreg = subprocess.check_output(
                    ["ioreg", "-n", "AGXAccelerator", "-r", "-d", "1"],
                    stderr=subprocess.DEVNULL, timeout=5
                ).decode()
                m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', ioreg)
                core_count = int(m.group(1)) if m else 0
            except Exception:
                core_count = 0

            gpu_name = chip if chip else "Apple Silicon GPU"
            if core_count:
                gpu_name += f" ({core_count}-core GPU)"

            gpus.append({
                "gpu_id":          0,
                "name":            gpu_name,
                "vram_total_mb":   vram_mb,
                "vram_used_mb":    None,
                "memory_free_mb":  None,  # Apple unified memory; not a pure free-VRAM number
                "driver_version":  "metal",
                "utilization_gpu": 0,
                "temperature_c":   None,
                "power_draw_w":    None,
            })
        except Exception:
            pass

    if not gpus:
        log.warning("No GPUs detected (nvidia-smi, AMD sysfs, and Apple Silicon all failed)")
    return gpus


def get_ollama_models() -> list:
    try:
        resp = http("GET", f"{OLLAMA_URL}/api/tags", timeout=5)
        return [m["name"] for m in resp.get("models", [])]
    except Exception:
        return []


def get_vllm_models() -> list:
    if not VLLM_URL:
        return []
    try:
        resp = http("GET", f"{VLLM_URL.rstrip('/')}/v1/models", timeout=5)
        data = resp.get("data") or []
        return [m.get("id") for m in data if m.get("id")]
    except Exception:
        return []


def vllm_healthy() -> bool:
    if not VLLM_URL:
        return False
    try:
        # vLLM exposes /health on most builds; fall back to /v1/models
        for path in ("/health", "/v1/models"):
            try:
                with urllib.request.urlopen(f"{VLLM_URL.rstrip('/')}{path}", timeout=3) as r:
                    if 200 <= r.status < 300:
                        return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def detect_sharding_caps(gpus: list) -> dict:
    """
    Decide what we advertise for tensor/pipeline parallelism.

    TP requires *symmetric* GPUs in practice — vLLM shards weights evenly,
    so the smaller card caps each shard and the larger card's extra VRAM is
    wasted (or worse, OOM if --gpu-memory-utilization is high). We therefore
    only advertise tp=true when:
      * vLLM is currently healthy (not just configured)
      * ≥2 GPUs detected
      * the GPUs are within ~15% of each other in vram_total_mb

    PP works fine with asymmetric GPUs (each layer is atomic; you can place
    more layers on the bigger card). It's also what Ollama does natively
    across multi-GPU machines. So pp=true whenever there are ≥2 GPUs
    (regardless of symmetry), unless explicitly turned off via env.
    """
    n_gpus = sum(1 for g in gpus if (g.get("vram_total_mb") or 0) > 0)
    sizes = sorted([g.get("vram_total_mb") or 0 for g in gpus if (g.get("vram_total_mb") or 0) > 0])
    largest = sizes[-1] if sizes else 0
    smallest = sizes[0] if sizes else 0
    symmetric = (n_gpus >= 2 and largest > 0 and (largest - smallest) / largest <= 0.15)

    is_vllm_healthy = bool(VLLM_URL) and vllm_healthy()

    # Backend kind
    backend = BACKEND_KIND_ENV
    if backend == "auto":
        has_ollama = bool(get_ollama_models())
        if is_vllm_healthy and has_ollama:
            backend = "mixed"
        elif is_vllm_healthy:
            backend = "vllm"
        else:
            backend = "ollama"

    # ── Tensor parallel ────────────────────────────────────────────────
    if TENSOR_PARALLEL == "on":
        tp = True
    elif TENSOR_PARALLEL == "off":
        tp = False
    else:  # auto
        tp = is_vllm_healthy and n_gpus >= 2 and symmetric

    if MAX_TP_SIZE_ENV.isdigit():
        max_tp = max(1, int(MAX_TP_SIZE_ENV))
    elif tp:
        # Cap at the count of *symmetric* GPUs (within 15% of largest)
        sym_count = sum(1 for s in sizes if (largest - s) / largest <= 0.15) if largest else 1
        max_tp = max(1, sym_count)
    else:
        max_tp = 1

    # ── Pipeline parallel ──────────────────────────────────────────────
    # Auto: any agent with ≥2 GPUs can do PP via Ollama's native multi-GPU
    # layer-split, or vLLM's --pipeline-parallel-size. Explicit env override
    # still wins.
    pp_env = os.environ.get("PIPELINE_PARALLEL", "auto").lower()
    if pp_env in ("true", "1", "yes", "on"):
        pp = True
    elif pp_env in ("false", "0", "no", "off"):
        pp = False
    else:  # auto
        pp = n_gpus >= 2

    return {
        "tensor_parallel_supported": tp,
        "max_tp_size":                max_tp,
        "pipeline_supported":         pp,
        "backend_kind":               backend,
        # Diagnostic fields the coordinator can log/observe:
        "gpu_symmetric":              symmetric,
        "n_gpus":                     n_gpus,
    }


def load_agent_id() -> str:
    if os.path.exists(AGENT_ID_FILE):
        with open(AGENT_ID_FILE) as f:
            return f.read().strip()
    agent_id = str(uuid.uuid4())
    fd = os.open(AGENT_ID_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(agent_id)
    except Exception:
        os.close(fd)
        raise
    try:
        os.chmod(AGENT_ID_FILE, 0o600)
    except Exception:
        pass
    return agent_id


# ── Core ──────────────────────────────────────────────────────────────────────

AGENT_ID = load_agent_id()

# ── v3-C ECDSA P-256 attestation (anti-sybil) ────────────────────────────────
#
# Per-node attestation key: ECDSA P-256, persisted at ATTEST_KEY_PATH (mode 600).
# Every register / heartbeat / job-complete carries:
#     attest_ts    — int unix seconds
#     attest_nonce — 16-byte hex
#     attest_sig   — base64 DER signature over f"{agent_id}|{ts}|{nonce}"
#     attestation_pubkey_b64 — only on register (DER SubjectPublicKeyInfo)
#     device_fingerprint     — only on register (sha256 of MAC+host+CPU+GPU IDs)
#
# Graceful degradation: if `cryptography` is unavailable OR key fails to load,
# attestation fields are omitted. Coordinator falls back to v2 (0.5x multiplier).

ATTEST_KEY_PATH       = os.path.join(SCRIPT_DIR, ".attest-key.pem")
ATTEST_DISABLED       = os.environ.get("MYAI_DISABLE_ATTESTATION", "").lower() in ("1", "true", "yes")
_attest_priv          = None
_attest_pub_b64       = ""   # SEC1 65-byte uncompressed (0x04 || X || Y), base64url no-pad
_attest_available     = False
_attest_fingerprint   = ""


def _b64url_nopad(data: bytes) -> str:
    """base64url encode, strip padding (matches browser/coordinator spec)."""
    import base64 as _b64
    return _b64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _attest_load_or_create() -> bool:
    """Load existing key or generate+persist. Returns True on success.

    Spec format (v3-C, aligns with browser agent + coordinator validator):
      - ECDSA P-256 PEM PKCS8 on disk, mode 600.
      - Public key shipped as SEC1 uncompressed (65 bytes: 0x04 || X(32) || Y(32))
        base64url no-pad.
      - Signatures shipped as raw r||s (64 bytes) base64url no-pad.
    """
    global _attest_priv, _attest_pub_b64, _attest_available
    if ATTEST_DISABLED:
        log.info("v3c-attest: disabled by MYAI_DISABLE_ATTESTATION=1; skipping")
        return False
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization
    except Exception as e:
        log.warning(f"v3c-attest: cryptography unavailable ({e}); skipping (legacy fallback active, 0.5x earnings)")
        return False

    try:
        if os.path.exists(ATTEST_KEY_PATH):
            with open(ATTEST_KEY_PATH, "rb") as f:
                _attest_priv = serialization.load_pem_private_key(f.read(), password=None)
        else:
            _attest_priv = ec.generate_private_key(ec.SECP256R1())
            pem = _attest_priv.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
            fd = os.open(ATTEST_KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(pem)
            except Exception:
                os.close(fd); raise
            try: os.chmod(ATTEST_KEY_PATH, 0o600)
            except Exception: pass
            log.info(f"v3c-attest: generated new ECDSA P-256 key at {ATTEST_KEY_PATH}")

        # SEC1 uncompressed point (65 bytes: 0x04 || X || Y) base64url no-pad
        pub_sec1 = _attest_priv.public_key().public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint,
        )
        _attest_pub_b64 = _b64url_nopad(pub_sec1)
        _attest_available = True
        return True
    except Exception as e:
        log.warning(f"v3c-attest: key load/create failed ({e}); skipping")
        _attest_available = False
        return False


def _attest_sign(agent_id: str) -> dict:
    """Return dict with attest_ts/attest_nonce/attest_sig, or {} if unavailable.

    Signature is raw r||s (64 bytes) base64url no-pad over the canonical
    string f"{agent_id}|{ts}|{nonce}" -- matches the browser-agent envelope
    that the coordinator api/security/attestation.py already parses.
    """
    if not _attest_available or _attest_priv is None:
        return {}
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
        import secrets as _secrets
        ts = int(time.time())
        nonce = _secrets.token_hex(16)
        msg = f"{agent_id}|{ts}|{nonce}".encode("utf-8")
        sig_der = _attest_priv.sign(msg, ec.ECDSA(hashes.SHA256()))
        r, s = decode_dss_signature(sig_der)
        raw = r.to_bytes(32, "big") + s.to_bytes(32, "big")
        return {
            "attest_ts": ts,
            "attest_nonce": nonce,
            "attest_sig": _b64url_nopad(raw),
        }
    except Exception as e:
        log.debug(f"v3c-attest: sign failed: {e}")
        return {}


def _compute_device_fingerprint() -> str:
    """sha256 hex of stable device identifiers (cached).

    Per v3-C spec: machine-id + hostname + first MAC + CPU model
    + GPU 0 name (or "no-gpu") + Python platform string.
    """
    global _attest_fingerprint
    if _attest_fingerprint:
        return _attest_fingerprint
    try:
        h = hashlib.sha256()
        # machine-id: /etc/machine-id on Linux, IOPlatformUUID via ioreg on macOS
        machine_id = ""
        for path in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
            try:
                if os.path.exists(path):
                    machine_id = open(path).read().strip()
                    break
            except Exception: pass
        if not machine_id:
            try:
                r = subprocess.run(
                    ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
                    capture_output=True, text=True, timeout=3,
                )
                for line in r.stdout.splitlines():
                    if "IOPlatformUUID" in line:
                        machine_id = line.split("=")[-1].strip().strip('"')
                        break
            except Exception: pass
        h.update(f"machine_id:{machine_id}|".encode())
        h.update(f"host:{socket.gethostname()}|".encode())
        # First MAC (uuid.getnode returns int, 48-bit)
        h.update(f"mac:{uuid.getnode():012x}|".encode())
        # CPU model
        cpu = platform.processor() or ""
        try:
            if os.path.exists("/proc/cpuinfo"):
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("model name"):
                            cpu = line.split(":", 1)[1].strip()
                            break
        except Exception: pass
        h.update(f"cpu:{cpu}|".encode())
        # GPU 0 name (nvidia-smi) -- or "no-gpu"
        gpu0 = "no-gpu"
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", "0"],
                capture_output=True, text=True, timeout=3,
            )
            n = r.stdout.strip().splitlines()
            if n and n[0]:
                gpu0 = n[0].strip()
        except Exception: pass
        h.update(f"gpu0:{gpu0}|".encode())
        # Python platform string
        h.update(f"pyplat:{platform.platform()}|".encode())
        _attest_fingerprint = h.hexdigest()
        return _attest_fingerprint
    except Exception:
        return ""


# Initialise key on import (safe -- best-effort)
_attest_load_or_create()




def build_register_payload() -> dict:
    gpus = get_gpu_info()
    ollama_models = get_ollama_models()
    vllm_models = get_vllm_models()
    is_vllm_up = vllm_healthy()
    caps = detect_sharding_caps(gpus)

    # Prefer the address derived from the private key; fall back to AGENT_WALLET env
    wallet_addr = ""
    if AGENT_WALLET_KEY:
        try:
            from eth_account import Account
            wallet_addr = Account.from_key(AGENT_WALLET_KEY).address
        except Exception:
            pass
    if not wallet_addr:
        wallet_addr = AGENT_WALLET

    payload = {
        "agent_id":        AGENT_ID,
        "agent_name":      AGENT_NAME,
        "version":         AGENT_VERSION,
        "platform":        "native",  # v3-A/v3-C: coordinator enum is {browser-webgpu,mobile,native}
        "platform_os":     platform.system(),
        "ollama_url":      OLLAMA_URL,
        "vllm_url":        VLLM_URL or None,
        "vllm_healthy":    is_vllm_up,
        "vllm_models":     vllm_models,
        "gpus":            gpus,
        "models":          ollama_models,
        "wallet_address":  wallet_addr,
        "price_per_hour_myai": 1.0,
        # Sharding caps (new)
        "tensor_parallel_supported": caps["tensor_parallel_supported"],
        "max_tp_size":                caps["max_tp_size"],
        "pipeline_supported":         caps["pipeline_supported"],
        "backend_kind":               caps["backend_kind"],
    }
    # v3-C: attach attestation envelope (best-effort, backward compatible)
    if _attest_available:
        payload["attestation_pubkey_b64"] = _attest_pub_b64
        payload["device_fingerprint"]     = _compute_device_fingerprint()
        payload["attest_alg"]             = "ecdsa-p256-sha256"
        payload.update(_attest_sign(AGENT_ID))
    return payload


def register() -> bool:
    payload = build_register_payload()
    # v3-C: detect 409 device-already-bound responses (different wallet owns this fingerprint).
    # We need raw response code, so do an inline request rather than the http() helper that swallows it.
    if _attest_available:
        try:
            import urllib.request as _ur, urllib.error as _ue
            data = json.dumps(payload).encode()
            _hdrs = {"Content-Type": "application/json"}
            try:
                _tok = _get_auth_header()
                if _tok:
                    _hdrs["Authorization"] = f"Bearer {_tok}"
            except Exception:
                pass
            req = _ur.Request(
                f"{COORDINATOR_URL}/api/v1/agents/register",
                data=data, method="POST", headers=_hdrs,
            )
            with _ur.urlopen(req, timeout=30) as r:
                resp = json.loads(r.read().decode())
        except _ue.HTTPError as e:
            if e.code == 409:
                body = e.read().decode()[:400]
                log.error(f"v3c-attest: REGISTRATION REJECTED 409 (device fingerprint or pubkey already bound to a different wallet): {body}")
                log.error("v3c-attest: This means another agent_id under another wallet has already claimed this hardware.")
                log.error("v3c-attest: If you legitimately moved hardware, delete the stale row via brain admin then run --rotate-attest-key.")
                log.error("v3c-attest: Exiting (operator action required).")
                sys.exit(1)
            log.warning(f"register http {e.code}: {e.read().decode()[:200]}")
            resp = {}
        except Exception as e:
            log.debug(f"register http error: {e}")
            resp = {}
    else:
        resp = http("POST", f"{COORDINATOR_URL}/api/v1/agents/register", payload)
    if resp.get("success"):
        log.info(f"Registered as '{AGENT_NAME}' (id={AGENT_ID})")
        if payload["gpus"]:
            log.info(f"  GPUs: {', '.join(g['name'] for g in payload['gpus'])}")
        if payload["models"]:
            log.info(f"  Ollama models: {', '.join(payload['models'])}")
        if payload["vllm_models"]:
            log.info(f"  vLLM models  : {', '.join(payload['vllm_models'])}")
        log.info(
            f"  Sharding: tp={payload['tensor_parallel_supported']}/{payload['max_tp_size']} "
            f"pp={payload['pipeline_supported']} backend={payload['backend_kind']}"
        )
        return True
    log.error(f"Registration failed: {resp}")
    return False


def heartbeat():
    """Send periodic heartbeat. Includes fresh sharding caps + per-GPU VRAM."""
    import random as _random
    while True:
        jitter = _random.uniform(-0.15, 0.15) * HEARTBEAT_INTERVAL
        time.sleep(max(1, HEARTBEAT_INTERVAL + jitter))

        gpus = get_gpu_info()
        caps = detect_sharding_caps(gpus)
        body = {
            "status": "online",
            "gpus": gpus,
            "vllm_url": VLLM_URL or None,
            "vllm_healthy": vllm_healthy(),
            "vllm_models": get_vllm_models(),
            "models": get_ollama_models(),
            "tensor_parallel_supported": caps["tensor_parallel_supported"],
            "max_tp_size":                caps["max_tp_size"],
            "pipeline_supported":         caps["pipeline_supported"],
            "backend_kind":               caps["backend_kind"],
        }
        # v3-C: per-heartbeat signed envelope
        if _attest_available:
            body.update(_attest_sign(AGENT_ID))
        resp = http("POST", f"{COORDINATOR_URL}/api/v1/agents/{AGENT_ID}/heartbeat", body)
        if resp.get("success"):
            log.debug("Heartbeat ok")
        else:
            log.warning("Heartbeat failed — attempting re-register")
            register()


# ── Inference (Ollama + vLLM) ────────────────────────────────────────────────

def _is_chat_payload(prompt: str) -> bool:
    return prompt.strip().startswith("[")


def run_ollama(model: str, prompt: str, timeout: int = 120) -> str:
    if _is_chat_payload(prompt):
        try:
            messages = json.loads(prompt)
            payload = {"model": model, "messages": messages, "stream": False}
            resp = http("POST", f"{OLLAMA_URL}/api/chat", payload, timeout=timeout)
            return resp.get("message", {}).get("content", "").strip()
        except Exception:
            pass
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = http("POST", f"{OLLAMA_URL}/api/generate", payload, timeout=timeout)
    return resp.get("response", "").strip()


def run_ollama_streaming(model: str, prompt: str, token_callback=None, timeout: int = 120) -> str:
    import urllib.request as _ur
    is_chat = _is_chat_payload(prompt)
    if is_chat:
        try:
            messages = json.loads(prompt)
            payload = {"model": model, "messages": messages, "stream": True}
            url = f"{OLLAMA_URL}/api/chat"
        except Exception:
            is_chat = False

    if not is_chat:
        payload = {"model": model, "prompt": prompt, "stream": True}
        url = f"{OLLAMA_URL}/api/generate"

    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    req = _ur.Request(url, data=data, headers=headers, method="POST")

    result_parts = []
    try:
        with _ur.urlopen(req, timeout=timeout) as resp:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                    token = (chunk.get("message", {}).get("content", "")
                             if is_chat else chunk.get("response", ""))
                    if token:
                        result_parts.append(token)
                        if token_callback:
                            token_callback(token)
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        log.warning(f"Ollama streaming error: {e}")

    return "".join(result_parts)


def run_vllm(model: str, prompt: str, timeout: int = 180) -> str:
    """OpenAI-compatible request to local vLLM."""
    if not VLLM_URL:
        return ""
    if _is_chat_payload(prompt):
        try:
            messages = json.loads(prompt)
        except Exception:
            messages = [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]
    payload = {"model": model, "messages": messages, "stream": False}
    resp = http("POST", f"{VLLM_URL.rstrip('/')}/v1/chat/completions", payload, timeout=timeout)
    try:
        return resp["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def run_vllm_streaming(model: str, prompt: str, token_callback=None, timeout: int = 180) -> str:
    if not VLLM_URL:
        return ""
    import urllib.request as _ur
    if _is_chat_payload(prompt):
        try:
            messages = json.loads(prompt)
        except Exception:
            messages = [{"role": "user", "content": prompt}]
    else:
        messages = [{"role": "user", "content": prompt}]
    payload = {"model": model, "messages": messages, "stream": True}
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    req = _ur.Request(f"{VLLM_URL.rstrip('/')}/v1/chat/completions",
                       data=data, headers=headers, method="POST")
    parts = []
    try:
        with _ur.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload_s = line[5:].strip()
                if payload_s == "[DONE]":
                    break
                try:
                    obj = json.loads(payload_s)
                    delta = obj["choices"][0].get("delta", {})
                    tok = delta.get("content", "")
                    if tok:
                        parts.append(tok)
                        if token_callback:
                            token_callback(tok)
                except Exception:
                    continue
    except Exception as e:
        log.warning(f"vLLM streaming error: {e}")
    return "".join(parts)


def _route_backend(model: str, shard_meta: dict) -> str:
    """Return 'vllm' or 'ollama' for this job."""
    if shard_meta:
        # Sharded jobs strongly prefer vLLM (Ollama doesn't expose TP/PP knobs)
        if VLLM_URL and model in get_vllm_models():
            return "vllm"
        if VLLM_URL:
            log.warning(f"shard job for {model} but vLLM doesn't serve it; falling back to Ollama")
        return "ollama"
    if PREFER_VLLM and VLLM_URL and model in get_vllm_models():
        return "vllm"
    return "ollama"


def complete_job(job_id: str, result: str, success: bool = True):
    body = {"success": success, "result": result}
    # v3-C: signed envelope on completion (so coordinator can verify earner)
    if _attest_available:
        body.update(_attest_sign(AGENT_ID))
    http("POST", f"{COORDINATOR_URL}/api/v1/agents/{AGENT_ID}/jobs/{job_id}/complete", body)


def stream_token_to_coordinator(job_id: str, token: str):
    http("POST", f"{COORDINATOR_URL}/api/v1/agents/{AGENT_ID}/jobs/{job_id}/stream", {
        "token": token,
    })


def process_job(job: dict):
    job_id  = job.get("job_id", "unknown")
    model   = job.get("model", "llama3.2")
    prompt  = job.get("prompt", "")
    stream  = job.get("stream", False)

    # Sharding metadata (any of these may be present)
    shard_meta = {
        k: job[k] for k in (
            "shard_role", "shard_group", "shard_world_size", "shard_rank",
            "tensor_parallel_size", "pipeline_parallel_size",
        ) if k in job and job[k] is not None
    }
    # ws_agent.py sends shard_meta as a nested dict too; merge
    if isinstance(job.get("shard_meta"), dict):
        for k, v in job["shard_meta"].items():
            if v is not None:
                shard_meta.setdefault(k, v)

    if not prompt:
        log.warning(f"Job {job_id} has no prompt — skipping")
        complete_job(job_id, "", success=False)
        return

    backend = _route_backend(model, shard_meta)
    log.info(
        f"Running job {job_id} | model={model} | backend={backend} | stream={stream} | "
        f"shard={shard_meta or '∅'} | prompt={prompt[:60]}..."
    )

    try:
        if stream:
            if backend == "vllm":
                result = run_vllm_streaming(
                    model, prompt,
                    token_callback=lambda tok: stream_token_to_coordinator(job_id, tok),
                )
            else:
                result = run_ollama_streaming(
                    model, prompt,
                    token_callback=lambda tok: stream_token_to_coordinator(job_id, tok),
                )
        else:
            result = run_vllm(model, prompt) if backend == "vllm" else run_ollama(model, prompt)
    except Exception as e:
        log.error(f"Job {job_id} crashed in backend={backend}: {e}")
        complete_job(job_id, f"agent-error: {e}", success=False)
        return

    if result:
        log.info(f"Job {job_id} done ({len(result)} chars, backend={backend})")
        complete_job(job_id, result, success=True)
    else:
        log.warning(f"Job {job_id} returned empty result (backend={backend})")
        complete_job(job_id, "No response from model", success=False)


# ── WebSocket dispatch ────────────────────────────────────────────────────────

def ws_loop():
    try:
        import websocket as ws_lib
    except ImportError:
        log.warning("websocket-client not installed — falling back to HTTP polling")
        log.warning("Install with: pip install websocket-client")
        return

    ws_url = COORDINATOR_URL.replace("https://", "wss://").replace("http://", "ws://")
    ws_url = f"{ws_url}/agents/{AGENT_ID}/ws"

    reconnect_delay = 1

    while True:
        try:
            log.info(f"Connecting WebSocket to {ws_url}...")
            ws = ws_lib.create_connection(ws_url, timeout=10)

            payload = build_register_payload()
            ws_data = {
                "agent_name":  payload["agent_name"],
                "version":     payload["version"],
                "gpus":        payload["gpus"],
                "models":      payload["models"],
                "vllm_url":    payload["vllm_url"],
                "vllm_healthy": payload["vllm_healthy"],
                "vllm_models": payload["vllm_models"],
                "tensor_parallel_supported": payload["tensor_parallel_supported"],
                "max_tp_size":                payload["max_tp_size"],
                "pipeline_supported":         payload["pipeline_supported"],
                "backend_kind":               payload["backend_kind"],
            }
            # v3-C: propagate attestation fields if present in HTTP payload
            for _k in ("attestation_pubkey_b64", "device_fingerprint", "attest_alg",
                       "attest_ts", "attest_nonce", "attest_sig"):
                if _k in payload:
                    ws_data[_k] = payload[_k]
            ws.send(json.dumps({"type": "register", "data": ws_data}))

            resp = json.loads(ws.recv())
            if resp.get("type") == "registered":
                log.info("WebSocket connected and registered")
                reconnect_delay = 1
            else:
                log.warning(f"Unexpected WS response: {resp}")

            while True:
                try:
                    msg = ws.recv()
                    if not msg:
                        continue

                    data = json.loads(msg)
                    msg_type = data.get("type")

                    if msg_type == "job.assign":
                        job = data.get("data", {})
                        if not job:
                            # ws_agent variant inlines fields at top level
                            job = {k: data[k] for k in ("job_id", "model", "prompt", "max_tokens",
                                                         "shard_meta", "shard_role", "shard_group",
                                                         "shard_world_size", "shard_rank",
                                                         "tensor_parallel_size",
                                                         "pipeline_parallel_size", "stream",
                                                         "is_synthesis", "chunk_index", "parent_id")
                                   if k in data}
                        log.info(f"[WS] Received job: {job.get('job_id', '?')[:12]}")
                        t = threading.Thread(target=process_job, args=(job,), daemon=True)
                        t.start()

                    elif msg_type == "heartbeat_ack":
                        log.debug("[WS] Heartbeat ack")

                    elif msg_type == "ack":
                        pass

                    else:
                        log.debug(f"[WS] Unknown message type: {msg_type}")

                except ws_lib.WebSocketTimeoutException:
                    gpus = get_gpu_info()
                    caps = detect_sharding_caps(gpus)
                    _hb_data = {
                        "gpus": gpus,
                        "vllm_url": VLLM_URL or None,
                        "vllm_healthy": vllm_healthy(),
                        "vllm_models": get_vllm_models(),
                        "models": get_ollama_models(),
                        **caps,
                    }
                    if _attest_available:
                        _hb_data.update(_attest_sign(AGENT_ID))
                    ws.send(json.dumps({"type": "heartbeat", "data": _hb_data}))

                except ws_lib.WebSocketConnectionClosedException:
                    log.warning("[WS] Connection closed by server")
                    break

        except Exception as e:
            log.warning(f"[WS] Connection error: {e}")

        import random as _random
        jitter = _random.uniform(0, reconnect_delay * 0.5)
        wait = reconnect_delay + jitter
        log.info(f"[WS] Reconnecting in {wait:.1f}s...")
        time.sleep(wait)
        reconnect_delay = min(reconnect_delay * 2, 60)


# ── HTTP polling fallback ─────────────────────────────────────────────────────

def poll_loop():
    log.info(f"Polling {COORDINATOR_URL} every {POLL_INTERVAL}s...")
    while True:
        try:
            resp = http("GET",
                        f"{COORDINATOR_URL}/api/v1/agents/{AGENT_ID}/jobs/pending",
                        timeout=10)
            jobs = resp.get("data", {}).get("jobs", [])
            for job in jobs:
                process_job(job)
        except Exception as e:
            log.debug(f"Poll error: {e}")
        time.sleep(POLL_INTERVAL)


# ── Auto-update ───────────────────────────────────────────────────────────────

def _version_tuple(v: str):
    """Tolerant numeric tuple; strips any non-digit suffix per component."""
    try:
        out = []
        for part in v.strip().split("."):
            num = ""
            for ch in part:
                if ch.isdigit():
                    num += ch
                else:
                    break
            out.append(int(num) if num else 0)
        return tuple(out) if out else (0, 0, 0)
    except Exception:
        return (0, 0, 0)


def check_for_update() -> bool:
    """Remote auto-update DISABLED — security fix (board #9671).

    The previous implementation polled the coordinator version endpoint for a
    {version, url} pair, downloaded the coordinator-supplied URL over the
    network with NO signature check and NO certificate/host pinning (only a
    substring "sanity check"), overwrote this running file with the downloaded
    bytes, and self-exec'd the interpreter. That is a remote-code-execution
    path: a MITM on the download or a compromised/rogue coordinator could run
    arbitrary code on every provider box. It has been removed.

    Updates are delivered out-of-band via the packaged installer or the
    canonical `myai-agent` PyPI package (`pip install -U myai-agent`), which
    performs no self-fetch-and-exec. This function is retained as an inert
    no-op so existing call sites keep working without any network fetch or
    code execution.
    """
    return False


def auto_update_loop():
    # Auto-update loop disabled (board #9671). Inert no-op: no polling,
    # no download, no exec. Kept so any thread target reference is harmless.
    return


# ── Entry point ───────────────────────────────────────────────────────────────

def _graceful_shutdown(signum, frame):
    log.info(f"Received signal {signum} — marking agent offline and exiting")
    try:
        http("POST",
             f"{COORDINATOR_URL}/api/v1/agents/{AGENT_ID}/heartbeat",
             {"status": "offline"}, timeout=5)
    except Exception:
        pass
    sys.exit(0)


if __name__ == "__main__":
    # v3-C: CLI flags
    if "--show-pubkey" in sys.argv:
        try:
            ok = _attest_load_or_create() if not _attest_available else True
            print(f"agent_id  : {AGENT_ID}")
            print(f"pubkey_b64: {_attest_pub_b64}")
            print(f"fingerprint: {_compute_device_fingerprint()}")
            print(f"alg       : ecdsa-p256-sha256 (SEC1 uncompressed pubkey, raw r||s sig, base64url no-pad)")
            print(f"key_path  : {ATTEST_KEY_PATH}")
            print(f"available : {ok}")
            sys.exit(0)
        except Exception as e:
            print(f"show-pubkey failed: {e}"); sys.exit(1)

    if "--rotate-attest-key" in sys.argv:
        try:
            if os.path.exists(ATTEST_KEY_PATH):
                os.unlink(ATTEST_KEY_PATH)
                print(f"Deleted {ATTEST_KEY_PATH}")
            # Force regen by clearing module globals
            globals()["_attest_priv"] = None
            globals()["_attest_pub_b64"] = ""
            globals()["_attest_available"] = False
            ok = _attest_load_or_create()
            print(f"Generated new attest key: {_attest_pub_b64[:48]}... (ok={ok})")
            print("IMPORTANT: the next register call will be REJECTED with HTTP 409")
            print("if your wallet already has a (different) attestation_pubkey on file.")
            print("Delete the stale agent row via brain admin BEFORE restarting:")
            print("  curl -X DELETE https://api.myaitoken.io/api/v1/admin/agents/<old_agent_id>")
            print("Then restart the agent.")
            sys.exit(0)
        except Exception as e:
            print(f"Rotate failed: {e}"); sys.exit(1)

    import signal as _signal
    _signal.signal(_signal.SIGTERM, _graceful_shutdown)
    _signal.signal(_signal.SIGINT, _graceful_shutdown)

    log.info(f"MyAI GPU Agent v{AGENT_VERSION} starting — {AGENT_NAME} ({platform.system()})")
    log.info(f"  Coordinator : {COORDINATOR_URL}")
    log.info(f"  Ollama      : {OLLAMA_URL}")
    log.info(f"  vLLM        : {VLLM_URL or '(none)'}")
    log.info(f"  Agent ID    : {AGENT_ID}")
    log.info(f"  WebSocket   : {'enabled' if USE_WEBSOCKET else 'disabled'}")
    if _attest_available:
        log.info(f"  Attestation : ECDSA P-256 (pubkey {_attest_pub_b64[:24]}...)")
    else:
        log.info(f"  Attestation : DISABLED (cryptography unavailable)")

    # ── Auth bootstrap ────────────────────────────────────────────────────────
    if AGENT_WALLET_KEY:
        wallet_address = _derive_wallet_address(AGENT_WALLET_KEY)
        log.info(f"  Auth        : wallet {wallet_address}")
        if not _wallet_authenticate():
            log.warning("  Wallet auth failed on startup — will retry during registration")
    elif MYAI_API_KEY:
        log.warning(
            f"  Auth        : API key (DEPRECATED — migrate to AGENT_WALLET_KEY before {API_KEY_AUTH_DEADLINE})"
        )
    else:
        log.warning("  Auth        : none configured — set AGENT_WALLET_KEY in agent.env")


    for attempt in range(5):
        if register():
            break
        log.warning(f"Registration attempt {attempt+1}/5 failed — retrying in 10s")
        time.sleep(10)
    else:
        log.error("Could not register with coordinator after 5 attempts. Exiting.")
        sys.exit(1)

    hb = threading.Thread(target=heartbeat, daemon=True); hb.start()

    if USE_WEBSOCKET:
        ws_thread = threading.Thread(target=ws_loop, daemon=True); ws_thread.start()
        log.info("WebSocket dispatch enabled — HTTP polling as fallback")

    poll_loop()
