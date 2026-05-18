"""
v3-C Native Agent Attestation -- anti-sybil ECDSA P-256.

Generates a per-node ECDSA P-256 keypair on first run, persists it at
$config_dir/.attest-key.pem mode 600, and signs every register / heartbeat /
job-complete envelope so the coordinator can prove the agent is the same
hardware that registered. Aligns with the browser-agent format that
api/security/attestation.py on the coordinator already validates:

  - public key  : SEC1 uncompressed (65 bytes: 0x04 || X(32) || Y(32))
                  base64url no-pad
  - signature   : raw r||s (64 bytes) base64url no-pad
  - canonical   : f"{agent_id}|{ts}|{nonce}"  (UTF-8)
  - hash        : SHA-256
  - clock skew  : +/- 60s enforced server-side

Fingerprint = SHA-256 hex of:
  machine-id + hostname + first MAC + CPU model
  + GPU 0 name (or "no-gpu") + Python platform.

Set MYAI_DISABLE_ATTESTATION=1 to opt out (legacy nodes during rollout
-- coordinator falls back to 0.5x earnings multiplier).
"""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import platform
import secrets
import socket
import subprocess
import time
import uuid

log = logging.getLogger("myai_agent.attestation")


def _b64url_nopad(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


class Attestation:
    """Owns the on-disk ECDSA P-256 key + signed-envelope generation."""

    def __init__(self, key_path: str):
        self.key_path = key_path
        self.disabled = os.environ.get(
            "MYAI_DISABLE_ATTESTATION", ""
        ).lower() in ("1", "true", "yes")
        self._priv = None
        self.pubkey_b64 = ""
        self.available = False
        self._fingerprint = ""
        if not self.disabled:
            self._load_or_create()

    # ------------------------------------------------------------------
    def _load_or_create(self) -> bool:
        try:
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives import serialization
        except Exception as e:
            log.warning(
                "v3c-attest: cryptography unavailable (%s); skipping "
                "(legacy 0.5x multiplier active)", e,
            )
            return False

        try:
            os.makedirs(os.path.dirname(self.key_path) or ".", exist_ok=True)
            if os.path.exists(self.key_path):
                with open(self.key_path, "rb") as f:
                    self._priv = serialization.load_pem_private_key(
                        f.read(), password=None,
                    )
            else:
                self._priv = ec.generate_private_key(ec.SECP256R1())
                pem = self._priv.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption(),
                )
                fd = os.open(
                    self.key_path,
                    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                    0o600,
                )
                try:
                    with os.fdopen(fd, "wb") as f:
                        f.write(pem)
                except Exception:
                    os.close(fd); raise
                try:
                    os.chmod(self.key_path, 0o600)
                except Exception:
                    pass
                log.info(
                    "v3c-attest: generated new ECDSA P-256 key at %s",
                    self.key_path,
                )

            # SEC1 uncompressed (65 bytes) base64url no-pad
            sec1 = self._priv.public_key().public_bytes(
                encoding=serialization.Encoding.X962,
                format=serialization.PublicFormat.UncompressedPoint,
            )
            self.pubkey_b64 = _b64url_nopad(sec1)
            self.available = True
            return True
        except Exception as e:
            log.warning("v3c-attest: key load/create failed (%s); skipping", e)
            self.available = False
            return False

    # ------------------------------------------------------------------
    def sign_envelope(self, agent_id: str) -> dict:
        """Return {attest_ts, attest_nonce, attest_sig} or {}."""
        if not self.available or self._priv is None:
            return {}
        try:
            from cryptography.hazmat.primitives.asymmetric import ec
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric.utils import (
                decode_dss_signature,
            )
            ts = int(time.time())
            nonce = secrets.token_hex(16)
            msg = f"{agent_id}|{ts}|{nonce}".encode("utf-8")
            sig_der = self._priv.sign(msg, ec.ECDSA(hashes.SHA256()))
            r, s = decode_dss_signature(sig_der)
            raw = r.to_bytes(32, "big") + s.to_bytes(32, "big")
            return {
                "attest_ts": ts,
                "attest_nonce": nonce,
                "attest_sig": _b64url_nopad(raw),
            }
        except Exception as e:
            log.debug("v3c-attest: sign failed: %s", e)
            return {}

    # ------------------------------------------------------------------
    def device_fingerprint(self) -> str:
        if self._fingerprint:
            return self._fingerprint
        try:
            h = hashlib.sha256()

            # machine-id (Linux) or IOPlatformUUID (macOS)
            machine_id = ""
            for p in ("/etc/machine-id", "/var/lib/dbus/machine-id"):
                try:
                    if os.path.exists(p):
                        machine_id = open(p).read().strip()
                        break
                except Exception:
                    pass
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
                except Exception:
                    pass
            h.update(f"machine_id:{machine_id}|".encode())
            h.update(f"host:{socket.gethostname()}|".encode())
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
            except Exception:
                pass
            h.update(f"cpu:{cpu}|".encode())

            # GPU 0 (nvidia-smi)
            gpu0 = "no-gpu"
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name",
                     "--format=csv,noheader", "-i", "0"],
                    capture_output=True, text=True, timeout=3,
                )
                lines = r.stdout.strip().splitlines()
                if lines and lines[0]:
                    gpu0 = lines[0].strip()
            except Exception:
                pass
            h.update(f"gpu0:{gpu0}|".encode())

            h.update(f"pyplat:{platform.platform()}|".encode())
            self._fingerprint = h.hexdigest()
            return self._fingerprint
        except Exception:
            return ""

    # ------------------------------------------------------------------
    def rotate(self) -> bool:
        """Delete on-disk key + regenerate. Caller must restart + re-register."""
        try:
            if os.path.exists(self.key_path):
                os.unlink(self.key_path)
            self._priv = None
            self.pubkey_b64 = ""
            self.available = False
            return self._load_or_create()
        except Exception as e:
            log.error("v3c-attest: rotate failed: %s", e)
            return False
