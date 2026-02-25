"""
GPU detection — NVIDIA (all platforms), Apple Silicon / AMD (macOS).
Returns a list of GPU dicts; never raises.
"""

from __future__ import annotations

import json
import logging
import platform
import re
import subprocess
from typing import List, Dict, Any

log = logging.getLogger(__name__)


def _run(cmd: list, timeout: int = 5) -> str:
    """Run a subprocess and return stdout, or '' on failure."""
    try:
        return subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=timeout
        ).decode().strip()
    except Exception:
        return ""


def _nvidia() -> List[Dict[str, Any]]:
    """Query nvidia-smi for GPU info."""
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version,utilization.gpu",
        "--format=csv,noheader,nounits",
    ])
    if not out:
        return []
    gpus = []
    for i, line in enumerate(out.splitlines()):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            gpus.append({
                "gpu_id": i,
                "name": parts[0],
                "vram_total_mb": int(parts[1]) if parts[1].isdigit() else 0,
                "driver_version": parts[2],
                "utilization_gpu": int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 0,
                "vendor": "NVIDIA",
            })
    return gpus


def _apple_silicon() -> List[Dict[str, Any]]:
    """
    Detect Apple Silicon GPU via system_profiler.
    M-series Macs expose a unified memory GPU.
    """
    out = _run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=10)
    if not out:
        return []
    try:
        data = json.loads(out)
        displays = data.get("SPDisplaysDataType", [])
        gpus = []
        for i, d in enumerate(displays):
            name = d.get("sppci_model", d.get("_name", "Apple GPU"))
            vram_str = d.get("spdisplays_vram", d.get("spdisplays_vram_shared", "0 MB"))
            # Parse "8192 MB" or "16 GB" etc.
            vram_mb = 0
            m = re.match(r"(\d+)\s*(MB|GB)", str(vram_str), re.IGNORECASE)
            if m:
                vram_mb = int(m.group(1))
                if m.group(2).upper() == "GB":
                    vram_mb *= 1024
            gpus.append({
                "gpu_id": i,
                "name": name,
                "vram_total_mb": vram_mb,
                "driver_version": "Metal",
                "utilization_gpu": 0,
                "vendor": "Apple",
            })
        return gpus
    except Exception as e:
        log.debug(f"Apple GPU detection failed: {e}")
        return []


def _rocm() -> List[Dict[str, Any]]:
    """Detect AMD GPUs via rocm-smi (Linux)."""
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--json"])
    if not out:
        return []
    try:
        data = json.loads(out)
        gpus = []
        for i, (card, info) in enumerate(data.items()):
            name = info.get("Card series", info.get("Card SKU", "AMD GPU"))
            vram_str = info.get("VRAM Total Memory (B)", "0")
            try:
                vram_mb = int(vram_str) // (1024 * 1024)
            except (ValueError, TypeError):
                vram_mb = 0
            gpus.append({
                "gpu_id": i,
                "name": name,
                "vram_total_mb": vram_mb,
                "driver_version": "ROCm",
                "utilization_gpu": 0,
                "vendor": "AMD",
            })
        return gpus
    except Exception as e:
        log.debug(f"ROCm GPU detection failed: {e}")
        return []


def detect() -> List[Dict[str, Any]]:
    """
    Detect all available GPUs. Returns list of GPU dicts.
    Priority: NVIDIA > Apple Silicon > AMD ROCm > empty list.
    """
    gpus = _nvidia()
    if gpus:
        return gpus

    if platform.system() == "Darwin":
        gpus = _apple_silicon()
        if gpus:
            return gpus

    if platform.system() == "Linux":
        gpus = _rocm()
        if gpus:
            return gpus

    # Last resort: check if Ollama is GPU-accelerated by checking /proc/driver/nvidia
    if platform.system() == "Linux":
        import os
        if os.path.exists("/proc/driver/nvidia"):
            return [{"gpu_id": 0, "name": "NVIDIA GPU (details unavailable)", "vram_total_mb": 0,
                     "driver_version": "unknown", "utilization_gpu": 0, "vendor": "NVIDIA"}]

    return []
