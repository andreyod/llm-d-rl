"""EPP file-discovery endpoints YAML writer."""

from __future__ import annotations

import fcntl
import logging
import os
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def model_label(model_config: Any) -> str:
    """Return model_config.path for EPP endpoint labels."""
    if hasattr(model_config, "path"):
        return str(model_config.path)
    # Plain dict (after OmegaConf.to_container)
    return str(model_config.get("path", "unknown"))


def split_address(server_address: str) -> tuple[str, str]:
    """Split ``host:port`` or ``[ipv6]:port`` into (host, port)."""
    if server_address.startswith("["):
        bracket_end = server_address.index("]")
        host = server_address[1:bracket_end]
        rest = server_address[bracket_end + 1:]
        if not rest.startswith(":"):
            raise ValueError(f"Bad IPv6 address: {server_address!r}")
        return host, rest[1:]
    if ":" not in server_address:
        raise ValueError(f"Bad address (expected host:port): {server_address!r}")
    host, port = server_address.rsplit(":", 1)
    return host, port


def _atomic_write(path: str, entries: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    lock_path = path + ".lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            existing: dict[str, dict] = {}
            if os.path.exists(path):
                with open(path) as f:
                    data = yaml.safe_load(f) or {}
                existing = {e["name"]: e for e in data.get("endpoints", [])}
            for entry in entries:
                existing[entry["name"]] = entry
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                yaml.dump({"endpoints": list(existing.values())}, f, sort_keys=False)
            os.replace(tmp, path)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def write_rollout_endpoints(path: str, server_addresses: list[str], model_config: Any) -> None:
    """Write standard (non-PD) endpoints YAML.

    Each replica gets ``name: vllm-replica-{i}`` and ``rankIndex: i``.
    """
    if not path or not server_addresses:
        return
    label = model_label(model_config)
    entries = []
    for i, addr in enumerate(server_addresses):
        host, port = split_address(addr)
        entries.append({
            "name": f"vllm-replica-{i}",
            "address": host,
            "port": port,
            "rankIndex": i,
            "labels": {"model": label},
        })
    _atomic_write(path, entries)
    logger.info("Wrote %d endpoints to %s", len(entries), path)


def write_pd_endpoints(
    path: str,
    server_addresses: list[str],
    server_roles: list[str],
    model_config: Any,
) -> None:
    """Write P/D endpoints YAML with per-replica role labels.

    Roles are ``"prefill"`` or ``"decode"``; sets ``llm-d.ai/role`` label.
    """
    if not path:
        return
    label = model_label(model_config)
    entries = []
    role_counters: dict[str, int] = {}
    for addr, role in zip(server_addresses, server_roles):
        host, port = split_address(addr)
        idx = role_counters.get(role, 0)
        role_counters[role] = idx + 1
        entries.append({
            "name": f"vllm-{role}-0-{idx}",
            "address": host,
            "port": port,
            "labels": {"model": label, "llm-d.ai/role": role},
        })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        yaml.dump({"endpoints": entries}, f, sort_keys=False)
    os.replace(tmp, path)
    logger.info("Wrote %d P/D endpoints to %s", len(entries), path)
