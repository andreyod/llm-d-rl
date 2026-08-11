"""EPP file-discovery endpoints YAML writer."""

from __future__ import annotations

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
    """Split ``host:port``, ``[ipv6]:port``, or ``scheme://host:port`` into (host, port)."""
    if "://" in server_address:
        server_address = server_address.split("://", 1)[1]
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
    """Replace the endpoints file with exactly `entries`.

    A full replace, not a merge, so removing an endpoint from the list removes it
    from the file. That makes this SINGLE-WRITER only: every caller must own the
    whole list. Two processes writing the same path will clobber each other (an
    earlier version merged under an flock; deletion support replaced it).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        yaml.dump({"endpoints": entries}, f, sort_keys=False)
    os.replace(tmp, path)


def write_rollout_endpoints(
    path: str,
    server_addresses: list[str],
    model_config: Any = None,
    engine_type: str = "vllm",
) -> None:
    """Write standard (non-PD) endpoints YAML.

    Each replica gets ``name: {engine_type}-replica-{i}`` and ``rankIndex: i``.
    The ``llm-d.ai/engine-type`` label tells EPP's metrics extractor which
    Prometheus metric-name mapping to use (vllm/sglang/trtllm-serve/...);
    defaults to "vllm" to preserve existing behaviour for callers that don't
    pass it. ``model_config`` is optional; if omitted the ``model`` label is
    not written.

    An empty ``server_addresses`` writes an empty list rather than doing nothing:
    a registry-style caller (see vime's router_shim) needs the last endpoint's
    removal to actually reach the file, or EPP keeps routing to an engine that is
    gone.
    """
    if not path:
        return
    label = model_label(model_config) if model_config is not None else None
    entries = []
    for i, addr in enumerate(server_addresses):
        host, port = split_address(addr)
        labels: dict[str, str] = {}
        if label is not None:
            labels["model"] = label
        labels["llm-d.ai/engine-type"] = engine_type
        entries.append({
            "name": f"{engine_type}-replica-{i}",
            "address": host,
            "port": port,
            "rankIndex": i,
            "labels": labels,
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
