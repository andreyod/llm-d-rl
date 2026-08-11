"""Framework-agnostic EPP / Envoy process launcher.

Owns the EPP and Envoy argv, binary-path resolution and readiness waiting, so
every caller starts the router stack identically:

  - verl: LlmdActor, a Ray actor pinned to the head node, wraps this.
  - a framework with no injectable hook: the ``llm-d-rl-router`` console script
    (see cli.py) runs it as a plain process from a pod lifecycle hook.

No ray, verl or other framework imports here - stdlib only.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_EPP_GRPC_PORT = 9002
DEFAULT_EPP_HEALTH_PORT = 9003
DEFAULT_EPP_METRICS_PORT = 9090
DEFAULT_ENVOY_PORT = 8081
DEFAULT_EPP_LOG = "/tmp/epp.log"
DEFAULT_ENVOY_LOG = "/tmp/envoy.log"
DEFAULT_START_TIMEOUT = 120.0

# EPP's pool-name parsing wants a pod-style name; this stand-in keeps a local run
# (no POD_NAME, no HOSTNAME) from tripping it.
_FALLBACK_POD_NAME = "verl-epp-abc12-xyz34"


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read LLMD_<name>, falling back to the older VERL_<name>, then default."""
    return os.environ.get(f"LLMD_{name}") or os.environ.get(f"VERL_{name}") or default


def epp_binary() -> str:
    """Resolve the EPP binary path.

    Overridable via LLMD_EPP_BINARY (or VERL_EPP_BINARY) so the binary can be
    injected at runtime (fetch-binaries initContainer / push-epp.sh) instead of
    being baked into an image; iterating on the EPP then needs no image rebuild.
    Read per call, so an env var set after import still applies.
    """
    return _env("EPP_BINARY", "/usr/local/bin/epp")


def envoy_binary() -> str:
    """Resolve the Envoy binary path (same runtime-injection rationale as EPP)."""
    return _env("ENVOY_BINARY", "/usr/local/bin/envoy")


async def wait_port(host: str, port: int, timeout: float = DEFAULT_START_TIMEOUT) -> None:
    """Wait until host:port accepts a TCP connection."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=1.0
            )
            writer.close()
            await writer.wait_closed()
            return
        except (OSError, asyncio.TimeoutError):
            await asyncio.sleep(0.5)
    raise RuntimeError(f"Timed out after {timeout}s waiting for {host}:{port}")


class RouterStack:
    """Owns the EPP and (optionally) Envoy child processes."""

    def __init__(self) -> None:
        self.epp_proc: Optional[subprocess.Popen] = None
        self.envoy_proc: Optional[subprocess.Popen] = None

    async def start_epp(
        self,
        config_file: str,
        *,
        grpc_port: int = DEFAULT_EPP_GRPC_PORT,
        health_port: int = DEFAULT_EPP_HEALTH_PORT,
        metrics_port: int = DEFAULT_EPP_METRICS_PORT,
        pool_name: str = "file-discovery",
        pool_namespace: str = "default",
        log_path: str = DEFAULT_EPP_LOG,
    ) -> tuple[int, int]:
        """Start EPP, wait for its health port, return (grpc_port, health_port)."""
        binary = epp_binary()
        if not config_file:
            raise ValueError("EPP config file is required")
        if not os.path.isfile(binary):
            raise RuntimeError(f"EPP binary not found at {binary!r}")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"EPP config not found: {config_file!r}")

        pod_name = os.environ.get("POD_NAME", os.environ.get("HOSTNAME", _FALLBACK_POD_NAME))
        cmd = [
            binary,
            "--config-file", config_file,
            "--pool-name", pool_name,
            "--pool-namespace", pool_namespace,
            "--grpc-port", str(grpc_port),
            "--grpc-health-port", str(health_port),
            "--metrics-port", str(metrics_port),
            "--secure-serving=false",
            "--tracing=false",
            # burst-prefix-cache-producer (used by every burst-based mode: epp,
            # epp-p2p, epp-inflight, epp-fc - not p2p-specific) is Alpha-stability
            # in current EPP builds; the runner hard-fails config loading
            # ("Plugin stability validation failed") unless this is set.
            "--allow-experimental-plugins",
            f"-v={_env('EPP_VERBOSITY', '1')}",
        ]
        logger.info("[router_stack] starting EPP: %s", " ".join(cmd))
        self.epp_proc = subprocess.Popen(
            cmd,
            stdout=open(log_path, "w"),
            stderr=subprocess.STDOUT,
            env={**os.environ, "POD_NAME": pod_name},
        )
        await wait_port("127.0.0.1", health_port,
                        timeout=float(_env("EPP_START_TIMEOUT", str(DEFAULT_START_TIMEOUT))))
        return grpc_port, health_port

    async def start_envoy(
        self,
        config_file: str,
        *,
        port: int = DEFAULT_ENVOY_PORT,
        service_node: str = "envoy-proxy",
        concurrency: int = 8,
        log_path: str = DEFAULT_ENVOY_LOG,
    ) -> int:
        """Start Envoy, wait for its listener, return the port."""
        binary = envoy_binary()
        if not config_file:
            raise ValueError("Envoy config file is required")
        if not os.path.isfile(binary):
            raise RuntimeError(f"Envoy binary not found at {binary!r}")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"Envoy config not found: {config_file!r}")

        cmd = [
            binary,
            "--service-node", service_node,
            "--log-level", _env("ENVOY_LOG_LEVEL", "info"),
            "--concurrency", str(concurrency),
            "--drain-strategy", "immediate",
            "--drain-time-s", "60",
            "--disable-hot-restart",
            "-c", config_file,
        ]
        logger.info("[router_stack] starting Envoy: %s", " ".join(cmd))
        self.envoy_proc = subprocess.Popen(
            cmd, stdout=open(log_path, "w"), stderr=subprocess.STDOUT
        )
        await wait_port("127.0.0.1", port)
        return port

    def stop(self, timeout: float = 10.0) -> None:
        """Stop whatever is running, Envoy first so no request outlives its router.

        Reaps each child before returning (SIGKILL after timeout), so a caller
        that exits right after this does not orphan an EPP still holding its
        gRPC port - the next start would then fail to bind.
        """
        for proc in (self.envoy_proc, self.epp_proc):
            if proc is None or proc.poll() is not None:
                continue
            proc.terminate()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning("[router_stack] pid %d ignored SIGTERM, killing", proc.pid)
                proc.kill()
                proc.wait()
