"""Shared Ray actor that starts EPP and optionally Envoy for llm-d integrations."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
from typing import Any, Optional

import ray

from llm_d_rl_common.endpoints import write_pd_endpoints, write_rollout_endpoints

logger = logging.getLogger(__name__)

# Path to the EPP binary. Overridable via VERL_EPP_BINARY so the binary can be
# injected at runtime (fetch-binaries initContainer / push-epp.sh) instead of being
# baked into the verl image; iterating on the EPP then needs no verl rebuild.
_EPP_BINARY = os.environ.get("VERL_EPP_BINARY", "/usr/local/bin/epp")
# Envoy is injected at runtime (fetch-binaries initContainer) instead of baked
# into the verl image, same as the EPP; iterating on Envoy then needs no rebuild.
_ENVOY_BINARY = os.environ.get("VERL_ENVOY_BINARY", "/usr/local/bin/envoy")
_EPP_LOG = "/tmp/epp.log"
_ENVOY_LOG = "/tmp/envoy.log"
_DEFAULT_EPP_GRPC_PORT = 9002
_DEFAULT_EPP_HEALTH_PORT = 9003
_DEFAULT_ENVOY_PORT = 8081


async def _wait_port(host: str, port: int, timeout: float = 120.0) -> None:
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


@ray.remote
class LlmdActor:
    """Ray actor pinned to the head node that starts EPP and optionally Envoy.

    llmd_epp integration:  start(..., with_envoy=False) → returns EPP gRPC address
    llmd_serving integration:  start(..., with_envoy=True)  → returns Envoy address
    """

    def __init__(self) -> None:
        self._epp_proc: Optional[subprocess.Popen] = None
        self._envoy_proc: Optional[subprocess.Popen] = None

    async def start(
        self,
        server_addresses: list[str],
        model_config: dict,
        rollout_config: dict,
        server_roles: Optional[list[str]] = None,
        with_envoy: bool = False,
        engine_type: str = "vllm",
    ) -> str:
        """Write endpoints, start EPP (and Envoy if with_envoy). Returns address for workers."""
        custom = rollout_config.get("custom") or {}

        # Write endpoints file on this node (co-located with EPP).
        endpoints_file = custom.get("epp_endpoints_file")
        if endpoints_file:
            pd_mode = rollout_config.get("name") == "vllm-llmd-pd"
            if pd_mode and server_roles and any(r is not None for r in server_roles):
                write_pd_endpoints(endpoints_file, server_addresses, server_roles, model_config)
            else:
                write_rollout_endpoints(endpoints_file, server_addresses, model_config, engine_type=engine_type)
            logger.info("[LlmdActor] wrote endpoints to %s", endpoints_file)

        epp_grpc_port, epp_health_port = await self._start_epp(rollout_config, custom)
        logger.info("[LlmdActor] EPP ready on grpc=%d health=%d", epp_grpc_port, epp_health_port)

        if with_envoy:
            envoy_port = await self._start_envoy(custom)
            host = ray.util.get_node_ip_address()
            logger.info("[LlmdActor] Envoy ready on :%d", envoy_port)
            return f"{host}:{envoy_port}"

        host = ray.util.get_node_ip_address()
        return f"{host}:{epp_grpc_port}"

    async def _start_epp(self, rollout_config: dict, custom: dict):
        epp_config_file = custom.get("epp_config_file")
        if not epp_config_file:
            raise RuntimeError("rollout.custom.epp_config_file is required")
        if not os.path.isfile(_EPP_BINARY):
            raise RuntimeError(f"EPP binary not found at {_EPP_BINARY!r}")
        if not os.path.isfile(epp_config_file):
            raise RuntimeError(f"EPP config not found: {epp_config_file!r}")

        grpc_port = int(custom.get("epp_grpc_port", _DEFAULT_EPP_GRPC_PORT))
        health_port = int(custom.get("epp_grpc_health_port", _DEFAULT_EPP_HEALTH_PORT))
        pool_name = custom.get("epp_pool_name", "file-discovery")
        pool_namespace = custom.get("epp_pool_namespace", "default")
        pod_name = os.environ.get("POD_NAME", os.environ.get("HOSTNAME", "verl-epp-abc12-xyz34"))

        cmd = [
            _EPP_BINARY,
            "--config-file", epp_config_file,
            "--pool-name", pool_name,
            "--pool-namespace", pool_namespace,
            "--grpc-port", str(grpc_port),
            "--grpc-health-port", str(health_port),
            "--metrics-port", "9090",
            "--secure-serving=false",
            "--tracing=false",
            # burst-prefix-cache-producer (used by every burst-based mode: epp,
            # epp-p2p, epp-inflight, epp-fc - not p2p-specific) is Alpha-stability
            # in current EPP builds; the runner now hard-fails config loading
            # ("Plugin stability validation failed") unless this is set.
            "--allow-experimental-plugins",
            f"-v={os.environ.get('VERL_EPP_VERBOSITY', '1')}",
        ]
        env = {**os.environ, "POD_NAME": pod_name}
        logger.info("[LlmdActor] starting EPP: %s", " ".join(cmd))
        self._epp_proc = subprocess.Popen(
            cmd, stdout=open(_EPP_LOG, "w"), stderr=subprocess.STDOUT, env=env
        )
        timeout = float(os.environ.get("VERL_EPP_START_TIMEOUT", "120"))
        await _wait_port("127.0.0.1", health_port, timeout=timeout)
        return grpc_port, health_port

    async def _start_envoy(self, custom: dict) -> int:
        envoy_config = custom.get("envoy_config")
        if not envoy_config:
            raise RuntimeError("rollout.custom.envoy_config is required")
        envoy_port = int(custom.get("envoy_port", _DEFAULT_ENVOY_PORT))

        if not os.path.isfile(_ENVOY_BINARY):
            raise RuntimeError(f"Envoy binary not found at {_ENVOY_BINARY!r}")
        if not os.path.isfile(envoy_config):
            raise RuntimeError(f"Envoy config not found: {envoy_config!r}")

        cmd = [
            _ENVOY_BINARY,
            "--service-node", "envoy-proxy",
            "--log-level", os.environ.get("VERL_ENVOY_LOG_LEVEL", "info"),
            "--concurrency", "8",
            "--drain-strategy", "immediate",
            "--drain-time-s", "60",
            "--disable-hot-restart",
            "-c", envoy_config,
        ]
        logger.info("[LlmdActor] starting Envoy: %s", " ".join(cmd))
        self._envoy_proc = subprocess.Popen(
            cmd, stdout=open(_ENVOY_LOG, "w"), stderr=subprocess.STDOUT
        )
        await _wait_port("127.0.0.1", envoy_port)
        return envoy_port
