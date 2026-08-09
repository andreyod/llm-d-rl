# P2P KV-cache-sharing replica: every rank is the same role (no prefill/decode
# split, unlike PD). Reuses PDDecodeVLLMHttpServer's local-sidecar mechanism as-is
# (launch_server(), generate(), get_server_address(), _prepare_sampling_params()) --
# the NIXL side-channel env vars it sets are harmless no-ops here since the engine
# uses OffloadingConnector, not NixlConnector. get_server_address() already returns
# the sidecar's own port (not vLLM's), so the endpoint file / EPP discovery list
# the sidecar's address automatically, with no endpoint-writing changes needed.
#
# Only _launch_sidecar() differs from PD: --kv-connector=offloading instead of
# nixlv2, plus --p2p-connector-port so the sidecar and vLLM's OffloadingConnector
# P2P tier agree on the pull port (p2pPullAvailable() in the sidecar is true
# whenever --kv-connector=offloading, so no --enable-p2p-pull or PD role split
# is needed for this aggregated case).
from __future__ import annotations

import logging
import os
import subprocess

import ray

from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from llm_d_rl_verl_integration.pd_replica import (
    _SIDECAR_BINARY,
    _find_free_port,
    PDDecodeVLLMHttpServer,
    PDServerAdapter as P2PServerAdapter,
)

logger = logging.getLogger(__name__)

_DEFAULT_P2P_CONNECTOR_PORT = 7777


class P2PVLLMHttpServer(PDDecodeVLLMHttpServer):
    def _launch_sidecar(self) -> None:
        sidecar_log_level = os.environ.get("VERL_SIDECAR_LOG_LEVEL", "1")
        vllm_port = self._server_port
        self._sidecar_port = _find_free_port()
        p2p_port = int(os.environ.get("VERL_P2P_CONNECTOR_PORT", _DEFAULT_P2P_CONNECTOR_PORT))
        cmd = [
            _SIDECAR_BINARY,
            f"--port={self._sidecar_port}",
            f"--vllm-port={vllm_port}",
            "--kv-connector=offloading",
            f"--p2p-connector-port={p2p_port}",
            "--secure-proxy=false",
            f"--zap-log-level={sidecar_log_level}",
        ]
        log_path = f"/tmp/sidecar-p2p-{self.replica_rank}.log"
        logger.info("Launching llm-d routing sidecar (P2P): %s (log: %s)", " ".join(cmd), log_path)
        self._sidecar_log = open(log_path, "w")
        self._sidecar_process = subprocess.Popen(
            cmd, stdout=self._sidecar_log, stderr=subprocess.STDOUT
        )


class P2PEngineReplica(vLLMReplica):
    def __init__(self, replica_rank, config, model_config, gpus_per_node=8, **kwargs):
        super().__init__(replica_rank, config, model_config, gpus_per_node, **kwargs)
        self.server_class = ray.remote(P2PVLLMHttpServer)
        self._engine_role = "p2p"

    async def launch_servers(self):
        await super().launch_servers()
        logger.info("P2P engine %s ready at %s (sidecar)", self.replica_rank, self._server_address)


def P2PEngineReplicaFactory(replica_rank, config, model_config, gpus_per_node=8, **kwargs):
    return P2PEngineReplica(replica_rank, config, model_config, gpus_per_node, **kwargs)
