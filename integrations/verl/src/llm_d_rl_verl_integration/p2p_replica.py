# P2P KV-cache-sharing replica: symmetric (no prefill/decode split, unlike PD).
# Subclasses PDDecodeVLLMHttpServer and overrides only the sidecar launch
# (--kv-connector=offloading) and peer addressing.
#
# Each replica binds its P2P control socket on its own loopback IP
# (p2p_addressing.p2p_listener_host) with a flat port, so both dispatch paths
# address peers by host.
#
# VERL_P2P_NOSIDECAR skips the sidecar and POSTs kv_transfer_params straight to
# vLLM's native /inference/v1/generate.
#
# Requires kv_connector_extra_config spec_name=TieringOffloadingSpec and
# secondary_tiers=[{type:p2p}] (set by run_test.sh); without both there is no
# P2P tier and remote_kv_source is silently ignored.
#
# Design rationale: HANDOVER.md "CODE RATIONALE INDEX".
from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Optional

import ray

from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from llm_d_rl_common.endpoints import model_label as model_label_for_epp
from llm_d_rl_verl_integration.p2p_addressing import (
    DEFAULT_P2P_CONNECTOR_PORT,
    p2p_listener_host,
)
from llm_d_rl_verl_integration.pd_replica import (
    _SIDECAR_BINARY,
    _find_free_port,
    PDDecodeVLLMHttpServer,
    PDServerAdapter as P2PServerAdapter,
)

logger = logging.getLogger(__name__)

class P2PVLLMHttpServer(PDDecodeVLLMHttpServer):
    async def launch_server(self, master_address=None, master_port=None, dp_rpc_port=None):
        # Bind vLLM's P2P control socket on this replica's own loopback IP.
        # admission.py derives the same address when naming a migration source.
        os.environ["VLLM_P2P_SIDE_CHANNEL_HOST"] = p2p_listener_host(self.replica_rank)
        os.environ["VLLM_P2P_SIDE_CHANNEL_PORT"] = str(self._p2p_listener_port())
        await super().launch_server(
            master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port,
        )

    def _p2p_listener_port(self) -> int:
        """Port vLLM's P2P tier binds on this replica. Flat across replicas."""
        return int(os.environ.get("VERL_P2P_CONNECTOR_PORT", DEFAULT_P2P_CONNECTOR_PORT))

    @staticmethod
    def _nosidecar() -> bool:
        # "enabled" is what run_test.sh sets; Ray's runtime_env.env_vars rejects
        # the bool Hydra would infer from a bare "true".
        return os.environ.get("VERL_P2P_NOSIDECAR", "false").strip().lower() in (
            "1", "true", "yes", "enabled",
        )

    def _launch_sidecar(self) -> None:
        if self._nosidecar():
            logger.info(
                "VERL_P2P_NOSIDECAR set: skipping sidecar launch for replica %s, "
                "dispatching directly to vLLM's native endpoint instead.",
                self.replica_rank,
            )
            self._sidecar_port = None
            return
        sidecar_log_level = os.environ.get("VERL_SIDECAR_LOG_LEVEL", "1")
        vllm_port = self._server_port
        self._sidecar_port = _find_free_port()
        # The sidecar names peers by host only: it keeps extractHost(source) from
        # our header and overwrites the port with this flat value.
        p2p_port = self._p2p_listener_port()
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
        # POD_IP lets the sidecar recognise itself as the source and skip the pull.
        sidecar_env = dict(os.environ)
        sidecar_env["POD_IP"] = self._server_address
        self._sidecar_process = subprocess.Popen(
            cmd, stdout=self._sidecar_log, stderr=subprocess.STDOUT, env=sidecar_env
        )

    def get_server_address(self):
        assert self._server_port is not None, "server not launched"
        if self._nosidecar():
            # No sidecar exists in this mode - register vLLM's own port as the
            # dispatch/discovery address instead (impacts anything that reads
            # this: write_rollout_endpoints()/EPP file-discovery, vllm_scrape.py).
            return self._server_address, self._server_port
        return self._server_address, self._sidecar_port

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        sidecar_headers: Optional[dict] = None,
        **kwargs,
    ) -> TokenOutput:
        if self._nosidecar():
            return await self._generate_direct(
                prompt_ids, sampling_params, request_id,
                kv_transfer_params=kwargs.get("kv_transfer_params"),
            )
        return await super().generate(
            prompt_ids, sampling_params, request_id, sidecar_headers=sidecar_headers, **kwargs
        )

    async def _generate_direct(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        *,
        kv_transfer_params: Optional[dict] = None,
    ) -> TokenOutput:
        """Direct dispatch to vLLM's native POST /inference/v1/generate.

        Used when VERL_P2P_NOSIDECAR is set. Callers pass kv_transfer_params as
        {"remote_kv_source": {remote_host, remote_port, kv_request_id}}.
        """
        url = f"http://localhost:{self._server_port}/inference/v1/generate"
        body: dict[str, Any] = {
            "model": model_label_for_epp(self.model_config),
            "token_ids": prompt_ids,
            "sampling_params": self._prepare_sampling_params(sampling_params, prompt_ids),
        }
        if kv_transfer_params:
            body["kv_transfer_params"] = kv_transfer_params
            # WARNING level: logger.info() is filtered inside the Ray actor.
            if os.environ.get("VERL_P2P_DEBUG_BODY"):
                logger.warning(
                    "P2P_DEBUG_BODY replica=%s url=%s kv_transfer_params=%s",
                    self.replica_rank, url, kv_transfer_params,
                )

        session = await self._get_sidecar_session()  # a plain shared aiohttp session, not sidecar-specific despite the name
        try:
            async with session.request("POST", url, json=body) as resp:
                if not resp.ok:
                    error_body = await resp.text()
                    raise RuntimeError(
                        f"vLLM returned {resp.status}: {error_body} | "
                        f"kv_transfer_params={kv_transfer_params}"
                    )
                data = await resp.json()
        except Exception as e:
            logger.error(
                "_generate_direct() raised %s: %s - request_id=%s url=%s kv_transfer_params=%s",
                type(e).__name__, e, request_id, url, kv_transfer_params,
            )
            raise

        choices = data.get("choices") or []
        if choices:
            choice = choices[0]
            token_ids = [int(t) for t in (choice.get("token_ids") or [])]
            finish_reason = choice.get("finish_reason")
            logprobs_content = (choice.get("logprobs") or {}).get("content") or []
            log_probs = [e["logprob"] for e in logprobs_content] if logprobs_content else None
        else:
            token_ids, finish_reason, log_probs = [], None, None

        self._completed_requests += 1
        return TokenOutput(
            token_ids=token_ids,
            stop_reason=finish_reason,
            log_probs=log_probs,
            extra_fields={
                "global_steps": self.global_steps,
                # vLLM's echo of kv_transfer_params; non-null when a pull ran.
                "kv_transfer_params_response": data.get("kv_transfer_params"),
                # Connector-agnostic prefix-hit ground truth.
                "cached_tokens": (
                    (data.get("usage") or {}).get("prompt_tokens_details") or {}
                ).get("cached_tokens"),
            },
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
