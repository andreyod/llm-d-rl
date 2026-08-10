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
#
# Optional VERL_P2P_NOSIDECAR mode: when our own client already knows the exact
# migration source (see wave_admission/admission.py's _continue_or_migrate - it
# tracks its own sticky `_resident` map, no EPP p2p-source-producer discovery
# needed), the sidecar's only remaining job is translating one HTTP header into
# vLLM's own `kv_transfer_params.remote_kv_source` body field before forwarding
# to vLLM's native `/inference/v1/generate` endpoint - confirmed via direct vLLM
# source inspection (entrypoints/scale_out/token_in_token_out/protocol.py) that
# `kv_transfer_params` is a first-class field on that endpoint's own request
# schema, and the response shape it returns is identical either way (same
# choices[0].token_ids/finish_reason/logprobs.content the sidecar-mediated path
# already parses - the sidecar does not reshape the response). So the sidecar
# hop is skippable for THIS specific use case (not the general EPP-driven one,
# where an external caller has no way to set kv_transfer_params itself).
# NOT LIVE-VALIDATED as of this writing - see _generate_direct()'s docstring for
# the specific unconfirmed assumption (which port a peer's P2P listener is
# actually reachable on).
from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Optional

import ray

from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMReplica

from llm_d_rl_common.endpoints import model_label as model_label_for_epp
from llm_d_rl_verl_integration.pd_replica import (
    _SIDECAR_BINARY,
    _find_free_port,
    PDDecodeVLLMHttpServer,
    PDServerAdapter as P2PServerAdapter,
)

logger = logging.getLogger(__name__)

_DEFAULT_P2P_CONNECTOR_PORT = 7777


class P2PVLLMHttpServer(PDDecodeVLLMHttpServer):
    @staticmethod
    def _nosidecar() -> bool:
        return os.environ.get("VERL_P2P_NOSIDECAR", "false").strip().lower() in ("1", "true", "yes")

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
        """Optional direct-to-vLLM dispatch, bypassing the sidecar entirely.

        Only used when VERL_P2P_NOSIDECAR is set. Talks straight to vLLM's own
        native `POST /inference/v1/generate` (confirmed route + request/response
        schema via direct vLLM source inspection of
        entrypoints/scale_out/token_in_token_out/{protocol,api_router}.py on
        this exact build) instead of going through the sidecar's HTTP proxy +
        header-to-body translation step.

        UNCONFIRMED ASSUMPTION (flagging explicitly, not verified by a live
        successful pull as of this writing): callers of this method are
        expected to build `kv_transfer_params` as
        `{"remote_kv_source": {"remote_host": ..., "remote_port": ...,
        "kv_request_id": ...}}` - the exact shape vLLM's own P2P manager parses
        (vllm/v1/kv_offload/tiering/p2p/manager.py's `_parse_source`). The
        `remote_port` value there is assumed to be vLLM's own P2P-tier listener
        port (`vllm.envs.VLLM_P2P_SIDE_CHANNEL_PORT`, default 5710 - NOT this
        module's `_DEFAULT_P2P_CONNECTOR_PORT`/7777, which is a sidecar-only
        proxy port unrelated to vLLM's native listener) - this repo has never
        needed to reason about that port directly before, since the sidecar
        mediated it. Whether vLLM's P2P listener is even the same code path the
        sidecar's mechanism has been validated against this session is NOT
        confirmed; the sidecar may be acting as more than a header-injection
        proxy. Treat any run using this path as unvalidated until confirmed via
        a live pull (e.g. checking the response's own `kv_transfer_params` echo,
        or a gen_s timing signature consistent with a real pull vs a silent
        fall-back to full recompute).
        """
        url = f"http://localhost:{self._server_port}/inference/v1/generate"
        body: dict[str, Any] = {
            "model": model_label_for_epp(self.model_config),
            "token_ids": prompt_ids,
            "sampling_params": self._prepare_sampling_params(sampling_params, prompt_ids),
        }
        if kv_transfer_params:
            body["kv_transfer_params"] = kv_transfer_params

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
                # Echoed back by vLLM verbatim when the pull actually happened -
                # the intended live-validation signal for this unconfirmed path.
                "kv_transfer_params_response": data.get("kv_transfer_params"),
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
