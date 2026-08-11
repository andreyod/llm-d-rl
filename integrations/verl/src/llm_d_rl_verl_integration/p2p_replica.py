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
#
# PEER ADDRESSING (changed 2026-08-11): replicas are separated by IP, not port.
# Each replica's P2P control socket binds its own loopback alias
# (p2p_addressing.p2p_listener_host: 127.0.7.<rank+1>) on the FLAT base port
# 7777. This matters because the sidecar keeps only the HOST from our
# x-kv-cache-source-host-port header and replaces the port with its single
# --p2p-connector-port, so while replicas were separated by port every
# sidecar-mediated pull was aimed at rank 0 - missing, or dialling itself and
# raising NIXL_ERR_INVALID_PARAM. Separated by IP, one flat port is correct and
# BOTH paths (sidecar and VERL_P2P_NOSIDECAR) address peers correctly with no
# change to the sidecar's Go source. Single-node only - see p2p_addressing.py.
#
# Both paths ALSO require kv_connector_extra_config to set
# spec_name=TieringOffloadingSpec and secondary_tiers=[{type:p2p}] (run_test.sh
# sets both for --mode epp-p2p and --mode wave-admission-p2p). Without them vLLM
# builds a CPU-only offloading spec, no P2P tier exists, and any
# kv_transfer_params.remote_kv_source is discarded silently - no error, no log,
# no metric. Every P2P run in this repo before 2026-08-10 was in that state.
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

# Shared with wave_admission/admission.py, which must derive the SAME address
# when it names a migration's source - see p2p_addressing.py for the full
# rationale (one IP per replica, flat port, single-node only).
_DEFAULT_P2P_CONNECTOR_PORT = DEFAULT_P2P_CONNECTOR_PORT


class P2PVLLMHttpServer(PDDecodeVLLMHttpServer):
    async def launch_server(self, master_address=None, master_port=None, dp_rpc_port=None):
        # vLLM's OWN P2P-tier listener (vllm/v1/kv_offload/tiering/p2p/manager.py,
        # only reachable when kv_connector_extra_config sets spec_name=
        # TieringOffloadingSpec + secondary_tiers=[{type:p2p}] - see run_test.sh)
        # defaults to VLLM_P2P_SIDE_CHANNEL_HOST="localhost"/PORT=5710, and both
        # need overriding: bind a per-replica loopback alias (see
        # p2p_listener_host()) on the FLAT base port, so a peer is identified by
        # IP rather than by port. admission.py builds kv_source from the same
        # helper, so the two agree on BOTH the sidecar and nosidecar paths.
        #
        # Only the ZMQ CONTROL identity is affected. vLLM keeps two decoupled
        # identities (manager.py's `_local_id` vs `_nixl_agent_name`): the KV
        # bytes move over NixlTransport, addressed by VLLM_NIXL_SIDE_CHANNEL_HOST
        # (still the node IP, set by PD's launch_server below) and a per-process
        # uuid agent name, with transport chosen from UCX_TLS
        # (cuda_ipc first -> same-host GPU-to-GPU IPC). So this does not move the
        # data path onto loopback, and Ray is untouched entirely.
        os.environ["VLLM_P2P_SIDE_CHANNEL_HOST"] = p2p_listener_host(self.replica_rank)
        os.environ["VLLM_P2P_SIDE_CHANNEL_PORT"] = str(self._p2p_listener_port())
        await super().launch_server(
            master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port,
        )

    def _p2p_listener_port(self) -> int:
        """The port vLLM's own P2P tier binds on THIS replica.

        FLAT across replicas (no rank offset): replicas are separated by IP now,
        see p2p_listener_host(). A flat port is what upstream assumes and what
        the sidecar's single --p2p-connector-port can express.
        """
        return int(os.environ.get("VERL_P2P_CONNECTOR_PORT", _DEFAULT_P2P_CONNECTOR_PORT))

    @staticmethod
    def _nosidecar() -> bool:
        # "enabled" (not "true") is what run_test.sh's WAVE_ADMISSION_P2P_NOSIDECAR
        # toggle actually sets, via a Ray runtime_env env var - Hydra/OmegaConf's
        # CLI override parser infers bare "true" as a native bool, which Ray's
        # runtime_env.env_vars (requires plain str values) rejects at ray.init()
        # time. "1"/"true"/"yes" still accepted for direct (non-hydra-routed) use.
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
        p2p_port = self._p2p_listener_port()
        # Why a single FLAT --p2p-connector-port is now CORRECT here.
        #
        # The sidecar injects this one value as remote_port for every P2P source
        # it is asked about, and it DISCARDS the port in our
        # x-kv-cache-source-host-port header - read from its source
        # (pkg/sidecar/proxy/connector_p2p.go): p2pSourceParams() keeps only
        # extractHost(source) and overwrites the port with p2pPortFor(), which
        # short-circuits to the flat base whenever --data-parallel-size <= 1.
        # So the sidecar can only ever name peers by HOST.
        #
        # That used to be fatal, because we separated replicas by PORT
        # (base+rank on one shared pod IP): every pull was aimed at rank 0,
        # missed, and silently recomputed - and when the destination WAS rank 0
        # it dialled itself, producing NIXL_ERR_INVALID_PARAM / "remote agent
        # name same as local agent". Replicas are now separated by IP instead
        # (see p2p_listener_host()), so host alone identifies a peer and the
        # flat port is exactly right. This is also why the sidecar's
        # --data-parallel-size rank-decoding mode is no longer needed - which is
        # just as well, since enabling it makes every sidecar spawn N-1 clones
        # (startDataParallel()) that collide on their siblings' ports.
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
        # POD_IP is REQUIRED for the sidecar to recognise itself as the KV source
        # and skip the pull - it is not optional decoration. From its source
        # (pkg/sidecar/proxy/connector_p2p.go): `self := normalizeEndpoint(
        # net.JoinHostPort(os.Getenv("POD_IP"), s.config.Port))`, and a request
        # whose source == self is dispatched normally instead of running the P2P
        # source protocol. With POD_IP unset, `self` degenerates to ":<port>",
        # matches nothing, and the sidecar happily asks a replica to pull from
        # ITSELF - which surfaces as NIXL_ERR_INVALID_PARAM / "remote agent name
        # same as local agent" and appears to hang the request. The reference
        # guide (llm-d PR #2067's patch-sidecar.yaml) sets POD_IP for exactly
        # this reason ("POD_IP lets the sidecar recognize itself as the source
        # and skip self-pulls"); we never did.
        #
        # Note `self` is IP *and this sidecar's own port*, so with N sidecars per
        # pod on distinct ports the comparison still discriminates correctly -
        # sharing one pod IP does not break it.
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
            # Diagnostic for the "nosidecar pulls silently do nothing" problem:
            # proves whether the request LEAVES here with a well-formed source.
            # WARNING level on purpose - logger.info() is filtered out inside the
            # Ray actor, so an info-level line here would never be visible. Gated
            # by an env var so normal runs stay quiet; only fires on migrated
            # turns anyway (a few dozen per run).
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
                # Echoed back by vLLM verbatim when the pull actually happened -
                # the intended live-validation signal for this unconfirmed path.
                "kv_transfer_params_response": data.get("kv_transfer_params"),
                # Connector-agnostic ground truth (see pd_replica.py's
                # generate() for the same field) - checks this independent of
                # the kv_transfer_params echo above.
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
