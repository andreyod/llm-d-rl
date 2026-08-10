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
# IMPORTANT (2026-08-10): the sidecar path can only ever pull from rank 0 in this
# repo's topology (N replicas sharing one pod/network namespace) - the sidecar
# advertises a single flat P2P port while vLLM must bind rank r at base+r. See
# _launch_sidecar()'s comment for the full reasoning and why the sidecar's own
# --data-parallel-size rank-decoding mode cannot be used here. VERL_P2P_NOSIDECAR
# is the only path in this repo that can address every rank correctly.
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
from llm_d_rl_verl_integration.pd_replica import (
    _SIDECAR_BINARY,
    _find_free_port,
    PDDecodeVLLMHttpServer,
    PDServerAdapter as P2PServerAdapter,
)

logger = logging.getLogger(__name__)

_DEFAULT_P2P_CONNECTOR_PORT = 7777


class P2PVLLMHttpServer(PDDecodeVLLMHttpServer):
    async def launch_server(self, master_address=None, master_port=None, dp_rpc_port=None):
        # vLLM's OWN P2P-tier listener (vllm/v1/kv_offload/tiering/p2p/manager.py,
        # only reachable when kv_connector_extra_config sets spec_name=
        # TieringOffloadingSpec + secondary_tiers=[{type:p2p}] - see run_test.sh)
        # defaults to VLLM_P2P_SIDE_CHANNEL_HOST="localhost"/PORT=5710, neither
        # of which is right here: (1) host must be the real node IP, matching
        # what admission.py's kv_source already uses to address this replica -
        # "localhost" would silently never be reachable by a peer dialing the
        # node IP: mirrors NIXL's own VLLM_NIXL_SIDE_CHANNEL_HOST pattern right
        # below. (2) port must be offset by replica_rank: all N replicas here
        # are independent engines sharing ONE pod/network-namespace (unlike the
        # llm-d reference guide's one-pod-per-replica topology, where distinct
        # pod IPs make a single shared port harmless) - without a per-replica
        # offset every replica's listener would collide on the same port.
        # admission.py encodes this SAME base+rank scheme into kv_source, so the
        # two agree on the VERL_P2P_NOSIDECAR path. The SIDECAR path cannot be
        # made to agree - see _launch_sidecar()'s warning for why (it advertises
        # a flat port, and its rank-decoding mode is unusable in this topology).
        p2p_port = self._p2p_listener_port()
        os.environ["VLLM_P2P_SIDE_CHANNEL_HOST"] = self._server_address
        os.environ["VLLM_P2P_SIDE_CHANNEL_PORT"] = str(p2p_port)
        await super().launch_server(
            master_address=master_address, master_port=master_port, dp_rpc_port=dp_rpc_port,
        )

    def _p2p_listener_port(self) -> int:
        """The port vLLM's own P2P tier binds on THIS replica.

        Per-replica offset is mandatory, not a choice: every replica here is an
        independent engine process inside ONE pod/network namespace, so a single
        flat port cannot be bound more than once. (The llm-d reference guide's
        one-pod-per-replica topology has a distinct pod IP per replica, which is
        why a flat port is fine there and why nothing upstream needs this.)
        """
        return _DEFAULT_P2P_CONNECTOR_PORT + self.replica_rank

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
        p2p_port = int(os.environ.get("VERL_P2P_CONNECTOR_PORT", _DEFAULT_P2P_CONNECTOR_PORT))
        # KNOWN, UNFIXABLE-FROM-HERE LIMITATION of the sidecar path in this
        # topology: --p2p-connector-port is a single FLAT value that the sidecar
        # injects as remote_port for every P2P source it is asked about, while
        # vLLM necessarily binds rank r at p2p_port+r here (see
        # _p2p_listener_port()). So a pull whose true source is rank s>0 is told
        # to dial rank 0's tier, where it misses and silently falls back to a
        # full recompute. Only rank 0 can actually serve a sidecar-mediated pull.
        #
        # The sidecar DOES have rank-decoding logic, and it is unusable here.
        # Read from its source (pkg/sidecar/proxy/{connector_p2p,data_parallel,
        # proxy}.go): p2pPortFor() returns the flat base unless
        # --data-parallel-size > 1, in which case it returns
        # base + (target_sidecar_port - dpBasePort) where dpBasePort is THIS
        # sidecar's own --port. That mode assumes ONE sidecar per pod which
        # clones itself across contiguous ports (startDataParallel() binds
        # --port+1 .. --port+N-1 and proxies clone r to vLLM port
        # <vllm-port>+r). Two hard mismatches with this repo: (1) we run N
        # SEPARATE sidecars, one per replica, so passing --data-parallel-size=N
        # would make every one of them spawn N-1 clones and collide on the same
        # ports; (2) that mode requires CONTIGUOUS vLLM ports, but _server_port
        # is assigned per replica by verl's own vLLMHttpServer, not by us.
        # Adopting it would mean one sidecar owning all N replicas plus taking
        # over verl's port assignment - a restructure of the shared PD-derived
        # launch path, not a flag.
        #
        # Practical consequence: use VERL_P2P_NOSIDECAR for any run that needs
        # correct per-rank pulls. That path builds kv_transfer_params itself with
        # the true source's base+rank port (admission.py's _p2p_direct_port) and
        # never consults this rank-math at all.
        logger.warning(
            "P2P sidecar mode on replica %s: this sidecar will advertise a FLAT "
            "remote_port=%s for every P2P source, but vLLM binds rank r at %s+r "
            "in this shared-network-namespace topology (this replica: %s). Only "
            "rank 0 can serve a sidecar-mediated pull; a source on any other "
            "rank misses and silently recomputes. Set VERL_P2P_NOSIDECAR=enabled "
            "for correct per-rank pulls.",
            self.replica_rank, p2p_port, _DEFAULT_P2P_CONNECTOR_PORT,
            self._p2p_listener_port(),
        )
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
