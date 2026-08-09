"""AgentLoopManager that launches EPP and routes via gRPC ext-proc.

To use, set in the training YAML config:

  Non-PD (standard vllm):
    actor_rollout_ref:
      rollout:
        name: vllm
        agent:
          agent_loop_manager_class: llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager
        custom:
          epp_config_file: /path/to/epp-config.yaml
          epp_endpoints_file: /tmp/epp-endpoints.yaml
          epp_grpc_port: 9002      # optional, default 9002
          epp_report_completion: true  # optional, default false - keep the ext_proc
                                        # stream open through generation and report
                                        # completion, so EPP's in-flight counter is
                                        # honest (needed for active-request-scorer /
                                        # a concurrency cap; adds one held-open stream
                                        # + 2 extra ext_proc messages per request)

  PD disaggregated (llm-d vllm):
    actor_rollout_ref:
      rollout:
        name: vllm-llmd-pd          # registers PDEngineReplicaFactory at import time
        disaggregation:
          prefill_replicas: 2       # do NOT set enabled=True (avoids NotImplementedError from verl)
          decode_replicas: 2
        agent:
          agent_loop_manager_class: llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager
        custom:
          epp_config_file: /path/to/epp-config.yaml
          epp_endpoints_file: /tmp/epp-endpoints.yaml
          sidecar_connector: nixlv2

  P2P KV-cache sharing (llm-d vllm, aggregated - every replica both pulls and serves):
    actor_rollout_ref:
      rollout:
        name: vllm-llmd-p2p         # registers P2PEngineReplicaFactory at import time
        agent:
          agent_loop_manager_class: llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager
        custom:
          epp_config_file: /path/to/epp-config-p2p.yaml
          epp_endpoints_file: /tmp/epp-endpoints.yaml
"""

from __future__ import annotations

import logging

import ray
from omegaconf import OmegaConf

from llm_d_rl_verl_integration.base_agent_loop_manager import LlmdBaseAgentLoopManager
from llm_d_rl_verl_integration.llmd_actor import LlmdActor
from llm_d_rl_verl_integration.llmd_epp.llm_client import EPPLLMClient
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import RolloutReplicaRegistry
from llm_d_rl_verl_integration.pd_replica import PDEngineReplicaFactory
from llm_d_rl_verl_integration.p2p_replica import P2PEngineReplicaFactory


def _load_llmd_pd():
    return PDEngineReplicaFactory


def _load_llmd_p2p():
    return P2PEngineReplicaFactory


# Register vllm-llmd-pd / vllm-llmd-p2p at import time — this module is imported
# before FSDP workers call get_rollout_class().
RolloutReplicaRegistry.register("vllm-llmd-pd", _load_llmd_pd)
RolloutReplicaRegistry.register("vllm-llmd-p2p", _load_llmd_p2p)

logger = logging.getLogger(__name__)


class LlmdRouterAgentLoopManager(LlmdBaseAgentLoopManager):
    """Launches EPP subprocess (via a Ray actor) and swaps in EPPLLMClient.

    Server actor handles are looked up by Ray actor name using the convention
    established by vLLMReplica.launch_servers(): ``"vllm_server_{rank}_0"``.
    server_addresses[i] from GlobalRequestLoadBalancer corresponds to
    replica_rank i (insertion order is preserved).
    """

    def _on_servers_ready(self, server_addresses: list[str]) -> None:
        rollout_cfg = self.rollout_config
        custom = OmegaConf.to_container(rollout_cfg.get("custom") or {}, resolve=True)
        endpoints_file = custom.get("epp_endpoints_file")

        # Detect PD mode by backend name (not disaggregation.enabled, which we
        # intentionally leave False to avoid verl's sglang-only guard). P2P mode
        # is detected the same way; unlike PD it has no role split, so it never
        # calls infer_roles() and server_roles stays None (LlmdActor.start() then
        # takes the plain write_rollout_endpoints() path, same as non-sidecar EPP).
        self._pd_mode = rollout_cfg.name == "vllm-llmd-pd"
        self._p2p_mode = rollout_cfg.name == "vllm-llmd-p2p"

        server_roles = None
        if self._pd_mode:
            server_roles = self.infer_roles(server_addresses, rollout_cfg)

        # Model name for EPP / generate body.
        self._model_name = self.model_config.path

        # Build address → actor handle map.
        # server_addresses[i] is the address for replica_rank i;
        # vLLMReplica names its node-0 server actor "vllm_server_{i}_0".
        self._address_to_handle = {}
        for i, addr in enumerate(server_addresses):
            actor_name = f"vllm_server_{i}_0"
            try:
                self._address_to_handle[addr] = ray.get_actor(actor_name)
            except ValueError:
                raise RuntimeError(
                    f"Could not find Ray actor {actor_name!r} for server {addr}. "
                    "Make sure the rollout backend is vllm and servers are started."
                )
        logger.info("[LlmdRouterAgentLoopManager] address→handle map: %s", list(self._address_to_handle.keys()))

        # Launch EPP via a Ray actor pinned to the head node.
        epp_actor = LlmdActor.options(
            scheduling_strategy=self.head_node_strategy()
        ).remote()

        self._grpc_addr = ray.get(
            epp_actor.start.remote(
                rollout_config=OmegaConf.to_container(rollout_cfg, resolve=True),
                server_addresses=server_addresses,
                model_config=OmegaConf.to_container(self.model_config, resolve=True),
                server_roles=server_roles,
            )
        )
        self._epp_actor = epp_actor
        logger.info("[LlmdRouterAgentLoopManager] EPP ready at %s", self._grpc_addr)

    def _create_llm_client(self) -> LLMServerClient:
        custom = OmegaConf.to_container(self.rollout_config.get("custom") or {}, resolve=True)
        return EPPLLMClient(
            config=self.config,
            load_balancer_handle=self.llm_client._load_balancer,
            grpc_addr=self._grpc_addr,
            address_to_handle=self._address_to_handle,
            model_name=self._model_name,
            # Both PD and P2P dispatch to a Ray actor whose generate() HTTP-calls a
            # local sidecar and needs EPP's response headers forwarded to build that
            # request; plain (non-sidecar) EPP mode calls the vLLM actor directly and
            # ignores sidecar_headers regardless of this flag.
            use_sidecar=self._pd_mode or self._p2p_mode,
            report_completion=bool(custom.get("epp_report_completion", False)),
        )


