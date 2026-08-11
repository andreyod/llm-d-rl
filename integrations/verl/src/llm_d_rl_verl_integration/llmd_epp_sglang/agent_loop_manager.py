"""AgentLoopManager that launches EPP and routes via gRPC ext-proc, for SGLang replicas.

SGLang variant of llmd_epp.agent_loop_manager - EPP as the endpoint picker,
verl/Ray dispatch directly to the picked replica, no proxy in the data path.
No PD/P2P support in this mode.

To use, set in the training YAML config:

    actor_rollout_ref:
      rollout:
        name: sglang               # verl built-in backend, no external_lib/registration needed
        agent:
          agent_loop_manager_class: llm_d_rl_verl_integration.llmd_epp_sglang.agent_loop_manager.SglangEPPRouterAgentLoopManager
        custom:
          epp_config_file: /path/to/epp-config.yaml   # same file as the vLLM path - backend-agnostic
          epp_endpoints_file: /tmp/epp-endpoints.yaml
          epp_grpc_port: 9002      # optional, default 9002
          epp_report_completion: true  # optional, default false - keep the ext_proc
                                        # stream open through generation and report
                                        # completion, so EPP's in-flight counter is
                                        # honest (needed for active-request-scorer /
                                        # a concurrency cap; adds one held-open stream
                                        # + 2 extra ext_proc messages per request)
"""

from __future__ import annotations

import logging

import ray
from omegaconf import OmegaConf

from llm_d_rl_verl_integration.base_agent_loop_manager import LlmdBaseAgentLoopManager
from llm_d_rl_verl_integration.llmd_actor import LlmdActor
from llm_d_rl_verl_integration.llmd_epp_sglang.llm_client import SglangEPPLLMClient
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)


class SglangEPPRouterAgentLoopManager(LlmdBaseAgentLoopManager):
    """Launches EPP subprocess (via a Ray actor) and swaps in SglangEPPLLMClient.

    Server actor handles are looked up by Ray actor name using the convention
    established by SGLangReplica.launch_servers(): ``"sglang_server_{rank}_0"``.
    server_addresses[i] from GlobalRequestLoadBalancer corresponds to
    replica_rank i (insertion order is preserved).
    """

    def _on_servers_ready(self, server_addresses: list[str]) -> None:
        rollout_cfg = self.rollout_config

        # Model name for EPP / generate body.
        self._model_name = self.model_config.path

        # Build address -> actor handle map.
        # server_addresses[i] is the address for replica_rank i;
        # SGLangReplica names its node-0 server actor "sglang_server_{i}_0".
        self._address_to_handle = {}
        for i, addr in enumerate(server_addresses):
            actor_name = f"sglang_server_{i}_0"
            try:
                self._address_to_handle[addr] = ray.get_actor(actor_name)
            except ValueError:
                raise RuntimeError(
                    f"Could not find Ray actor {actor_name!r} for server {addr}. "
                    "Make sure the rollout backend is sglang and servers are started."
                )
        logger.info("[SglangEPPRouterAgentLoopManager] address->handle map: %s", list(self._address_to_handle.keys()))

        # Launch EPP via a Ray actor pinned to the head node. LlmdActor.start()
        # writes the endpoints YAML itself (co-located with EPP); engine_type="sglang"
        # tags each entry so EPP's metrics extractor picks the SGLang Prometheus
        # metric-name mapping instead of the vLLM default.
        #
        # num_cpus=0: the SGLang manifest sets the head's Ray num-cpus to 0 (see
        # ray-cluster-sglang.yaml.tmpl) so verl's unpinned TaskRunnerV1 driver never
        # lands on this GPU-less node - importing SGLangReplica there crashes
        # (sgl_kernel needs libcuda.so.1). LlmdActor's default @ray.remote requests
        # 1 CPU, which a 0-CPU head can never satisfy, making the NodeAffinity pin
        # below infeasible; num_cpus=0 here means "pin to this node, need no CPU
        # slot" so EPP can still start there.
        epp_actor = LlmdActor.options(
            scheduling_strategy=self.head_node_strategy(),
            num_cpus=0,
        ).remote()

        self._grpc_addr = ray.get(
            epp_actor.start.remote(
                rollout_config=OmegaConf.to_container(rollout_cfg, resolve=True),
                server_addresses=server_addresses,
                model_config=OmegaConf.to_container(self.model_config, resolve=True),
                engine_type="sglang",
            )
        )
        self._epp_actor = epp_actor
        logger.info("[SglangEPPRouterAgentLoopManager] EPP ready at %s", self._grpc_addr)

    def _create_llm_client(self) -> LLMServerClient:
        custom = OmegaConf.to_container(self.rollout_config.get("custom") or {}, resolve=True)
        return SglangEPPLLMClient(
            config=self.config,
            load_balancer_handle=self.llm_client._load_balancer,
            grpc_addr=self._grpc_addr,
            address_to_handle=self._address_to_handle,
            model_name=self._model_name,
            report_completion=bool(custom.get("epp_report_completion", False)),
        )
