"""EPP as the endpoint picker, with SGLang replicas instead of vLLM.

Same flow as llmd_epp.agent_loop_manager (EPP picks, verl/Ray dispatches straight
to the picked replica, no proxy in the data path); only the four class attributes
below differ. No PD/P2P support in this mode.

To use, set in the training YAML config:

    actor_rollout_ref:
      rollout:
        name: sglang               # verl built-in backend, no external_lib needed
        agent:
          agent_loop_manager_class: llm_d_rl_verl_integration.llmd_epp_sglang.agent_loop_manager.SglangEPPRouterAgentLoopManager
        custom:
          epp_config_file: /path/to/epp-config.yaml   # the same file the vLLM path uses
          epp_endpoints_file: /tmp/epp-endpoints.yaml
          epp_grpc_port: 9002      # optional, default 9002
          epp_report_completion: true  # optional, default false - see llmd_epp
"""

from __future__ import annotations

from llm_d_rl_verl_integration.llmd_epp.agent_loop_manager import LlmdRouterAgentLoopManager
from llm_d_rl_verl_integration.llmd_epp_sglang.llm_client import SglangEPPLLMClient


class SglangEPPRouterAgentLoopManager(LlmdRouterAgentLoopManager):
    """EPP routing over SGLang replicas.

    epp_actor_options num_cpus=0: with --engine sglang the manifest sets the head's
    Ray num-cpus to "0" so verl's unpinned TaskRunnerV1 driver never lands on the
    GPU-less head (importing SGLangReplica needs libcuda.so.1). LlmdActor's default
    1-CPU request would then make its NodeAffinity pin to that head infeasible;
    num_cpus=0 means "pin here, need no CPU slot".

    PD/P2P detection in the parent is a no-op here: it keys off rollout.name being
    vllm-llmd-pd / vllm-llmd-p2p, and this backend is plain "sglang".
    """

    server_actor_prefix = "sglang_server"
    engine_type = "sglang"
    epp_actor_options = {"num_cpus": 0}
    client_cls = SglangEPPLLMClient
