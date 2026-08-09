"""Rollout backend registration for llm-d P2P KV-cache sharing (aggregated, no
prefill/decode split - every replica both pulls and serves via a local sidecar
in --kv-connector=offloading mode).

Import this module via verl's model.external_lib hook so both of verl's rollout
registries have the "vllm-llmd-p2p" entry before anything looks it up:

  - _ROLLOUT_REGISTRY (verl.workers.rollout.base): consulted early, during FSDP
    worker model-config instantiation (HFModelConfig etc.) - before FSDP workers
    call get_rollout_class().
  - RolloutReplicaRegistry (verl.workers.rollout.replica): consulted later, when
    TaskRunnerV1 actually launches the replica servers - this happens BEFORE
    llmd_epp.agent_loop_manager gets imported (that module's own identical
    RolloutReplicaRegistry.register call is too late to matter for this lookup;
    kept there too since it's a harmless duplicate registration, not the
    authoritative one).

In the run script:
    actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_p2p
"""

from verl.workers.rollout.base import _ROLLOUT_REGISTRY
from verl.workers.rollout.replica import RolloutReplicaRegistry

_ROLLOUT_REGISTRY[("vllm-llmd-p2p", "async")] = (
    "llm_d_rl_verl_integration.p2p_replica.P2PServerAdapter"
)


def _load_llmd_p2p():
    from llm_d_rl_verl_integration.p2p_replica import P2PEngineReplicaFactory
    return P2PEngineReplicaFactory


RolloutReplicaRegistry.register("vllm-llmd-p2p", _load_llmd_p2p)
