"""Rollout backend registration for llm-d PD disaggregated vLLM.

Import this module via verl's model.external_lib hook so that the
("vllm-llmd-pd", "async") entry is present in _ROLLOUT_REGISTRY before
FSDP workers call get_rollout_class().

In the run script:
    actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_pd
"""

from verl.workers.rollout.base import _ROLLOUT_REGISTRY

_ROLLOUT_REGISTRY[("vllm-llmd-pd", "async")] = (
    "llm_d_rl_verl_integration.pd_replica.PDServerAdapter"
)
