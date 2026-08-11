"""Base AgentLoopManager for llm-d integrations.

Subclasses override ``_create_llm_client()`` to return a custom client,
and optionally override ``_on_servers_ready()`` for extra setup work
(e.g. launching EPP).

No changes to verl core required — wire in via YAML:
    actor_rollout_ref.rollout.agent.agent_loop_manager_class: llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager
"""

from __future__ import annotations

import logging
import socket
from typing import Any

import ray
from omegaconf import DictConfig
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from verl.trainer.ppo.v1.agent_loop_tq import AgentLoopManagerTQ
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)


class LlmdBaseAgentLoopManager(AgentLoopManagerTQ):
    """Base class for llm-d AgentLoopManager variants.

    Lifecycle (runs in ``__init__``, before workers are spawned):
    1. ``super().__init__()`` — stores config, original llm_client.
    2. Retrieve server addresses from the GlobalRequestLoadBalancer actor.
    3. Call ``_on_servers_ready(server_addresses)`` — subclass hook for
       writing endpoints YAML, launching EPP, etc.
    4. Build a replacement client via ``_create_llm_client(server_addresses)``.
    5. Replace ``self.llm_client`` so workers receive the new client.
    """

    def __init__(
        self,
        config: DictConfig,
        llm_client: LLMServerClient,
        teacher_client=None,
        reward_loop_worker_handles=None,
    ):
        super().__init__(
            config=config,
            llm_client=llm_client,
            teacher_client=teacher_client,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        server_addresses: list[str] = ray.get(llm_client._load_balancer.get_all_servers.remote())
        logger.info("[LlmdBaseAgentLoopManager] servers: %s", server_addresses)

        self._on_servers_ready(server_addresses)

        new_client = self._create_llm_client()
        if new_client is not None:
            self.llm_client = new_client

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _on_servers_ready(self, server_addresses: list[str]) -> None:
        """Called after server addresses are retrieved, before client creation.

        Override to write YAML files, launch EPP, etc.
        Default implementation does nothing.
        """

    def _create_llm_client(self) -> LLMServerClient | None:
        """Create and return the replacement LLMServerClient.

        Return ``None`` to keep the original verl client unchanged.
        """
        return None

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def head_node_strategy() -> NodeAffinitySchedulingStrategy:
        """Return a scheduling strategy that pins a Ray actor to the head node."""
        gcs_address = ray.get_runtime_context().gcs_address
        head_host = gcs_address.split(":")[0]
        # KubeRay sets RAY_ADDRESS to the head Service DNS name (not an IP), which
        # ray.get_runtime_context().gcs_address then reflects verbatim. Resolving
        # only matters when this call runs on a node other than the head itself
        # (e.g. verl's TaskRunnerV1 driver, unpinned, landing on a GPU worker) -
        # from the head, gcs_address is already the head's own IP. Resolve via DNS
        # first so both cases match a NodeManagerAddress; fall back to the literal
        # string if it's already an IP (gethostbyname is a no-op on IP literals).
        try:
            head_ip = socket.gethostbyname(head_host)
        except socket.gaierror:
            head_ip = head_host
        for node in ray.nodes():
            if node["Alive"] and node["NodeManagerAddress"] == head_ip:
                return NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        raise RuntimeError(
            f"Could not find a live Ray node with GCS IP {head_ip} (resolved from {head_host!r}). "
            f"Nodes: {[n['NodeManagerAddress'] for n in ray.nodes()]}"
        )

    @staticmethod
    def infer_roles(server_addresses: list[str], rollout_cfg: Any) -> list[str]:
        """Infer prefill/decode roles from disaggregation config."""
        disagg = rollout_cfg.disaggregation
        n_prefill = int(getattr(disagg, "prefill_replicas", 1))
        n_decode = int(getattr(disagg, "decode_replicas", 1))
        roles = ["prefill"] * n_prefill + ["decode"] * n_decode
        if len(roles) < len(server_addresses):
            roles += ["decode"] * (len(server_addresses) - len(roles))
        return roles[: len(server_addresses)]
