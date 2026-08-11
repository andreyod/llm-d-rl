"""Ray actor that starts EPP and optionally Envoy for the llm-d integrations.

Process launching lives in llm_d_rl_common.router_stack (framework-agnostic, no
ray import); this wrapper adds the Ray actor and maps verl's
``rollout.custom.*`` config onto it.
"""

from __future__ import annotations

import logging
from typing import Optional

import ray

from llm_d_rl_common.endpoints import write_pd_endpoints, write_rollout_endpoints
from llm_d_rl_common.router_stack import (
    DEFAULT_ENVOY_PORT,
    DEFAULT_EPP_GRPC_PORT,
    DEFAULT_EPP_HEALTH_PORT,
    RouterStack,
)

logger = logging.getLogger(__name__)


@ray.remote
class LlmdActor:
    """Ray actor pinned to the head node that starts EPP and optionally Envoy.

    llmd_epp integration:  start(..., with_envoy=False) -> returns EPP gRPC address
    llmd_serving integration:  start(..., with_envoy=True)  -> returns Envoy address
    """

    def __init__(self) -> None:
        self._stack = RouterStack()

    async def start(
        self,
        server_addresses: list[str],
        model_config: dict,
        rollout_config: dict,
        server_roles: Optional[list[str]] = None,
        with_envoy: bool = False,
        engine_type: str = "vllm",
    ) -> str:
        """Write endpoints, start EPP (and Envoy if with_envoy). Returns address for workers."""
        custom = rollout_config.get("custom") or {}

        # Write endpoints file on this node (co-located with EPP).
        endpoints_file = custom.get("epp_endpoints_file")
        if endpoints_file:
            pd_mode = rollout_config.get("name") == "vllm-llmd-pd"
            if pd_mode and server_roles and any(r is not None for r in server_roles):
                write_pd_endpoints(endpoints_file, server_addresses, server_roles, model_config)
            else:
                write_rollout_endpoints(endpoints_file, server_addresses, model_config, engine_type=engine_type)
            logger.info("[LlmdActor] wrote endpoints to %s", endpoints_file)

        if not custom.get("epp_config_file"):
            raise RuntimeError("rollout.custom.epp_config_file is required")
        epp_grpc_port, epp_health_port = await self._stack.start_epp(
            custom["epp_config_file"],
            grpc_port=int(custom.get("epp_grpc_port", DEFAULT_EPP_GRPC_PORT)),
            health_port=int(custom.get("epp_grpc_health_port", DEFAULT_EPP_HEALTH_PORT)),
            pool_name=custom.get("epp_pool_name", "file-discovery"),
            pool_namespace=custom.get("epp_pool_namespace", "default"),
        )
        logger.info("[LlmdActor] EPP ready on grpc=%d health=%d", epp_grpc_port, epp_health_port)

        if with_envoy:
            if not custom.get("envoy_config"):
                raise RuntimeError("rollout.custom.envoy_config is required")
            envoy_port = await self._stack.start_envoy(
                custom["envoy_config"],
                port=int(custom.get("envoy_port", DEFAULT_ENVOY_PORT)),
            )
            logger.info("[LlmdActor] Envoy ready on :%d", envoy_port)
            return f"{ray.util.get_node_ip_address()}:{envoy_port}"

        return f"{ray.util.get_node_ip_address()}:{epp_grpc_port}"
