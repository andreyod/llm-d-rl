"""AgentLoopManager for wave-based admission control (Phase 1 of the
wave-admission + migration/offload plan -
see ``~/.claude/plans/steady-splashing-blanket.md``). Gates NEW conversation
admission by an ESTIMATE of per-replica free KV budget instead of routing
through verl's own load balancer or an external EPP; no cross-replica
migration (sticky after admission). Implements findings #5/#6 of
``~/work/rl-work/agentic/simulator2/FINDINGS.md`` for real, against live
vLLM replicas.

To use, set in the training YAML config:

    actor_rollout_ref:
      rollout:
        name: vllm
        agent:
          agent_loop_manager_class: llm_d_rl_verl_integration.wave_admission.agent_loop_manager.WaveAdmissionAgentLoopManager
        custom:
          epp_endpoints_file: /tmp/epp-endpoints.yaml       # optional, for the /metrics scraper
          wave_admission_wave1_size: 128                    # optional, default 128
          wave_admission_gpu_capacity_gb: 139.8              # optional
          wave_admission_gpu_util: 0.6                       # optional
          wave_admission_weights_gb: 15.2                    # optional
          wave_admission_kv_bytes_per_token: 57344           # optional
          wave_admission_initial_growth_guess: 100000        # optional
          wave_admission_prior_weight: 15                    # optional
          wave_admission_max_wait_s: 60                      # optional
          wave_admission_poll_interval_s: 0.5                # optional
          wave_admission_allow_migration: true               # optional, default true
          wave_admission_reserve_mode: size                  # optional, "turn"|"size"|"turn_size", default "size"
          wave_admission_reserve_z: 1.5                      # optional, default 1.5
          wave_admission_migration_cost_ratio: 1.0           # optional, default 1.0
          wave_admission_p2p_kv_available: false             # optional, default false - set true with --mode wave-admission-p2p
          wave_admission_p2p_connector_port: 7777             # optional, must match P2PVLLMHttpServer's --p2p-connector-port
          wave_admission_migration_cost_ratio_p2p: 0.0        # optional, default 0.0 (benchmarking assumption: ~free P2P pull)
          wave_admission_p2p_nosidecar: false                 # optional, default false - EXPERIMENTAL, not live-validated (see p2p_replica.py's _generate_direct() docstring); bypasses the sidecar and calls vLLM's native endpoint directly
          wave_admission_p2p_direct_port: 7777                # optional, default 7777; only used when p2p_nosidecar=true. vLLM's P2P tier binds this FLAT port on a per-replica loopback IP (see p2p_addressing.py), so it normally equals wave_admission_p2p_connector_port
"""

from __future__ import annotations

import logging
from typing import Any

import ray
from omegaconf import OmegaConf

from llm_d_rl_verl_integration.base_agent_loop_manager import LlmdBaseAgentLoopManager
from llm_d_rl_verl_integration.p2p_addressing import DEFAULT_P2P_CONNECTOR_PORT
from llm_d_rl_verl_integration.wave_admission.admission import (
    AdmissionLedger,
    compute_budget_tokens_per_replica,
)
from llm_d_rl_verl_integration.wave_admission.llm_client import WaveAdmissionLLMClient
from llm_d_rl_common.endpoints import write_rollout_endpoints
from verl.workers.rollout.llm_server import LLMServerClient

logger = logging.getLogger(__name__)

# Default matches what vllm_scrape.py reads and what EPP/native modes write.
_DEFAULT_ENDPOINTS_FILE = "/tmp/epp-endpoints.yaml"


def _custom_get(custom: dict[str, Any], key: str, default: Any) -> Any:
    val = custom.get(key, default)
    return default if val is None else val


class WaveAdmissionAgentLoopManager(LlmdBaseAgentLoopManager):
    """Wave-based admission control, sticky-after-admit, no migration (Phase 1)."""

    def _on_servers_ready(self, server_addresses: list[str]) -> None:
        custom = OmegaConf.to_container(self.rollout_config.get("custom") or {}, resolve=True)
        endpoints_file = _custom_get(custom, "epp_endpoints_file", _DEFAULT_ENDPOINTS_FILE)
        write_rollout_endpoints(endpoints_file, server_addresses, self.model_config)

        # Build address -> actor handle map, same convention
        # LlmdRouterAgentLoopManager uses: vLLMReplica.launch_servers() names
        # each replica's node-0 server actor "vllm_server_{rank}_0", and
        # server_addresses[i] corresponds to replica_rank i.
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

        budget = compute_budget_tokens_per_replica(
            gpu_capacity_gb=_custom_get(custom, "wave_admission_gpu_capacity_gb", None),
            gpu_memory_utilization=_custom_get(custom, "wave_admission_gpu_util", None),
            weights_gb=_custom_get(custom, "wave_admission_weights_gb", None),
            bytes_per_token=_custom_get(custom, "wave_admission_kv_bytes_per_token", None),
        )
        wave1_size = int(_custom_get(custom, "wave_admission_wave1_size", 128))
        initial_growth_guess = float(_custom_get(custom, "wave_admission_initial_growth_guess", 100_000.0))
        prior_weight = float(_custom_get(custom, "wave_admission_prior_weight", 15.0))
        max_wait_s = float(_custom_get(custom, "wave_admission_max_wait_s", 60.0))
        poll_interval_s = float(_custom_get(custom, "wave_admission_poll_interval_s", 0.5))
        allow_reactive_migration = bool(_custom_get(custom, "wave_admission_allow_migration", True))
        reserve_mode = str(_custom_get(custom, "wave_admission_reserve_mode", "size"))
        reserve_z = float(_custom_get(custom, "wave_admission_reserve_z", 1.5))
        migration_cost_ratio = float(_custom_get(custom, "wave_admission_migration_cost_ratio", 1.0))
        p2p_kv_available = bool(_custom_get(custom, "wave_admission_p2p_kv_available", False))
        p2p_connector_port = int(_custom_get(custom, "wave_admission_p2p_connector_port", 7777))
        migration_cost_ratio_p2p = float(_custom_get(custom, "wave_admission_migration_cost_ratio_p2p", 0.0))
        p2p_nosidecar = bool(_custom_get(custom, "wave_admission_p2p_nosidecar", False))
        # Default 7777 (DEFAULT_P2P_CONNECTOR_PORT), NOT vLLM's own 5710: with
        # replicas separated by loopback IP the tier binds the flat llm-d/sidecar
        # convention port on every replica - see p2p_addressing.py.
        p2p_direct_port = int(
            _custom_get(custom, "wave_admission_p2p_direct_port", DEFAULT_P2P_CONNECTOR_PORT)
        )
        # Read by _create_llm_client() below, which has no access to `custom`.
        self._p2p_nosidecar = p2p_nosidecar

        logger.info(
            "[WaveAdmissionAgentLoopManager] %d replicas, budget=%.0f tok/replica, "
            "wave1_size=%d, initial_growth_guess=%.0f, prior_weight=%.1f, max_wait_s=%.0f, "
            "allow_reactive_migration=%s, reserve_mode=%s, reserve_z=%.1f, migration_cost_ratio=%.1f, "
            "p2p_kv_available=%s, migration_cost_ratio_p2p=%.2f, p2p_nosidecar=%s, p2p_direct_port=%d",
            len(server_addresses), budget, wave1_size, initial_growth_guess, prior_weight, max_wait_s,
            allow_reactive_migration, reserve_mode, reserve_z, migration_cost_ratio,
            p2p_kv_available, migration_cost_ratio_p2p, p2p_nosidecar, p2p_direct_port,
        )

        # Pin the ledger to the head node (same rationale as LlmdActor: one
        # long-lived actor the whole fleet of AgentLoopWorkers shares).
        self._admission_ledger = AdmissionLedger.options(
            scheduling_strategy=self.head_node_strategy()
        ).remote(
            replicas=server_addresses,
            budget_tokens_per_replica=budget,
            wave1_size=wave1_size,
            initial_growth_guess=initial_growth_guess,
            prior_weight=prior_weight,
            max_wait_s=max_wait_s,
            poll_interval_s=poll_interval_s,
            allow_reactive_migration=allow_reactive_migration,
            reserve_mode=reserve_mode,
            reserve_z=reserve_z,
            migration_cost_ratio=migration_cost_ratio,
            p2p_kv_available=p2p_kv_available,
            p2p_connector_port=p2p_connector_port,
            migration_cost_ratio_p2p=migration_cost_ratio_p2p,
            p2p_nosidecar=p2p_nosidecar,
            p2p_direct_port=p2p_direct_port,
        )

    def _create_llm_client(self) -> LLMServerClient:
        return WaveAdmissionLLMClient(
            config=self.config,
            load_balancer_handle=self.llm_client._load_balancer,
            address_to_handle=self._address_to_handle,
            admission_ledger=self._admission_ledger,
            p2p_nosidecar=self._p2p_nosidecar,
        )
