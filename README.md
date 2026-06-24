# llm-d RL verl Integration

Integrates [llm-d](https://github.com/llm-d/llm-d) into [verl](https://github.com/volcengine/verl) RL training rollouts, introducing llm-d's inference router and PD capabilities via llm-d's PD sidecar.

Both integrations are wired in through Hydra config - no verl source changes.
This repo provides two approaches:
1. EPP as the rollout router.
2. The llm-d stack as the inference backend.

---

## Integration points

During each training step verl drives generation through the following component hierarchy:

![verl generate call flow](diagrams/verl-generate-call-flow.png)

`LLMServerClient` is the object `AgentLoopWorker` calls for every generation request. verl's default implementation uses `GlobalRequestLoadBalancer` to select replicas by least in-flight requests, with sticky sessions for multi-turn continuity.

This integration replaces two components:

- **`AgentLoopManager`** - extended to start EPP (and optionally Envoy) as Ray actors pinned to the head node, and to inject a custom `LLMServerClient` into each `AgentLoopWorker`.
- **`LLMServerClient`** - replaced with `EPPLLMClient` or `EnvoyLLMClient`, both of which route through EPP's scoring system.

---

## Integration 1 - EPP as a router (direct gRPC)

### Overview

The goal of this integration is to use EPP as the routing strategy.
Each generation request is sent to the **Endpoint Picker Plugin (EPP)** via gRPC ext_proc. EPP scores all available vLLM replicas (prefix-cache hit rate, queue depth, KV utilization) and injects the chosen backend address as a header. The `EPPLLMClient` reads that header and forwards the request directly to the selected vLLM replica.

![epp generate call flow](diagrams/epp-generate-call-flow.png)

### How the lifecycle works

After all vLLM replicas are up:

1. `EPPAgentLoopManager` creates `LlmdActor` - a Ray actor **pinned to the head node** that starts EPP.
2. The actor:
   a. Writes the EPP endpoints YAML on the head node.
   b. Starts the EPP subprocess.
   c. Returns the EPP gRPC address.
3. `_create_llm_client` builds `EPPLLMClient` with that address.

**EPPLLMClient:**
When `.generate()` is called on `EPPLLMClient`, it sends a gRPC ext_proc request to the EPP subprocess. After receiving the selected endpoint, it maps the address to the matching inference engine replica's Ray actor handle and calls `.generate()` on it directly - the same path as verl's built-in `LLMClient`, but with EPP-driven replica selection instead of least-in-flight load balancing.

### Config variables

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `rollout.agent.agent_loop_manager_class` | yes | - | `llm_d_rl_verl_integration.epp_router.agent_loop_manager.EPPAgentLoopManager` |
| `rollout.custom.epp_config_file` | yes | - | Path to EPP YAML config (plugin list, scorers) |
| `rollout.custom.epp_endpoints_file` | yes | - | Path where the endpoints YAML is written; must match `epp_config_file` discovery path |
| `rollout.custom.epp_grpc_port` | no | `9002` | EPP gRPC ext_proc port |
| `rollout.custom.epp_grpc_health_port` | no | `9003` | EPP gRPC health check port |
| `rollout.custom.epp_pool_name` | no | `file-discovery` | EPP pool name |
| `rollout.custom.epp_pool_namespace` | no | `default` | EPP pool namespace |

Supports PD - see [PD Disaggregation](#pd-disaggregation---vllm-llmd-pd).

---

## Integration 2 - llm-d stack (Envoy + EPP, HTTP proxy)

### Overview

This integration uses the llm-d stack as the rollout backend: Envoy is treated as the single rollout endpoint (note: the LLM inference engines are still launched by verl).
All generation requests are sent to a single **Envoy** proxy endpoint. Envoy calls EPP via gRPC ext_proc to pick the best replica, then forwards the request to it. verl workers only ever speak HTTP to one address; all routing intelligence lives inside Envoy + EPP on the head node.

### How the lifecycle works

After all vLLM replicas are up:

1. `EnvoyAgentLoopManager` creates `LlmdActor` - a Ray actor **pinned to the head node** that starts EPP and Envoy.
2. The actor:
   a. Writes the EPP endpoints YAML on the head node.
   b. Starts the EPP subprocess.
   c. Starts the Envoy subprocess.
   d. Returns `<head-node-ip>:8081` as the Envoy address.
3. `_create_llm_client` builds `EnvoyLLMClient` with that address.

**EnvoyLLMClient:**
When `.generate()` is called on `EnvoyLLMClient`, it sends an HTTP request to the local Envoy proxy. Envoy calls EPP via gRPC ext_proc to select a replica, then forwards the request to it using an `ORIGINAL_DST` cluster driven by the address EPP injects into the response header.

### Config variables

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `rollout.agent.agent_loop_manager_class` | yes | - | `llm_d_rl_verl_integration.llmd_stack.agent_loop_manager.EnvoyAgentLoopManager` |
| `rollout.custom.epp_config_file` | yes | - | Path to EPP YAML config |
| `rollout.custom.epp_endpoints_file` | yes | - | Path where endpoints YAML is written |
| `rollout.custom.envoy_config` | no | bundled `envoy.yaml` | Path to Envoy config YAML |
| `rollout.custom.envoy_port` | no | `8081` | Envoy listener port |
| `rollout.custom.epp_grpc_port` | no | `9002` | EPP gRPC ext_proc port |
| `rollout.custom.epp_grpc_health_port` | no | `9003` | EPP gRPC health check port |
| `rollout.custom.epp_pool_name` | no | `file-discovery` | EPP pool name |
| `rollout.custom.epp_pool_namespace` | no | `default` | EPP pool namespace |

For PD disaggregated mode see [PD Disaggregation](#pd-disaggregation---vllm-llmd-pd).

---

## PD Disaggregation - `vllm-llmd-pd`

Both integrations support PD (prefill-decode) disaggregation via `rollout.name=vllm-llmd-pd`.

Replicas are split into prefill and decode roles by `PDEngineReplicaFactory` (registered as the `vllm-llmd-pd` backend in verl's `RolloutReplicaRegistry`). The first `prefill_replicas` ranks become prefill; the remaining become decode. `world_size / tp_size` must equal `prefill_replicas + decode_replicas`.

- **Prefill replicas** (`PDPrefillVLLMHttpServer`) - launch vLLM with NIXL side-channel env vars so the decode sidecar can pull KV blocks from them. They never serve generate requests directly.
- **Decode replicas** (`PDDecodeVLLMHttpServer`) - launch vLLM with NIXL env vars, then spawn `llm-d-routing-sidecar` alongside it. The sidecar is the public endpoint: it receives the request, fetches the prompt KV cache from the prefill replica via NIXL, then decodes locally. `get_server_address()` returns the sidecar port, so EPP routes to the sidecar, not to vLLM directly.

Role labels (`llm-d.ai/role: prefill` / `decode`) are written to the EPP endpoints YAML so EPP's `prefill-filter` and `decode-filter` plugins route correctly.

### Config

| Key | Required | Description |
|-----|----------|-------------|
| `rollout.name` | yes | `vllm-llmd-pd` |
| `rollout.disaggregation.prefill_replicas` | yes | Number of prefill replicas |
| `rollout.disaggregation.decode_replicas` | yes | Number of decode replicas |
| `rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector` | yes | `NixlConnector` |
| `rollout.engine_kwargs.vllm.kv_transfer_config.kv_role` | yes | `kv_both` |
| `rollout.engine_kwargs.vllm.no_disable_hybrid_kv_cache_manager` | yes | `true` |
| `rollout.custom.sidecar_connector` | no | KV connector type passed to `llm-d-routing-sidecar` (default: `nixlv2`) |
| `model.external_lib` | yes | `llm_d_rl_verl_integration.register_pd` - registers `vllm-llmd-pd` in FSDP workers |

The EPP config must use the PD-aware profile (example in `examples/configmap.yaml` - `epp-config-pd.yaml`).

---

## Code structure

```
.
├── push-epp.sh            # Push a freshly built / image-extracted EPP binary into the
│                          # running head pod with no verl rebuild and no pod recreation.
├── rl_orchestrate.sh      # Autonomous A/B run orchestrator with collection and
│                          # md5-verified cleanup.
│
├── docker/
│   └── Dockerfile.verl-vllm018-llm-d-integration  # Image with base verl, Envoy and sidecar.
│                          # EPP is NOT baked in - it is supplied at runtime (see below).
│
├── examples/
│   ├── README.md          # KubeRay deployment walkthrough - training script examples for all
│   │                      # four modes (EPP / Envoy x non-PD / PD) and 4-GPU variants
│   ├── ray-cluster.yaml   # RayCluster manifest - head + worker nodes
│   └── configmap.yaml     # ConfigMap with two EPP configs:
│                          #   epp-config.yaml    - standard prefix-cache routing
│                          #   epp-config-pd.yaml - PD-aware routing
│
└── llm_d_rl_verl_integration/           # Python package (pip install -e .)
    ├── base_agent_loop_manager.py        # LlmdAgentLoopManager base - Both integrations extend this.
    ├── llmd_actor.py                     # Ray actor pinned to the head node. Starts EPP and
    │                                     # optionally Envoy as subprocesses, writes the
    │                                     # endpoints YAML, returns service addresses.
    ├── endpoints.py                      # Builds the EPP endpoints YAML from vLLM server addresses.
    ├── pd_replica.py                     # PD-aware vLLM server classes.
    ├── register_pd.py                    # Imported via model.external_lib to register the
    │                                     # "vllm-llmd-pd" backend in FSDP worker processes.
    │
    ├── epp_router/                       # Integration 1 - EPP direct gRPC routing
    │   ├── agent_loop_manager.py         # EPPAgentLoopManager: starts LlmdActor and injects EPPLLMClient.
    │   ├── llm_client.py                 # EPPLLMClient: calls EPP gRPC to pick an endpoint,
    │   │                                 # then calls actor.generate() on the chosen replica.
    │   └── grpc_client.py                # Async gRPC ext_proc client.
    │
    └── llmd_stack/                       # Integration 2 - Envoy + EPP HTTP proxy routing
        ├── agent_loop_manager.py         # EnvoyAgentLoopManager: starts LlmdActor, injects EnvoyLLMClient.
        ├── llm_client.py                 # EnvoyLLMClient: sends all generate() calls as HTTP
        │                                 # to Envoy; Envoy calls EPP and forwards onward.
        └── envoy.yaml                    # Envoy config.
```

---

## Supplying the EPP at runtime

The EPP binary is intentionally **not** baked into the verl image (a ~28 GB build),
so iterating on the EPP never triggers a verl rebuild. `LlmdActor` launches whatever
binary `VERL_EPP_BINARY` points at (default `/usr/local/bin/epp`, set to
`/opt/epp-bin/epp` in `examples/ray-cluster.yaml`). There are two ways to get a binary
there:

- **On pod start (cold path):** the `fetch-epp` init container in `ray-cluster.yaml`
  extracts `/app/epp` from a separate, public EPP image (set via the `EPP_IMAGE` env in
  that container) into a shared `epp-bin` emptyDir. To change the EPP version a recreated
  pod boots with, bump that image tag and recreate the pod.
- **Into a running pod (fast inner loop):** `./push-epp.sh` builds the EPP from a local
  scheduler checkout (or extracts it from an image with `--from-image REF`) and
  `kubectl cp`s it into the head pod's `/opt/epp-bin/epp` - no verl rebuild and no pod
  recreation. Restart the verl run to pick it up, since EPP is started once per job.

The sidecar and Envoy stay baked into the image; override the sidecar at runtime via
`VERL_SIDECAR_BINARY` if you need to iterate on it too.

---

## How to run

See [examples/](examples/README.md) for a step-by-step KubeRay deployment walkthrough including manifests, EPP config setup, and training script examples.
