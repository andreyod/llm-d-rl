# llm-d RL verl Integration

This package integrates [llm-d](https://github.com/llm-d/llm-d)'s inference router into [verl](https://github.com/volcengine/verl) RL training rollouts. During each training step verl generates completions from a set of vLLM replicas; this integration replaces verl's default round-robin replica selection with llm-d's **Endpoint Picker Plugin (EPP)**, which routes each request to the replica most likely to have its KV cache already warm — reducing redundant computation and improving throughput in large-group RL workloads (GRPO, PPO with large rollout groups).

No verl source changes are required. Everything is wired in through Hydra config.

---

## Setup

### Prerequisites

- Ray cluster with at least one GPU worker node
- verl with a compatible vLLM version (available in verl's [official images](https://hub.docker.com/r/verlai/verl))

### 1. Set up verl

The examples in this repo use `verlai/verl:vllm018.dev1`, tested on verl commit `334d9f8b03816382cbd72e898bc8ae04efca6fbe`. Other versions may work but have not been validated.

### 2. Install the integration Python package

The package must be installed on every node in the Ray cluster — both the head node and all worker nodes.

Install from PyPI or a git ref:

```bash
pip install git+https://github.com/llm-d-incubation/llm-d-rl.git
```

Or, without installing, clone the repo and add the source to your Python path:

```bash
git clone https://github.com/llm-d-incubation/llm-d-rl.git
export PYTHONPATH=$(pwd)/llm-d-rl/src:$PYTHONPATH
```

### 3. Get the EPP, Envoy, and sidecar binaries

The integration launches three external processes at runtime. None of these binaries are baked into the verl image (a ~28 GB build), so iterating on any of them never triggers a verl rebuild. Obtain them from the published images or build from source. At runtime, `LlmdActor` on the head node launches EPP and Envoy via `VERL_EPP_BINARY` and `VERL_ENVOY_BINARY`; on each worker, decode replicas launch the sidecar via `VERL_SIDECAR_BINARY` (PD mode only). Point the integration at the binaries via env vars (set before starting Ray, or in `ray.init` runtime env):

| Env var | Default | Binary | Node | Required for |
|---------|---------|--------|------|-------------|
| `VERL_EPP_BINARY` | `/usr/local/bin/epp` | EPP scorer | head | All integrations |
| `VERL_ENVOY_BINARY` | `/usr/local/bin/envoy` | Envoy proxy | head | Integration 2 only |
| `VERL_SIDECAR_BINARY` | `/opt/llm-d-bins/pd-sidecar` | llm-d routing sidecar | workers (decode replicas) | PD disaggregation only |

### 4. Copy the config files

The following starting-point configs are provided in this repo — copy them to somewhere accessible on the head node and modify as needed for your workload:

- `deploy/epp-config.yaml` — EPP scorer config (standard routing)
- `deploy/epp-config-pd.yaml` — EPP scorer config (PD disaggregated routing)
- `deploy/envoy.yaml` — Envoy proxy config (Integration 2 only)

The endpoints YAML holds the list of available vLLM replica addresses. `LlmdActor` writes it at startup to the path set by the `epp_endpoints_file` Hydra override (step 5). The EPP config's `file-discovery` plugin has a `path:` field that tells EPP where to read it from — these two paths must match. The default in the provided config is `/tmp/epp-endpoints.yaml`.

### 5. Run training

Running the integration requires adding a few Hydra overrides to your existing verl training command/script — no other changes needed. The overrides tell verl which `AgentLoopManager` class to use and where to find the config files. See the Hydra overrides under [Integration 1](#integration-1--epp-router-direct-grpc) and [Integration 2](#integration-2--llm-d-stack-envoy--epp-http-proxy) below.


### KubeRay example

[`kuberay/`](kuberay/README.md) contains a complete end-to-end example on Kubernetes using KubeRay with automated scripts for deployment, training, and benchmarking.

---

## Integrations

During each training step, verl drives generation through the following component hierarchy:

![verl generate call flow](docs/diagrams/verl-generate-call-flow.png)

`LLMServerClient` is the object `AgentLoopWorker` calls for every generation request. verl's default implementation uses `GlobalRequestLoadBalancer` to select replicas by least in-flight requests.

This integration replaces two components, both wired in via a single Hydra key with no verl patches required:

- **`AgentLoopManager`** — extended to start EPP (and optionally Envoy) as Ray actors pinned to the head node, and to inject a custom `LLMServerClient` into each `AgentLoopWorker`.
- **`LLMServerClient`** — replaced with `EPPLLMClient` (Integration 1) or `EnvoyLLMClient` (Integration 2), both of which route through EPP's scoring system.

### Integration 1 — EPP router (direct gRPC)

Each generation request is sent to EPP via gRPC ext_proc. EPP scores all available vLLM replicas (prefix-cache hit rate, queue depth, KV utilization) and returns the chosen backend address. `EPPLLMClient` then calls that replica's Ray actor directly — the same path as verl's built-in client, but with EPP-driven selection.

![epp generate call flow](docs/diagrams/epp-generate-call-flow.png)

**Startup sequence**

After all vLLM replicas are up:

1. `LlmdRouterAgentLoopManager` spawns `LlmdActor` — a Ray actor pinned to the head node.
2. `LlmdActor` writes the EPP endpoints YAML, starts the EPP subprocess, waits for its health port, and returns the gRPC address.
3. `LlmdRouterAgentLoopManager` builds `EPPLLMClient` with that address and replaces `self.llm_client` — workers receive it before any generation begins.

#### Hydra overrides

To use this integration, pass these as Hydra overrides to your verl training command. Required keys must be set; optional keys have working defaults.

```bash
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.LlmdRouterAgentLoopManager \
+actor_rollout_ref.rollout.custom.epp_config_file=/path/to/epp-config.yaml \
+actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml
```

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `rollout.agent.agent_loop_manager_class` | yes | — | `llm_d_rl_verl_integration.epp_router.agent_loop_manager.LlmdRouterAgentLoopManager` |
| `rollout.custom.epp_config_file` | yes | — | Path to EPP YAML config (plugin list, scorers). Use `deploy/epp-config.yaml` as a starting point. |
| `rollout.custom.epp_endpoints_file` | yes | — | Path where the endpoints YAML is written; must match the `path` in the EPP config's `file-discovery` plugin |
| `rollout.custom.epp_grpc_port` | no | `9002` | EPP gRPC ext_proc port |
| `rollout.custom.epp_grpc_health_port` | no | `9003` | EPP gRPC health check port |
| `rollout.custom.epp_pool_name` | no | `file-discovery` | EPP pool name |
| `rollout.custom.epp_pool_namespace` | no | `default` | EPP pool namespace |

Supports PD — see [PD Disaggregation](#pd-disaggregation).

### Integration 2 — llm-d stack (Envoy + EPP, HTTP proxy)

All generation requests are sent to a single **Envoy** proxy endpoint. Envoy calls EPP via gRPC ext_proc to pick the best replica, then forwards the request to it. verl workers only ever speak HTTP to one address; all routing intelligence lives inside Envoy + EPP on the head node.

**Startup sequence**

After all vLLM replicas are up:

1. `LlmdAgentLoopManager` spawns `LlmdActor` on the head node.
2. `LlmdActor` writes the EPP endpoints YAML, starts EPP, starts Envoy, and returns `<head-node-ip>:8081`.
3. `LlmdAgentLoopManager` builds `EnvoyLLMClient` with that address.

#### Hydra overrides

To use this integration, pass these as Hydra overrides to your verl training command. Required keys must be set; optional keys have working defaults.

```bash
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_stack.agent_loop_manager.LlmdAgentLoopManager \
+actor_rollout_ref.rollout.custom.epp_config_file=/path/to/epp-config.yaml \
+actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
+actor_rollout_ref.rollout.custom.envoy_config=/path/to/envoy.yaml
```

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `rollout.agent.agent_loop_manager_class` | yes | — | `llm_d_rl_verl_integration.llmd_stack.agent_loop_manager.LlmdAgentLoopManager` |
| `rollout.custom.epp_config_file` | yes | — | Path to EPP YAML config |
| `rollout.custom.epp_endpoints_file` | yes | — | Path where endpoints YAML is written |
| `rollout.custom.envoy_config` | yes | — | Path to Envoy config YAML. Use `deploy/envoy.yaml` as a starting point. |
| `rollout.custom.envoy_port` | no | `8081` | Envoy listener port |
| `rollout.custom.epp_grpc_port` | no | `9002` | EPP gRPC ext_proc port |
| `rollout.custom.epp_grpc_health_port` | no | `9003` | EPP gRPC health check port |

For PD disaggregated mode see [PD Disaggregation](#pd-disaggregation).

---

## PD Disaggregation

PD disaggregation requires patches to address vLLM and verl issues in the base image. [`deploy/Dockerfile.verl-vllm018-llm-d-integration`](deploy/Dockerfile.verl-vllm018-llm-d-integration) builds a ready-to-use image on top of `verlai/verl:vllm018.dev1` with those patches applied. This image is only needed for PD — the standard `verlai/verl:vllm018.dev1` image works for Integrations 1 and 2 without PD.

Both integrations support prefill-decode (PD) disaggregation via `rollout.name=vllm-llmd-pd`.

Replicas are split into prefill and decode roles by `PDEngineReplicaFactory`. The first `prefill_replicas` ranks become prefill replicas; the remaining become decode replicas. `world_size / tp_size` must equal `prefill_replicas + decode_replicas`.

- **Prefill replicas** — launch vLLM with NIXL side-channel env vars. They never serve `generate()` calls directly; the decode sidecar pulls KV blocks from them.
- **Decode replicas** — launch vLLM with NIXL env vars, then spawn `llm-d-routing-sidecar` alongside. The sidecar is the public endpoint: it fetches the KV cache from the prefill replica over NIXL, then decodes locally.

Role labels (`llm-d.ai/role: prefill` / `decode`) are written to the EPP endpoints YAML so EPP's `prefill-filter` and `decode-filter` plugins route correctly.

### Hydra overrides

To enable PD, add these overrides on top of whichever integration you are using. Required keys must be set; optional keys have working defaults.

```bash
INFER_BACKEND=vllm-llmd-pd \
actor_rollout_ref.rollout.disaggregation.prefill_replicas=<N> \
actor_rollout_ref.rollout.disaggregation.decode_replicas=<N> \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=NixlConnector \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
+actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_pd
```

Use `deploy/epp-config-pd.yaml` as the EPP config file for PD runs.

| Key | Required | Description |
|-----|----------|-------------|
| `rollout.name` | yes | `vllm-llmd-pd` |
| `rollout.disaggregation.prefill_replicas` | yes | Number of prefill replicas |
| `rollout.disaggregation.decode_replicas` | yes | Number of decode replicas |
| `rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector` | yes | `NixlConnector` |
| `rollout.engine_kwargs.vllm.kv_transfer_config.kv_role` | yes | `kv_both` |
| `rollout.custom.sidecar_connector` | no | KV connector type for `llm-d-routing-sidecar` (default: `nixlv2`) |
| `model.external_lib` | yes | `llm_d_rl_verl_integration.register_pd` — registers `vllm-llmd-pd` in FSDP worker processes |

### When to use which integration

| | EPP router (Integration 1) | llm-d stack (Integration 2) |
|---|---|---|
| **How routing works** | verl workers call EPP over gRPC, get back a replica address, call that replica's Ray actor directly | verl workers HTTP POST to Envoy; Envoy calls EPP and forwards to the replica |
| **Extra processes** | EPP only | EPP + Envoy |
| **Best for** | Performance-critical runs; fewer moving parts | Closer to a production llm-d deployment; full HTTP stack |
| **PD disaggregation** | ✓ | ✓ |

If you are unsure, **start with Integration 1 (EPP router)** — it is simpler and has lower latency.

---

## Observability

### Per-request JSONL logging

`EPPLLMClient` writes a JSONL timing record per request when `VERL_REQLOG_DIR` is set. The file is `$VERL_REQLOG_DIR/reqlog-<pid>.jsonl`, one file per worker process, line-buffered.

Fields: `ts, request_id, endpoint, prompt_hash, prompt_tokens, output_tokens, pick_s, gen_s`

`prompt_hash` is a BLAKE2b-8 digest of the token IDs — requests in the same GRPO group (same prompt, different samples) will share the same hash across replicas.

The logging is a no-op when `VERL_REQLOG_DIR` is unset, so it is safe to ship in all builds.

### Debug logging

All components default to quiet output. Increase verbosity with these env vars:

| Env var | Component | Debug value |
|---------|-----------|-------------|
| `VERL_VLLM_LOG_LEVEL` | vLLM inside replicas | `DEBUG` |
| `VERL_SIDECAR_LOG_LEVEL` | llm-d routing sidecar | `5` |
| `VERL_EPP_VERBOSITY` | EPP subprocess | `5` |
| `VERL_ENVOY_LOG_LEVEL` | Envoy proxy | `debug` |

Log files on the pod: `/tmp/epp.log`, `/tmp/envoy.log`, `/tmp/sidecar-decode-{rank}.log`.

*Note: Ray actors are spawned as new processes and do not inherit the launching shell's environment. Set these in the pod spec `env:` section (KubeRay) or in your `ray.init` runtime env.*
