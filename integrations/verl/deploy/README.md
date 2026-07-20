# Deploying the integration

This is the general guide for wiring the integration into any Ray cluster, plus the full Hydra
override and environment-variable reference. For a ready-to-run, end-to-end example, use the
**[KubeRay walkthrough](kuberay/README.md)** - it is the concrete instantiation of every step below.

For how the integration works (the two modes, the mandatory core, PD), see
[`../docs/architecture.md`](../docs/architecture.md).

## Prerequisites

- A Ray cluster with at least one GPU worker node.
- verl on a compatible vLLM version (the [official verl images](https://hub.docker.com/r/verlai/verl)
  provide the environment).

## Steps

### 1. Provide verl

The official `verlai/verl` images are **environment** images: they ship the vLLM/CUDA/torch stack
but not verl itself, so verl is installed at runtime. The examples here use
`verlai/verl:vllm018.dev1` and verl commit `334d9f8b03816382cbd72e898bc8ae04efca6fbe`:

```bash
git clone https://github.com/volcengine/verl.git /tmp/verl/verl
cd /tmp/verl/verl && git checkout 334d9f8b03816382cbd72e898bc8ae04efca6fbe
pip install --no-deps -e .
```

Do this on **every** node (head and all workers). On KubeRay this runs from the `postStart` hook in
[`kuberay/ray-cluster.yaml.tmpl`](kuberay/ray-cluster.yaml.tmpl).

### 2. Install the integration package

Install on **every** node (head and all workers) - Ray does not propagate a pip install across
nodes:

```bash
pip install "git+https://github.com/llm-d-incubation/llm-d-rl.git#subdirectory=integrations/verl"
```

Or add the source to `PYTHONPATH` without installing:

```bash
git clone https://github.com/llm-d-incubation/llm-d-rl.git
export PYTHONPATH=$(pwd)/llm-d-rl/integrations/verl/src:$PYTHONPATH
```

### 3. Get the EPP, Envoy, and sidecar binaries

The integration launches these as external processes at runtime; none are baked into the verl image,
so iterating on them never triggers a verl rebuild. Obtain them from the published llm-d images or
build from source, then point the integration at them via env vars (set before Ray starts, or in the
`ray.init` runtime env). On the head, `LlmdActor` launches EPP (and, in serving mode, Envoy); on each
worker, decode replicas launch the sidecar (PD only).

| Env var | Default | Binary | Node | Required for |
|---------|---------|--------|------|--------------|
| `VERL_EPP_BINARY` | `/usr/local/bin/epp` | EPP (endpoint picker) | head | every run |
| `VERL_ENVOY_BINARY` | `/usr/local/bin/envoy` | Envoy proxy | head | llm-d serving mode only |
| `VERL_SIDECAR_BINARY` | `/opt/llm-d-bins/pd-sidecar` | llm-d routing sidecar | workers (decode) | PD disaggregation only |

### 4. Place the config files

Copy these starting-point configs (in this directory) to any path readable on the head node, edit as
needed, and pass their paths as the Hydra overrides in step 5:

- [`deploy/epp-config.yaml`](epp-config.yaml) - EPP scorer config (standard routing)
- [`deploy/epp-config-pd.yaml`](epp-config-pd.yaml) - EPP scorer config (PD disaggregated)
- [`deploy/envoy.yaml`](envoy.yaml) - Envoy proxy config (llm-d serving mode only)

The EPP config's `file-discovery` plugin `path:` must match the `epp_endpoints_file` override -
`LlmdActor` writes the replica list there and EPP reads it (default `/tmp/epp-endpoints.yaml`).

### 5. Add the Hydra overrides and run

Running the integration is just a few Hydra overrides on your existing verl training command - see
the reference below. The KubeRay walkthrough ([`kuberay/README.md`](kuberay/README.md)) shows the
full commands for each mode.

## Hydra override reference

### EPP as the endpoint picker

```bash
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager \
+actor_rollout_ref.rollout.custom.epp_config_file=/path/to/epp-config.yaml \
+actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml
```

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `rollout.agent.agent_loop_manager_class` | yes | - | `llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager` |
| `rollout.custom.epp_config_file` | yes | - | Path to the EPP YAML config (plugin list, scorers). Start from `deploy/epp-config.yaml`. |
| `rollout.custom.epp_endpoints_file` | yes | - | Path where the endpoints YAML is written; must match the `path` in the EPP config's `file-discovery` plugin |
| `rollout.custom.epp_grpc_port` | no | `9002` | EPP gRPC ext_proc port |
| `rollout.custom.epp_grpc_health_port` | no | `9003` | EPP gRPC health check port |
| `rollout.custom.epp_pool_name` | no | `file-discovery` | EPP pool name |
| `rollout.custom.epp_pool_namespace` | no | `default` | EPP pool namespace |

### llm-d serving (Envoy + EPP)

```bash
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_serving.agent_loop_manager.LlmdAgentLoopManager \
+actor_rollout_ref.rollout.custom.epp_config_file=/path/to/epp-config.yaml \
+actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
+actor_rollout_ref.rollout.custom.envoy_config=/path/to/envoy.yaml
```

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `rollout.agent.agent_loop_manager_class` | yes | - | `llm_d_rl_verl_integration.llmd_serving.agent_loop_manager.LlmdAgentLoopManager` |
| `rollout.custom.epp_config_file` | yes | - | Path to the EPP YAML config |
| `rollout.custom.epp_endpoints_file` | yes | - | Path where the endpoints YAML is written |
| `rollout.custom.envoy_config` | yes | - | Path to the Envoy config YAML. Start from `deploy/envoy.yaml`. |
| `rollout.custom.envoy_port` | no | `8081` | Envoy listener port |
| `rollout.custom.epp_grpc_port` | no | `9002` | EPP gRPC ext_proc port |
| `rollout.custom.epp_grpc_health_port` | no | `9003` | EPP gRPC health check port |

### PD disaggregation

Add these on top of whichever mode you use. Use `deploy/epp-config-pd.yaml` as the EPP config, and
the `deploy/Dockerfile.pd` image (PD needs NIXL and vLLM/verl patches not in the stock image).

```bash
INFER_BACKEND=vllm-llmd-pd \
actor_rollout_ref.rollout.disaggregation.prefill_replicas=<N> \
actor_rollout_ref.rollout.disaggregation.decode_replicas=<N> \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=NixlConnector \
+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
+actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_pd
```

| Key | Required | Description |
|-----|----------|-------------|
| `rollout.name` | yes | `vllm-llmd-pd` |
| `rollout.disaggregation.prefill_replicas` | yes | Prefill replica count; `prefill_replicas + decode_replicas == world_size / tp_size` |
| `rollout.disaggregation.decode_replicas` | yes | Decode replica count (same constraint) |
| `rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector` | yes | `NixlConnector` |
| `rollout.engine_kwargs.vllm.kv_transfer_config.kv_role` | yes | `kv_both` |
| `rollout.custom.sidecar_connector` | no | KV connector for `llm-d-routing-sidecar` (default `nixlv2`) |
| `model.external_lib` | yes | `llm_d_rl_verl_integration.register_pd` - registers `vllm-llmd-pd` in FSDP worker processes |

## Observability

### Per-request JSONL logging (reqlog)

All routing clients (`llmd_epp`, `llmd_serving`, `native_logging`) write a JSONL timing record per
request when `VERL_REQLOG_DIR` is set: `$VERL_REQLOG_DIR/reqlog-<pid>.jsonl`, one file per worker
process, line-buffered. It is a no-op when `VERL_REQLOG_DIR` is unset.

| Field | Description | Modes |
|-------|-------------|-------|
| `ts` | Wall-clock timestamp (Unix seconds) | all |
| `request_id` | verl request ID | all |
| `turn` | 0-based turn index within a multi-turn trajectory | all |
| `endpoint` | Backend pod that served the request (`host:port`) | all |
| `prompt_hash` | BLAKE2b-8 hex digest of the input token IDs | all |
| `prompt_tokens` | Number of input tokens | all |
| `output_tokens` | Number of generated tokens | all |
| `pick_s` | Time spent on the routing decision (EPP gRPC call or load-balancer acquire) | `llmd_epp`, `native_logging` |
| `gen_s` | Generation time — actor call only for `llmd_epp`/`native_logging`; full round-trip (routing + inference) for `llmd_serving` | all |

### Debug logging

All components default to quiet output. Increase verbosity with these env vars:

| Env var | Component | Debug value |
|---------|-----------|-------------|
| `VERL_VLLM_LOG_LEVEL` | vLLM inside replicas | `DEBUG` |
| `VERL_SIDECAR_LOG_LEVEL` | llm-d routing sidecar | `5` |
| `VERL_EPP_VERBOSITY` | EPP subprocess | `5` |
| `VERL_ENVOY_LOG_LEVEL` | Envoy proxy | `debug` |

Log files on the pod: `/tmp/epp.log`, `/tmp/envoy.log`, `/tmp/sidecar-decode-{rank}.log`.

Note: Ray actors are spawned as new processes and do not inherit the launching shell's environment.
Set these in the pod spec `env:` section (KubeRay) or in your `ray.init` runtime env.
