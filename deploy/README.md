# Running the KubeRay Example

Single-node GRPO training on GSM8K with Qwen3-4B using the llm-d RL verl integration.
As shipped, `ray-cluster.yaml.tmpl` has the **4-GPU** worker option active; an 8-GPU option is
also provided (commented out in the manifest). The run commands below are grouped the same
way: the first set targets 8 GPUs, and a [4-GPU Option](#4-gpu-option) section follows. Pick
the set that matches the worker `resources` block you enabled in the manifest.

## Prerequisites

- Kubernetes cluster with GPU nodes
- KubeRay CRD and operator installed (see [setting-kuberay.md](setting-kuberay.md) for instructions)

## Directory structure

```
deploy/
  epp-config.yaml         # EPP config — standard prefix-cache routing (source of truth)
  epp-config-pd.yaml      # EPP config — PD-aware routing (source of truth)
  envoy.yaml              # Envoy proxy config for the llm-d stack integration (source of truth)
  configmap.yaml          # Reference stub; the ConfigMap is built by deploy.sh
  ray-cluster.yaml.tmpl   # RayCluster definition (template; values come from deploy.env)
  deploy.env              # Single source of truth for the namespace and every runtime image
  deploy.sh               # Render ray-cluster.yaml.tmpl with deploy.env, create ConfigMap, apply/delete
  setting-kuberay.md      # KubeRay operator / CRD install instructions
```

## Step 1 - Set deploy.env and edit the manifest

All deployment config that changes per environment lives in `deploy.env`:

- **Namespace (required)** - set `NAMESPACE` in `deploy.env`. It has no default;
  `deploy.sh`, `push-epp.sh`, and `rl_orchestrate.sh` all read it from here and refuse to run
  until it is set. This is the single place the namespace is defined.
- **Images** - every runtime image (verl, crane, EPP, Envoy) is defined in `deploy.env` too.
  Edit tags there rather than in the manifest; `deploy.sh` substitutes them (and `NAMESPACE`)
  into `ray-cluster.yaml.tmpl` at apply time.

The manifest template itself only needs edits for node/GPU layout:

- **GPU count** - the worker `resources` block ships with the 4-GPU option active and the
  8-GPU option commented out. Enable whichever matches your node.
- **Node placement** - the head co-locates onto the worker's node via `podAffinity`, and the
  worker is anchored to a GPU node by its `nvidia.com/gpu` request. The worker `nodeAffinity`
  has a `NotIn` list excluding known-faulty GPU hosts (e.g. `pokprod-b93r44s3`) - edit that
  list for your cluster.

Neither the EPP nor the Envoy binary is baked into the verl image. The `fetch-binaries` init
container in `ray-cluster.yaml.tmpl` extracts both from the public images set in `deploy.env`
(`IMG_EPP`, `IMG_ENVOY`) on pod start; use `scripts/utils/push-epp.sh` to push a new EPP into a
running pod without recreating it. See
[Supplying the EPP and Envoy at runtime](../README.md#supplying-the-epp-and-envoy-at-runtime)
in the main README.

## Step 2 - Deploy

`deploy.sh apply` does everything: it builds the `llmd-epp-configs` ConfigMap from the
standalone config files (`epp-config.yaml`, `epp-config-pd.yaml`, and `envoy.yaml` are the
source of truth - **do not** apply `configmap.yaml` directly, it has no `data:` block) and
applies the rendered cluster manifest, both into the namespace from `deploy.env`:

```bash
bash deploy/deploy.sh apply
```

Useful sub-commands: `deploy.sh configmap` ((re)create just the ConfigMap),
`deploy.sh render` (print the rendered manifest without applying), `deploy.sh delete`
(tear down the cluster).

Wait for both pods to be ready:
```bash
kubectl get pods -w
```

The `postStart` hook on each pod installs the integration package with pip install and pre-downloads GSM8K and Qwen3-4B. Training should not start until both pods report `Ready`.

## Step 3 - Run training

Exec into the head pod, then run one of the commands below.

```bash
kubectl exec -it <head-pod-name> -- bash
cd /opt/verl/examples/grpo_trainer
```

All commands use verl's own `run_qwen3_4b_fsdp.sh` as the base script and pass the integration overrides via `$@`. `hydra.run.dir` is required because the default `./outputs/` path is read-only in the container.

### EPP - direct gRPC routing

```bash
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_epp_fsdp_8gpu \
bash /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=50 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.LlmdRouterAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### EPP - direct gRPC routing, PD disaggregated

```bash
INFER_BACKEND=vllm-llmd-pd \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_epp_pd_fsdp_8gpu \
bash /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    actor_rollout_ref.rollout.disaggregation.prefill_replicas=2 \
    actor_rollout_ref.rollout.disaggregation.decode_replicas=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=NixlConnector \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=80 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_pd \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.LlmdRouterAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### llm-d stack (Envoy + EPP - HTTP proxy routing)

```bash
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_envoy_fsdp_8gpu \
bash /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=50 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_stack.agent_loop_manager.LlmdAgentLoopManager \
    +actor_rollout_ref.rollout.custom.envoy_config=/etc/llmd-configs/envoy.yaml \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### llm-d stack (Envoy + EPP - HTTP proxy routing, PD disaggregated)

```bash
INFER_BACKEND=vllm-llmd-pd \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_envoy_pd_fsdp_8gpu \
bash /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    actor_rollout_ref.rollout.disaggregation.prefill_replicas=2 \
    actor_rollout_ref.rollout.disaggregation.decode_replicas=2 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=NixlConnector \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=80 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_pd \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_stack.agent_loop_manager.LlmdAgentLoopManager \
    +actor_rollout_ref.rollout.custom.envoy_config=/etc/llmd-configs/envoy.yaml \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

## EPP config

`deploy/epp-config.yaml` (standard) and `deploy/epp-config-pd.yaml` (PD disaggregated) are the starting-point configs. Customize scorer weights or swap plugins to tune routing for your workload.

The path is passed per run via `+actor_rollout_ref.rollout.custom.epp_config_file=...` (see the commands above). You can point to any file accessible on the head node — mount your own ConfigMap, copy a file to `/tmp`, or use the sample directly in non-k8s environments.

To update a running cluster after editing a config file, re-run the `kubectl create configmap` command from Step 2 above, then recreate the pod.

See the [main README](../README.md) for the full config reference and architecture overview.


## 4-GPU Option

The same scripts work with a 4-GPU Ray cluster by adjusting a few parameters. Run from inside the head pod (`kubectl exec -it <head-pod> -- bash`).

### EPP - direct gRPC routing

```bash
NGPUS_PER_NODE=4 \
TRAIN_BATCH_SIZE=256 \
PPO_MINI_BATCH_SIZE=128 \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_epp_fsdp_4gpu \
bash /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=50 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.LlmdRouterAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### EPP - direct gRPC routing, PD disaggregated

```bash
NGPUS_PER_NODE=4 \
TRAIN_BATCH_SIZE=256 \
PPO_MINI_BATCH_SIZE=128 \
INFER_BACKEND=vllm-llmd-pd \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
ROLLOUT_GPU_MEM_UTIL=0.6 \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_epp_pd_fsdp_4gpu \
bash /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    actor_rollout_ref.rollout.disaggregation.prefill_replicas=1 \
    actor_rollout_ref.rollout.disaggregation.decode_replicas=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=NixlConnector \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=80 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_pd \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.LlmdRouterAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### llm-d stack (Envoy + EPP - HTTP proxy routing)

```bash
NGPUS_PER_NODE=4 \
TRAIN_BATCH_SIZE=256 \
PPO_MINI_BATCH_SIZE=128 \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_envoy_fsdp_4gpu \
bash /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=50 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_stack.agent_loop_manager.LlmdAgentLoopManager \
    +actor_rollout_ref.rollout.custom.envoy_config=/etc/llmd-configs/envoy.yaml \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### llm-d stack (Envoy + EPP - HTTP proxy routing, PD disaggregated)

```bash
NGPUS_PER_NODE=4 \
TRAIN_BATCH_SIZE=256 \
PPO_MINI_BATCH_SIZE=128 \
INFER_BACKEND=vllm-llmd-pd \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 \
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_envoy_pd_fsdp_4gpu \
bash /opt/verl/examples/grpo_trainer/run_qwen3_4b_fsdp.sh \
    actor_rollout_ref.rollout.disaggregation.prefill_replicas=1 \
    actor_rollout_ref.rollout.disaggregation.decode_replicas=1 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=NixlConnector \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
    trainer.logger='["console","file"]' \
    trainer.default_local_dir=/tmp/checkpoints \
    trainer.total_training_steps=80 \
    '+ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs' \
    +actor_rollout_ref.model.external_lib=llm_d_rl_verl_integration.register_pd \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_stack.agent_loop_manager.LlmdAgentLoopManager \
    +actor_rollout_ref.rollout.custom.envoy_config=/etc/llmd-configs/envoy.yaml \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

## Logs

#### Training logs (verl)

verl's file logger writes per-step training metrics (rewards, loss, timing) to the directory set by `VERL_FILE_LOGGER_ROOT`. In the example commands this is `/tmp/verl/logs` on the **head pod**. Each training step appends a JSON line to a file in that directory - useful for plotting reward curves or diagnosing training instability.

The file path is:
```
<VERL_FILE_LOGGER_ROOT>/<trainer.project_name>/<trainer.experiment_name>.jsonl
```

`trainer.project_name` and `trainer.experiment_name` are Hydra config fields, overridden in the run script via the `PROJECT_NAME` and `EXPERIMENT_NAME` env vars. In the example commands above these are set explicitly, for example:
```bash
PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=qwen3_4b_grpo_vllm_epp_pd_fsdp_8gpu \
```
which produces:
```
/tmp/verl/logs/verl_grpo_gsm8k_examples/qwen3_4b_grpo_vllm_epp_pd_fsdp_8gpu.jsonl
```

```bash
kubectl exec <head-pod> -- tail -f /tmp/verl/logs/*.jsonl
```

#### Component log files

Each integration component writes its output to a fixed file path on the pod it runs on:

| File | Pod | Component | Contents |
|------|-----|-----------|----------|
| `/tmp/epp.log` | head | EPP subprocess | Endpoint scoring decisions, plugin output, gRPC ext_proc traffic |
| `/tmp/envoy.log` | head | Envoy proxy | HTTP request routing, upstream selection, connection errors |
| `/tmp/sidecar-decode-{rank}.log` | worker | llm-d routing sidecar (one per decode replica) | NIXL V2 protocol - prefill calls, `kv_transfer_params` received, decode forwarding |
| `/tmp/ray/session_latest/logs/worker-*.out` | worker | vLLM prefill and decode engines | vLLM engine logs including NIXL KV transfer traces when `VERL_VLLM_LOG_LEVEL=DEBUG` |

To stream a log live:
```bash
kubectl exec <head-pod> -- tail -f /tmp/epp.log
kubectl exec <worker-pod> -- tail -f /tmp/sidecar-decode-0.log
```

#### Increasing verbosity

All components default to quiet logging. Set these env vars to increase verbosity - either in the shell before launching training, or in the `env:` section of your KubeRay `RayCluster` / `RayJob` container spec.

| Env var | Component | Default | Debug value |
|---------|-----------|---------|-------------|
| `VERL_VLLM_LOG_LEVEL` | vLLM inside prefill and decode replicas (`VLLM_LOGGING_LEVEL`) | unset (vLLM default) | `DEBUG` |
| `VERL_SIDECAR_LOG_LEVEL` | llm-d routing sidecar (`--zap-log-level`) | `0` | `5` |
| `VERL_EPP_VERBOSITY` | EPP subprocess (`-v`) | `0` | `5` |
| `VERL_ENVOY_LOG_LEVEL` | Envoy proxy (`--log-level`) | `info` | `debug` |

*Note: Ray actors are spawned as new processes on remote nodes and do not inherit the launching shell's environment.*

With *KubeRay* - set in the container spec; vars are present before Ray starts:

```yaml
containers:
  - name: ray-worker
    env:
      - name: VERL_VLLM_LOG_LEVEL
        value: "DEBUG"
      - name: VERL_EPP_VERBOSITY
        value: "5"
```


## Saving Rollout Generations (Optional)

To save the model's generated outputs during training and validation, add these overrides to any command above:

```
trainer.validation_data_dir=/tmp/verl/generations/val \
trainer.rollout_data_dir=/tmp/verl/generations/train \
```

Outputs are written as parquet files to the specified directories on the head node. This is useful for inspecting model behavior or offline reward analysis.
Make sure you have write permission to the destination path.