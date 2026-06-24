# Running the KubeRay Example

Single-node GRPO training on GSM8K with Qwen3-4B using the llm-d RL verl integration.
As shipped, `ray-cluster.yaml` has the **4-GPU** worker option active; an 8-GPU option is
also provided (commented out in the manifest). The run commands below are grouped the same
way: the first set targets 8 GPUs, and a [4-GPU Option](#4-gpu-option) section follows. Pick
the set that matches the worker `resources` block you enabled in the manifest.

## Prerequisites

- Kubernetes cluster with GPU nodes
- KubeRay CRD and operator installed (see [examples/setting-kuberay.md](setting-kuberay.md) for instructions. )

## Directory structure

```
examples/
  configmap.yaml         # Kubernetes ConfigMap bundling both EPP configs
  ray-cluster.yaml       # RayCluster definition
  setting-kuberay.md     # KubeRay operator / CRD install instructions
```

## Step 1 — Edit the deployment manifests

`configmap.yaml` and `ray-cluster.yaml` both default to the `ezraverl` namespace. Adjust to
match your environment:

- **Namespace** - replace `ezraverl` (appears in each manifest's `metadata.namespace`, and
  the ConfigMap reference) with your target namespace.
- **GPU count** - the worker `resources` block ships with the 4-GPU option active and the
  8-GPU option commented out. Enable whichever matches your node.
- **Node placement** - the head co-locates onto the worker's node via `podAffinity`, and the
  worker is anchored to a GPU node by its `nvidia.com/gpu` request. The worker `nodeAffinity`
  has a `NotIn` list excluding known-faulty GPU hosts (e.g. `pokprod-b93r44s3`) - edit that
  list for your cluster.

The EPP binary is **not** baked into the verl image. The `fetch-epp` init container in
`ray-cluster.yaml` extracts it from the public image set by its `EPP_IMAGE` env on pod start;
use `push-epp.sh` (repo root) to push a new EPP into a running pod without recreating it. See
[Supplying the EPP at runtime](../README.md#supplying-the-epp-at-runtime) in the main README.

## Step 2 — Deploy

```bash
# ConfigMap must exist before the cluster starts (pods mount it at /etc/llmd-configs/)
kubectl apply -f examples/configmap.yaml
kubectl apply -f examples/ray-cluster.yaml
```

Wait for both pods to be ready:
```bash
kubectl get pods -w
```

The `postStart` hook on each pod installs the integration package with pip install and pre-downloads GSM8K and Qwen3-4B. Training should not start until both pods report `Ready`.

## Step 3 — Run training

Exec into the head pod, then run one of the commands below.

```bash
kubectl exec -it <head-pod-name> -- bash
cd /opt/verl/examples/grpo_trainer
```

All commands use verl's own `run_qwen3_4b_fsdp.sh` as the base script and pass the integration overrides via `$@`. `hydra.run.dir` is required because the default `./outputs/` path is read-only in the container.

### EPP — direct gRPC routing

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
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.EPPAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### EPP — direct gRPC routing, PD disaggregated

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
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.EPPAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### Llm-d stack (Envoy + EPP — HTTP proxy routing)

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
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_stack.agent_loop_manager.EnvoyAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### Llm-d stack (Envoy + EPP — HTTP proxy routing, PD disaggregated)

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
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_stack.agent_loop_manager.EnvoyAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

## EPP config

The two configs bundled in `examples/configmap.yaml` (`epp-config.yaml` and `epp-config-pd.yaml`) are starting points. Customize the scorer weights or swap plugins to tune routing for your workload. The path is passed per run via the Hydra override `+actor_rollout_ref.rollout.custom.epp_config_file=...` (see the commands above) — you can mount your own ConfigMap or point to any file accessible on the head node.

See the [main README](../README.md) for the full config reference and architecture overview.


## 4-GPU Option

The same scripts work with a 4-GPU Ray cluster by adjusting a few parameters. Run from inside the head pod (`kubectl exec -it <head-pod> -- bash`).

### EPP — direct gRPC routing

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
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.EPPAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### EPP — direct gRPC routing, PD disaggregated

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
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.EPPAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### Llm-d stack (Envoy + EPP — HTTP proxy routing)

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
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_stack.agent_loop_manager.EnvoyAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

### Llm-d stack (Envoy + EPP — HTTP proxy routing, PD disaggregated)

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
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.llmd_stack.agent_loop_manager.EnvoyAgentLoopManager \
    +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config-pd.yaml \
    +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml \
    +actor_rollout_ref.rollout.custom.sidecar_connector=nixlv2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true' \
    'hydra.run.dir=/tmp/hydra-outputs'
```

## Logs

#### Training logs (verl)

verl's file logger writes per-step training metrics (rewards, loss, timing) to the directory set by `VERL_FILE_LOGGER_ROOT`. In the example commands this is `/tmp/verl/logs` on the **head pod**. Each training step appends a JSON line to a file in that directory — useful for plotting reward curves or diagnosing training instability.

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
| `/tmp/sidecar-decode-{rank}.log` | worker | llm-d routing sidecar (one per decode replica) | NIXL V2 protocol — prefill calls, `kv_transfer_params` received, decode forwarding |
| `/tmp/ray/session_latest/logs/worker-*.out` | worker | vLLM prefill and decode engines | vLLM engine logs including NIXL KV transfer traces when `VERL_VLLM_LOG_LEVEL=DEBUG` |

To stream a log live:
```bash
kubectl exec <head-pod> -- tail -f /tmp/epp.log
kubectl exec <worker-pod> -- tail -f /tmp/sidecar-decode-0.log
```

#### Increasing verbosity

All components default to quiet logging. Set these env vars to increase verbosity — either in the shell before launching training, or in the `env:` section of your KubeRay `RayCluster` / `RayJob` container spec.

| Env var | Component | Default | Debug value |
|---------|-----------|---------|-------------|
| `VERL_VLLM_LOG_LEVEL` | vLLM inside prefill and decode replicas (`VLLM_LOGGING_LEVEL`) | unset (vLLM default) | `DEBUG` |
| `VERL_SIDECAR_LOG_LEVEL` | llm-d routing sidecar (`--zap-log-level`) | `0` | `5` |
| `VERL_EPP_VERBOSITY` | EPP subprocess (`-v`) | `0` | `5` |
| `VERL_ENVOY_LOG_LEVEL` | Envoy proxy (`--log-level`) | `info` | `debug` |

*Note: Ray actors are spawned as new processes on remote nodes and do not inherit the launching shell's environment.*

With *KubeRay* — set in the container spec; vars are present before Ray starts:

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
Make sure you have write premission to the destination path!!