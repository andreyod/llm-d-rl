#!/usr/bin/env bash
# run_test.sh  --mode <native|epp|epp-inflight|epp-fc|epp-p2p|wave-admission|llm-d>  [options]
#
# Usage examples:
#   bash run_test.sh --mode native
#   bash run_test.sh --mode epp
#   bash run_test.sh --mode epp --steps 20 --tp 2 --n 4
#   bash run_test.sh --mode epp-fc --task weka   # EPP routing + per-endpoint concurrency CAP (flow-control queue)
#   bash run_test.sh --mode epp-p2p              # EPP routing + P2P KV-cache sharing (every replica pulls/serves)
#   bash run_test.sh --mode wave-admission --task weka   # estimation-gated admission, no EPP (see wave_admission/)
#   bash run_test.sh --mode llm-d          # (not yet implemented)
#
# Options:
#   --mode   native | epp | epp-inflight | epp-fc | epp-p2p | wave-admission | llm-d (required)
#   --steps  total_training_steps          (default: 40)
#   --tp     tensor-parallel size          (default: 1)
#   --n      rollout group size            (default: 8)
#   --task   any folder under workloads/ (gsm8k | hotpotqa | musique | quality |
#            searchr1 | scotus_xl | arxiv | geo3k)   (default: gsm8k)
#   --name   override experiment name      (default: auto-generated)
#   --reqlog enable per-request JSONL log  (default: on for all modes)

set -euo pipefail

# -- defaults -----------------------------------------------------------------
MODE=""
STEPS=40
TP=1
N=8
CUSTOM_NAME=""
REQLOG=""          # empty = auto (on for non-native modes)
TASK="gsm8k"       # name of a folder under workloads/ (each has a task.env)

# -- arg parsing ---------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)   MODE="$2";        shift 2 ;;
    --steps)  STEPS="$2";       shift 2 ;;
    --tp)     TP="$2";          shift 2 ;;
    --n)      N="$2";           shift 2 ;;
    --task)   TASK="$2";        shift 2 ;;   # any folder name under workloads/
    --name)   CUSTOM_NAME="$2"; shift 2 ;;
    --reqlog) REQLOG="$2";      shift 2 ;;   # "on" or "off"
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "ERROR: --mode is required  (native | epp | epp-inflight | epp-fc | epp-p2p | wave-admission | llm-d)"
  exit 1
fi

# -- per-mode config -----------------------------------------------------------
# Each branch only sets which agent-loop manager class to use, (for EPP modes)
# which EPP config file to mount, and (for the completion-reporting modes)
# whether to enable epp_report_completion; the actual hydra overrides are
# assembled once, identically, right after the case statement.
AGENT_LOOP_MANAGER_CLASS=""
EPP_CONFIG_FILE=""
EPP_REPORT_COMPLETION=""
ROLLOUT_NAME=""        # only epp-p2p sets this (registers a non-default rollout backend)
EXTERNAL_LIB=""        # only epp-p2p sets this (model.external_lib import hook)
P2P_ENGINE_HYDRA=""    # only epp-p2p sets this (OffloadingConnector engine_kwargs)

case "$MODE" in
  native)
    DEFAULT_NAME="qwen3_4b_grpo_baseline_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    # Native verl routing (GlobalRequestLoadBalancer), but with a logging client so
    # the run produces the same per-request reqlog as EPP, plus the endpoints YAML
    # for the vLLM /metrics scraper. Routing behaviour is unchanged from stock native.
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.native_logging.agent_loop_manager.NativeLoggingAgentLoopManager"
    ;;

  epp)
    DEFAULT_NAME="qwen3_4b_grpo_epp_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager"
    EPP_CONFIG_FILE="epp-config.yaml"
    ;;

  epp-inflight)
    # EPP routing on the in-flight counter, NO cap (routing only).
    # epp_report_completion=true keeps the ext_proc stream open through
    # generation and reports completion, so EPP's inflight-load-producer is
    # honest and active-request-scorer routes to the least-in-flight endpoint.
    # Config epp-config-inflight.yaml (inflight-load-producer + active-request-scorer,
    # no flowControl / no concurrency-detector). Intended for --task weka.
    DEFAULT_NAME="qwen3_4b_grpo_eppinflight_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager"
    EPP_CONFIG_FILE="epp-config-inflight.yaml"
    EPP_REPORT_COMPLETION="true"
    ;;

  epp-fc)
    # EPP routing + a per-endpoint concurrency CAP enforced by EPP's flow-control
    # layer (over-cap requests queued via EnqueueAndWait). "fc" = flow control =
    # the cap+queue. Builds on epp-inflight's honest in-flight counter
    # (epp_report_completion=true). The flow-control layer is enabled the
    # NON-deprecated way: featureGates: [flowControl] inside the config (NOT the
    # deprecated env var). The cap config is overridable via EPP_CAP_CONFIG
    # (default epp-config-inflight-cap.yaml) to sweep the cap C without editing
    # the mode.
    DEFAULT_NAME="qwen3_4b_grpo_eppfc_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager"
    EPP_CONFIG_FILE="${EPP_CAP_CONFIG:-epp-config-inflight-cap.yaml}"
    EPP_REPORT_COMPLETION="true"
    ;;

  wave-admission)
    # Estimation-gated admission control (no EPP): NEW conversations are
    # gated by an in-process AdmissionLedger's estimate of per-replica free
    # KV budget (causal per-turn-index growth estimator, wave1_size
    # unconditional first admits); sticky placement after admission, no
    # migration. See wave_admission/admission.py and
    # ~/.claude/plans/steady-splashing-blanket.md. Tunables (wave1_size, GPU
    # budget formula inputs, ...) are set via
    # actor_rollout_ref.rollout.custom.wave_admission_* overrides (EXTRA_HYDRA
    # below, or pass through EXTRA_OVERRIDES) - defaults match the H200 /
    # Qwen2.5-7B-class weka ctxc64k_n256 testbed.
    DEFAULT_NAME="qwen3_4b_grpo_waveadmission_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.wave_admission.agent_loop_manager.WaveAdmissionAgentLoopManager"
    ;;

  epp-p2p)
    # P2P KV-cache sharing (llm-d/llm-d PR #2067): every replica runs a local llm-d
    # routing sidecar (--kv-connector=offloading) so EPP's p2p-source-producer
    # header (x-kv-cache-source-host-port) becomes kv_transfer_params.remote_kv_source
    # before the request reaches vLLM's OffloadingConnector P2P tier, instead of
    # recomputing the cached prefix. rollout.name=vllm-llmd-p2p registers
    # P2PEngineReplicaFactory (register_p2p.py / p2p_replica.py) - every replica is
    # symmetric (both pulls and serves), unlike PD's prefill/decode split, so no
    # disaggregation.* replica counts are needed. See deploy/epp-config-p2p.yaml.
    DEFAULT_NAME="qwen3_4b_grpo_eppp2p_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    AGENT_LOOP_MANAGER_CLASS="llm_d_rl_verl_integration.llmd_epp.agent_loop_manager.LlmdRouterAgentLoopManager"
    # Override via EPP_P2P_CONFIG=epp-config-p2p-load.yaml for the load-only/no-burst
    # arm (destination picking decoupled from prefix locality - see that file's
    # header comment for why this matters for actually exercising the P2P pull path).
    EPP_CONFIG_FILE="${EPP_P2P_CONFIG:-epp-config-p2p.yaml}"
    ROLLOUT_NAME="vllm-llmd-p2p"
    EXTERNAL_LIB="llm_d_rl_verl_integration.register_p2p"
    # --block-size must match exactly across every replica (guide requirement: a
    # mismatch makes the whole pull path silently inert / vLLM rejects the transfer).
    # offload_prompt_only=false: the default (true) never offloads decode-phase
    # (generated) blocks, so a peer pull of freshly-generated content would miss -
    # see p2psource/README.md's "Deployment Requirements".
    #
    # VERL_USE_EXTERNAL_MODULES (not just model.external_lib above): verl/__init__.py
    # reads this env var and imports it at `import verl` time, in EVERY process that
    # touches verl - including the TaskRunnerV1 driver, which resolves rollout.name
    # via RolloutReplicaRegistry.get() long before any FSDP worker (where
    # model.external_lib is actually consumed, inside HFModelConfig instantiation)
    # ever starts. Without this, the driver never imports register_p2p at all and
    # `rollout.name=vllm-llmd-p2p` fails with "Unknown rollout mode" - model.external_lib
    # alone only reaches the later _ROLLOUT_REGISTRY lookup, not this earlier one.
    # cpu_bytes_to_use: the ONLY strictly-required kv_connector_extra_config key
    # (vllm/v1/kv_offload/cpu/spec.py raises if unset - everything else there has
    # a default). 4GiB per replica x n_gpus_per_node=8 replicas = 32GiB host RAM,
    # comfortably inside this cluster's node RAM; not tuned/sized from measured
    # KV capacity, just a safe smoke-test default. Override via
    # P2P_CPU_BYTES_TO_USE for a real sizing pass.
    P2P_ENGINE_HYDRA="
      +ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES=llm_d_rl_verl_integration.register_p2p \
      +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=64 \
      +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector=OffloadingConnector \
      +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_role=kv_both \
      +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.offload_prompt_only=false \
      +actor_rollout_ref.rollout.engine_kwargs.vllm.kv_transfer_config.kv_connector_extra_config.cpu_bytes_to_use=${P2P_CPU_BYTES_TO_USE:-4294967296}"
    ;;

  llm-d)
    echo "ERROR: --mode llm-d is not yet implemented"
    exit 1
    ;;

  *)
    echo "ERROR: unknown mode '${MODE}'. Choose: native | epp | epp-inflight | epp-fc | epp-p2p | wave-admission | llm-d"
    exit 1
    ;;
esac

# Hydra overrides common to every routing mode above: the agent-loop manager class
# and endpoints file are always set; epp_config_file is added only when the mode
# set one (native has none).
#
# trainer.use_v1=true is mandatory here: verl's own default (false) routes through
# the legacy main_ppo_v0.TaskRunner/RayPPOTrainer, whose _validate() unconditionally
# constructs and passes a DataProto to async_rollout_manager.generate_sequences().
# Every llm-d mode's manager class chain (LlmdBaseAgentLoopManager) subclasses verl's
# AgentLoopManagerTQ (see base_agent_loop_manager.py), whose generate_sequences expects
# a TensorDict and returns None (real outputs go through TransferQueue) - calling it
# with a DataProto crashes inside AgentLoopWorkerTQ.generate_sequences with
# `AttributeError: 'NoneType' object has no attribute 'keys'` (DataProto.pop() called
# with the wrong signature). TaskRunnerV1 (use_v1=true) is the pipeline actually
# designed for AgentLoopManagerTQ end-to-end, including validation - not just an
# unrelated flag; no key prefixed with `+` since trainer.use_v1 already exists in
# verl's schema.
EXTRA_HYDRA="
  trainer.use_v1=true \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=${AGENT_LOOP_MANAGER_CLASS}"
if [[ -n "$EPP_CONFIG_FILE" ]]; then
  EXTRA_HYDRA="${EXTRA_HYDRA} \
  +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/${EPP_CONFIG_FILE}"
fi
if [[ -n "$EPP_REPORT_COMPLETION" ]]; then
  EXTRA_HYDRA="${EXTRA_HYDRA} \
  +actor_rollout_ref.rollout.custom.epp_report_completion=${EPP_REPORT_COMPLETION}"
fi
if [[ -n "$ROLLOUT_NAME" ]]; then
  EXTRA_HYDRA="${EXTRA_HYDRA} \
  actor_rollout_ref.rollout.name=${ROLLOUT_NAME}"
fi
if [[ -n "$EXTERNAL_LIB" ]]; then
  EXTRA_HYDRA="${EXTRA_HYDRA} \
  +actor_rollout_ref.model.external_lib=${EXTERNAL_LIB}"
fi
if [[ -n "$P2P_ENGINE_HYDRA" ]]; then
  EXTRA_HYDRA="${EXTRA_HYDRA} \
  ${P2P_ENGINE_HYDRA}"
fi
EXTRA_HYDRA="${EXTRA_HYDRA} \
  +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml"

EXPERIMENT_NAME="${CUSTOM_NAME:-$DEFAULT_NAME}"

# -- reqlog override -----------------------------------------------------------
if [[ "$REQLOG" == "on" ]]; then
  EXTRA_HYDRA="
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_REQLOG_DIR=/tmp/verl/reqlog${EXTRA_HYDRA}"
fi

# -- task config: sourced from the self-contained workload folder --------------
# Each workloads/<name>/task.env sets FSDP_SCRIPT, DEF_MODEL, DEF_PROJECT, DEF_TRAIN,
# DEF_TEST, DEF_MAXP, DEF_MAXR and the TASK_OVERRIDES array (fully - including any
# env-var-driven logic like QUALITY_SHUFFLE or the geo3k image sizing). Adding a
# workload means adding a folder; this driver does not change.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Resolve the workloads dir: explicit WORKLOADS_DIR override, else the repo layout
# (benchmarks/scripts -> ../workloads), else /tmp/workloads (where run_on_head.sh copies
# the selected workload folder alongside run_test.sh on the head pod).
WORKLOADS_DIR="${WORKLOADS_DIR:-}"
if [[ -z "$WORKLOADS_DIR" ]]; then
  if [[ -d "$SCRIPT_DIR/../workloads" ]]; then
    WORKLOADS_DIR="$(cd "$SCRIPT_DIR/../workloads" && pwd)"
  elif [[ -d /tmp/workloads ]]; then
    WORKLOADS_DIR=/tmp/workloads
  fi
fi
TASK_ENV="$WORKLOADS_DIR/$TASK/task.env"
if [[ ! -f "$TASK_ENV" ]]; then
  echo "ERROR: no task.env for --task '$TASK' (looked at: $TASK_ENV)"
  echo "       available workloads: $(ls -1 "$WORKLOADS_DIR" 2>/dev/null | tr '\n' ' ')"
  exit 1
fi
TASK_OVERRIDES=()
# shellcheck disable=SC1090
source "$TASK_ENV"

TRAIN_RESOLVED=${TRAIN_FILE:-$DEF_TRAIN}
TEST_RESOLVED=${TEST_FILE:-$DEF_TEST}

# Optional extra hydra overrides, appended LAST so they win over the per-task defaults
# (e.g. raise ppo/log_prob token budgets for a bigger max_prompt). Space-separated;
# values must not contain spaces. Empty by default.
read -r -a EXTRA_OV <<< "${EXTRA_OVERRIDES:-}"

# -- launch --------------------------------------------------------------------
cd /tmp/verl/verl/examples/grpo_trainer

# TRAIN_BATCH_SIZE / PPO_MINI_BATCH_SIZE are env-overridable: workloads with fewer
# prompts than the default batch (e.g. weka replays N<256 conversations) must set
# them to the conversation count, or the dataloader cannot fill a step.
ROLLOUT_N=$N ROLLOUT_TP=$TP NGPUS_PER_NODE=8 \
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-128} \
MODEL_PATH=${MODEL_PATH:-$DEF_MODEL} \
TRAIN_FILE=$TRAIN_RESOLVED \
TEST_FILE=$TEST_RESOLVED \
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-$DEF_MAXP} MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-$DEF_MAXR} \
SAVE_FREQ=-1 PROJECT_NAME=${PROJECT_NAME:-$DEF_PROJECT} \
EXPERIMENT_NAME=$EXPERIMENT_NAME \
bash "$FSDP_SCRIPT" \
  trainer.logger='["console","file"]' \
  trainer.total_training_steps=$STEPS \
  trainer.default_local_dir=/tmp/checkpoints \
  trainer.rollout_data_dir=/tmp/verl/generations/train \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs \
  actor_rollout_ref.rollout.disable_log_stats=False \
  actor_rollout_ref.rollout.n=$N \
  ${TASK_OVERRIDES[@]+"${TASK_OVERRIDES[@]}"} \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true \
  ${EXTRA_OV[@]+"${EXTRA_OV[@]}"} \
  hydra.run.dir=/tmp/hydra-outputs${EXTRA_HYDRA}
