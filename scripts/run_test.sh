#!/usr/bin/env bash
# run_test.sh  --mode <native|epp|llm-d>  [options]
#
# Usage examples:
#   bash run_test.sh --mode native
#   bash run_test.sh --mode epp
#   bash run_test.sh --mode epp --steps 20 --tp 2 --n 4
#   bash run_test.sh --mode llm-d          # (not yet implemented)
#
# Options:
#   --mode   native | epp | llm-d          (required)
#   --steps  total_training_steps          (default: 40)
#   --tp     tensor-parallel size          (default: 1)
#   --n      rollout group size            (default: 8)
#   --task   gsm8k | hotpotqa | geo3k      (default: gsm8k)
#   --name   override experiment name      (default: auto-generated)
#   --reqlog enable per-request JSONL log  (default: on for all modes)

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────────────────────
MODE=""
STEPS=40
TP=1
N=8
CUSTOM_NAME=""
REQLOG=""          # empty = auto (on for non-native modes)
TASK="gsm8k"       # gsm8k (text, Qwen3-4B) | geo3k (multimodal, Qwen2.5-VL-7B)

# ── arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)   MODE="$2";        shift 2 ;;
    --steps)  STEPS="$2";       shift 2 ;;
    --tp)     TP="$2";          shift 2 ;;
    --n)      N="$2";           shift 2 ;;
    --task)   TASK="$2";        shift 2 ;;   # gsm8k | geo3k
    --name)   CUSTOM_NAME="$2"; shift 2 ;;
    --reqlog) REQLOG="$2";      shift 2 ;;   # "on" or "off"
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "ERROR: --mode is required  (native | epp | llm-d)"
  exit 1
fi

# ── per-mode config ───────────────────────────────────────────────────────────
EXTRA_HYDRA=""

case "$MODE" in
  native)
    DEFAULT_NAME="qwen3_4b_grpo_baseline_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    # Native verl routing (GlobalRequestLoadBalancer), but with a logging client so
    # the run produces the same per-request reqlog as EPP, plus the endpoints YAML
    # for the vLLM /metrics scraper. Routing behaviour is unchanged from stock native.
    EXTRA_HYDRA="
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.native_logging.agent_loop_manager.NativeLoggingAgentLoopManager \
  +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml"
    ;;

  epp)
    DEFAULT_NAME="qwen3_4b_grpo_epp_tp${TP}_n${N}_${STEPS}s"
    [[ -z "$REQLOG" ]] && REQLOG="on"
    EXTRA_HYDRA="
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=llm_d_rl_verl_integration.epp_router.agent_loop_manager.LlmdRouterAgentLoopManager \
  +actor_rollout_ref.rollout.custom.epp_config_file=/etc/llmd-configs/epp-config.yaml \
  +actor_rollout_ref.rollout.custom.epp_endpoints_file=/tmp/epp-endpoints.yaml"
    ;;

  llm-d)
    echo "ERROR: --mode llm-d is not yet implemented"
    exit 1
    ;;

  *)
    echo "ERROR: unknown mode '${MODE}'. Choose: native | epp | llm-d"
    exit 1
    ;;
esac

EXPERIMENT_NAME="${CUSTOM_NAME:-$DEFAULT_NAME}"

# ── reqlog override ───────────────────────────────────────────────────────────
if [[ "$REQLOG" == "on" ]]; then
  EXTRA_HYDRA="
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_REQLOG_DIR=/tmp/verl/reqlog${EXTRA_HYDRA}"
fi

# ── task config: model / dataset / launch script ─────────────────────────────
case "$TASK" in
  gsm8k)
    FSDP_SCRIPT=run_qwen3_4b_fsdp.sh
    DEF_MODEL=/tmp/verl/models/Qwen3-4B
    DEF_PROJECT=verl_grpo_gsm8k_examples
    DEF_TRAIN=/tmp/verl/data/gsm8k/train.parquet
    DEF_TEST=/tmp/verl/data/gsm8k/test.parquet
    DEF_MAXP=1024; DEF_MAXR=1024
    TASK_OVERRIDES=()
    ;;
  hotpotqa)
    # Text Qwen3-4B on single-turn context-in-prompt HotpotQA (real long-prompt / short-answer
    # reading comprehension; reward = EM via data_source=searchR1_hotpotqa baked into the parquet).
    # Same launch path as gsm8k (4B script reads TRAIN_FILE/TEST_FILE env), just different data
    # and a KV-reuse profile (long context prompt, short decode).
    FSDP_SCRIPT=run_qwen3_4b_fsdp.sh
    DEF_MODEL=/tmp/verl/models/Qwen3-4B
    DEF_PROJECT=verl_grpo_hotpotqa
    DEF_TRAIN=/tmp/verl/data/hotpotqa/train.parquet
    DEF_TEST=/tmp/verl/data/hotpotqa/test.parquet
    DEF_MAXP=4096; DEF_MAXR=256
    # The 4B script's default actor.ppo_max_token_len_per_gpu=3000 (and rollout log_prob 4096)
    # is < the longest prompt+response here (up to ~4096+256), which trips a max_token_len
    # assertion in the actor update. Raise the dynamic-bsz token budgets above max_prompt+response.
    TASK_OVERRIDES=(
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
    )
    ;;
  geo3k)
    # Multimodal Qwen2.5-VL-7B on geo3k. KV-reuse profile: short decode + room for big images.
    # The VL launch script HARDCODES data.train_files/val_files (unlike the 4B script which reads
    # $TRAIN_FILE), so we override them on the CLI. Use big-image parquet via $TRAIN_FILE/$TEST_FILE.
    FSDP_SCRIPT=run_qwen2_5_vl_7b_fsdp.sh
    DEF_MODEL=/tmp/verl/models/Qwen2.5-VL-7B-Instruct
    DEF_PROJECT=verl_grpo_geo3k
    DEF_TRAIN=/tmp/verl/data/geo3k/train.parquet
    DEF_TEST=/tmp/verl/data/geo3k/test.parquet
    DEF_MAXP=8192; DEF_MAXR=256
    ;;
  *) echo "ERROR: unknown --task '$TASK' (expected gsm8k|hotpotqa|geo3k)"; exit 1 ;;
esac
TRAIN_RESOLVED=${TRAIN_FILE:-$DEF_TRAIN}
TEST_RESOLVED=${TEST_FILE:-$DEF_TEST}
if [[ "$TASK" == "geo3k" ]]; then
  # actor.strategy=fsdp (FSDP1): the VL launch script forces fsdp2 + use_fused_kernels=True, which
  # trips "aten.mm.default got mixed torch.Tensor and DTensor" in compute_log_prob (verl #5633 - the
  # fused-logits matmul mixes a plain tensor with an fsdp2-sharded DTensor). FSDP1 shards params as
  # flat plain tensors (no DTensor), so the mixed-type matmul cannot occur; this unblocks geo3k
  # training while keeping fused kernels on (disabling them instead risks OOM per the same issue).
  TASK_OVERRIDES=(data.train_files="$TRAIN_RESOLVED" data.val_files="$TEST_RESOLVED" data.image_key=images
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384)
fi

# ── launch ────────────────────────────────────────────────────────────────────
cd /opt/verl/examples/grpo_trainer

ROLLOUT_N=$N ROLLOUT_TP=$TP NGPUS_PER_NODE=8 TRAIN_BATCH_SIZE=256 PPO_MINI_BATCH_SIZE=128 \
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
  hydra.run.dir=/tmp/hydra-outputs${EXTRA_HYDRA}
