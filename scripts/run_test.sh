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
#   --name   override experiment name      (default: auto-generated)
#   --reqlog enable per-request JSONL log  (default: on for epp/llm-d, off for native)

set -euo pipefail

# ── defaults ─────────────────────────────────────────────────────────────────
MODE=""
STEPS=40
TP=1
N=8
CUSTOM_NAME=""
REQLOG=""          # empty = auto (on for non-native modes)

# ── arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)   MODE="$2";        shift 2 ;;
    --steps)  STEPS="$2";       shift 2 ;;
    --tp)     TP="$2";          shift 2 ;;
    --n)      N="$2";           shift 2 ;;
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
    [[ -z "$REQLOG" ]] && REQLOG="off"
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

# ── launch ────────────────────────────────────────────────────────────────────
cd /opt/verl/examples/grpo_trainer

ROLLOUT_N=$N ROLLOUT_TP=$TP NGPUS_PER_NODE=4 TRAIN_BATCH_SIZE=256 PPO_MINI_BATCH_SIZE=128 \
MODEL_PATH=/tmp/verl/models/Qwen3-4B \
TRAIN_FILE=/tmp/verl/data/gsm8k/train.parquet \
TEST_FILE=/tmp/verl/data/gsm8k/test.parquet \
SAVE_FREQ=-1 PROJECT_NAME=verl_grpo_gsm8k_examples \
EXPERIMENT_NAME=$EXPERIMENT_NAME \
bash run_qwen3_4b_fsdp.sh \
  trainer.logger='["console","file"]' \
  trainer.total_training_steps=$STEPS \
  trainer.default_local_dir=/tmp/checkpoints \
  trainer.rollout_data_dir=/tmp/verl/generations/train \
  +ray_kwargs.ray_init.runtime_env.env_vars.VERL_FILE_LOGGER_ROOT=/tmp/verl/logs \
  actor_rollout_ref.rollout.disable_log_stats=False \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_prompt_tokens_details=true \
  hydra.run.dir=/tmp/hydra-outputs${EXTRA_HYDRA}
