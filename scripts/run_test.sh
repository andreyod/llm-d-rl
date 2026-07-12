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
#   --task   gsm8k | hotpotqa | musique | quality | searchr1 | geo3k   (default: gsm8k)
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
  musique)
    # Text Qwen3-4B on single-turn context-in-prompt MuSiQue (real multi-hop long-prompt /
    # short-answer reading comprehension; reward = EM via data_source=searchR1_musique baked into
    # the parquet). Same launch path as hotpotqa, but MuSiQue ships ~20 candidate paragraphs per
    # example (vs HotpotQA's 10), so prompts are ~2x longer (measured p50=2459 p90=3208 max=7271
    # tok on Qwen3-4B) with the same short decode -> a higher prefill:decode ratio, the regime
    # where EPP burst prefix-cache co-location wins hardest.
    FSDP_SCRIPT=run_qwen3_4b_fsdp.sh
    DEF_MODEL=/tmp/verl/models/Qwen3-4B
    DEF_PROJECT=verl_grpo_musique
    DEF_TRAIN=/tmp/verl/data/musique/train.parquet
    DEF_TEST=/tmp/verl/data/musique/test.parquet
    # max_prompt=5120 keeps 99.93% of examples (filter_overlong_prompts drops the rest); the p99 is
    # 4047 so almost nothing is lost. Response cap 1024 (NOT 256): measured on the honest HotpotQA
    # 1024 run the model's natural reasoning is p50~252 p90~567 mean~320, and a 256 cap clips p90+
    # (output pinned at 256) which artificially inflates the prefill:decode ratio and the EPP win.
    # We keep decode uncapped-in-practice and let MuSiQue's LARGE prompt (p50=2459, ~1.6x HotpotQA)
    # carry the prefill dominance honestly: ~2459/~320 = ~7-8:1, vs HotpotQA-1024's ~4.6:1, so a
    # bigger EPP win from bigger input alone - no decode clipping. Directly comparable to the
    # hotpotqa_*_1024 pair (same 1024 cap). Budgets >= max_prompt+response (5120+1024=6144).
    DEF_MAXP=5120; DEF_MAXR=1024
    TASK_OVERRIDES=(
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192
    )
    ;;
  quality)
    # Text Qwen3-4B on QuALITY long-context multiple-choice reading comprehension (real, convergent).
    # This is the "large input compensates for CoT" workload: article-first prompts are ~6.8k tok
    # (p50; p99 7648, max 7668), the answer is one option letter, and REASONING STAYS ON (<think>) -
    # even a few-hundred-token CoT leaves prefill:decode well above ~5:1 purely from the huge article.
    # Reward = EM via data_source=searchR1_nq alias baked into the parquet (letter match).
    #
    # DOCUMENT CO-LOCATION: the article is the long LEADING prefix, rows are sorted by article, and we
    # set data.shuffle=False so all ~17 questions of an article land in the same rollout step; the EPP
    # burst longest-prefix placement then co-locates the whole document on one replica (article KV
    # prefilled once, reused across its questions). NOTE: shuffle=False + article order means correlated
    # batches - a deliberate RL-dynamics tradeoff for the cross-question KV-reuse study.
    FSDP_SCRIPT=run_qwen3_4b_fsdp.sh
    DEF_MODEL=/tmp/verl/models/Qwen3-4B
    DEF_PROJECT=verl_grpo_quality
    DEF_TRAIN=/tmp/verl/data/quality/train.parquet
    DEF_TEST=/tmp/verl/data/quality/test.parquet
    # max_prompt=8192 keeps 100% (max 7668 + chat template < 8192). Response cap 2048 is a generous
    # NON-clipping ceiling so we measure the model's natural CoT (not clip it). Budgets >= 8192+2048.
    DEF_MAXP=8192; DEF_MAXR=2048
    # data.shuffle is env-controlled (QUALITY_SHUFFLE, default False): False = document co-location
    # (article-ordered, questions cluster in one step); True = the controlled counter-run that removes
    # article clustering so native cannot ambiently warm the article cache (isolates the pigeonhole
    # effect vs the shuffle=False run - keep everything else identical).
    TASK_OVERRIDES=(
      data.shuffle=${QUALITY_SHUFFLE:-False}
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288
    )
    ;;
  searchr1)
    # Text Qwen3-4B on MULTI-TURN AGENTIC Search-R1: the model interleaves <think> reasoning with
    # `search` tool calls (served by the in-cluster BM25 retriever over wiki-18) and emits a final
    # <answer>; reward = EM via data_source=searchR1_hotpotqa baked into the parquet. This is the
    # genuinely prefill-heavy regime EPP targets - each turn re-appends the growing conversation +
    # retrieved passages, so keeping a trajectory's turns on one replica reuses that KV instead of
    # re-prefilling it. First task to use verl's multi-turn tool loop.
    #
    # Activation: default_agent_loop=tool_agent selects verl's ToolAgentLoop per sample; our
    # native/epp agent_loop_manager_class still supplies the logging/routing client via
    # server_manager.generate (orthogonal - the loop is host-side, generation stays on vLLM).
    # format=hermes is the Qwen3 tool-call format; the chat template injects the tool schema via
    # tools=, so make_searchr1.py's prompt only specifies <think>/<answer>, not a raw tool-call syntax.
    # tool_config_path = searchr1_tool_config.yaml, mounted from the llmd-epp-configs ConfigMap.
    #
    # Length: the initial prompt is just the question (small -> DEF_MAXP=2048); DEF_MAXR must cover
    # ALL turns' generations PLUS the appended tool observations (response_mask=0), hence large (4096).
    # max_tool_response_length caps each turn's passages. Token-len budgets >= max_prompt+response.
    # These are first-guess values - tune after the smoke run reveals the real turn/length spread.
    FSDP_SCRIPT=run_qwen3_4b_fsdp.sh
    DEF_MODEL=/tmp/verl/models/Qwen3-4B
    DEF_PROJECT=verl_grpo_searchr1
    DEF_TRAIN=/tmp/verl/data/searchr1/train.parquet
    DEF_TEST=/tmp/verl/data/searchr1/test.parquet
    DEF_MAXP=2048; DEF_MAXR=4096
    TASK_OVERRIDES=(
      actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent
      actor_rollout_ref.rollout.multi_turn.enable=True
      actor_rollout_ref.rollout.multi_turn.format=hermes
      actor_rollout_ref.rollout.multi_turn.tool_config_path=/etc/llmd-configs/searchr1_tool_config.yaml
      actor_rollout_ref.rollout.multi_turn.max_assistant_turns=4
      actor_rollout_ref.rollout.multi_turn.max_user_turns=4
      actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
      actor_rollout_ref.rollout.multi_turn.max_tool_response_length=512
      data.return_raw_chat=True
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu=12288
      actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=12288
      actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=12288
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
  *) echo "ERROR: unknown --task '$TASK' (expected gsm8k|hotpotqa|musique|quality|searchr1|geo3k)"; exit 1 ;;
esac
TRAIN_RESOLVED=${TRAIN_FILE:-$DEF_TRAIN}
TEST_RESOLVED=${TEST_FILE:-$DEF_TEST}
if [[ "$TASK" == "geo3k" ]]; then
  # actor.strategy=fsdp (FSDP1): the VL launch script forces fsdp2 + use_fused_kernels=True, which
  # trips "aten.mm.default got mixed torch.Tensor and DTensor" in compute_log_prob (verl #5633 - the
  # fused-logits matmul mixes a plain tensor with an fsdp2-sharded DTensor). FSDP1 shards params as
  # flat plain tensors (no DTensor), so the mixed-type matmul cannot occur; this unblocks geo3k
  # training while keeping fused kernels on (disabling them instead risks OOM per the same issue).
  #
  # Image size for the KV-reuse study: geo3k source diagrams are tiny (~64 image tokens), a weak
  # KV-reuse workload. Qwen2.5-VL image tokens = pixels / (28*28); verl stores images raw and only
  # the processor resizes, bounded by min_pixels/max_pixels (data.mm_processor_kwargs, forwarded to
  # both training and vLLM rollout so they tokenize identically). smart_resize UPSCALES an image up
  # to min_pixels, so pinning min=max forces every image to ~GEO3K_IMG_TOKENS tokens regardless of
  # native size -> a large shared image+question prefix per GRPO group = a strong prefill-heavy
  # KV-reuse workload. This is synthetic inflation (upsampled diagrams add no visual info), a systems
  # benchmark, not a quality result. GEO3K_IMG_TOKENS default 4096; sweep it to chart the crossover.
  GEO3K_IMG_TOKENS=${GEO3K_IMG_TOKENS:-4096}
  GEO3K_IMG_PIXELS=$(( GEO3K_IMG_TOKENS * 784 ))   # 784 = 28*28 pixels per visual token
  # Host-RAM shaping for large images: the head pod (driver, collates all gen results) is cgroup
  # capped at 96 GB. Large-image pixel_values tensors blow past that during validation (the whole
  # test set is generated as one batch - val_batch_size is deprecated/ignored in this verl) unless
  # we shrink the footprint:
  #  - cap the geo3k TEST set to ~256 examples when building the data (see make/preprocess) so
  #    validation collates far fewer image tensors on the driver.
  #  - filter_overlong_prompts=False: True processes every image through the processor at dataset
  #    init to measure length; at large token counts that is a big upfront cost. Pinned images + a
  #    generous max_prompt are never overlong, so filtering is unnecessary here.
  #  - train_batch_size 256 -> 128: halve the images the driver holds per training step.
  GEO3K_TRAIN_BATCH=${GEO3K_TRAIN_BATCH:-64}
  TASK_OVERRIDES=(data.train_files="$TRAIN_RESOLVED" data.val_files="$TEST_RESOLVED" data.image_key=images
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=16384
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=16384
    +data.mm_processor_kwargs.min_pixels=$GEO3K_IMG_PIXELS
    +data.mm_processor_kwargs.max_pixels=$GEO3K_IMG_PIXELS
    data.filter_overlong_prompts=False
    data.train_batch_size=$GEO3K_TRAIN_BATCH)
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
