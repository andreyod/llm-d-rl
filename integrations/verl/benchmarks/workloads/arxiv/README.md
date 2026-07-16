# arxiv workload

Large-input arXiv subject-area classification (scientific-domain analog of `scotus_xl`): read a full
arXiv paper and assign exactly one of 11 arXiv subject categories. Single-turn; very large input,
short output. Reward = normalized exact-match (`data_source=searchR1_nq`; `ground_truth.target=[code,
name]`, so the arXiv category code or its descriptive name scores). Builder: `make_arxiv.py`.

## Task and data

- Dataset: `ccdv/arxiv-classification`, config **`no_ref`** - full paper text (title, abstract, body)
  from the Long-Document-Dataset (He et al. 2019). 33k papers: ~28k train / 2.5k val / 2.5k test.
  The `no_ref` config strips in-document class references (e.g. `[cs.LG]` -> `[]`) so the label
  cannot leak into the prompt; the default config does not, so we do not use it.
- Labels: the 11 arXiv categories (math.AC Commutative Algebra, cs.CV Computer Vision and Pattern
  Recognition, cs.AI Artificial Intelligence, cs.SY Systems and Control, math.GR Group Theory,
  cs.CE Computational Engineering Finance and Science, cs.PL Programming Languages, cs.IT Information
  Theory, cs.DS Data Structures and Algorithms, cs.NE Neural and Evolutionary Computing,
  math.ST Statistics Theory). Slightly unbalanced.
- Input sizes: every document is > 4k tokens, with a long tail (raw text ranges from ~2.85k to
  ~2.55M characters). The prompt is capped at 24,576 tokens; the long tail is truncated (see below).
  Exact prompt-token percentiles are printed by the builder's `token_stats` when the data is built.
- Output: brief CoT then a single label; response capped at 2,048 tokens (non-clipping for a short
  label + brief reasoning). Use `--no_cot` for the direct single-label variant.

## Test setup

- Cluster: 1 worker node, 8x NVIDIA H200; vLLM TP=1 -> 8 independent replicas (one per GPU).
- Model: Qwen3-4B (FSDP1 actor/ref). GRPO `rollout.n=8`, `train_batch_size=256`
  -> 256 prompts x 8 samples = 2048 requests/step.
- Sequence budget: `max_prompt_length=24576`, `max_response_length=2048`;
  `filter_overlong_prompts=True` + `truncation=right` drop/clip the long paper tail.
- Routing compared:
  - native = verl least-in-flight load balancer (sticky session, request_id -> replica).
  - EPP as the verl endpoint picker = llm-d burst prefix-cache producer with `balanceBy: tokens`
    and `windowDurationMs: 1000` (covers the per-step rollout arrival burst).

## Results summary

Runs: `arxiv_{native,epp}_5s` (5 steps each, 8 replicas). Using EPP as the verl endpoint picker
(vs native least-in-flight), steady-state over steps 2-5:

1. Mean rollout time per step reduced ~28% (149.2s -> 107.9s).
2. Slowest-replica (straggler) generation time reduced ~28% (144.9s -> 103.7s).
3. Per-request generation is faster at every percentile (the straggler tail shrinks).
4. Validation accuracy tracked together (native 0.681 -> 0.713, EPP 0.639 -> 0.699 over 5 steps;
   parity within run-to-run noise at this step count) - no accuracy change from routing.

| metric | native | EPP (token-balanced burst) | diff |
|---|---|---|---|
| mean rollout / step (verl `timing_s/gen`) | 149.2 s | 107.9 s | -27.7% |
| mean rollout / step (reqlog full-span) | ~145 s | ~104 s | ~-28% |
| generate_sequences mean / replica | 80.9 s | 61.2 s | -24.3% |
| generate_sequences slowest (straggler) | 144.9 s | 103.7 s | -28.5% |
| straggler ratio (slowest / mean) | 1.79x | 1.68x | -6.1% |
| full step time (`timing_s/step`, training-dominated) | 958.6 s | 918.8 s | -4.1% |
| prompt_length mean (sanity: same data) | 12,967 | 12,967 | - |
| response_length mean (non-clipping) | 608 | 615 | - |
| val accuracy (step 0 -> 5) | 0.681 -> 0.713 | 0.639 -> 0.699 | parity |

Notes: rollout generation is only a fraction of the training-dominated full step (update_actor alone
is ~550 s/step), so the ~28% rollout-generation win nets ~4% on the full step. The prefix-cache hit
rate and per-replica KV utilization (from the vLLM `/metrics` scrape) are not reported for this
5-step pair; the scotus_xl workload (the legal analog of this task) documents them for the same
routing mechanism (native re-prefills each document 8x and saturates KV; EPP co-locates the group and
prefills once).

## Reproduce

```bash
# build the data once (no_ref config, CoT), then run each arm
python3 benchmarks/workloads/arxiv/make_arxiv.py --local_dir /tmp/verl/data/arxiv
benchmarks/scripts/run_test.sh --task arxiv --mode <native|epp> --steps 30
```

The EPP arm needs `deploy/epp-config.yaml` with `windowDurationMs: 1000` + `balanceBy: tokens` and
the token-balanced EPP binary.
