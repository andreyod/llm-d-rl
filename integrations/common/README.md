# llm-d-rl-common

Framework-agnostic utilities for talking to [llm-d](https://github.com/llm-d/llm-d)'s
**Endpoint Picker (EPP)** from an RL training loop. No dependency on any specific RL
framework (verl, Ray, etc.) - only `grpcio` and `pyyaml`.

Used today by [`integrations/verl`](../verl/README.md); intended to be reused by any
other framework integration that needs to talk to EPP.

## Contents

- `epp_grpc_client.py` - minimal hand-rolled EPP ext-proc gRPC client (`EPPGrpcClient`).
  `route()` is the entry point: it picks between EPP's fire-and-forget and
  tracked-completion protocols and returns a `RoutingResult` with a `.complete()`
  hook that is a no-op unless completion tracking was requested.
- `reqlog.py` - shared per-request JSONL logging helpers.
- `endpoints.py` - EPP file-discovery endpoints YAML writer.
- `router_stack.py` - `RouterStack`: owns the EPP / Envoy argv, binary-path resolution
  (`LLMD_EPP_BINARY`, falling back to `VERL_EPP_BINARY`) and readiness waiting. Stdlib
  only, no ray. verl's `LlmdActor` wraps it in a head-pinned Ray actor.
- `cli.py` - the `llm-d-rl-router` console script: runs the same stack in the
  foreground, for frameworks that start EPP from a pod lifecycle hook instead of from
  inside the training process.
