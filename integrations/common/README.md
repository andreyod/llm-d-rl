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
