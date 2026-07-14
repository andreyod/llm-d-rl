"""LLMServerClient that routes via EPP gRPC, then delegates inference to the
chosen vLLM actor handle exactly as original verl does.

Non-PD: EPP picks endpoint → call actor.generate.remote() → vLLM handles it.
PD:     EPP picks decode endpoint + sidecar headers → call actor.generate.remote(sidecar_headers=...)
        → PDDecodeVLLMHttpServer.generate() → HTTP to local sidecar.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional

import ray

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)


def _phash(prompt_ids) -> str:
    try:
        b = b",".join(str(int(t)).encode() for t in prompt_ids)
        return hashlib.blake2b(b, digest_size=8).hexdigest()
    except Exception:
        return ""


class EPPLLMClient(LLMServerClient):
    """Routes each request through EPP gRPC to pick a server, then calls
    that server's Ray actor directly — same as original verl flow.

    Args:
        config: verl DictConfig.
        load_balancer_handle: original GlobalRequestLoadBalancer (kept for
            compatibility but not used for routing decisions).
        grpc_addr: EPP gRPC address (``host:port``).
        address_to_handle: ``{server_address: ray_actor_handle}`` map built
            at startup. server_address must match what EPP returns as the
            ``x-gateway-destination-endpoint`` header.
        model_name: model identifier sent in the EPP request body.
        pd_mode: if True, forward sidecar_headers returned by EPP to
            actor.generate.remote() so PDDecodeVLLMHttpServer can reach the sidecar.
    """

    def __init__(
        self,
        config,
        load_balancer_handle=None,
        *,
        grpc_addr: str,
        address_to_handle: dict[str, ray.actor.ActorHandle],
        model_name: str,
        pd_mode: bool = False,
        **kwargs,
    ):
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        self._grpc_addr = grpc_addr
        self._address_to_handle = address_to_handle
        self._model_name = model_name
        self._pd_mode = pd_mode
        self._epp_client = None  # created on workers after unpickling via __setstate__

    def __setstate__(self, state):
        self.__dict__.update(state)
        from llm_d_rl_verl_integration.llmd_epp.grpc_client import EPPGrpcClient
        self._epp_client = EPPGrpcClient(self._grpc_addr)
        self._reqlog_f = self._open_reqlog()
        # Per-trajectory turn counter keyed by the (stable) incoming request_id;
        # see NativeLogging twin. 0-based turn index per trajectory (0 for single-turn).
        self._turn_counts: dict[str, int] = {}

    @staticmethod
    def _open_reqlog():
        """Open the per-process JSONL log file if VERL_REQLOG_DIR is set."""
        d = os.environ.get("VERL_REQLOG_DIR")
        if not d:
            return None
        try:
            os.makedirs(d, exist_ok=True)
            return open(os.path.join(d, f"reqlog-{os.getpid()}.jsonl"), "a", buffering=1)
        except Exception:
            return None

    def _log_request(self, rec: dict) -> None:
        """Write a record to the per-process JSONL log. No-op if logging is disabled."""
        if self._reqlog_f is None:
            return
        try:
            self._reqlog_f.write(json.dumps(rec) + "\n")
        except Exception:
            pass

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data=None,
        video_data=None,
        **kwargs,
    ) -> TokenOutput:
        t0 = time.monotonic()
        endpoint, sidecar_headers = await self._epp_client.pick(self._model_name, prompt_ids)
        t_pick = time.monotonic()

        if endpoint is None:
            raise RuntimeError(f"EPP returned no endpoint for request {request_id}")

        actor = self._address_to_handle.get(endpoint)
        if actor is None:
            raise RuntimeError(
                f"EPP returned endpoint {endpoint!r} which is not in the known server map. "
                f"Known: {list(self._address_to_handle.keys())}"
            )

        extra_kwargs: dict[str, Any] = {}
        if self._pd_mode and sidecar_headers:
            extra_kwargs["sidecar_headers"] = sidecar_headers

        out = await actor.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
            image_data=image_data,
            video_data=video_data,
            **extra_kwargs,
        )
        t_end = time.monotonic()

        try:
            ntok = len(out.token_ids) if getattr(out, "token_ids", None) is not None else None
        except Exception:
            ntok = None
        rid = str(request_id)
        turn = self._turn_counts.get(rid, 0)
        self._turn_counts[rid] = turn + 1
        self._log_request({
            "ts": time.time(),
            "request_id": rid,
            "turn": turn,
            "endpoint": endpoint,
            "prompt_hash": _phash(prompt_ids),
            "prompt_tokens": len(prompt_ids),
            "output_tokens": ntok,
            "pick_s": round(t_pick - t0, 5),
            "gen_s": round(t_end - t_pick, 5),
        })
        return out
