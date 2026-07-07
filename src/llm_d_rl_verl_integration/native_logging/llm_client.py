"""LLMServerClient that keeps verl's native routing but logs each request.

This is the native-mode twin of epp_router.EPPLLMClient: it does NOT change
routing at all - it uses verl's stock GlobalRequestLoadBalancer (least in-flight)
via ``_acquire_server``/``_release_server`` exactly as the base LLMServerClient
does. The only addition is a per-request JSONL record (endpoint, timings, token
counts) written to VERL_REQLOG_DIR, byte-compatible with the EPP reqlog so the
same analysis tooling works for both modes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Optional
from uuid import uuid4

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)


def _phash(prompt_ids) -> str:
    try:
        b = b",".join(str(int(t)).encode() for t in prompt_ids)
        return hashlib.blake2b(b, digest_size=8).hexdigest()
    except Exception:
        return ""


class LoggingLLMClient(LLMServerClient):
    """Native verl routing + per-request reqlog.

    Routing is unchanged from the base class: each request is dispatched to the
    server chosen by the GlobalRequestLoadBalancer, and released in a finally
    block. We only measure and record; behaviour matches stock native rollout.
    """

    def __setstate__(self, state):
        # File handles do not pickle; the reqlog is (re)opened on the worker
        # process after unpickling, same as EPPLLMClient.
        self.__dict__.update(state)
        self._reqlog_f = self._open_reqlog()

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
        if getattr(self, "_reqlog_f", None) is None:
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
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput:
        # Same acquire/generate/release flow as the base LLMServerClient; we only
        # wrap it with timing and a reqlog write. server_id is the endpoint addr.
        t0 = time.monotonic()
        server_id, server = await self._acquire_server(request_id)
        t_pick = time.monotonic()
        try:
            multimodal_kwargs = {}
            if audio_data is not None:
                multimodal_kwargs["audio_data"] = audio_data
            if mm_processor_kwargs:
                multimodal_kwargs["mm_processor_kwargs"] = mm_processor_kwargs
            # priority is only supported by vLLM rollout server.
            priority = kwargs.pop("priority", 0)
            priority_kwargs = (
                {"priority": priority}
                if priority != 0 and self.config.actor_rollout_ref.rollout.name == "vllm"
                else {}
            )
            output: TokenOutput = await server.generate.remote(
                request_id=uuid4().hex,  # use new request_id for each turn
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=image_data,
                video_data=video_data,
                **multimodal_kwargs,
                **priority_kwargs,
                **kwargs,
            )
            global_steps = output.extra_fields.get("global_steps")
            output.extra_fields.setdefault("min_global_steps", global_steps)
            output.extra_fields.setdefault("max_global_steps", global_steps)
            t_end = time.monotonic()

            try:
                ntok = len(output.token_ids) if getattr(output, "token_ids", None) is not None else None
            except Exception:
                ntok = None
            self._log_request({
                "ts": time.time(),
                "request_id": str(request_id),
                "endpoint": server_id,
                "prompt_hash": _phash(prompt_ids),
                "prompt_tokens": len(prompt_ids),
                "output_tokens": ntok,
                "pick_s": round(t_pick - t0, 5),
                "gen_s": round(t_end - t_pick, 5),
            })
            return output
        finally:
            self._release_server(server_id)
