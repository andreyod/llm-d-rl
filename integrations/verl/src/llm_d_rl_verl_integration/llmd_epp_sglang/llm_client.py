"""EPP-routed client for SGLang replicas.

Identical routing to the vLLM client - EPP picks the endpoint, we call that
replica's Ray actor - so this only overrides the actor kwargs. Everything else,
including the reqlog schema, comes from llmd_epp.llm_client.
"""

from __future__ import annotations

from typing import Any

from llm_d_rl_verl_integration.llmd_epp.llm_client import EPPLLMClient


class SglangEPPLLMClient(EPPLLMClient):
    """EPPLLMClient with SGLang's narrower generate() signature."""

    def _actor_kwargs(self, sidecar_headers, kwargs: dict[str, Any]) -> dict[str, Any]:
        # SGLangHttpServer.generate() declares only prompt_ids / sampling_params /
        # request_id / image_data / video_data plus its own PD bootstrap_* kwargs, and
        # like vLLM's server has no **kwargs catch-all - but it declares fewer of the
        # multimodal ones. verl's AgentLoopWorkerTQ calls every
        # LLMServerClient.generate() with a fixed kwarg set regardless of backend, so
        # drop what this server cannot take rather than raising on every request.
        # No sidecar path: no PD/P2P for SGLang in this mode.
        for key in ("audio_data", "mm_processor_kwargs", "priority"):
            kwargs.pop(key, None)
        return kwargs
