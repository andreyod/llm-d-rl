"""LLMServerClient that sends all requests to a single Envoy endpoint.

Envoy handles EPP routing internally; verl just does HTTP POST to Envoy.
"""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)

_GENERATE_PATH = "/inference/v1/generate"


class EnvoyLLMClient(LLMServerClient):
    """Sends all generation requests to a single Envoy endpoint.

    Envoy internally queries EPP and routes to the appropriate vLLM replica.

    Args:
        config: verl DictConfig.
        load_balancer_handle: original GlobalRequestLoadBalancer (kept for
            compatibility).
        envoy_address: Envoy endpoint as ``host:port`` (no scheme).
        model_name: model identifier sent in the request body.
    """

    def __init__(
        self,
        config,
        load_balancer_handle=None,
        *,
        envoy_address: str,
        model_name: str,
        **kwargs,
    ):
        super().__init__(config=config, load_balancer_handle=load_balancer_handle, **kwargs)
        base = envoy_address if envoy_address.startswith("http") else f"http://{envoy_address}"
        self._url = f"{base}{_GENERATE_PATH}"
        self._model_name = model_name
        self._max_tokens = int(
            config.actor_rollout_ref.rollout.response_length
        )
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=36000),
            )
        return self._session

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
        body = {
            "model": self._model_name,
            "token_ids": prompt_ids,
            "sampling_params": {**sampling_params, "max_tokens": self._max_tokens},
        }
        session = await self._get_session()
        async with session.post(self._url, json=body) as resp:
            if not resp.ok:
                raise RuntimeError(
                    f"Envoy returned HTTP {resp.status} for {self._url}"
                )
            data = await resp.json()

        return _parse_response(data)


def _parse_response(data: dict) -> TokenOutput:
    choice = data["choices"][0]
    token_ids = [int(t) for t in (choice.get("token_ids") or [])]
    stop_reason = choice.get("finish_reason")
    logprobs_content = (choice.get("logprobs") or {}).get("content") or []
    log_probs = [e.get("logprob") for e in logprobs_content] if logprobs_content else None
    return TokenOutput(token_ids=token_ids, log_probs=log_probs, stop_reason=stop_reason)
