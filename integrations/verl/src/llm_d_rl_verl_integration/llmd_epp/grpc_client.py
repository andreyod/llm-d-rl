"""Minimal EPP gRPC ext-proc client.

Sends token IDs to EPP and reads back the chosen endpoint
(x-gateway-destination-endpoint) plus any sidecar headers.
Hand-rolled protobuf — no generated stubs needed.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

import grpc.aio

logger = logging.getLogger(__name__)

_EXT_PROC_METHOD = "/envoy.service.ext_proc.v3.ExternalProcessor/Process"
DESTINATION_HEADER = "x-gateway-destination-endpoint"


# ---------------------------------------------------------------------------
# Minimal protobuf encoder
# ---------------------------------------------------------------------------

def _varint(n: int) -> bytes:
    buf = []
    while n > 0x7F:
        buf.append((n & 0x7F) | 0x80)
        n >>= 7
    buf.append(n & 0x7F)
    return bytes(buf)


def _lv(field: int, data: bytes) -> bytes:
    tag = _varint((field << 3) | 2)
    return tag + _varint(len(data)) + data


def _bool_field(field: int, value: bool) -> bytes:
    return _varint((field << 3) | 0) + bytes([1 if value else 0])


def _encode_header_map(headers: list[tuple[str, bytes]]) -> bytes:
    out = b""
    for k, v in headers:
        hv = _lv(1, k.encode()) + _lv(3, v)
        out += _lv(1, hv)
    return out


def _encode_request_headers(headers: list[tuple[str, bytes]]) -> bytes:
    http_headers = _lv(1, _encode_header_map(headers)) + _bool_field(3, False)
    return _lv(2, http_headers)


def _encode_request_body(body: bytes) -> bytes:
    http_body = _lv(1, body) + _bool_field(2, True)
    return _lv(4, http_body)


# ---------------------------------------------------------------------------
# Minimal protobuf decoder
# ---------------------------------------------------------------------------

def _decode_varint(data: bytes, pos: int) -> tuple[int, int]:
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _decode_fields(data: bytes) -> dict[int, list[bytes]]:
    fields: dict[int, list[bytes]] = {}
    pos = 0
    while pos < len(data):
        tag, pos = _decode_varint(data, pos)
        wire_type = tag & 0x7
        field_number = tag >> 3
        if wire_type == 0:
            _, pos = _decode_varint(data, pos)
        elif wire_type == 2:
            length, pos = _decode_varint(data, pos)
            fields.setdefault(field_number, []).append(data[pos: pos + length])
            pos += length
        elif wire_type == 5:
            pos += 4
        elif wire_type == 1:
            pos += 8
        else:
            break
    return fields


def _extract_headers(response_bytes: bytes) -> dict[str, str]:
    result: dict[str, str] = {}
    top = _decode_fields(response_bytes)
    for hr_bytes in top.get(1, []):
        for cr_bytes in _decode_fields(hr_bytes).get(1, []):
            for hm_bytes in _decode_fields(cr_bytes).get(2, []):
                for hvo_bytes in _decode_fields(hm_bytes).get(1, []):
                    for hv_bytes in _decode_fields(hvo_bytes).get(1, []):
                        hv = _decode_fields(hv_bytes)
                        key = (hv.get(1, [b""])[0]).decode(errors="ignore")
                        val = (hv.get(3, [b""])[0]).decode(errors="ignore")
                        if key:
                            result[key] = val
    return result


# ---------------------------------------------------------------------------
# EPP gRPC client
# ---------------------------------------------------------------------------

class EPPGrpcClient:
    """Thin gRPC client for EPP's ext-proc endpoint.

    Creates a persistent channel per instance. Each AgentLoopWorker
    should create its own instance (lazy, after unpickling).
    """

    def __init__(self, grpc_addr: str) -> None:
        self._addr = grpc_addr
        self._channel = grpc.aio.insecure_channel(grpc_addr)
        self._method = self._channel.stream_stream(
            _EXT_PROC_METHOD,
            request_serializer=lambda x: x,
            response_deserializer=lambda x: x,
        )

    async def pick(self, model: str, prompt_ids: list[int]) -> tuple[Optional[str], dict[str, str]]:
        """Ask EPP which endpoint to route this request to.

        Returns:
            (endpoint, sidecar_headers) where endpoint is ``host:port`` or None.
            sidecar_headers contains all EPP-set headers except the destination header.
        """
        body = json.dumps({"model": model, "token_ids": prompt_ids}).encode()
        req_headers = _encode_request_headers([
            (":method", b"POST"),
            (":path", b"/inference/v1/generate"),
            ("content-type", b"application/json"),
            ("content-length", str(len(body)).encode()),
        ])
        req_body = _encode_request_body(body)

        async def _iter():
            yield req_headers
            yield req_body

        async for response_bytes in self._method(_iter()):
            headers = _extract_headers(response_bytes)
            endpoint = headers.get(DESTINATION_HEADER)
            if endpoint:
                sidecar_headers = {k: v for k, v in headers.items() if k != DESTINATION_HEADER}
                return endpoint, sidecar_headers

        return None, {}

    async def close(self) -> None:
        await self._channel.close()
