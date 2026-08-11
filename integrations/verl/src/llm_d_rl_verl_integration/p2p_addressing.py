# P2P peer addressing, shared by the engine side (p2p_replica.py, which binds
# the listener) and the client side (wave_admission/admission.py, which names the
# source in kv_source). Both MUST derive the address the same way or a pull is
# aimed at the wrong process and silently falls back to a full re-prefill.
#
# Deliberately dependency-free (stdlib only, no verl/vLLM/ray imports) so the
# AdmissionLedger actor can import it without pulling the rollout stack in.
from __future__ import annotations

# vLLM's own default is VLLM_P2P_SIDE_CHANNEL_PORT=5710; the llm-d guide and the
# routing sidecar's --p2p-connector-port both use 7777, so we follow them.
DEFAULT_P2P_CONNECTOR_PORT = 7777

# Per-replica loopback alias block for the P2P control socket. NOT 127.0.0.x, so
# that replica 0 never lands on 127.0.0.1 where it could collide with anything
# else in the namespace bound to localhost on the same port.
P2P_LOOPBACK_NET = "127.0.7."


def p2p_listener_host(replica_index: int) -> str:
    """The IP vLLM's P2P control socket binds on the replica at `replica_index`.

    ONE IP PER REPLICA IS THE WHOLE POINT. Everything in the P2P stack
    identifies a peer as `host:port`, and both the llm-d routing sidecar and the
    reference guide assume ONE ENGINE PER POD - i.e. that a pod IP fully
    identifies a peer, so every engine can share one flat tier port. This repo
    violates that: N replicas are independent engine processes inside ONE
    pod/network namespace and therefore share a pod IP.

    Separating replicas by IP restores the upstream model instead of forking it.
    Linux routes all of 127.0.0.0/8 to `lo`, so any 127.x.y.z can be bound with
    no extra address, no NET_ADMIN and no CNI support - verified on kermit: 8
    sockets bound the SAME port 7777 on distinct 127.0.7.x and dialled each
    other inside the namespace.

    Why this matters concretely: the sidecar keeps only the HOST from our
    `x-kv-cache-source-host-port` header and overwrites the port with its single
    `--p2p-connector-port` (pkg/sidecar/proxy/connector_p2p.go's
    p2pSourceParams/p2pPortFor). While replicas were separated by PORT, every
    sidecar-mediated pull was therefore aimed at rank 0 - missing, or dialling
    itself and raising NIXL_ERR_INVALID_PARAM. Separated by IP, the flat port is
    correct and the sidecar needs no patch.

    Only the ZMQ CONTROL identity uses this. The KV bytes travel over NIXL,
    addressed by VLLM_NIXL_SIDE_CHANNEL_HOST (the real node IP) plus a
    per-process uuid agent name, with the transport picked from UCX_TLS
    (cuda_ipc first, i.e. same-host GPU-to-GPU). Ray is not involved at all.

    SINGLE-NODE ONLY: loopback is not routable off-box. If replicas ever span
    nodes this must become real per-replica addresses (secondary pod IPs /
    Multus, or one pod per replica), or peers silently fail to connect.
    """
    if replica_index < 0 or replica_index > 253:
        raise ValueError(
            f"replica_index {replica_index} outside the {P2P_LOOPBACK_NET}0/24 "
            "alias block; use real per-replica IPs for a fleet this large"
        )
    return f"{P2P_LOOPBACK_NET}{replica_index + 1}"
