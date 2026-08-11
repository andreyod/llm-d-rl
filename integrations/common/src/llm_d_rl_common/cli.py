"""``llm-d-rl-router``: run the EPP (and optionally Envoy) as a foreground process.

For frameworks that cannot start the router stack from inside the training
process the way verl's LlmdActor does, and would otherwise hand-roll the argv in
a pod lifecycle hook. Same binaries, same flags, one implementation
(router_stack.py).

    llm-d-rl-router --epp-config /etc/llmd-configs/epp-config.yaml \
                    [--envoy-config /etc/llmd-configs/envoy.yaml]

Blocks until a child exits or a signal arrives, so a container runtime can
supervise it and stop it cleanly.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys

from llm_d_rl_common.router_stack import (
    DEFAULT_ENVOY_PORT,
    DEFAULT_EPP_GRPC_PORT,
    DEFAULT_EPP_HEALTH_PORT,
    DEFAULT_EPP_METRICS_PORT,
    RouterStack,
)

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="llm-d-rl-router", description=__doc__)
    p.add_argument("--epp-config", required=True, help="path to the EPP YAML config")
    p.add_argument("--envoy-config", help="path to an Envoy YAML config; omit to run EPP only")
    p.add_argument("--grpc-port", type=int, default=DEFAULT_EPP_GRPC_PORT)
    p.add_argument("--grpc-health-port", type=int, default=DEFAULT_EPP_HEALTH_PORT)
    p.add_argument("--metrics-port", type=int, default=DEFAULT_EPP_METRICS_PORT)
    p.add_argument("--pool-name", default="file-discovery")
    p.add_argument("--pool-namespace", default="default")
    p.add_argument("--envoy-port", type=int, default=DEFAULT_ENVOY_PORT)
    p.add_argument("--log-level", default="INFO", help="python logging level for this script")
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> int:
    stack = RouterStack()
    grpc_port, health_port = await stack.start_epp(
        args.epp_config,
        grpc_port=args.grpc_port,
        health_port=args.grpc_health_port,
        metrics_port=args.metrics_port,
        pool_name=args.pool_name,
        pool_namespace=args.pool_namespace,
    )
    logger.info("EPP ready on grpc=%d health=%d", grpc_port, health_port)

    if args.envoy_config:
        envoy_port = await stack.start_envoy(args.envoy_config, port=args.envoy_port)
        logger.info("Envoy ready on :%d", envoy_port)

    stopping = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stopping.set)

    # Stop if a child dies on its own, so the container restarts rather than
    # sitting there looking healthy with no router.
    died: list[str] = []

    async def _watch() -> None:
        while True:
            for name, proc in (("epp", stack.epp_proc), ("envoy", stack.envoy_proc)):
                if proc is not None and proc.poll() is not None:
                    logger.error("%s exited with code %s", name, proc.returncode)
                    died.append(name)
                    stopping.set()
                    return
            await asyncio.sleep(1.0)

    watcher = asyncio.create_task(_watch())
    await stopping.wait()          # returns on signal or on a child dying
    watcher.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await watcher

    stack.stop()
    return 1 if died else 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
