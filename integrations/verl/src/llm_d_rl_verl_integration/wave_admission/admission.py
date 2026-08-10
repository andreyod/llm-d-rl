"""Wave-based admission control: causal per-turn-index growth estimator + a
per-replica admission ledger that gates NEW conversation starts by ESTIMATED
free KV budget, per findings #5/#6 of the scheduling simulator
(``~/work/rl-work/agentic/simulator2/FINDINGS.md``). Ported to run for real
against live vLLM replicas instead of a discrete-event simulation - see
``~/.claude/plans/steady-splashing-blanket.md`` for the design writeup.

``AdmissionLedger`` runs as a single Ray actor (the same pattern verl's own
``GlobalRequestLoadBalancer`` uses) so every ``AgentLoopWorker`` process shares
one consistent ledger instead of each worker holding its own private,
inconsistent copy after ``LLMServerClient`` gets pickled to it.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import time

import ray

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class GrowthEstimator:
    """Causal, cross-conversation running estimate of "remaining KV growth",
    generalized per the simulator's follow-up finding
    (``~/work/rl-work/agentic/simulator2/FINDINGS.md`` #8-10): keyed by
    ``reserve_mode`` instead of turn index alone, with an optional
    ``reserve_z`` tail-risk margin on top of the point estimate.

    - ``reserve_mode="turn"``: key = turn index k (the original design).
    - ``reserve_mode="size"``: key = log2 bucket of the CURRENT context size
      - a directly observable, generally stronger signal for "how much more
      will this grow" than turn count alone. This is the mode that won the
      simulator's 12-seed A/B (-7.9% vs sticky at reserve_z~1.5-2.0, vs -4.7%
      for turn+mean).
    - ``reserve_mode="turn_size"``: both jointly (fragments the sample more;
      pairs badly with a large reserve_z per the simulator's finding 9).

    ``reserve_z`` adds ``reserve_z * std`` on top of the mean, where std is a
    SEPARATE, unblended Welford running standard deviation over REAL samples
    only at that key (no principled prior variance to seed it with) - it is
    exactly 0.0 until >=2 real samples land in a key, so it has no effect
    during bootstrap and only kicks in once real variance data exists.
    ``reserve_mode="turn", reserve_z=0.0`` reproduces the original turn-only,
    mean-only estimator exactly.

    Guardrail: only ever fed via ``observe()`` from conversations that have
    ALREADY fully completed (see ``AdmissionLedger.on_trajectory_done``) -
    never from a conversation's own future. Feeding it from a conversation's
    own future turns reproduces the leakage bug the simulator caught: it
    inflated the measured improvement by ~4%.
    """

    def __init__(
        self,
        initial_guess: float,
        prior_weight: float,
        reserve_mode: str = "size",
        reserve_z: float = 0.0,
    ):
        if reserve_mode not in ("turn", "size", "turn_size"):
            raise ValueError(f"unknown reserve_mode: {reserve_mode!r}")
        self._initial_guess = initial_guess
        self._prior_weight = prior_weight
        self._reserve_mode = reserve_mode
        self._reserve_z = reserve_z

        # Prior-blended running mean (unchanged design from the turn-only
        # estimator), keyed per reserve_mode.
        self._growth_mean: dict = {}
        self._growth_mean_n: dict = {}
        # Separate, unblended Welford running mean/variance over REAL samples
        # only, used purely for the std margin.
        self._growth_rmean: dict = {}
        self._growth_m2: dict = {}
        self._growth_count: dict = {}

    @staticmethod
    def _size_bucket(size: float) -> int:
        """log2 bucket of current context size - coarse enough that a few
        hundred conversations still land enough real samples per bucket."""
        return int(math.log2(max(size, 1)))

    def key(self, context_size: float, turn_index: int):
        """The estimator key for a resident at this turn/size, per
        reserve_mode. Callers compute this once and pass it to both
        ``estimate()`` and (once the conversation completes) ``observe()``.
        """
        if self._reserve_mode == "turn":
            return turn_index
        if self._reserve_mode == "size":
            return self._size_bucket(context_size)
        return (turn_index, self._size_bucket(context_size))  # "turn_size"

    def _mean(self, key) -> float:
        return self._growth_mean.get(key, self._initial_guess)

    def _std(self, key) -> float:
        n = self._growth_count.get(key, 0)
        if n < 2:
            return 0.0
        return (self._growth_m2[key] / (n - 1)) ** 0.5

    def estimate(self, key) -> float:
        return self._mean(key) + self._reserve_z * self._std(key)

    def observe(self, key, remaining_growth: float) -> None:
        n = self._growth_mean_n.get(key, self._prior_weight)
        old_mean = self._mean(key)
        self._growth_mean[key] = old_mean + (remaining_growth - old_mean) / (n + 1)
        self._growth_mean_n[key] = n + 1

        c = self._growth_count.get(key, 0) + 1
        rold = self._growth_rmean.get(key, 0.0)
        rnew = rold + (remaining_growth - rold) / c
        self._growth_m2[key] = self._growth_m2.get(key, 0.0) + (remaining_growth - rold) * (remaining_growth - rnew)
        self._growth_rmean[key] = rnew
        self._growth_count[key] = c


@ray.remote
class AdmissionLedger:
    """Per-fleet admission state shared (via Ray actor handle) by every
    ``AgentLoopWorker``'s ``WaveAdmissionLLMClient``.

    Placement prefers staying resident (cheap, incremental) once admitted; if
    ``allow_reactive_migration`` is True (default) and the resident replica
    can no longer fit a conversation's next turn, it falls back to whichever
    other replica currently has room, rather than blocking or overshooting.
    This is reactive (only triggers when the resident replica genuinely lacks
    room), not proactive load-balancing (which would compare every replica's
    projected completion time on every turn regardless of fit).
    """

    def __init__(
        self,
        replicas: list[str],
        *,
        budget_tokens_per_replica: float,
        wave1_size: int,
        initial_growth_guess: float,
        prior_weight: float,
        max_wait_s: float,
        poll_interval_s: float,
        allow_reactive_migration: bool = True,
        reserve_mode: str = "size",
        reserve_z: float = 0.0,
        migration_cost_ratio: float = 1.0,
        p2p_kv_available: bool = False,
        p2p_connector_port: int = 7777,
        migration_cost_ratio_p2p: float = 0.0,
        p2p_nosidecar: bool = False,
        p2p_direct_port: int = 7777,
    ):
        self._replicas = list(replicas)
        self._budget = budget_tokens_per_replica
        self._wave1_size = wave1_size
        self._max_wait_s = max_wait_s
        self._poll_interval_s = poll_interval_s
        self._allow_reactive_migration = allow_reactive_migration
        self._migration_cost_ratio = migration_cost_ratio
        # A migration is only "expensive" (full context_size recompute) when
        # there is no cheap way to move the KV. With the P2P KV-cache-sharing
        # rollout backend (P2PVLLMHttpServer / OffloadingConnector, see
        # p2p_replica.py), every replica offloads its KV to a CPU tier every
        # other replica can pull from - so once we KNOW the exact source
        # replica (our own sticky `_resident` map, not EPP's p2p-source-producer
        # discovery), the migration cost collapses toward a P2P pull instead of
        # a full re-prefill. migration_cost_ratio_p2p default 0.0 is a
        # deliberate benchmarking assumption (per user instruction) to measure
        # the UPPER BOUND of what breaking session-affinity is worth once
        # cross-GPU KV transfer is ~free - not a measured real transfer cost.
        self._p2p_kv_available = p2p_kv_available
        self._p2p_connector_port = p2p_connector_port
        self._migration_cost_ratio_p2p = migration_cost_ratio_p2p
        # p2p_nosidecar: kv_source must carry vLLM's OWN P2P-tier listener port,
        # which p2p_replica.py's launch_server() sets to p2p_direct_port (base,
        # default 7777, matching the llm-d reference guide's convention - NOT
        # vllm.envs.VLLM_P2P_SIDE_CHANNEL_PORT's own unrelated default of 5710)
        # PLUS that replica's own rank (self._replicas.index(...) below) - every
        # replica here shares one node/network-namespace (unlike the reference
        # guide's one-pod-per-replica topology), so a flat, unoffset port would
        # collide across all of them. p2p_connector_port (also 7777 by default,
        # a DIFFERENT, sidecar-only setting) is used instead when going through
        # the sidecar, which has its own separate, incompatible rank-offset
        # scheme - see p2p_replica.py's _launch_sidecar() for why that one is
        # deliberately NOT engaged.
        self._p2p_nosidecar = p2p_nosidecar
        self._p2p_direct_port = p2p_direct_port

        self._used: dict[str, float] = {r: 0.0 for r in replicas}
        self._estimator = GrowthEstimator(
            initial_growth_guess, prior_weight, reserve_mode=reserve_mode, reserve_z=reserve_z,
        )

        # request_id -> assigned replica (residency, sticky for the trajectory).
        self._resident: dict[str, str] = {}
        # request_id -> {turn_index: context_size measured after that turn}.
        self._history: dict[str, dict[int, float]] = {}
        # request_id -> current estimated remaining growth (cached from the
        # estimator at the request's most-recently-observed turn index).
        self._reserve_charge: dict[str, float] = {}

        self._admitted_count = 0
        self._delayed_polls = 0
        self._forced_admissions = 0
        self._migrations = 0

        logger.info(
            "[AdmissionLedger] %d replicas, budget=%.0f tok/replica, wave1_size=%d, "
            "allow_reactive_migration=%s, reserve_mode=%s, reserve_z=%.1f, migration_cost_ratio=%.1f, "
            "p2p_kv_available=%s, migration_cost_ratio_p2p=%.2f, p2p_nosidecar=%s, p2p_direct_port=%d",
            len(replicas), budget_tokens_per_replica, wave1_size, allow_reactive_migration,
            reserve_mode, reserve_z, migration_cost_ratio,
            p2p_kv_available, migration_cost_ratio_p2p, p2p_nosidecar, p2p_direct_port,
        )

    def _reserve(self, replica: str) -> float:
        return sum(
            self._reserve_charge.get(rid, 0.0)
            for rid, r in self._resident.items()
            if r == replica
        )

    def _estimated_free(self, replica: str) -> float:
        return self._budget - self._used[replica] - self._reserve(replica)

    def _least_loaded(self) -> str:
        return max(self._replicas, key=self._estimated_free)

    def _book(self, request_id: str, replica: str, turn_index: int, context_size: float) -> None:
        """Pre-book an ESTIMATED context_size at turn_index on replica.

        If ``replica`` is already this request's resident replica and it has
        a recorded size for turn_index - 1, only the INCREMENTAL delta over
        that prior size is added (the rest is already counted - sticky
        continuation, cheap). Otherwise (turn 0, or landing on a NEW replica
        via migration) the FULL context_size is added fresh - a cold load,
        matching the simulator's full_prefill_cost vs incr_cost distinction.
        ``record_turn`` later corrects this estimate to the actual
        post-generation size.
        """
        history = self._history.setdefault(request_id, {})
        if self._resident.get(request_id) == replica and (turn_index - 1) in history:
            prev = history[turn_index - 1]
            self._used[replica] += max(0.0, context_size - prev)
        else:
            self._used[replica] += context_size
        self._resident[request_id] = replica
        history[turn_index] = context_size
        self._reserve_charge[request_id] = self._estimator.estimate(
            self._estimator.key(context_size, turn_index)
        )

    def _release_booking(self, request_id: str, replica: str, turn_index: int) -> None:
        """Release the booking recorded for request_id at turn_index on
        replica (used when migrating a resident conversation away)."""
        size = self._history.get(request_id, {}).get(turn_index, 0.0)
        self._used[replica] = max(0.0, self._used[replica] - size)

    async def acquire(self, request_id: str, *, turn_index: int, context_size: float) -> dict:
        """Return {"replica": <address>, "kv_source": <address-or-None>} for
        this request to dispatch to. ``kv_source`` is only ever non-None on a
        migration into a NEW replica while ``p2p_kv_available`` is set - it
        names the replica the conversation's KV was previously resident on
        (known exactly, from our own sticky ``_resident`` map), for the
        caller to stamp as ``x-kv-cache-source-host-port`` so the P2P sidecar
        pulls it instead of the destination recomputing from scratch.

        turn_index == 0 (a brand-new conversation): may WAIT (poll) until a
        replica has enough estimated headroom, unless still inside the
        unconditional first wave.
        turn_index > 0: prefers staying on the resident replica (cheap,
        incremental) if it fits; otherwise, if ``allow_reactive_migration``,
        falls back to whichever OTHER replica currently has room for the
        full context - a reactive migration, not a proactive one (it only
        triggers when the resident replica genuinely can't fit the next
        turn, mirroring the simulator's plain "online" policy, not
        "online_lb"'s every-turn projected-completion comparison). If
        migration is disabled, or nowhere fits, it stays resident and
        overshoots rather than blocking an already-running conversation -
        real vLLM's own asymmetry: preemption protects continuations, it
        never stalls them waiting for room that isn't there.
        """
        if turn_index > 0:
            return self._continue_or_migrate(request_id, turn_index, context_size)

        self._admitted_count += 1
        if self._admitted_count <= self._wave1_size:
            replica = self._least_loaded()
        else:
            deadline = time.monotonic() + self._max_wait_s
            replica = None
            while time.monotonic() < deadline:
                candidate = self._least_loaded()
                if self._estimated_free(candidate) >= context_size:
                    replica = candidate
                    break
                self._delayed_polls += 1
                await asyncio.sleep(self._poll_interval_s)
            if replica is None:
                # Safety valve: never hang a training run forever waiting for
                # headroom that may never appear (mirrors the simulator's own
                # deadlock-avoidance concern, FINDINGS.md #1).
                self._forced_admissions += 1
                replica = self._least_loaded()
                logger.warning(
                    "[AdmissionLedger] forced admission of %s after %.1fs wait "
                    "(no replica reached estimated_free>=%.0f)",
                    request_id, self._max_wait_s, context_size,
                )

        self._book(request_id, replica, 0, context_size)
        return {"replica": replica, "kv_source": None}

    def _continue_or_migrate(self, request_id: str, turn_index: int, context_size: float) -> dict:
        resident = self._resident.get(request_id)
        if resident is None:
            # Shouldn't happen (turn 0 always assigns first) but fail safe
            # rather than crash a training run.
            replica = self._least_loaded()
            self._book(request_id, replica, turn_index, context_size)
            return {"replica": replica, "kv_source": None}

        prev_size = self._history.get(request_id, {}).get(turn_index - 1, 0.0)
        incremental_need = max(0.0, context_size - prev_size)
        if self._estimated_free(resident) >= incremental_need:
            self._book(request_id, resident, turn_index, context_size)
            return {"replica": resident, "kv_source": None}

        # Resident can't fit the incremental growth - but migrating is not
        # free in general: with no offload/P2P available, a migrated turn
        # pays a FULL re-prefill of context_size, measured empirically this
        # session at ~33% more wall-clock than an incremental continuation at
        # typical sizes (~2.5s extra at ~50K tokens). Only worth paying that
        # real cost if the shortfall it relieves is itself large relative to
        # that cost - otherwise migration just adds recompute on top of a
        # marginal (possibly false, since our budget estimate is
        # conservative) crisis. migration_cost_ratio=1.0 (default, no P2P)
        # requires the deficit to be at least as large as the full re-prefill
        # it would cost to relieve it.
        #
        # When p2p_kv_available, the destination can PULL the resident's KV
        # over the P2P tier instead of recomputing it (we know the exact
        # source replica - our own `resident` var - so no discovery step is
        # needed), so migration_cost_ratio_p2p (default 0.0, near-free) is
        # used instead: the deficit threshold to justify migrating collapses
        # toward "migrate whenever it doesn't fit and somewhere else does",
        # the real-cluster analog of the simulator's cheap-transfer
        # "online_lb" finding (see FINDINGS.md section D: -29% vs sticky).
        effective_cost_ratio = (
            self._migration_cost_ratio_p2p if self._p2p_kv_available else self._migration_cost_ratio
        )
        deficit = incremental_need - self._estimated_free(resident)
        migration_worth_it = (
            self._allow_reactive_migration
            and deficit >= effective_cost_ratio * context_size
        )
        if migration_worth_it:
            others = [g for g in self._replicas if g != resident]
            target = max(others, key=self._estimated_free, default=None)
            if target is not None and self._estimated_free(target) >= context_size:
                self._migrations += 1
                self._release_booking(request_id, resident, turn_index - 1)
                self._book(request_id, target, turn_index, context_size)
                kv_source = None
                if self._p2p_kv_available:
                    resident_host = resident.rsplit(":", 1)[0]
                    if self._p2p_nosidecar:
                        # Must match p2p_replica.py's launch_server(): vLLM's own
                        # P2P listener on `resident` binds to p2p_direct_port +
                        # resident's own rank (its index in the same ordered
                        # replicas list every AgentLoopManager builds
                        # server_addresses from - see agent_loop_manager.py).
                        port = self._p2p_direct_port + self._replicas.index(resident)
                    else:
                        port = self._p2p_connector_port
                    kv_source = f"{resident_host}:{port}"
                logger.info(
                    "[AdmissionLedger] migrated %s: %s -> %s at turn %d "
                    "(deficit %.0f >= %.2fx migration cost %.0f, kv_source=%s)",
                    request_id, resident, target, turn_index,
                    deficit, effective_cost_ratio, context_size, kv_source,
                )
                return {"replica": target, "kv_source": kv_source}

        # Nowhere (including resident) has room, migration is disabled, or
        # the deficit isn't severe enough to justify the recompute cost:
        # stay resident and overshoot rather than blocking an in-flight
        # conversation or paying to relieve a marginal shortfall.
        self._book(request_id, resident, turn_index, context_size)
        return {"replica": resident, "kv_source": None}

    def record_turn(self, request_id: str, *, turn_index: int, context_size: float) -> None:
        """Called after each turn's generation completes (any turn_index) to
        true up the self-tracked ledger with the ACTUAL context size against
        the ESTIMATE ``acquire()``/``_book`` already pre-booked for this
        turn_index (on whichever replica it ended up on, including a
        mid-conversation migration).
        """
        replica = self._resident.get(request_id)
        if replica is None:
            return
        history = self._history.setdefault(request_id, {})
        prev_size = history.get(turn_index, context_size)
        delta = context_size - prev_size
        if delta:
            self._used[replica] += delta
        history[turn_index] = context_size
        self._reserve_charge[request_id] = self._estimator.estimate(
            self._estimator.key(context_size, turn_index)
        )

    def on_trajectory_done(self, request_id: str) -> None:
        """Release this conversation's ledger entry and feed the growth
        estimator from its now-fully-observed history - ONLY for use by
        other, still-in-flight or future conversations (the causal guardrail
        described on ``GrowthEstimator``).
        """
        replica = self._resident.pop(request_id, None)
        history = self._history.pop(request_id, {})
        self._reserve_charge.pop(request_id, None)
        if replica is None or not history:
            return
        final_size = history[max(history.keys())]
        for turn_idx, size_at_turn in history.items():
            key = self._estimator.key(size_at_turn, turn_idx)
            self._estimator.observe(key, final_size - size_at_turn)
        self._used[replica] = max(0.0, self._used[replica] - final_size)

    def stats(self) -> dict:
        return {
            "used": dict(self._used),
            "estimated_free": {r: self._estimated_free(r) for r in self._replicas},
            "admitted": self._admitted_count,
            "delayed_polls": self._delayed_polls,
            "forced_admissions": self._forced_admissions,
            "migrations": self._migrations,
            "resident_count": len(self._resident),
        }


def compute_budget_tokens_per_replica(
    *,
    gpu_capacity_gb: float | None = None,
    gpu_memory_utilization: float | None = None,
    weights_gb: float | None = None,
    bytes_per_token: float | None = None,
) -> float:
    """Per-GPU KV token budget, same formula validated on-cluster in
    ``~/work/rl-work/weka-sweep-2026-07-24/CTXC.md`` sec 3:
    ``(gpu_memory_utilization * gpu_capacity_gb - weights_gb) / bytes_per_token``.
    All inputs fall back to env vars, then to defaults matching the H200 /
    Qwen2.5-7B-class setup that dataset was calibrated against - override per
    model/util via the ``custom.wave_admission_*`` YAML keys.
    """
    if gpu_capacity_gb is None:
        gpu_capacity_gb = _env_float("WAVE_ADMISSION_GPU_CAPACITY_GB", 139.8)
    if gpu_memory_utilization is None:
        gpu_memory_utilization = _env_float("WAVE_ADMISSION_GPU_UTIL", 0.6)
    if weights_gb is None:
        weights_gb = _env_float("WAVE_ADMISSION_WEIGHTS_GB", 15.2)
    if bytes_per_token is None:
        bytes_per_token = _env_float("WAVE_ADMISSION_KV_BYTES_PER_TOKEN", 57344.0)

    usable_gb = max(0.0, gpu_memory_utilization * gpu_capacity_gb - weights_gb)
    return (usable_gb * 1e9) / bytes_per_token
