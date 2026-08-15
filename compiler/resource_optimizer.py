from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import ceil, sqrt
from typing import Any

from orqis.compiler.ir import (
    AnalysisBundle,
    GraphIR0,
    GraphIR2,
    PartitionIR2,
    ResourceCandidateIR,
    ResourceIR,
    ResourceOptimizationIR,
    StepTraceIR,
)
from orqis.compiler.utils import payload_units


LAMBDA_MEMORY_CANDIDATES_MB = (128, 256, 512, 1024, 1536, 2048, 3008)
DEFAULT_EXECUTOR_IDS = {"lambda_default", "lambda_container_default", "mcp_default"}
RESOURCE_POLICIES = ("baseline", "om", "om2")
DEFAULT_SLO_PROFILE_ID = "prototype"
DEFAULT_OBJECTIVE_WEIGHTS = {
    "cost": 0.48,
    "latency": 0.42,
    "error": 0.10,
}


@dataclass(frozen=True)
class SLOProfile:
    profile_id: str
    description: str
    max_error_rate: float = 0.05
    min_memory_headroom: float = 1.15
    timeout_p99_fraction: float = 0.90
    error_headroom_target: float = 1.25
    error_timeout_pressure_fraction: float = 0.75
    helper_p95_ms: int = 1000
    loop_p95_ms: int = 700
    send_p95_ms: int = 1300
    capability_p95_ms: int = 5000
    llm_p95_ms: int = 3600
    hot_llm_p95_ms: int = 3400


SLO_PROFILES = {
    DEFAULT_SLO_PROFILE_ID: SLOProfile(
        profile_id=DEFAULT_SLO_PROFILE_ID,
        description="Current prototype constraints with moderate safety headroom.",
    ),
    "cost_relaxed": SLOProfile(
        profile_id="cost_relaxed",
        description="Slightly looser safety and latency targets that favor cheaper memory tiers.",
        max_error_rate=0.08,
        min_memory_headroom=1.10,
        timeout_p99_fraction=0.95,
        error_headroom_target=1.15,
        helper_p95_ms=1200,
        loop_p95_ms=850,
        send_p95_ms=1500,
        capability_p95_ms=5600,
        llm_p95_ms=3800,
        hot_llm_p95_ms=3600,
    ),
    "latency_tight": SLOProfile(
        profile_id="latency_tight",
        description="Tighter latency guardrails that penalize slower low-memory candidates.",
        max_error_rate=0.04,
        min_memory_headroom=1.18,
        timeout_p99_fraction=0.88,
        error_headroom_target=1.25,
        helper_p95_ms=850,
        loop_p95_ms=600,
        send_p95_ms=1100,
        capability_p95_ms=4300,
        llm_p95_ms=3000,
        hot_llm_p95_ms=3200,
    ),
    "reliability_tight": SLOProfile(
        profile_id="reliability_tight",
        description="Stricter headroom and failure-risk constraints that preserve more buffer.",
        max_error_rate=0.03,
        min_memory_headroom=1.25,
        timeout_p99_fraction=0.85,
        error_headroom_target=1.30,
        capability_p95_ms=5000,
        llm_p95_ms=3400,
        hot_llm_p95_ms=3300,
    ),
}


def get_slo_profile(profile_id: str = DEFAULT_SLO_PROFILE_ID) -> SLOProfile:
    try:
        return SLO_PROFILES[profile_id]
    except KeyError as exc:
        available = ", ".join(sorted(SLO_PROFILES))
        raise ValueError(f"unknown SLO profile: {profile_id} (available: {available})") from exc


def describe_slo_profile(profile_id: str = DEFAULT_SLO_PROFILE_ID) -> dict[str, Any]:
    return asdict(get_slo_profile(profile_id))


def optimize_partition_resources(
    lgir0: GraphIR0,
    analysis: AnalysisBundle,
    lgir2: GraphIR2,
    runtime_trace: list[StepTraceIR] | None = None,
    *,
    policy_id: str = "om2",
    slo_profile_id: str = DEFAULT_SLO_PROFILE_ID,
) -> dict[str, ResourceOptimizationIR]:
    if policy_id not in RESOURCE_POLICIES:
        raise ValueError(f"unknown resource policy: {policy_id}")
    slo_profile = get_slo_profile(slo_profile_id)
    observed = build_observed_workloads(analysis, lgir2, runtime_trace or [])
    return {
        partition_id: optimize_partition(
            partition,
            observed.get(partition_id, {}),
            policy_id=policy_id,
            slo_profile=slo_profile,
        )
        for partition_id, partition in lgir2.partitions.items()
    }


def optimize_partition(
    partition: PartitionIR2,
    observed: dict[str, Any],
    *,
    policy_id: str = "om2",
    slo_profile: SLOProfile | None = None,
) -> ResourceOptimizationIR:
    slo_profile = slo_profile or get_slo_profile()
    resource_hint = partition.resources or ResourceIR()
    initial_memory = resource_hint.memory_mb or default_initial_memory(partition)
    initial_timeout = resource_hint.timeout_sec or default_initial_timeout(partition)
    initial_concurrency = resource_hint.concurrency_limit
    workload = build_partition_workload(partition, observed)
    peak_memory = estimate_peak_memory_mb(partition, workload)
    if policy_id == "baseline":
        return baseline_partition_optimization(
            partition=partition,
            resource_hint=resource_hint,
            workload=workload,
            initial_memory=initial_memory,
            initial_timeout=initial_timeout,
            initial_concurrency=initial_concurrency,
            peak_memory_mb=peak_memory,
            slo_profile=slo_profile,
        )

    selected_concurrency = choose_concurrency_limit(
        workload,
        initial_concurrency,
        preserve_initial=policy_id == "om2",
    )
    objective_weights = choose_objective_weights(workload, policy_id=policy_id)
    constraints = {
        "slo_profile_id": slo_profile.profile_id,
        "timeout_sec": initial_timeout,
        "slo_p95_latency_ms": latency_slo_ms(partition, initial_timeout, workload, slo_profile),
        "max_error_rate": slo_profile.max_error_rate,
        "min_memory_headroom": slo_profile.min_memory_headroom,
        "timeout_p99_fraction": slo_profile.timeout_p99_fraction,
        "error_headroom_target": slo_profile.error_headroom_target,
        "error_timeout_pressure_fraction": slo_profile.error_timeout_pressure_fraction,
    }
    if policy_id == "om2":
        constraints.update(
            {
                "profile_min_memory_mb": minimum_profile_memory_mb(initial_memory, workload),
                "profile_max_p95_growth": maximum_profile_p95_growth(workload),
                "profile_max_p99_growth": maximum_profile_p99_growth(workload),
            }
        )
    candidates = evaluate_candidates(
        partition=partition,
        workload=workload,
        initial_memory=initial_memory,
        selected_concurrency=selected_concurrency,
        peak_memory_mb=peak_memory,
        constraints=constraints,
        objective_weights=objective_weights,
        apply_profile_guardrails_enabled=policy_id == "om2",
    )
    selected = choose_candidate(candidates)
    selected_timeout = choose_timeout_sec(partition, selected, initial_timeout, workload)
    total_compute_mb = selected.memory_mb * (selected_concurrency or 1)
    reason = build_reason(initial_memory, selected, candidates)
    notes = build_notes(partition, workload, selected, selected_concurrency, policy_id=policy_id)
    return ResourceOptimizationIR(
        partition_id=partition.partition_id,
        policy_id=policy_id,
        strategy="profile_aware_memory_candidate_evaluation"
        if policy_id == "om2"
        else "memory_candidate_evaluation",
        initial_memory_mb=resource_hint.memory_mb,
        initial_timeout_sec=resource_hint.timeout_sec,
        initial_concurrency_limit=initial_concurrency,
        selected_memory_mb=selected.memory_mb,
        selected_timeout_sec=selected_timeout,
        selected_concurrency_limit=selected_concurrency,
        total_compute_mb=total_compute_mb,
        workload=workload,
        constraints=constraints,
        objective_weights=dict(objective_weights),
        candidates=candidates,
        reason=reason,
        notes=notes,
    )


def baseline_partition_optimization(
    *,
    partition: PartitionIR2,
    resource_hint: ResourceIR,
    workload: dict[str, Any],
    initial_memory: int,
    initial_timeout: int,
    initial_concurrency: int | None,
    peak_memory_mb: float,
    slo_profile: SLOProfile,
) -> ResourceOptimizationIR:
    baseline_candidate = evaluate_candidate(
        partition=partition,
        workload=workload,
        memory_mb=initial_memory,
        selected_concurrency=initial_concurrency,
        peak_memory_mb=peak_memory_mb,
        constraints={
            "timeout_sec": initial_timeout,
            "max_error_rate": slo_profile.max_error_rate,
            "min_memory_headroom": slo_profile.min_memory_headroom,
            "timeout_p99_fraction": slo_profile.timeout_p99_fraction,
            "error_headroom_target": slo_profile.error_headroom_target,
            "error_timeout_pressure_fraction": slo_profile.error_timeout_pressure_fraction,
        },
    )
    baseline_candidate.objective_score = 0.0
    baseline_notes = [
        "baseline profile copies the partition resource hints instead of searching Lambda memory candidates",
    ]
    if resource_hint.memory_mb is None or resource_hint.timeout_sec is None:
        baseline_notes.append("missing explicit resource hints fall back to the compiler's default Lambda tier or timeout")
    baseline_notes.extend(baseline_candidate.notes)
    return ResourceOptimizationIR(
        partition_id=partition.partition_id,
        policy_id="baseline",
        strategy="baseline_passthrough_partition_resources",
        initial_memory_mb=resource_hint.memory_mb,
        initial_timeout_sec=resource_hint.timeout_sec,
        initial_concurrency_limit=resource_hint.concurrency_limit,
        selected_memory_mb=initial_memory,
        selected_timeout_sec=initial_timeout,
        selected_concurrency_limit=initial_concurrency,
        total_compute_mb=initial_memory * (initial_concurrency or 1),
        workload=workload,
        constraints={
            "slo_profile_id": slo_profile.profile_id,
            "timeout_sec": initial_timeout,
            "slo_p95_latency_ms": latency_slo_ms(partition, initial_timeout, workload, slo_profile),
            "max_error_rate": slo_profile.max_error_rate,
            "min_memory_headroom": slo_profile.min_memory_headroom,
            "timeout_p99_fraction": slo_profile.timeout_p99_fraction,
            "error_headroom_target": slo_profile.error_headroom_target,
            "error_timeout_pressure_fraction": slo_profile.error_timeout_pressure_fraction,
        },
        objective_weights={},
        candidates=[baseline_candidate],
        reason=(
            f"kept memory at {initial_memory}MB and timeout at {initial_timeout}s by copying the partition's "
            "resource metadata without candidate search"
        ),
        notes=baseline_notes,
    )


def build_observed_workloads(
    analysis: AnalysisBundle,
    lgir2: GraphIR2,
    runtime_trace: list[StepTraceIR],
) -> dict[str, dict[str, Any]]:
    node_to_partition = {
        member: partition_id
        for partition_id, partition in lgir2.partitions.items()
        for member in partition.members
    }
    observations: dict[str, dict[str, Any]] = {
        partition_id: {
            "observed_invocations": 0,
            "observed_peak_concurrency": 0,
            "input_units": [],
            "result_units": [],
            "task_kinds": Counter(),
        }
        for partition_id in lgir2.partitions
    }
    per_step_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for step in runtime_trace:
        for task in step.tasks:
            partition_id = node_to_partition.get(task.node_id)
            if partition_id is None:
                continue
            item = observations[partition_id]
            item["observed_invocations"] += 1
            item["input_units"].append(payload_units(task.input_slice))
            item["result_units"].append(payload_units(task.result))
            item["task_kinds"][task.task_kind] += 1
            per_step_counts[partition_id][str(step.step)] += 1

    for partition_id, item in observations.items():
        peak = max(per_step_counts[partition_id].values(), default=0)
        item["observed_peak_concurrency"] = peak
        item["observed_invocations"] = item["observed_invocations"] or 1
        item["observed_peak_concurrency"] = item["observed_peak_concurrency"] or 1
        item["avg_input_units"] = average(item["input_units"])
        item["max_input_units"] = max(item["input_units"], default=0.0)
        item["avg_result_units"] = average(item["result_units"])
        item["max_result_units"] = max(item["result_units"], default=0.0)
        item["task_kinds"] = dict(item["task_kinds"])

    annotate_fanout_workloads(analysis, lgir2, observations)
    return observations


def annotate_fanout_workloads(
    analysis: AnalysisBundle,
    lgir2: GraphIR2,
    observations: dict[str, dict[str, Any]],
) -> None:
    node_to_partition = {
        member: partition_id
        for partition_id, partition in lgir2.partitions.items()
        for member in partition.members
    }
    for fanout in analysis.fanout_regions:
        source_partition = node_to_partition.get(fanout.fanout_source)
        map_partitions = sorted(
            {
                node_to_partition[map_node]
                for map_node in fanout.map_nodes
                if map_node in node_to_partition
            }
        )
        fanout_width = sum(observations[partition_id]["observed_invocations"] for partition_id in map_partitions) or len(
            fanout.map_nodes
        )
        if source_partition in observations:
            observations[source_partition]["fanout_width"] = max(1, fanout_width)
            observations[source_partition]["fanout_map_partitions"] = map_partitions
        for partition_id in map_partitions:
            observations[partition_id]["is_dynamic_fanout_map"] = True
            observations[partition_id]["fanout_width"] = max(1, observations[partition_id]["observed_invocations"])


def build_partition_workload(partition: PartitionIR2, observed: dict[str, Any]) -> dict[str, Any]:
    effect_domains = partition.side_effects.effect_domains if partition.side_effects else []
    is_effectful = bool(partition.side_effects and partition.side_effects.purity == "Effectful")
    is_llm = "llm" in effect_domains
    is_capability_wrapper = bool(
        partition.tool_binding_ids
        or partition.subagent_ids
        or partition.asset_ids
        or (partition.executor_id and partition.executor_id not in DEFAULT_EXECUTOR_IDS)
    )
    observed_invocations = int(observed.get("observed_invocations", 1) or 1)
    peak_concurrency = int(observed.get("observed_peak_concurrency", 1) or 1)
    fanout_width = int(observed.get("fanout_width", peak_concurrency) or 1)
    max_input_units = float(observed.get("max_input_units", 0.0) or 0.0)
    max_result_units = float(observed.get("max_result_units", 0.0) or 0.0)
    static_units = (
        len(partition.checkpoint_read_set) * 8
        + len(partition.task_input_keys) * 6
        + len(partition.write_set) * 8
        + len(partition.attached_routes) * 10
    )
    work_units = max(1.0, max_input_units + max_result_units + static_units)
    if partition.emits_send:
        work_units += fanout_width * 8
    if partition.loop_component is not None:
        work_units *= 1.25
    if is_llm:
        work_units *= 1.5
    is_remote_bound = bool(set(effect_domains) & {"llm", "db", "network", "http"})
    is_hot_repeated = bool(
        partition.loop_component is not None
        or observed.get("is_dynamic_fanout_map")
        or observed_invocations >= 3
        or peak_concurrency > 1
    )
    return {
        "profile": classify_partition_profile(
            partition=partition,
            is_capability_wrapper=is_capability_wrapper,
            is_hot_repeated=is_hot_repeated,
            is_remote_bound=is_remote_bound,
        ),
        "member_count": len(partition.members),
        "route_count": len(partition.attached_routes),
        "checkpoint_read_keys": len(partition.checkpoint_read_set),
        "task_input_keys": len(partition.task_input_keys),
        "write_keys": len(partition.write_set),
        "emits_send": partition.emits_send,
        "loop_component": partition.loop_component,
        "effect_domains": list(effect_domains),
        "is_effectful": is_effectful,
        "is_llm": is_llm,
        "is_capability_wrapper": is_capability_wrapper,
        "is_remote_bound": is_remote_bound,
        "is_hot_repeated": is_hot_repeated,
        "is_dynamic_fanout_map": bool(observed.get("is_dynamic_fanout_map")),
        "observed_invocations": observed_invocations,
        "observed_peak_concurrency": peak_concurrency,
        "fanout_width": fanout_width,
        "avg_input_units": round(float(observed.get("avg_input_units", 0.0) or 0.0), 3),
        "max_input_units": round(max_input_units, 3),
        "avg_result_units": round(float(observed.get("avg_result_units", 0.0) or 0.0), 3),
        "max_result_units": round(max_result_units, 3),
        "work_units": round(work_units, 3),
    }


def evaluate_candidates(
    *,
    partition: PartitionIR2,
    workload: dict[str, Any],
    initial_memory: int,
    selected_concurrency: int | None,
    peak_memory_mb: float,
    constraints: dict[str, Any],
    objective_weights: dict[str, float],
    apply_profile_guardrails_enabled: bool = True,
) -> list[ResourceCandidateIR]:
    candidates = candidate_memory_values(initial_memory, peak_memory_mb, float(constraints["min_memory_headroom"]))
    evaluated: list[ResourceCandidateIR] = []
    for memory_mb in candidates:
        candidate = evaluate_candidate(
            partition=partition,
            workload=workload,
            memory_mb=memory_mb,
            selected_concurrency=selected_concurrency,
            peak_memory_mb=peak_memory_mb,
            constraints=constraints,
        )
        evaluated.append(candidate)
    baseline = next((candidate for candidate in evaluated if candidate.memory_mb == initial_memory), None)
    reference_cost = max(
        (baseline.estimated_cost_units if baseline is not None else 0.0),
        0.001,
    )
    if baseline is None:
        feasible_costs = [candidate.estimated_cost_units for candidate in evaluated if candidate.feasible]
        reference_cost = min(feasible_costs, default=1.0)

    slo_ms = float(constraints["slo_p95_latency_ms"])
    max_error_rate = float(constraints["max_error_rate"])
    for candidate in evaluated:
        if apply_profile_guardrails_enabled:
            profile_notes = apply_profile_guardrails(candidate, baseline, constraints)
            if profile_notes:
                candidate.feasible = False
                candidate.notes.extend(profile_notes)
        cost_norm = candidate.estimated_cost_units / reference_cost
        latency_norm = candidate.estimated_p95_latency_ms / max(slo_ms, 1.0)
        error_norm = candidate.estimated_error_rate / max(max_error_rate, 0.001)
        infeasible_penalty = 10.0 if not candidate.feasible else 0.0
        candidate.objective_score = round(
            objective_weights["cost"] * cost_norm
            + objective_weights["latency"] * latency_norm
            + objective_weights["error"] * error_norm
            + infeasible_penalty,
            4,
        )
    return evaluated


def evaluate_candidate(
    *,
    partition: PartitionIR2,
    workload: dict[str, Any],
    memory_mb: int,
    selected_concurrency: int | None,
    peak_memory_mb: float,
    constraints: dict[str, Any],
) -> ResourceCandidateIR:
    timeout_sec = int(constraints["timeout_sec"])
    cpu_share = memory_mb / 512.0
    cpu_ms = estimate_cpu_ms(partition, workload)
    io_ms = estimate_io_ms(partition, workload)
    warm_latency = io_ms + cpu_ms / max(cpu_share, 0.1)
    cold_start_ms = estimate_cold_start_ms(partition, memory_mb, cpu_share, workload)
    cold_probability = estimate_cold_start_probability(workload, selected_concurrency)
    p50 = warm_latency
    p95 = warm_latency + cold_start_ms * (0.75 if cold_probability >= 0.05 else 0.25)
    p99 = warm_latency + cold_start_ms
    invocations = max(1, int(workload["observed_invocations"]))
    expected_latency = warm_latency + cold_start_ms * cold_probability
    cost_units = invocations * (memory_mb / 1024.0) * (expected_latency / 1000.0)
    error_rate, error_notes = estimate_error_rate(
        memory_mb=memory_mb,
        peak_memory_mb=peak_memory_mb,
        p99_ms=p99,
        timeout_sec=timeout_sec,
        selected_concurrency=selected_concurrency,
        workload=workload,
        min_memory_headroom=float(constraints["min_memory_headroom"]),
        error_headroom_target=float(constraints.get("error_headroom_target", constraints["min_memory_headroom"])),
        error_timeout_pressure_fraction=float(constraints.get("error_timeout_pressure_fraction", 0.75)),
    )
    min_memory_headroom = float(constraints["min_memory_headroom"])
    timeout_p99_fraction = float(constraints.get("timeout_p99_fraction", 0.9))
    max_error_rate = float(constraints["max_error_rate"])
    feasible = (
        memory_mb >= peak_memory_mb * min_memory_headroom
        and p99 < timeout_sec * 1000 * timeout_p99_fraction
        and error_rate <= max_error_rate
    )
    notes = list(error_notes)
    if memory_mb < peak_memory_mb * min_memory_headroom:
        notes.append("insufficient memory headroom")
    if p99 >= timeout_sec * 1000 * timeout_p99_fraction:
        notes.append("p99 too close to timeout")
    return ResourceCandidateIR(
        memory_mb=memory_mb,
        concurrency_limit=selected_concurrency,
        cpu_share=round(cpu_share, 3),
        estimated_peak_memory_mb=round(peak_memory_mb, 3),
        estimated_p50_latency_ms=round(p50, 3),
        estimated_p95_latency_ms=round(p95, 3),
        estimated_p99_latency_ms=round(p99, 3),
        estimated_cost_units=round(cost_units, 6),
        estimated_error_rate=round(error_rate, 5),
        estimated_cold_start_ms=round(cold_start_ms, 3),
        cold_start_probability=round(cold_probability, 3),
        feasible=feasible,
        notes=notes,
    )


def estimate_peak_memory_mb(partition: PartitionIR2, workload: dict[str, Any]) -> float:
    peak = 72.0
    peak += 18.0 * workload["member_count"]
    peak += 6.0 * workload["checkpoint_read_keys"]
    peak += 5.0 * workload["task_input_keys"]
    peak += 8.0 * workload["write_keys"]
    peak += 10.0 * workload["route_count"]
    peak += min(96.0, workload["work_units"] / 2.0)
    if workload["emits_send"]:
        peak += 28.0 + min(192.0, workload["fanout_width"] * 3.0)
    if workload["is_llm"]:
        peak += 170.0
    elif workload["is_effectful"]:
        peak += 80.0
    if partition.loop_component is not None:
        peak += 24.0
    return max(64.0, peak)


def estimate_cpu_ms(partition: PartitionIR2, workload: dict[str, Any]) -> float:
    cpu_ms = 18.0 * workload["member_count"]
    cpu_ms += 12.0 * workload["checkpoint_read_keys"]
    cpu_ms += 10.0 * workload["write_keys"]
    cpu_ms += 6.0 * workload["task_input_keys"]
    cpu_ms += 9.0 * workload["route_count"]
    cpu_ms += workload["work_units"] * 0.55
    if workload["emits_send"]:
        cpu_ms += 22.0 + workload["fanout_width"] * 2.5
    if workload["is_llm"]:
        llm_cpu_ms, _llm_io_ms = estimate_llm_latency_overheads(partition, workload)
        cpu_ms += llm_cpu_ms
    elif workload["is_effectful"]:
        cpu_ms += 65.0
    if partition.loop_component is not None:
        cpu_ms += 24.0
    return cpu_ms


def estimate_io_ms(partition: PartitionIR2, workload: dict[str, Any]) -> float:
    io_ms = 18.0
    io_ms += 8.0 * workload["checkpoint_read_keys"]
    io_ms += 11.0 * workload["write_keys"]
    io_ms += 5.0 * workload["task_input_keys"]
    io_ms += 8.0 * workload["route_count"]
    if workload["emits_send"]:
        io_ms += 22.0 + min(120.0, workload["fanout_width"] * 4.0)
    if workload["is_llm"]:
        _llm_cpu_ms, llm_io_ms = estimate_llm_latency_overheads(partition, workload)
        io_ms += llm_io_ms
    elif workload["is_effectful"]:
        io_ms += 90.0
    if partition.loop_component is not None:
        io_ms += 20.0
    return io_ms


def estimate_llm_latency_overheads(partition: PartitionIR2, workload: dict[str, Any]) -> tuple[float, float]:
    # Calibrated against the live Bedrock smoke runs for the final router,
    # skills, and subagent examples. The goal here is to lift the absolute
    # latency scale for LLM-heavy partitions without inventing an aggressive
    # memory-speedup curve that we have not measured yet.
    cpu_ms = 280.0 + workload["work_units"] * 0.55
    io_ms = 1800.0
    if partition.loop_component is not None:
        cpu_ms += 80.0
        io_ms += 350.0
    if workload["is_capability_wrapper"] and partition.loop_component is None:
        cpu_ms += 140.0
        io_ms += 700.0
    if workload["task_input_keys"] > 0 and workload["write_keys"] >= 2:
        cpu_ms += 140.0
        io_ms += 900.0
    return cpu_ms, io_ms


def estimate_cold_start_ms(
    partition: PartitionIR2,
    memory_mb: int,
    cpu_share: float,
    workload: dict[str, Any],
) -> float:
    cold_start = 135.0 + 165.0 / sqrt(max(cpu_share, 0.1)) + memory_mb * 0.01
    cold_start += 12.0 * max(0, workload["member_count"] - 1)
    if workload["is_llm"]:
        cold_start += 70.0
    elif workload["is_effectful"]:
        cold_start += 35.0
    if partition.loop_component is not None:
        cold_start -= 25.0
    return max(80.0, cold_start)


def estimate_cold_start_probability(workload: dict[str, Any], selected_concurrency: int | None) -> float:
    invocations = max(1, int(workload["observed_invocations"]))
    peak_concurrency = selected_concurrency or int(workload["observed_peak_concurrency"])
    if invocations > 1 and peak_concurrency <= 1:
        probability = 0.08
    elif peak_concurrency > 1:
        probability = 0.18 + min(0.30, peak_concurrency * 0.035)
    else:
        probability = 0.18
    if workload["is_llm"]:
        probability += 0.04
    return min(0.60, max(0.04, probability))


def estimate_error_rate(
    *,
    memory_mb: int,
    peak_memory_mb: float,
    p99_ms: float,
    timeout_sec: int,
    selected_concurrency: int | None,
    workload: dict[str, Any],
    min_memory_headroom: float,
    error_headroom_target: float,
    error_timeout_pressure_fraction: float,
) -> tuple[float, list[str]]:
    notes: list[str] = []
    error = 0.001
    headroom_target = peak_memory_mb * max(error_headroom_target, min_memory_headroom)
    if memory_mb < headroom_target:
        pressure = (headroom_target - memory_mb) / max(headroom_target, 1.0)
        error += min(0.08, pressure * 0.10)
        notes.append("memory pressure raises retry/oom risk")
    timeout_budget_ms = timeout_sec * 1000.0
    timeout_pressure_floor = timeout_budget_ms * error_timeout_pressure_fraction
    timeout_pressure_window = max(timeout_budget_ms * (1.0 - error_timeout_pressure_fraction), 1.0)
    if p99_ms > timeout_pressure_floor:
        pressure = (p99_ms - timeout_pressure_floor) / timeout_pressure_window
        error += min(0.10, max(0.0, pressure) * 0.08)
        notes.append("latency tail approaches timeout")
    peak_concurrency = int(workload["observed_peak_concurrency"])
    if selected_concurrency is not None and selected_concurrency < peak_concurrency:
        error += 0.02
        notes.append("concurrency limit below observed peak")
    return error, notes


def choose_candidate(candidates: list[ResourceCandidateIR]) -> ResourceCandidateIR:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    pool = feasible or candidates
    return min(pool, key=lambda candidate: (candidate.objective_score or 999.0, candidate.memory_mb))


def choose_concurrency_limit(
    workload: dict[str, Any],
    initial_concurrency: int | None,
    *,
    preserve_initial: bool = True,
) -> int | None:
    if preserve_initial and initial_concurrency is not None:
        return initial_concurrency
    if workload["is_dynamic_fanout_map"]:
        demand = max(1, int(workload["observed_peak_concurrency"]), int(workload["fanout_width"]))
        return min(64, demand)
    return initial_concurrency


def choose_timeout_sec(
    partition: PartitionIR2,
    selected: ResourceCandidateIR,
    initial_timeout_sec: int,
    workload: dict[str, Any],
) -> int:
    timeout = ceil(selected.estimated_p99_latency_ms / 1000.0 + 1.0)
    if workload["is_capability_wrapper"]:
        timeout = max(timeout, initial_timeout_sec)
    elif partition.side_effects and partition.side_effects.purity == "Effectful":
        timeout = max(timeout, 15)
    elif partition.emits_send or partition.loop_component is not None:
        timeout = max(timeout, 5)
    else:
        timeout = max(timeout, 3)
    return int(timeout)


def candidate_memory_values(initial_memory: int, peak_memory_mb: float, min_memory_headroom: float) -> list[int]:
    candidates = set(LAMBDA_MEMORY_CANDIDATES_MB)
    candidates.add(initial_memory)
    required = int(ceil((peak_memory_mb * min_memory_headroom) / 64.0) * 64)
    if required > max(candidates):
        candidates.add(min(10240, required))
    return sorted(memory for memory in candidates if memory >= 128)


def latency_slo_ms(
    partition: PartitionIR2,
    initial_timeout_sec: int,
    workload: dict[str, Any],
    slo_profile: SLOProfile,
) -> int:
    if partition.loop_component is not None and workload["is_llm"]:
        return min(initial_timeout_sec * 1000, slo_profile.hot_llm_p95_ms)
    if workload["is_capability_wrapper"]:
        return min(initial_timeout_sec * 1000, slo_profile.capability_p95_ms)
    if partition.side_effects and "llm" in partition.side_effects.effect_domains:
        return min(initial_timeout_sec * 1000, slo_profile.llm_p95_ms)
    if partition.emits_send:
        return min(initial_timeout_sec * 1000, slo_profile.send_p95_ms)
    if partition.loop_component is not None:
        return min(initial_timeout_sec * 1000, slo_profile.loop_p95_ms)
    return min(initial_timeout_sec * 1000, slo_profile.helper_p95_ms)


def default_initial_memory(partition: PartitionIR2) -> int:
    if partition.side_effects and partition.side_effects.purity == "Effectful":
        return 1536
    if partition.emits_send:
        return 1024
    return 512


def default_initial_timeout(partition: PartitionIR2) -> int:
    if partition.side_effects and partition.side_effects.purity == "Effectful":
        return 90
    if partition.emits_send or partition.loop_component is not None:
        return 30
    return 15


def build_reason(
    initial_memory: int,
    selected: ResourceCandidateIR,
    candidates: list[ResourceCandidateIR],
) -> str:
    feasible_count = sum(1 for candidate in candidates if candidate.feasible)
    direction = "kept"
    if selected.memory_mb < initial_memory:
        direction = "down-sized"
    elif selected.memory_mb > initial_memory:
        direction = "up-sized"
    return (
        f"{direction} memory from {initial_memory}MB to {selected.memory_mb}MB after evaluating "
        f"{len(candidates)} Lambda memory candidates ({feasible_count} feasible); "
        f"selected J={selected.objective_score}, p95={selected.estimated_p95_latency_ms}ms, "
        f"cost={selected.estimated_cost_units}"
    )


def build_notes(
    partition: PartitionIR2,
    workload: dict[str, Any],
    selected: ResourceCandidateIR,
    selected_concurrency: int | None,
    *,
    policy_id: str,
) -> list[str]:
    notes = [
        "CPU share is modeled as proportional to Lambda memory.",
    ]
    if policy_id == "om2":
        notes.append("Objective combines normalized cost, p95 latency, and timeout/oom error risk with profile-aware weights.")
    else:
        notes.append("Objective uses fixed weights J = 0.48 cost + 0.42 latency + 0.10 error with no profile guardrails.")
    if selected_concurrency is not None:
        total_compute = selected.memory_mb * selected_concurrency
        notes.append(f"total compute envelope is concurrency x memory = {total_compute}MB")
    if workload["is_dynamic_fanout_map"]:
        notes.append("fanout map width from the trace is treated as a lower bound, not a reason to shrink explicit parallelism")
    if partition.loop_component is not None:
        notes.append("loop partition receives a tighter latency target because it repeats across iterations")
    if workload["is_hot_repeated"]:
        notes.append("hot repeated work keeps latency close to the initial memory tier so per-invocation slowdown does not amplify across fanout or loops")
    if workload["is_capability_wrapper"]:
        notes.append("capability wrappers preserve the annotated timeout floor because remote model/tool variance is not fully visible in one sample trace")
    if selected.estimated_peak_memory_mb < selected.memory_mb * 0.65:
        notes.append("selected memory has substantial headroom for warm reuse and payload variance")
    return notes


def choose_objective_weights(workload: dict[str, Any], *, policy_id: str) -> dict[str, float]:
    if policy_id == "om":
        return dict(DEFAULT_OBJECTIVE_WEIGHTS)
    if workload["is_hot_repeated"] and workload["is_capability_wrapper"]:
        return {"cost": 0.25, "latency": 0.45, "error": 0.30}
    if workload["is_hot_repeated"]:
        return {"cost": 0.30, "latency": 0.55, "error": 0.15}
    if workload["is_capability_wrapper"]:
        return {"cost": 0.35, "latency": 0.35, "error": 0.30}
    if workload["emits_send"]:
        return {"cost": 0.40, "latency": 0.45, "error": 0.15}
    return dict(DEFAULT_OBJECTIVE_WEIGHTS)


def classify_partition_profile(
    *,
    partition: PartitionIR2,
    is_capability_wrapper: bool,
    is_hot_repeated: bool,
    is_remote_bound: bool,
) -> str:
    if is_hot_repeated and is_capability_wrapper:
        return "hot_capability_wrapper"
    if is_hot_repeated and is_remote_bound:
        return "hot_remote_worker"
    if is_capability_wrapper:
        return "capability_wrapper"
    if partition.emits_send:
        return "fanout_controller"
    if is_hot_repeated:
        return "hot_repeated_worker"
    return "helper_once"


def minimum_profile_memory_mb(initial_memory: int, workload: dict[str, Any]) -> int | None:
    allowed_downshift = None
    if workload["is_hot_repeated"] or workload["is_capability_wrapper"] or workload["is_remote_bound"]:
        allowed_downshift = 1
    elif workload["emits_send"]:
        allowed_downshift = 2
    if allowed_downshift is None:
        return None
    floor = initial_memory
    for _ in range(allowed_downshift):
        next_floor = previous_memory_tier(floor)
        if next_floor == floor:
            break
        floor = next_floor
    return floor


def maximum_profile_p95_growth(workload: dict[str, Any]) -> float | None:
    if workload["is_hot_repeated"] and workload["is_capability_wrapper"]:
        return 1.25
    if workload["is_hot_repeated"]:
        return 1.12
    if workload["is_capability_wrapper"]:
        return 1.10
    if workload["emits_send"]:
        return 1.20
    return None


def maximum_profile_p99_growth(workload: dict[str, Any]) -> float | None:
    if workload["is_hot_repeated"] and workload["is_capability_wrapper"]:
        return 1.30
    if workload["is_hot_repeated"]:
        return 1.15
    if workload["is_capability_wrapper"]:
        return 1.12
    if workload["emits_send"]:
        return 1.25
    return None


def apply_profile_guardrails(
    candidate: ResourceCandidateIR,
    baseline: ResourceCandidateIR | None,
    constraints: dict[str, Any],
) -> list[str]:
    notes: list[str] = []
    profile_min_memory_mb = constraints.get("profile_min_memory_mb")
    if profile_min_memory_mb is not None and candidate.memory_mb < int(profile_min_memory_mb):
        notes.append("below OM2 profile floor for memory down-sizing")
    if baseline is None:
        return notes
    max_p95_growth = constraints.get("profile_max_p95_growth")
    if max_p95_growth is not None and candidate.estimated_p95_latency_ms > baseline.estimated_p95_latency_ms * float(
        max_p95_growth
    ):
        notes.append("p95 degrades too much relative to the initial memory tier")
    max_p99_growth = constraints.get("profile_max_p99_growth")
    if max_p99_growth is not None and candidate.estimated_p99_latency_ms > baseline.estimated_p99_latency_ms * float(
        max_p99_growth
    ):
        notes.append("p99 degrades too much relative to the initial memory tier")
    return notes


def previous_memory_tier(memory_mb: int) -> int:
    tiers = sorted(set(LAMBDA_MEMORY_CANDIDATES_MB + (memory_mb,)))
    lower = [tier for tier in tiers if tier < memory_mb]
    return max(lower, default=memory_mb)


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)
