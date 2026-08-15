from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import statistics
import time
import traceback
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from orqis.examples.final_experiments import subagents as subagents_example
from orqis.examples.final_experiments.bedrock import bedrock_enabled, bedrock_telemetry_session
from orqis.examples.final_experiments.router_experiment import (
    CandidateConfig,
    aggregate_partition_means,
    aggregate_telemetry,
    estimate_lambda_cost_usd,
    expand_dataset_ids,
    json_compact,
    summarize_phase_entries,
    worker_spec_payload,
    write_json,
    write_measurements_csv,
    ws,
)


DEFAULT_RESULTS_JSON = Path("artifacts/final_experiments/incident_response_swarm_results.json")
DEFAULT_MEASUREMENTS_JSON = Path("artifacts/final_experiments/incident_response_swarm_measurements.json")
DEFAULT_OUTPUT_CSV = Path("artifacts/final_experiments/incident_response_swarm_measurements.csv")
DEFAULT_LATENCY_TARGET_MS = 3000.0
DEFAULT_COORDINATOR_MEMORY_MB = 512
WORKFLOW_ID = "incident_response_swarm"
BASELINE_CANDIDATE = "IR-B"
DATASET_RANGE = "IR1-IR6"

SUBAGENT_LOGICAL_IDS = {
    "identity_subagent": "p_identity_subagent",
    "network_subagent": "p_network_subagent",
    "communications_subagent": "p_communications_subagent",
}
SUBAGENT_FUNCS = {
    "p_identity_subagent": subagents_example.identity_subagent,
    "p_network_subagent": subagents_example.network_subagent,
    "p_communications_subagent": subagents_example.communications_subagent,
}
REDUCER_KEYS = {"findings", "containment_actions", "communication_drafts"}

SUBAGENT_CANDIDATES = [
    CandidateConfig(
        WORKFLOW_ID,
        "compiler baseline",
        "IR-B",
        ["d", "i", "n", "c", "s", "f"],
        [512, 1024, 4096, 512, 512, 256],
        DATASET_RANGE,
        "Delegated-worker structure without memory search; identity and network dominate the baseline footprint.",
        [
            ws("p_ingest_alert_plan_response_fanout", 512, timeout_sec=15),
            ws("p_identity_subagent", 1024, timeout_sec=30),
            ws("p_network_subagent", 4096, timeout_sec=300),
            ws("p_communications_subagent", 512, timeout_sec=15),
            ws("p_synthesize_recommendation", 512, timeout_sec=20),
            ws("p_finalize_incident", 256, timeout_sec=10),
        ],
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "compiler SLO",
        "IR-C",
        ["d", "i", "n", "c", "s", "f"],
        [512, 512, 512, 512, 512, 256],
        DATASET_RANGE,
        "Same partitioning as IR-B but with reduced delegated-worker tiers under the cost-oriented SLO profile.",
        [
            ws("p_ingest_alert_plan_response_fanout", 512, timeout_sec=15),
            ws("p_identity_subagent", 512, timeout_sec=30),
            ws("p_network_subagent", 512, timeout_sec=300),
            ws("p_communications_subagent", 512, timeout_sec=15),
            ws("p_synthesize_recommendation", 512, timeout_sec=20),
            ws("p_finalize_incident", 256, timeout_sec=10),
        ],
        slo_profile="cost_relaxed",
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "compiler SLO",
        "IR-L",
        ["d", "i", "n", "c", "s", "f"],
        [512, 512, 512, 512, 512, 256],
        DATASET_RANGE,
        "Same partitioning and memory outcome as IR-C under the latency-oriented SLO profile.",
        [
            ws("p_ingest_alert_plan_response_fanout", 512, timeout_sec=15),
            ws("p_identity_subagent", 512, timeout_sec=30),
            ws("p_network_subagent", 512, timeout_sec=300),
            ws("p_communications_subagent", 512, timeout_sec=15),
            ws("p_synthesize_recommendation", 512, timeout_sec=20),
            ws("p_finalize_incident", 256, timeout_sec=10),
        ],
        slo_profile="latency_tight",
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "compiler SLO",
        "IR-R",
        ["d", "i", "n", "c", "s", "f"],
        [512, 512, 512, 512, 512, 256],
        DATASET_RANGE,
        "Reliability-oriented rerun still keeps the delegated workers at the reduced tiers once oversizing is removed.",
        [
            ws("p_ingest_alert_plan_response_fanout", 512, timeout_sec=15),
            ws("p_identity_subagent", 512, timeout_sec=30),
            ws("p_network_subagent", 512, timeout_sec=300),
            ws("p_communications_subagent", 512, timeout_sec=15),
            ws("p_synthesize_recommendation", 512, timeout_sec=20),
            ws("p_finalize_incident", 256, timeout_sec=10),
        ],
        slo_profile="reliability_tight",
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "user scenario",
        "IR-U1",
        ["ig", "pl", "i", "n", "c", "s", "f"],
        [1024, 1024, 1024, 1024, 1024, 1024, 1024],
        DATASET_RANGE,
        "Direct node-by-node lift with extra control-path boundaries instead of compiler fusion.",
        [
            ws("p_ingest_alert", 1024, timeout_sec=10),
            ws("p_plan_response_fanout", 1024, timeout_sec=15),
            ws("p_identity_subagent", 1024, timeout_sec=30),
            ws("p_network_subagent", 1024, timeout_sec=300),
            ws("p_communications_subagent", 1024, timeout_sec=15),
            ws("p_synthesize_recommendation", 1024, timeout_sec=20),
            ws("p_finalize_incident", 1024, timeout_sec=10),
        ],
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "user scenario",
        "IR-U2",
        ["(ig+pl+i+n+c+s+f)"],
        [4096],
        DATASET_RANGE,
        "Monolith that collapses delegated parallelism and mixes divergent subagent resource requirements.",
        [ws("p_monolith", 4096, timeout_sec=300)],
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "user scenario",
        "IR-U3",
        ["d", "i", "n", "c", "s", "f"],
        [1024, 2048, 4096, 1024, 1024, 1024],
        DATASET_RANGE,
        "Same partitioning as the compiler rows, but with conservative manual memory sizing.",
        [
            ws("p_ingest_alert_plan_response_fanout", 1024, timeout_sec=15),
            ws("p_identity_subagent", 2048, timeout_sec=30),
            ws("p_network_subagent", 4096, timeout_sec=300),
            ws("p_communications_subagent", 1024, timeout_sec=15),
            ws("p_synthesize_recommendation", 1024, timeout_sec=20),
            ws("p_finalize_incident", 1024, timeout_sec=10),
        ],
    ),
]

CANDIDATE_BY_ID = {candidate.candidate: candidate for candidate in SUBAGENT_CANDIDATES}

SUBAGENT_DATASETS = {
    "IR1": {
        "dataset_id": "IR1",
        "incident_id": "inc-ir1",
        "severity": "medium",
        "alert_summary": (
            "Multiple sign-in failures and a short burst of suspicious VPN activity were detected "
            "against a shared support account."
        ),
        "expected_parallel_branches": 3,
        "scenario": "medium_identity_network_review",
    },
    "IR2": {
        "dataset_id": "IR2",
        "incident_id": "inc-ir2",
        "severity": "high",
        "alert_summary": (
            "Multiple MFA resets were followed by impossible-travel sign-ins and unusual VPN "
            "activity against an administrative account."
        ),
        "expected_parallel_branches": 3,
        "scenario": "high_admin_identity_compromise",
    },
    "IR3": {
        "dataset_id": "IR3",
        "incident_id": "inc-ir3",
        "severity": "critical",
        "alert_summary": (
            "Endpoint telemetry and proxy logs show a coordinated burst of outbound traffic after "
            "an authentication anomaly on a privileged user, suggesting lateral movement and urgent "
            "customer-facing communications needs."
        ),
        "expected_parallel_branches": 3,
        "scenario": "critical_lateral_movement",
    },
    "IR4": {
        "dataset_id": "IR4",
        "incident_id": "inc-ir4",
        "severity": "low",
        "alert_summary": (
            "A customer reported an isolated login issue after a routine SSO change, with minor proxy "
            "noise but no confirmed privilege change."
        ),
        "expected_parallel_branches": 3,
        "scenario": "low_sso_triage",
    },
    "IR5": {
        "dataset_id": "IR5",
        "incident_id": "inc-ir5",
        "severity": "high",
        "alert_summary": (
            "A suspicious administrator sign-in was followed by repeated VPN reconnects, endpoint "
            "telemetry drift, and customer concern about whether the incident requires external updates."
        ),
        "expected_parallel_branches": 3,
        "scenario": "high_admin_vpn_escalation",
    },
    "IR6": {
        "dataset_id": "IR6",
        "incident_id": "inc-ir6",
        "severity": "critical",
        "alert_summary": (
            "Critical alert: impossible-travel sign-ins, proxy anomalies, and endpoint evidence suggest "
            "an active incident affecting a privileged account, with likely containment actions and a "
            "stakeholder-safe communication draft needed before the next executive review."
        ),
        "expected_parallel_branches": 3,
        "scenario": "critical_exec_review_bundle",
    },
}

for dataset in SUBAGENT_DATASETS.values():
    dataset["workflow"] = WORKFLOW_ID
    dataset["input_work_units"] = len(str(dataset["alert_summary"]))


def filter_candidates(candidate_ids: list[str] | None) -> list[CandidateConfig]:
    if not candidate_ids:
        return list(SUBAGENT_CANDIDATES)
    selected = set(candidate_ids)
    filtered = [candidate for candidate in SUBAGENT_CANDIDATES if candidate.candidate in selected]
    found = {candidate.candidate for candidate in filtered}
    missing = sorted(selected - found)
    if missing:
        available = ", ".join(sorted(CANDIDATE_BY_ID))
        raise ValueError(f"unknown candidate id(s): {', '.join(missing)}; available: {available}")
    return filtered


def filter_dataset_ids(dataset_ids: list[str] | None) -> list[str]:
    if not dataset_ids:
        return sorted(SUBAGENT_DATASETS)
    selected = [dataset_id for dataset_id in dataset_ids]
    missing = [dataset_id for dataset_id in selected if dataset_id not in SUBAGENT_DATASETS]
    if missing:
        available = ", ".join(sorted(SUBAGENT_DATASETS))
        raise ValueError(f"unknown dataset id(s): {', '.join(missing)}; available: {available}")
    return selected


def json_size_bytes(value: Any) -> int:
    return len(json_compact(value).encode("utf-8"))


def state_slice(state: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: state.get(key) for key in keys if key in state}


def build_initial_state(dataset_profile: dict[str, Any], invocation_label: str) -> dict[str, Any]:
    state = dict(subagents_example.get_sample_input())
    state.update(
        {
            "incident_id": f"{dataset_profile['incident_id']}::{invocation_label}",
            "severity": dataset_profile["severity"],
            "alert_summary": dataset_profile["alert_summary"],
            "evidence_scope": "",
            "investigation_objective": "",
            "findings": [],
            "containment_actions": [],
            "communication_drafts": [],
            "executive_summary": "",
            "final_recommendation": "",
        }
    )
    return state


def merge_state(state: dict[str, Any], writes: dict[str, Any]) -> None:
    for key, value in writes.items():
        if key in REDUCER_KEYS:
            existing = list(state.get(key) or [])
            existing.extend(list(value or []))
            state[key] = existing
        else:
            state[key] = value


def merge_preview(state: dict[str, Any], writes: dict[str, Any]) -> dict[str, Any]:
    preview = dict(state)
    for key, value in writes.items():
        if key in REDUCER_KEYS:
            existing = list(preview.get(key) or [])
            existing.extend(list(value or []))
            preview[key] = existing
        else:
            preview[key] = value
    return preview


def accumulate_writes(target: dict[str, Any], writes: dict[str, Any]) -> None:
    for key, value in writes.items():
        if key in REDUCER_KEYS:
            existing = list(target.get(key) or [])
            existing.extend(list(value or []))
            target[key] = existing
        else:
            target[key] = value


def execute_partition(
    *,
    spec,
    base_state: dict[str, Any],
    read_keys: list[str],
    invoker: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
    task_input: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    request_payload = dict(state_slice(base_state, read_keys))
    if task_input:
        request_payload.update(task_input)
    request_payload_bytes = json_size_bytes(request_payload)
    state_bytes_before = json_size_bytes(base_state)
    telemetry_events: list[dict[str, Any]] = []
    started = perf_counter()
    writes_payload: dict[str, Any] = {}
    send_descriptors: list[dict[str, Any]] = []
    function_error: str | None = None
    error_trace: str | None = None
    try:
        with bedrock_telemetry_session(telemetry_events.append):
            result = invoker(request_payload, base_state)
        writes_payload = dict(result.get("writes") or {})
        send_descriptors = list(result.get("sends") or [])
    except Exception as exc:
        function_error = f"{type(exc).__name__}: {exc}"
        error_trace = traceback.format_exc(limit=8)
    duration_ms = round((perf_counter() - started) * 1000, 3)
    billed_duration_ms = max(1, int(math.ceil(duration_ms)))
    timed_out = duration_ms > (spec.timeout_sec * 1000.0)
    if timed_out and function_error is None:
        function_error = f"TimeoutError: exceeded configured timeout of {spec.timeout_sec}s"
    if function_error is not None:
        writes_payload = {}
        send_descriptors = []
        writes_bytes = 0
        response_payload_bytes = 0
        preview_state = dict(base_state)
    else:
        writes_bytes = json_size_bytes(writes_payload)
        response_payload_bytes = writes_bytes
        preview_state = merge_preview(base_state, writes_payload)
    send_payload_bytes = sum(json_size_bytes(item) for item in send_descriptors)
    remote_metrics = aggregate_telemetry(telemetry_events)
    checkpoint_trace = {
        "logical_id": spec.logical_id,
        "read_keys": list(read_keys),
        "task_input_keys": sorted(task_input) if task_input else [],
        "write_keys": sorted(writes_payload),
        "request_payload_bytes": request_payload_bytes,
        "writes_bytes": writes_bytes,
        "response_payload_bytes": response_payload_bytes,
        "send_count": len(send_descriptors),
        "send_targets": [descriptor.get("logical_id") or descriptor.get("node") for descriptor in send_descriptors],
        "send_payload_bytes": send_payload_bytes,
        "state_bytes_before": state_bytes_before,
        "state_bytes_after": json_size_bytes(preview_state),
    }
    metric = {
        "logical_id": spec.logical_id,
        "configured_memory_mb": spec.memory_mb,
        "configured_timeout_sec": spec.timeout_sec,
        "configured_concurrency_limit": spec.concurrency_limit,
        "duration_ms": duration_ms,
        "billed_duration_ms": billed_duration_ms,
        "init_duration_ms": None,
        "max_memory_used_mb": None,
        "estimated_cost_usd": estimate_lambda_cost_usd(spec.memory_mb, billed_duration_ms),
        "request_payload_bytes": request_payload_bytes,
        "response_payload_bytes": response_payload_bytes,
        "writes_bytes": writes_bytes,
        "state_bytes_before": state_bytes_before,
        "state_bytes_after": checkpoint_trace["state_bytes_after"],
        "send_count": len(send_descriptors),
        "send_payload_bytes": send_payload_bytes,
        "function_error": function_error,
        "timed_out": timed_out,
        "remote_call_count": remote_metrics["remote_call_count"],
        "remote_latency_ms": remote_metrics["remote_latency_ms"],
        "remote_input_tokens": remote_metrics["remote_input_tokens"],
        "remote_output_tokens": remote_metrics["remote_output_tokens"],
        "remote_total_tokens": remote_metrics["remote_total_tokens"],
        "error_trace": error_trace,
    }
    return metric, writes_payload, send_descriptors, checkpoint_trace


def partition_summary(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        grouped.setdefault(str(metric["logical_id"]), []).append(metric)
    summary: dict[str, dict[str, Any]] = {}
    for logical_id, items in grouped.items():
        summary[logical_id] = {
            "count": len(items),
            "avg_duration_ms": round(statistics.mean(float(item.get("duration_ms") or 0.0) for item in items), 3),
            "avg_billed_duration_ms": round(
                statistics.mean(float(item.get("billed_duration_ms") or 0.0) for item in items),
                3,
            ),
            "avg_request_payload_bytes": round(
                statistics.mean(float(item.get("request_payload_bytes") or 0.0) for item in items),
                3,
            ),
            "avg_writes_bytes": round(
                statistics.mean(float(item.get("writes_bytes") or 0.0) for item in items),
                3,
            ),
            "estimated_cost_usd_total": round(sum(float(item.get("estimated_cost_usd") or 0.0) for item in items), 10),
        }
    return summary


def find_spec(candidate: CandidateConfig, logical_id: str):
    for spec in candidate.worker_defs:
        if spec.logical_id == logical_id:
            return spec
    raise KeyError(f"candidate {candidate.candidate} does not define {logical_id}")


def invoke_dispatch_partition(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
    writes = dict(
        subagents_example.ingest_alert(
            {
                "severity": str(request_payload["severity"]),
                "alert_summary": str(request_payload["alert_summary"]),
            }
        )
    )
    interim = dict(request_payload)
    interim.update(writes)
    writes.update(
        subagents_example.plan_response(
            {
                "incident_id": str(request_payload["incident_id"]),
                "severity": str(request_payload["severity"]),
                "alert_summary": str(interim["alert_summary"]),
                "evidence_scope": str(interim["evidence_scope"]),
            }
        )
    )
    routed = dict(request_payload)
    routed.update(writes)
    sends = []
    for send in subagents_example.fanout_subagents(routed):
        sends.append(
            {
                "node": send.node,
                "logical_id": SUBAGENT_LOGICAL_IDS[send.node],
                "task_input": dict(send.arg),
            }
        )
    return {"writes": writes, "sends": sends}


def invoke_ingest_partition(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
    return {
        "writes": subagents_example.ingest_alert(
            {
                "severity": str(request_payload["severity"]),
                "alert_summary": str(request_payload["alert_summary"]),
            }
        ),
        "sends": [],
    }


def invoke_plan_partition(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
    writes = dict(
        subagents_example.plan_response(
            {
                "incident_id": str(request_payload["incident_id"]),
                "severity": str(request_payload["severity"]),
                "alert_summary": str(request_payload["alert_summary"]),
                "evidence_scope": str(request_payload["evidence_scope"]),
            }
        )
    )
    routed = dict(request_payload)
    routed.update(writes)
    sends = []
    for send in subagents_example.fanout_subagents(routed):
        sends.append(
            {
                "node": send.node,
                "logical_id": SUBAGENT_LOGICAL_IDS[send.node],
                "task_input": dict(send.arg),
            }
        )
    return {"writes": writes, "sends": sends}


def make_subagent_invoker(logical_id: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    subagent_fn = SUBAGENT_FUNCS[logical_id]

    def _invoke(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        return {"writes": subagent_fn(dict(request_payload)), "sends": []}

    return _invoke


def invoke_synthesis_partition(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
    synthesis_input = {
        "findings": list(request_payload.get("findings") or []),
        "containment_actions": list(request_payload.get("containment_actions") or []),
        "communication_drafts": list(request_payload.get("communication_drafts") or []),
    }
    return {"writes": subagents_example.synthesize_recommendation(synthesis_input), "sends": []}


def invoke_finalize_partition(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
    finalize_input = {
        "executive_summary": str(request_payload["executive_summary"]),
        "final_recommendation": str(request_payload["final_recommendation"]),
        "communication_drafts": list(request_payload.get("communication_drafts") or []),
    }
    return {"writes": subagents_example.finalize_incident(finalize_input), "sends": []}


def invoke_monolith(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
    local_state = dict(request_payload)
    writes_total: dict[str, Any] = {}
    step_writes = dict(
        subagents_example.ingest_alert(
            {"severity": str(local_state["severity"]), "alert_summary": str(local_state["alert_summary"])}
        )
    )
    merge_state(local_state, step_writes)
    accumulate_writes(writes_total, step_writes)

    step_writes = dict(
        subagents_example.plan_response(
            {
                "incident_id": str(local_state["incident_id"]),
                "severity": str(local_state["severity"]),
                "alert_summary": str(local_state["alert_summary"]),
                "evidence_scope": str(local_state["evidence_scope"]),
            }
        )
    )
    merge_state(local_state, step_writes)
    accumulate_writes(writes_total, step_writes)

    for send in subagents_example.fanout_subagents(local_state):
        subagent_writes = SUBAGENT_FUNCS[SUBAGENT_LOGICAL_IDS[send.node]](dict(send.arg))
        merge_state(local_state, subagent_writes)
        accumulate_writes(writes_total, dict(subagent_writes))

    step_writes = dict(
        subagents_example.synthesize_recommendation(
            {
                "findings": list(local_state.get("findings") or []),
                "containment_actions": list(local_state.get("containment_actions") or []),
                "communication_drafts": list(local_state.get("communication_drafts") or []),
            }
        )
    )
    merge_state(local_state, step_writes)
    accumulate_writes(writes_total, step_writes)

    step_writes = dict(
        subagents_example.finalize_incident(
            {
                "executive_summary": str(local_state["executive_summary"]),
                "final_recommendation": str(local_state["final_recommendation"]),
                "communication_drafts": list(local_state.get("communication_drafts") or []),
            }
        )
    )
    merge_state(local_state, step_writes)
    accumulate_writes(writes_total, step_writes)
    return {"writes": writes_total, "sends": []}


def execute_subagent_stage(
    candidate: CandidateConfig,
    state: dict[str, Any],
    sends: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], str | None]:
    metrics: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    error: str | None = None
    snapshot = dict(state)
    results: list[tuple[int, dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sends)) as executor:
        futures = []
        for index, send in enumerate(sends):
            spec = find_spec(candidate, str(send["logical_id"]))
            futures.append(
                (
                    index,
                    executor.submit(
                        execute_partition,
                        spec=spec,
                        base_state=snapshot,
                        read_keys=["incident_id", "severity", "evidence_scope"],
                        invoker=make_subagent_invoker(str(send["logical_id"])),
                        task_input=dict(send.get("task_input") or {}),
                    ),
                )
            )
        for index, future in futures:
            metric, writes, _, trace = future.result()
            results.append((index, metric, writes, [], trace))
    results.sort(key=lambda item: item[0])
    for _, metric, writes, _, trace in results:
        metrics.append(metric)
        traces.append(trace)
        if metric.get("function_error") and error is None:
            error = str(metric["function_error"])
        if not metric.get("function_error"):
            merge_state(state, writes)
    return metrics, traces, sends, error


def execute_workflow_run(
    candidate: CandidateConfig,
    dataset_profile: dict[str, Any],
    args: argparse.Namespace,
    invocation_label: str,
) -> dict[str, Any]:
    state = build_initial_state(dataset_profile, invocation_label)
    metrics: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    invoked_partitions: list[str] = []
    started = perf_counter()
    error: str | None = None

    def run_step(
        logical_id: str,
        read_keys: list[str],
        invoker: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
        *,
        task_input: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        spec = find_spec(candidate, logical_id)
        metric, writes, sends, trace = execute_partition(
            spec=spec,
            base_state=state,
            read_keys=read_keys,
            invoker=invoker,
            task_input=task_input,
        )
        metrics.append(metric)
        traces.append(trace)
        invoked_partitions.append(logical_id)
        return metric, writes, sends, trace

    if candidate.candidate in {"IR-B", "IR-C", "IR-L", "IR-R", "IR-U3"}:
        metric, writes, sends, _ = run_step(
            "p_ingest_alert_plan_response_fanout",
            ["incident_id", "severity", "alert_summary"],
            invoke_dispatch_partition,
        )
        if metric.get("function_error"):
            error = str(metric["function_error"])
        else:
            merge_state(state, writes)
        if error is None:
            branch_metrics, branch_traces, _, branch_error = execute_subagent_stage(candidate, state, sends)
            metrics.extend(branch_metrics)
            traces.extend(branch_traces)
            invoked_partitions.extend(str(item["logical_id"]) for item in branch_metrics)
            if branch_error is not None:
                error = branch_error
        if error is None:
            metric, writes, _, _ = run_step(
                "p_synthesize_recommendation",
                ["findings", "containment_actions", "communication_drafts"],
                invoke_synthesis_partition,
            )
            if metric.get("function_error"):
                error = str(metric["function_error"])
            else:
                merge_state(state, writes)
        if error is None:
            metric, writes, _, _ = run_step(
                "p_finalize_incident",
                ["executive_summary", "final_recommendation", "communication_drafts"],
                invoke_finalize_partition,
            )
            if metric.get("function_error"):
                error = str(metric["function_error"])
            else:
                merge_state(state, writes)
    elif candidate.candidate == "IR-U1":
        metric, writes, _, _ = run_step(
            "p_ingest_alert",
            ["severity", "alert_summary"],
            invoke_ingest_partition,
        )
        if metric.get("function_error"):
            error = str(metric["function_error"])
        else:
            merge_state(state, writes)
        sends: list[dict[str, Any]] = []
        if error is None:
            metric, writes, sends, _ = run_step(
                "p_plan_response_fanout",
                ["incident_id", "severity", "alert_summary", "evidence_scope"],
                invoke_plan_partition,
            )
            if metric.get("function_error"):
                error = str(metric["function_error"])
            else:
                merge_state(state, writes)
        if error is None:
            branch_metrics, branch_traces, _, branch_error = execute_subagent_stage(candidate, state, sends)
            metrics.extend(branch_metrics)
            traces.extend(branch_traces)
            invoked_partitions.extend(str(item["logical_id"]) for item in branch_metrics)
            if branch_error is not None:
                error = branch_error
        if error is None:
            metric, writes, _, _ = run_step(
                "p_synthesize_recommendation",
                ["findings", "containment_actions", "communication_drafts"],
                invoke_synthesis_partition,
            )
            if metric.get("function_error"):
                error = str(metric["function_error"])
            else:
                merge_state(state, writes)
        if error is None:
            metric, writes, _, _ = run_step(
                "p_finalize_incident",
                ["executive_summary", "final_recommendation", "communication_drafts"],
                invoke_finalize_partition,
            )
            if metric.get("function_error"):
                error = str(metric["function_error"])
            else:
                merge_state(state, writes)
    elif candidate.candidate == "IR-U2":
        metric, writes, _, _ = run_step(
            "p_monolith",
            ["incident_id", "severity", "alert_summary"],
            invoke_monolith,
        )
        if metric.get("function_error"):
            error = str(metric["function_error"])
        else:
            merge_state(state, writes)
    else:
        raise ValueError(f"unsupported candidate {candidate.candidate}")

    workflow_elapsed_ms = round((perf_counter() - started) * 1000, 3)
    worker_cost_total = round(sum(float(metric.get("estimated_cost_usd") or 0.0) for metric in metrics), 10)
    coordinator_duration_ms = max(0.0, round(workflow_elapsed_ms - sum(float(metric.get("duration_ms") or 0.0) for metric in metrics), 3))
    coordinator_billed_duration_ms = max(1, int(math.ceil(coordinator_duration_ms))) if coordinator_duration_ms > 0 else 0
    subagent_invocations = sum(1 for metric in metrics if metric["logical_id"] in SUBAGENT_FUNCS)
    result_correct = (
        subagent_invocations == dataset_profile["expected_parallel_branches"]
        and bool(state.get("executive_summary"))
        and bool(state.get("final_recommendation"))
        and len(list(state.get("findings") or [])) >= 3
    )
    result_meta = {
        "route_match": result_correct,
        "parallel_branch_count": subagent_invocations,
        "finding_count": len(list(state.get("findings") or [])),
        "containment_action_count": len(list(state.get("containment_actions") or [])),
        "communication_draft_count": len(list(state.get("communication_drafts") or [])),
        "final_state_bytes": json_size_bytes(state),
        "deployed_partition_count": len(candidate.worker_defs),
        "invoked_partition_count": len(metrics),
        "partition_sequence": invoked_partitions,
    }
    return {
        "workflow_elapsed_ms": workflow_elapsed_ms,
        "estimated_cost_usd": worker_cost_total,
        "error_count": sum(1 for metric in metrics if metric.get("function_error")),
        "timeout_count": sum(1 for metric in metrics if metric.get("timed_out")),
        "metrics": metrics,
        "partition_summary": partition_summary(metrics),
        "coordinator_metric": {
            "duration_ms": coordinator_duration_ms,
            "billed_duration_ms": coordinator_billed_duration_ms,
            "init_duration_ms": None,
            "estimated_cost_usd": estimate_lambda_cost_usd(args.coordinator_memory_mb, coordinator_billed_duration_ms)
            if coordinator_billed_duration_ms > 0
            else 0.0,
        },
        "checkpoint_trace": traces,
        "error": error,
        "result_meta": result_meta,
    }


def timed_execution(candidate: CandidateConfig, dataset_profile: dict[str, Any], args: argparse.Namespace, invocation_label: str) -> dict[str, Any]:
    started = perf_counter()
    payload = execute_workflow_run(candidate, dataset_profile, args, invocation_label)
    return {"payload": payload, "client_elapsed_ms": round((perf_counter() - started) * 1000, 3)}


def run_load_batch(
    candidate: CandidateConfig,
    dataset_profile: dict[str, Any],
    args: argparse.Namespace,
    *,
    batch_index: int,
    concurrency: int,
) -> list[dict[str, Any]]:
    started = perf_counter()
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(timed_execution, candidate, dataset_profile, args, f"load-b{batch_index:02d}-s{slot:02d}"): slot
            for slot in range(concurrency)
        }
        for future in concurrent.futures.as_completed(futures):
            slot = futures[future]
            timed = future.result()
            records.append(
                {
                    "phase": "load_measured",
                    "run_index": batch_index,
                    "slot_index": slot,
                    "payload": timed["payload"],
                    "client_elapsed_ms": timed["client_elapsed_ms"],
                }
            )
    batch_makespan_ms = round((perf_counter() - started) * 1000, 3)
    batch_throughput_rps = round(concurrency / (batch_makespan_ms / 1000.0), 6) if batch_makespan_ms > 0 else None
    for record in records:
        record["batch_index"] = batch_index
        record["batch_makespan_ms"] = batch_makespan_ms
        record["batch_throughput_rps"] = batch_throughput_rps
        record["batch_concurrency"] = concurrency
    return sorted(records, key=lambda item: int(item["slot_index"]))


def build_measurements_json(
    detailed_results: list[dict[str, Any]],
    *,
    candidates: list[CandidateConfig],
    args: argparse.Namespace,
) -> dict[str, Any]:
    measurements: list[dict[str, Any]] = []
    baseline_costs: dict[tuple[str, str], float] = {}
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    profiles: dict[tuple[str, str, str], dict[str, Any]] = {}
    completed_map: dict[tuple[str, str, str], bool] = {}
    for item in detailed_results:
        key = (item["workflow"], item["candidate"], item["dataset_id"])
        grouped[key] = list(item["runs"])
        profiles[key] = dict(item["dataset_profile"])
        completed_map[key] = bool(item.get("completed", True))

    for candidate in candidates:
        for dataset_id in expand_dataset_ids(candidate.run_on):
            key = (candidate.workflow, candidate.candidate, dataset_id)
            if key not in grouped:
                continue
            runs = grouped[key]
            dataset_profile = profiles[key]
            single_entries = [entry for entry in runs if entry["phase"] == "measured"]
            cold_entries = [entry for entry in runs if entry["phase"] == "cold_probe"]
            load_entries = [entry for entry in runs if entry["phase"] == "load_measured"]
            measured = [entry["payload"] for entry in single_entries]
            latencies = [payload["workflow_elapsed_ms"] for payload in measured if payload.get("workflow_elapsed_ms") is not None]
            costs = [payload["estimated_cost_usd"] for payload in measured if payload.get("estimated_cost_usd") is not None]
            error_total = sum(int(payload.get("error_count", 0) or 0) for payload in measured)
            timeout_total = sum(int(payload.get("timeout_count", 0) or 0) for payload in measured)
            total_invocations = sum(len(payload.get("metrics", [])) for payload in measured)
            partition_means = aggregate_partition_means(measured)
            single_summary = summarize_phase_entries(
                single_entries,
                modeled_stepfn_transition_usd=float(args.modeled_stepfn_transition_usd),
                modeled_checkpoint_read_request_usd=float(args.modeled_checkpoint_read_request_usd),
                modeled_checkpoint_write_request_usd=float(args.modeled_checkpoint_write_request_usd),
                modeled_checkpoint_read_gb_usd=float(args.modeled_checkpoint_read_gb_usd),
                modeled_checkpoint_write_gb_usd=float(args.modeled_checkpoint_write_gb_usd),
                modeled_bedrock_input_1k_token_usd=float(args.modeled_bedrock_input_1k_token_usd),
                modeled_bedrock_output_1k_token_usd=float(args.modeled_bedrock_output_1k_token_usd),
            )
            cold_summary = summarize_phase_entries(
                cold_entries,
                modeled_stepfn_transition_usd=float(args.modeled_stepfn_transition_usd),
                modeled_checkpoint_read_request_usd=float(args.modeled_checkpoint_read_request_usd),
                modeled_checkpoint_write_request_usd=float(args.modeled_checkpoint_write_request_usd),
                modeled_checkpoint_read_gb_usd=float(args.modeled_checkpoint_read_gb_usd),
                modeled_checkpoint_write_gb_usd=float(args.modeled_checkpoint_write_gb_usd),
                modeled_bedrock_input_1k_token_usd=float(args.modeled_bedrock_input_1k_token_usd),
                modeled_bedrock_output_1k_token_usd=float(args.modeled_bedrock_output_1k_token_usd),
            )
            load_summary = summarize_phase_entries(
                load_entries,
                modeled_stepfn_transition_usd=float(args.modeled_stepfn_transition_usd),
                modeled_checkpoint_read_request_usd=float(args.modeled_checkpoint_read_request_usd),
                modeled_checkpoint_write_request_usd=float(args.modeled_checkpoint_write_request_usd),
                modeled_checkpoint_read_gb_usd=float(args.modeled_checkpoint_read_gb_usd),
                modeled_checkpoint_write_gb_usd=float(args.modeled_checkpoint_write_gb_usd),
                modeled_bedrock_input_1k_token_usd=float(args.modeled_bedrock_input_1k_token_usd),
                modeled_bedrock_output_1k_token_usd=float(args.modeled_bedrock_output_1k_token_usd),
            )
            findings_mean = (
                round(statistics.mean(payload["result_meta"]["finding_count"] for payload in measured), 3)
                if measured
                else None
            )
            containment_mean = (
                round(statistics.mean(payload["result_meta"]["containment_action_count"] for payload in measured), 3)
                if measured
                else None
            )
            comms_mean = (
                round(statistics.mean(payload["result_meta"]["communication_draft_count"] for payload in measured), 3)
                if measured
                else None
            )
            branch_mean = (
                round(statistics.mean(payload["result_meta"]["parallel_branch_count"] for payload in measured), 3)
                if measured
                else None
            )
            load_concurrency = load_summary["batch_concurrency"]
            throughput_target_rps = None
            if load_concurrency and float(args.latency_target_ms) > 0:
                throughput_target_rps = round(load_concurrency / (float(args.latency_target_ms) / 1000.0), 6)

            row = {
                "workflow": candidate.workflow,
                "group": candidate.group,
                "candidate": candidate.candidate,
                "slo_profile": candidate.slo_profile,
                "dataset_id": dataset_id,
                "dataset_scenario": dataset_profile.get("scenario"),
                "dataset_severity": dataset_profile.get("severity"),
                "result_completed": completed_map.get(key, True),
                "partition_count": len(candidate.worker_defs),
                "partitioning_vector": candidate.partitioning_vector,
                "memory_vector_mb_full": candidate.memory_vector_mb,
                "timeout_vector_sec_full": [spec.timeout_sec for spec in candidate.worker_defs],
                "concurrency_vector_full": [spec.concurrency_limit for spec in candidate.worker_defs],
                "worker_resource_plan_json": json_compact([worker_spec_payload(spec) for spec in candidate.worker_defs]),
                "what_it_should_show": candidate.what_it_should_show,
                "latency_p95_ms": statistics.quantiles(latencies, n=100)[94] if len(latencies) >= 2 else (latencies[0] if latencies else None),
                "cost_usd": round(statistics.mean(costs), 10) if costs else None,
                "error_rate": round(error_total / total_invocations, 6) if total_invocations else 0.0,
                "timeout_rate": round(timeout_total / total_invocations, 6) if total_invocations else 0.0,
                "latency_target_ms": float(args.latency_target_ms),
                "partition_runtime_ms": [partition_means[key] for key in [spec.logical_id for spec in candidate.worker_defs] if key in partition_means],
                "partition_memory_mb": [value for value in candidate.memory_vector_mb if value is not None],
                "partition_timeout_sec": [spec.timeout_sec for spec in candidate.worker_defs],
                "partition_concurrency_limit": [spec.concurrency_limit for spec in candidate.worker_defs],
                "declared_total_memory_mb": sum(int(spec.memory_mb) for spec in candidate.worker_defs),
                "declared_max_partition_memory_mb": max(int(spec.memory_mb) for spec in candidate.worker_defs),
                "input_work_units": dataset_profile.get("input_work_units"),
                "cost_total_modeled_usd": single_summary["modeled_total_cost_mean_usd"],
                "cost_component_breakdown_modeled_usd_json": json_compact(
                    {
                        "lambda_worker": single_summary["cost_mean_usd"] or 0.0,
                        "lambda_coordinator": single_summary["coordinator_cost_mean_usd"] or 0.0,
                        "stepfunctions_modeled": single_summary["modeled_stepfn_cost_mean_usd"] or 0.0,
                        "checkpoint_modeled": single_summary["modeled_checkpoint_cost_mean_usd"] or 0.0,
                        "bedrock_modeled": single_summary["modeled_bedrock_cost_mean_usd"] or 0.0,
                    }
                ),
                "single_run_count": single_summary["run_count"],
                "single_client_latency_mean_ms": single_summary["client_latency_mean_ms"],
                "single_client_latency_p95_ms": single_summary["client_latency_p95_ms"],
                "single_client_latency_cv": single_summary["client_latency_cv"],
                "single_workflow_failure_rate": single_summary["workflow_failure_rate"],
                "single_timeout_run_rate": single_summary["timeout_run_rate"],
                "single_worker_invocations_mean": single_summary["worker_invocations_mean"],
                "single_worker_timeout_rate": single_summary["worker_timeout_rate"],
                "single_modeled_total_cost_mean_usd": single_summary["modeled_total_cost_mean_usd"],
                "single_modeled_bedrock_cost_mean_usd": single_summary["modeled_bedrock_cost_mean_usd"],
                "single_remote_call_count_mean": single_summary["remote_call_count_mean"],
                "single_remote_latency_mean_ms": single_summary["remote_latency_total_mean_ms"],
                "single_remote_latency_total_mean_ms": single_summary["remote_latency_total_mean_ms"],
                "single_remote_total_tokens_mean": single_summary["remote_total_tokens_mean"],
                "single_partition_duration_mean_ms_json": single_summary["partition_duration_mean_ms_json"],
                "single_partition_cost_mean_usd_json": single_summary["partition_cost_mean_usd_json"],
                "single_partition_request_bytes_mean_json": single_summary["partition_request_bytes_mean_json"],
                "single_partition_write_bytes_mean_json": single_summary["partition_write_bytes_mean_json"],
                "single_partition_send_count_mean_json": single_summary["partition_send_count_mean_json"],
                "single_result_correct_rate": single_summary["route_match_rate"],
                "single_final_state_bytes_mean": single_summary["final_state_bytes_mean"],
                "single_send_payload_bytes_mean": single_summary["send_payload_bytes_mean"],
                "single_parallel_branch_count_mean": branch_mean,
                "single_finding_count_mean": findings_mean,
                "single_containment_action_count_mean": containment_mean,
                "single_communication_draft_count_mean": comms_mean,
                "cold_probe_run_count": cold_summary["run_count"],
                "cold_client_latency_mean_ms": cold_summary["client_latency_mean_ms"],
                "cold_client_latency_p95_ms": cold_summary["client_latency_p95_ms"],
                "load_request_count": load_summary["run_count"],
                "load_batch_count": load_summary["batch_count"],
                "load_concurrency": load_concurrency,
                "load_client_latency_mean_ms": load_summary["client_latency_mean_ms"],
                "load_client_latency_p95_ms": load_summary["client_latency_p95_ms"],
                "load_client_latency_cv": load_summary["client_latency_cv"],
                "load_batch_makespan_mean_ms": load_summary["batch_makespan_mean_ms"],
                "load_throughput_mean_rps": load_summary["batch_throughput_mean_rps"],
                "load_remote_latency_mean_ms": load_summary["remote_latency_total_mean_ms"],
                "load_modeled_total_cost_mean_usd": load_summary["modeled_total_cost_mean_usd"],
                "load_result_correct_rate": load_summary["route_match_rate"],
                "throughput_target_rps": throughput_target_rps,
                "notes": "",
            }
            measurements.append(row)
            baseline_cost_value = row["cost_total_modeled_usd"] if row["cost_total_modeled_usd"] is not None else row["cost_usd"]
            if candidate.candidate == BASELINE_CANDIDATE and baseline_cost_value is not None:
                baseline_costs[(candidate.workflow, dataset_id)] = baseline_cost_value

    for row in measurements:
        baseline_cost = baseline_costs.get((row["workflow"], row["dataset_id"]))
        if baseline_cost is not None:
            row["cost_baseline_usd"] = baseline_cost

    return {
        "workflow": WORKFLOW_ID,
        "latency_targets_ms": {WORKFLOW_ID: float(args.latency_target_ms)},
        "pricing_model": {
            "stepfn_transition_usd": float(args.modeled_stepfn_transition_usd),
            "checkpoint_read_request_usd": float(args.modeled_checkpoint_read_request_usd),
            "checkpoint_write_request_usd": float(args.modeled_checkpoint_write_request_usd),
            "checkpoint_read_gb_usd": float(args.modeled_checkpoint_read_gb_usd),
            "checkpoint_write_gb_usd": float(args.modeled_checkpoint_write_gb_usd),
            "bedrock_input_1k_token_usd": float(args.modeled_bedrock_input_1k_token_usd),
            "bedrock_output_1k_token_usd": float(args.modeled_bedrock_output_1k_token_usd),
        },
        "notes": [
            "The incident-response candidate matrix matches the thesis experiment table for IR-B, IR-C, IR-L, IR-R, IR-U1, IR-U2, and IR-U3.",
            "single_* fields summarize measured single-request runs.",
            "cold_* fields summarize first-touch probes before warmups.",
            "load_* fields summarize measured concurrent batches when load flags are enabled.",
        ],
        "measurements": measurements,
    }


def build_experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "workflow": WORKFLOW_ID,
        "candidate_ids": list(args.candidate or []),
        "dataset_ids": list(args.dataset or []),
        "runs": int(args.runs),
        "warmup_runs": int(args.warmup_runs),
        "cold_runs": int(args.cold_runs),
        "load_batches": int(args.load_batches),
        "load_concurrency": int(args.load_concurrency),
        "sleep_sec": float(args.sleep_sec),
        "load_sleep_sec": float(args.load_sleep_sec),
        "resume": bool(args.resume),
        "latency_target_ms": float(args.latency_target_ms),
        "coordinator_memory_mb": int(args.coordinator_memory_mb),
        "modeled_stepfn_transition_usd": float(args.modeled_stepfn_transition_usd),
        "modeled_checkpoint_read_request_usd": float(args.modeled_checkpoint_read_request_usd),
        "modeled_checkpoint_write_request_usd": float(args.modeled_checkpoint_write_request_usd),
        "modeled_checkpoint_read_gb_usd": float(args.modeled_checkpoint_read_gb_usd),
        "modeled_checkpoint_write_gb_usd": float(args.modeled_checkpoint_write_gb_usd),
        "modeled_bedrock_input_1k_token_usd": float(args.modeled_bedrock_input_1k_token_usd),
        "modeled_bedrock_output_1k_token_usd": float(args.modeled_bedrock_output_1k_token_usd),
        "bedrock_enabled": bedrock_enabled(),
        "require_bedrock": bool(args.require_bedrock),
        "thesis_exact_matrix": True,
    }


def checkpoint_results(args: argparse.Namespace, detailed_results: list[dict[str, Any]], candidates: list[CandidateConfig]) -> None:
    write_json(args.results_json, {"experiment_config": build_experiment_config(args), "results": detailed_results})
    measurements_payload = build_measurements_json(detailed_results, candidates=candidates, args=args)
    write_json(args.measurements_json, measurements_payload)
    write_measurements_csv(args.output_csv, measurements_payload["measurements"])


def ensure_result_entry(detailed_results: list[dict[str, Any]], candidate: CandidateConfig, dataset_profile: dict[str, Any]) -> dict[str, Any]:
    key = (candidate.workflow, candidate.candidate, dataset_profile["dataset_id"])
    for row in detailed_results:
        if (row["workflow"], row["candidate"], row["dataset_id"]) == key:
            return row
    created = {
        "workflow": candidate.workflow,
        "candidate": candidate.candidate,
        "dataset_id": dataset_profile["dataset_id"],
        "dataset_profile": dict(dataset_profile),
        "runs": [],
        "completed": False,
    }
    detailed_results.append(created)
    return created


def count_phase_entries(run_entries: list[dict[str, Any]], phase: str) -> int:
    return sum(1 for entry in run_entries if entry.get("phase") == phase)


def cmd_catalog(_: argparse.Namespace) -> int:
    print("Candidates")
    for candidate in SUBAGENT_CANDIDATES:
        print(
            f"- {candidate.candidate}: group={candidate.group}, "
            f"slo_profile={candidate.slo_profile or 'baseline/user'}, "
            f"pi={candidate.partitioning_vector}, m={candidate.memory_vector_mb}"
        )
    print("\nDatasets")
    for dataset_id in sorted(SUBAGENT_DATASETS):
        dataset = SUBAGENT_DATASETS[dataset_id]
        print(
            f"- {dataset_id}: severity={dataset['severity']}, "
            f"work_units={dataset['input_work_units']}, scenario={dataset['scenario']}"
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if args.require_bedrock and not bedrock_enabled():
        raise SystemExit(
            "Bedrock is not enabled. Set ORQIS_BEDROCK_MODEL_ID (and optionally ORQIS_BEDROCK_REGION) before using --require-bedrock."
        )
    if int(args.load_batches) > 0 and int(args.load_concurrency) <= 0:
        raise SystemExit("--load-concurrency must be positive when --load-batches is greater than zero.")

    candidates = filter_candidates(args.candidate)
    selected_dataset_ids = filter_dataset_ids(args.dataset)
    detailed_results: list[dict[str, Any]] = []
    if args.resume and args.results_json.exists():
        existing_payload = json.loads(args.results_json.read_text(encoding="utf-8"))
        detailed_results = list(existing_payload.get("results", []))
        print(f"loaded {len(detailed_results)} rows from {args.results_json}")

    for candidate in candidates:
        allowed_dataset_ids = set(expand_dataset_ids(candidate.run_on))
        for dataset_id in selected_dataset_ids:
            if dataset_id not in allowed_dataset_ids:
                continue
            dataset_profile = dict(SUBAGENT_DATASETS[dataset_id])
            row = ensure_result_entry(detailed_results, candidate, dataset_profile)
            if args.resume and row.get("completed"):
                print(f"skip completed {candidate.candidate}/{dataset_id}")
                continue

            cold_done = count_phase_entries(row["runs"], "cold_probe")
            for run_index in range(cold_done, int(args.cold_runs)):
                timed = timed_execution(candidate, dataset_profile, args, f"cold-{run_index:02d}")
                row["runs"].append({"phase": "cold_probe", "run_index": run_index, **timed})
                checkpoint_results(args, detailed_results, candidates)
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)

            warmup_done = count_phase_entries(row["runs"], "warmup")
            for run_index in range(warmup_done, int(args.warmup_runs)):
                timed = timed_execution(candidate, dataset_profile, args, f"warmup-{run_index:02d}")
                row["runs"].append({"phase": "warmup", "run_index": run_index, **timed})
                checkpoint_results(args, detailed_results, candidates)
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)

            measured_done = count_phase_entries(row["runs"], "measured")
            for run_index in range(measured_done, int(args.runs)):
                timed = timed_execution(candidate, dataset_profile, args, f"measured-{run_index:02d}")
                row["runs"].append({"phase": "measured", "run_index": run_index, **timed})
                checkpoint_results(args, detailed_results, candidates)
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)

            existing_batches = sorted(
                {
                    int(entry["batch_index"])
                    for entry in row["runs"]
                    if entry.get("phase") == "load_measured" and entry.get("batch_index") is not None
                }
            )
            next_batch_index = existing_batches[-1] + 1 if existing_batches else 0
            while next_batch_index < int(args.load_batches):
                records = run_load_batch(
                    candidate,
                    dataset_profile,
                    args,
                    batch_index=next_batch_index,
                    concurrency=int(args.load_concurrency),
                )
                row["runs"].extend(records)
                checkpoint_results(args, detailed_results, candidates)
                next_batch_index += 1
                if args.load_sleep_sec > 0:
                    time.sleep(args.load_sleep_sec)

            row["completed"] = True
            checkpoint_results(args, detailed_results, candidates)
            print(
                f"completed {candidate.candidate}/{dataset_id}: "
                f"cold={args.cold_runs}, warmup={args.warmup_runs}, measured={args.runs}, load_batches={args.load_batches}"
            )

    print(f"wrote {args.results_json}")
    print(f"wrote {args.measurements_json}")
    print(f"wrote {args.output_csv}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the final incident-response swarm experiment matrix locally.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("catalog", help="List available subagents candidates and datasets.")

    run = subparsers.add_parser("run", help="Execute the subagents candidate matrix and write resumable results.")
    run.add_argument("--candidate", action="append", help="Candidate id to run. Repeat to select multiple candidates.")
    run.add_argument("--dataset", action="append", help="Dataset id to run. Repeat to select multiple datasets.")
    run.add_argument("--cold-runs", type=int, default=1)
    run.add_argument("--warmup-runs", type=int, default=1)
    run.add_argument("--runs", type=int, default=3)
    run.add_argument("--load-batches", type=int, default=0)
    run.add_argument("--load-concurrency", type=int, default=0)
    run.add_argument("--sleep-sec", type=float, default=0.0)
    run.add_argument("--load-sleep-sec", type=float, default=0.0)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--require-bedrock", action="store_true")
    run.add_argument("--latency-target-ms", type=float, default=DEFAULT_LATENCY_TARGET_MS)
    run.add_argument("--coordinator-memory-mb", type=int, default=DEFAULT_COORDINATOR_MEMORY_MB)
    run.add_argument("--modeled-stepfn-transition-usd", type=float, default=0.000025)
    run.add_argument("--modeled-checkpoint-read-request-usd", type=float, default=0.0)
    run.add_argument("--modeled-checkpoint-write-request-usd", type=float, default=0.0)
    run.add_argument("--modeled-checkpoint-read-gb-usd", type=float, default=0.0)
    run.add_argument("--modeled-checkpoint-write-gb-usd", type=float, default=0.0)
    run.add_argument("--modeled-bedrock-input-1k-token-usd", type=float, default=0.0)
    run.add_argument("--modeled-bedrock-output-1k-token-usd", type=float, default=0.0)
    run.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON)
    run.add_argument("--measurements-json", type=Path, default=DEFAULT_MEASUREMENTS_JSON)
    run.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "catalog":
        return cmd_catalog(args)
    if args.command == "run":
        return cmd_run(args)
    raise SystemExit(f"unknown command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
