from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from orqis.examples.final_experiments import router as router_example
from orqis.examples.final_experiments.bedrock import bedrock_enabled, bedrock_telemetry_session


AWS_LAMBDA_GB_SECOND_USD = 0.0000166667
AWS_LAMBDA_REQUEST_USD = 0.20 / 1_000_000
DEFAULT_MODELED_STEPFN_TRANSITION_USD = 0.000025
DEFAULT_MODELED_CHECKPOINT_READ_REQUEST_USD = 0.0
DEFAULT_MODELED_CHECKPOINT_WRITE_REQUEST_USD = 0.0
DEFAULT_MODELED_CHECKPOINT_READ_GB_USD = 0.0
DEFAULT_MODELED_CHECKPOINT_WRITE_GB_USD = 0.0
DEFAULT_MODELED_BEDROCK_INPUT_1K_TOKEN_USD = 0.0
DEFAULT_MODELED_BEDROCK_OUTPUT_1K_TOKEN_USD = 0.0
DEFAULT_RESULTS_JSON = Path("artifacts/final_experiments/service_desk_router_results.json")
DEFAULT_MEASUREMENTS_JSON = Path("artifacts/final_experiments/service_desk_router_measurements.json")
DEFAULT_OUTPUT_CSV = Path("artifacts/final_experiments/service_desk_router_measurements.csv")
DEFAULT_LATENCY_TARGET_MS = 2500.0
DEFAULT_COORDINATOR_MEMORY_MB = 512
WORKFLOW_ID = "service_desk_router"
BASELINE_CANDIDATE = "RT-B"
ROUTER_DATASET_RANGE = "RT1-RT6"

ROUTE_TO_NODE = {
    "billing": "billing_specialist",
    "identity_access": "identity_specialist",
    "vendor_security": "vendor_security_specialist",
}
NODE_TO_LOGICAL_ID = {
    "billing_specialist": "p_billing_specialist",
    "identity_specialist": "p_identity_specialist",
    "vendor_security_specialist": "p_vendor_security_specialist",
}
LOGICAL_ID_TO_SPECIALIST_FN = {
    "p_billing_specialist": router_example.billing_specialist,
    "p_identity_specialist": router_example.identity_specialist,
    "p_vendor_security_specialist": router_example.vendor_security_specialist,
}


@dataclass(frozen=True)
class WorkerSpec:
    logical_id: str
    memory_mb: int
    timeout_sec: int = 180
    concurrency_limit: int | None = None


@dataclass(frozen=True)
class CandidateConfig:
    workflow: str
    group: str
    candidate: str
    partitioning_vector: list[str]
    memory_vector_mb: list[int | None]
    run_on: str
    what_it_should_show: str
    worker_defs: list[WorkerSpec]
    slo_profile: str | None = None


def ws(
    logical_id: str,
    memory_mb: int,
    *,
    timeout_sec: int = 180,
    concurrency_limit: int | None = None,
) -> WorkerSpec:
    return WorkerSpec(
        logical_id=logical_id,
        memory_mb=memory_mb,
        timeout_sec=timeout_sec,
        concurrency_limit=concurrency_limit,
    )


ROUTER_CANDIDATES = [
    CandidateConfig(
        WORKFLOW_ID,
        "compiler baseline",
        "RT-B",
        ["d", "b", "i", "v", "f"],
        [256, 512, 768, 4096, 256],
        ROUTER_DATASET_RANGE,
        "Partitioning benefit without memory optimization; the isolated vendor branch dominates the baseline footprint.",
        [
            ws("p_intake_request_triage_request_fanout", 256, timeout_sec=10),
            ws("p_billing_specialist", 512, timeout_sec=20),
            ws("p_identity_specialist", 768, timeout_sec=25),
            ws("p_vendor_security_specialist", 4096, timeout_sec=180),
            ws("p_finalize_response", 256, timeout_sec=10),
        ],
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "compiler SLO",
        "RT-C",
        ["d", "b", "i", "v", "f"],
        [256, 512, 512, 512, 256],
        ROUTER_DATASET_RANGE,
        "Same partitioning as RT-B but lower memory tiers under the cost-oriented SLO profile.",
        [
            ws("p_intake_request_triage_request_fanout", 256, timeout_sec=10),
            ws("p_billing_specialist", 512, timeout_sec=20),
            ws("p_identity_specialist", 512, timeout_sec=25),
            ws("p_vendor_security_specialist", 512, timeout_sec=180),
            ws("p_finalize_response", 256, timeout_sec=10),
        ],
        slo_profile="cost_relaxed",
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "compiler SLO",
        "RT-L",
        ["d", "b", "i", "v", "f"],
        [256, 512, 512, 512, 256],
        ROUTER_DATASET_RANGE,
        "Same partitioning and memory outcome as RT-C under the latency-oriented SLO profile.",
        [
            ws("p_intake_request_triage_request_fanout", 256, timeout_sec=10),
            ws("p_billing_specialist", 512, timeout_sec=20),
            ws("p_identity_specialist", 512, timeout_sec=25),
            ws("p_vendor_security_specialist", 512, timeout_sec=180),
            ws("p_finalize_response", 256, timeout_sec=10),
        ],
        slo_profile="latency_tight",
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "compiler SLO",
        "RT-R",
        ["d", "b", "i", "v", "f"],
        [512, 512, 512, 512, 256],
        ROUTER_DATASET_RANGE,
        "Reliability pressure only raises the fused dispatch worker; the routed specialists remain at 512 MB.",
        [
            ws("p_intake_request_triage_request_fanout", 512, timeout_sec=10),
            ws("p_billing_specialist", 512, timeout_sec=20),
            ws("p_identity_specialist", 512, timeout_sec=25),
            ws("p_vendor_security_specialist", 512, timeout_sec=180),
            ws("p_finalize_response", 256, timeout_sec=10),
        ],
        slo_profile="reliability_tight",
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "user scenario",
        "RT-U1",
        ["in", "tr", "b", "i", "v", "f"],
        [1024, 1024, 1024, 1024, 1024, 1024],
        ROUTER_DATASET_RANGE,
        "Direct node-by-node lift with extra control-path boundaries and higher orchestration overhead.",
        [
            ws("p_intake_request", 1024, timeout_sec=10),
            ws("p_triage_request", 1024, timeout_sec=10),
            ws("p_billing_specialist", 1024, timeout_sec=20),
            ws("p_identity_specialist", 1024, timeout_sec=25),
            ws("p_vendor_security_specialist", 1024, timeout_sec=180),
            ws("p_finalize_response", 1024, timeout_sec=10),
        ],
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "user scenario",
        "RT-U2",
        ["(in+tr)", "(b+i+v)", "f"],
        [512, 3008, 256],
        ROUTER_DATASET_RANGE,
        "Merged specialist worker that forces every route to pay for the shared heavy branch footprint.",
        [
            ws("p_intake_request_triage_request", 512, timeout_sec=10),
            ws("p_merged_specialist", 3008, timeout_sec=180),
            ws("p_finalize_response", 256, timeout_sec=10),
        ],
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "user scenario",
        "RT-U3",
        ["d", "b", "i", "v", "f"],
        [512, 1024, 1024, 4096, 512],
        ROUTER_DATASET_RANGE,
        "Same partitioning as the compiler rows, but with conservative manual memory sizing.",
        [
            ws("p_intake_request_triage_request_fanout", 512, timeout_sec=10),
            ws("p_billing_specialist", 1024, timeout_sec=20),
            ws("p_identity_specialist", 1024, timeout_sec=25),
            ws("p_vendor_security_specialist", 4096, timeout_sec=180),
            ws("p_finalize_response", 512, timeout_sec=10),
        ],
    ),
]

CANDIDATE_BY_ID = {candidate.candidate: candidate for candidate in ROUTER_CANDIDATES}

ROUTER_DATASETS = {
    "RT1": {
        "dataset_id": "RT1",
        "case_id": "case-rt1",
        "account_tier": "standard",
        "request_text": (
            "Our latest invoice still shows the canceled add-on. Please review the billing charge, "
            "confirm whether a credit is pending, and tell us when the refund will appear."
        ),
        "expected_route": "billing",
        "scenario": "billing_credit_follow_up",
    },
    "RT2": {
        "dataset_id": "RT2",
        "case_id": "case-rt2",
        "account_tier": "enterprise",
        "request_text": (
            "Finance found an invoice delta after the renewal amendment. The customer is asking why the "
            "billing total changed, whether a retroactive credit memo exists, and who owns the refund "
            "reconciliation if the original charge was duplicated across two cost centers."
        ),
        "expected_route": "billing",
        "scenario": "billing_enterprise_reconciliation",
    },
    "RT3": {
        "dataset_id": "RT3",
        "case_id": "case-rt3",
        "account_tier": "business",
        "request_text": (
            "Users cannot log in after the SSO change. Please inspect the login failures and the MFA prompt "
            "loop and tell the identity admin what access step to try next."
        ),
        "expected_route": "identity_access",
        "scenario": "identity_sso_login_failure",
    },
    "RT4": {
        "dataset_id": "RT4",
        "case_id": "case-rt4",
        "account_tier": "enterprise",
        "request_text": (
            "After the Okta routing update, the admin can reach the tenant but several users are blocked by "
            "a repeated MFA challenge and intermittent access denials. Please review the SSO and login trail, "
            "check whether the problem is tenant-specific, and return the next remediation action for the "
            "identity access owner."
        ),
        "expected_route": "identity_access",
        "scenario": "identity_okta_mfa_escalation",
    },
    "RT5": {
        "dataset_id": "RT5",
        "case_id": "case-rt5",
        "account_tier": "enterprise",
        "request_text": (
            "Procurement needs the latest vendor due diligence package and the security questionnaire for "
            "renewal review. Please assemble the vendor security packet and clarify the review owner."
        ),
        "expected_route": "vendor_security",
        "scenario": "vendor_packet_refresh",
    },
    "RT6": {
        "dataset_id": "RT6",
        "case_id": "case-rt6",
        "account_tier": "enterprise",
        "request_text": (
            "The customer renewal depends on a fresh vendor due diligence package, questionnaire evidence, "
            "and a coordinated response across procurement, legal, and security. They requested the control "
            "mapping packet, the review owner, blockers for the renewal timeline, and the target completion "
            "date for the vendor security review."
        ),
        "expected_route": "vendor_security",
        "scenario": "vendor_due_diligence_coordination",
    },
}

for dataset in ROUTER_DATASETS.values():
    dataset["workflow"] = WORKFLOW_ID
    dataset["input_work_units"] = len(str(dataset["request_text"]))


def mean_or_none(values: list[float], digits: int = 6) -> float | None:
    if not values:
        return None
    return round(statistics.mean(values), digits)


def coefficient_of_variation(values: list[float], digits: int = 6) -> float | None:
    if not values:
        return None
    mean = statistics.mean(values)
    if mean == 0:
        return None
    if len(values) == 1:
        return 0.0
    return round(statistics.pstdev(values) / mean, digits)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return round(ordered[index], 3)


def max_partition_share(metrics: list[dict[str, Any]], value_key: str) -> float | None:
    totals: dict[str, float] = {}
    total = 0.0
    for metric in metrics:
        value = metric.get(value_key)
        if value is None:
            continue
        numeric = float(value)
        if numeric < 0:
            continue
        logical_id = str(metric.get("logical_id", ""))
        totals[logical_id] = totals.get(logical_id, 0.0) + numeric
        total += numeric
    if total <= 0 or not totals:
        return None
    return round(max(totals.values()) / total, 6)


def aggregate_metric_map(metrics: list[dict[str, Any]], value_key: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for metric in metrics:
        value = metric.get(value_key)
        if value is None:
            continue
        logical_id = str(metric.get("logical_id", ""))
        totals[logical_id] = totals.get(logical_id, 0.0) + float(value)
    return totals


def mean_mapping(mappings: list[dict[str, float]], digits: int = 6) -> dict[str, float]:
    keys = sorted({key for mapping in mappings for key in mapping})
    result: dict[str, float] = {}
    for key in keys:
        values = [mapping[key] for mapping in mappings if key in mapping]
        if values:
            result[key] = round(statistics.mean(values), digits)
    return result


def json_compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def estimate_lambda_cost_usd(memory_mb: int | float, billed_duration_ms: int | float | None) -> float | None:
    if billed_duration_ms is None:
        return None
    billed_seconds = float(billed_duration_ms) / 1000.0
    memory_gb = float(memory_mb) / 1024.0
    return round((memory_gb * billed_seconds * AWS_LAMBDA_GB_SECOND_USD) + AWS_LAMBDA_REQUEST_USD, 10)


def modeled_checkpoint_cost_usd(
    *,
    read_count: float,
    write_count: float,
    read_bytes: float,
    write_bytes: float,
    read_request_usd: float,
    write_request_usd: float,
    read_gb_usd: float,
    write_gb_usd: float,
) -> float:
    gib = 1024.0 * 1024.0 * 1024.0
    return (
        (read_count * read_request_usd)
        + (write_count * write_request_usd)
        + ((read_bytes / gib) * read_gb_usd)
        + ((write_bytes / gib) * write_gb_usd)
    )


def expand_dataset_ids(run_on: str) -> list[str]:
    start_text, end_text = run_on.split("-", 1)
    prefix = "".join(ch for ch in start_text if not ch.isdigit())
    start = int(start_text[len(prefix):])
    end = int(end_text[len(prefix):])
    return [f"{prefix}{index}" for index in range(start, end + 1)]


def filter_candidates(candidate_ids: list[str] | None) -> list[CandidateConfig]:
    if not candidate_ids:
        return list(ROUTER_CANDIDATES)
    selected = set(candidate_ids)
    filtered = [candidate for candidate in ROUTER_CANDIDATES if candidate.candidate in selected]
    found = {candidate.candidate for candidate in filtered}
    missing = sorted(selected - found)
    if missing:
        available = ", ".join(sorted(CANDIDATE_BY_ID))
        raise ValueError(f"unknown candidate id(s): {', '.join(missing)}; available: {available}")
    return filtered


def filter_dataset_ids(dataset_ids: list[str] | None) -> list[str]:
    if not dataset_ids:
        return sorted(ROUTER_DATASETS)
    selected = [dataset_id for dataset_id in dataset_ids]
    missing = [dataset_id for dataset_id in selected if dataset_id not in ROUTER_DATASETS]
    if missing:
        available = ", ".join(sorted(ROUTER_DATASETS))
        raise ValueError(f"unknown dataset id(s): {', '.join(missing)}; available: {available}")
    return selected


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def normalize_csv_value(value: Any) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return value
    return json_compact(value)


def write_measurements_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: normalize_csv_value(row.get(key)) for key in fieldnames})


def worker_spec_payload(spec: WorkerSpec) -> dict[str, Any]:
    return {
        "logical_id": spec.logical_id,
        "memory_mb": spec.memory_mb,
        "timeout_sec": spec.timeout_sec,
        "concurrency_limit": spec.concurrency_limit,
    }


def build_initial_state(dataset_profile: dict[str, Any], invocation_label: str) -> dict[str, Any]:
    state = dict(router_example.get_sample_input())
    state.update(
        {
            "case_id": f"{dataset_profile['case_id']}::{invocation_label}",
            "account_tier": dataset_profile["account_tier"],
            "request_text": dataset_profile["request_text"],
            "normalized_text": "",
            "route_decision": "",
            "evidence_refs": [],
            "follow_up_plan": [],
            "response_draft": "",
            "final_response": "",
        }
    )
    return state


def state_slice(state: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: state.get(key) for key in keys if key in state}


def json_size_bytes(value: Any) -> int:
    return len(json_compact(value).encode("utf-8"))


def json_roundtrip(value: Any) -> tuple[Any, int]:
    encoded = json_compact(value)
    return json.loads(encoded), len(encoded.encode("utf-8"))


def aggregate_telemetry(events: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "remote_call_count": float(len(events)),
        "remote_latency_ms": round(sum(float(event.get("latency_ms", 0.0) or 0.0) for event in events), 3),
        "remote_input_tokens": float(sum(int(event.get("input_tokens", 0) or 0) for event in events)),
        "remote_output_tokens": float(sum(int(event.get("output_tokens", 0) or 0) for event in events)),
        "remote_total_tokens": float(sum(int(event.get("total_tokens", 0) or 0) for event in events)),
    }


def state_digest(state: dict[str, Any]) -> str:
    return hashlib.sha1(json_compact(state).encode("utf-8")).hexdigest()[:12]


def invoke_dispatch_partition(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
    writes = dict(router_example.intake_request({"request_text": str(request_payload["request_text"])}))
    interim = dict(request_payload)
    interim.update(writes)
    writes.update(
        router_example.triage_request(
            {
                "account_tier": str(interim["account_tier"]),
                "normalized_text": str(interim["normalized_text"]),
            }
        )
    )
    routed = dict(interim)
    routed.update(writes)
    target_node = router_example.route_specialist(routed)
    return {
        "writes": writes,
        "sends": [{"node": target_node, "logical_id": NODE_TO_LOGICAL_ID[target_node]}],
    }


def invoke_intake_partition(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
    return {"writes": router_example.intake_request({"request_text": str(request_payload["request_text"])}), "sends": []}


def invoke_triage_partition(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
    writes = dict(
        router_example.triage_request(
            {
                "account_tier": str(request_payload["account_tier"]),
                "normalized_text": str(request_payload["normalized_text"]),
            }
        )
    )
    routed = dict(request_payload)
    routed.update(writes)
    target_node = router_example.route_specialist(routed)
    return {
        "writes": writes,
        "sends": [{"node": target_node, "logical_id": NODE_TO_LOGICAL_ID[target_node]}],
    }


def invoke_intake_triage_partition(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
    writes = dict(router_example.intake_request({"request_text": str(request_payload["request_text"])}))
    triage_input = {
        "account_tier": str(request_payload["account_tier"]),
        "normalized_text": str(writes["normalized_text"]),
    }
    writes.update(router_example.triage_request(triage_input))
    return {
        "writes": writes,
        "sends": [{"node": "merged_specialist", "logical_id": "p_merged_specialist"}],
    }


def make_specialist_invoker(logical_id: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    specialist_fn = LOGICAL_ID_TO_SPECIALIST_FN[logical_id]

    def _invoke(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        specialist_input = {
            "case_id": str(request_payload["case_id"]),
            "account_tier": str(request_payload["account_tier"]),
            "normalized_text": str(request_payload["normalized_text"]),
        }
        return {"writes": specialist_fn(specialist_input), "sends": []}

    return _invoke


def invoke_merged_specialist(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
    route_decision = str(request_payload["route_decision"])
    target_node = ROUTE_TO_NODE[route_decision]
    specialist_fn = LOGICAL_ID_TO_SPECIALIST_FN[NODE_TO_LOGICAL_ID[target_node]]
    specialist_input = {
        "case_id": str(request_payload["case_id"]),
        "account_tier": str(request_payload["account_tier"]),
        "normalized_text": str(request_payload["normalized_text"]),
    }
    return {"writes": specialist_fn(specialist_input), "sends": []}


def invoke_finalize_partition(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
    finalize_input = {
        "route_decision": str(request_payload["route_decision"]),
        "response_draft": str(request_payload["response_draft"]),
        "evidence_refs": list(request_payload.get("evidence_refs") or []),
        "follow_up_plan": list(request_payload.get("follow_up_plan") or []),
    }
    return {"writes": router_example.finalize_response(finalize_input), "sends": []}


def run_partition(
    *,
    spec: WorkerSpec,
    state: dict[str, Any],
    read_keys: list[str],
    invoker: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    request_slice = state_slice(state, read_keys)
    request_payload, request_payload_bytes = json_roundtrip(request_slice)
    telemetry_events: list[dict[str, Any]] = []
    state_bytes_before = json_size_bytes(state)
    started = perf_counter()
    writes_payload: dict[str, Any] = {}
    send_descriptors: list[dict[str, Any]] = []
    function_error: str | None = None
    error_trace: str | None = None
    try:
        with bedrock_telemetry_session(telemetry_events.append):
            result = invoker(request_payload, state)
        writes_payload = dict(result.get("writes") or {})
        send_descriptors = list(result.get("sends") or [])
    except Exception as exc:
        function_error = f"{type(exc).__name__}: {exc}"
        error_trace = traceback.format_exc(limit=8)
    duration_ms = round((perf_counter() - started) * 1000, 3)
    billed_duration_ms = max(1, int(math.ceil(duration_ms)))
    timeout_ms = spec.timeout_sec * 1000.0
    timed_out = duration_ms > timeout_ms
    if timed_out and function_error is None:
        function_error = f"TimeoutError: exceeded configured timeout of {spec.timeout_sec}s"
    if function_error is not None:
        writes_payload = {}
        send_descriptors = []
        writes_bytes = 0
        response_payload_bytes = 0
    else:
        writes_payload, writes_bytes = json_roundtrip(writes_payload)
        response_payload_bytes = writes_bytes
        state.update(writes_payload)
    send_payload_bytes = sum(json_size_bytes(descriptor) for descriptor in send_descriptors)
    remote_metrics = aggregate_telemetry(telemetry_events)
    state_bytes_after = json_size_bytes(state)
    checkpoint_trace = {
        "logical_id": spec.logical_id,
        "read_keys": list(read_keys),
        "write_keys": sorted(writes_payload),
        "request_payload_bytes": request_payload_bytes,
        "writes_bytes": writes_bytes,
        "response_payload_bytes": response_payload_bytes,
        "send_count": len(send_descriptors),
        "send_targets": [descriptor.get("logical_id") or descriptor.get("node") for descriptor in send_descriptors],
        "send_payload_bytes": send_payload_bytes,
        "state_bytes_before": state_bytes_before,
        "state_bytes_after": state_bytes_after,
        "state_digest_after": state_digest(state),
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
        "state_bytes_after": state_bytes_after,
        "send_count": len(send_descriptors),
        "send_payload_bytes": send_payload_bytes,
        "function_error": function_error,
        "timed_out": timed_out,
        "remote_call_count": remote_metrics["remote_call_count"],
        "remote_latency_ms": remote_metrics["remote_latency_ms"],
        "remote_input_tokens": remote_metrics["remote_input_tokens"],
        "remote_output_tokens": remote_metrics["remote_output_tokens"],
        "remote_total_tokens": remote_metrics["remote_total_tokens"],
        "route_decision": state.get("route_decision"),
        "checkpoint_read_keys": list(read_keys),
        "checkpoint_write_keys": sorted(writes_payload),
        "error_trace": error_trace,
    }
    return metric, send_descriptors, checkpoint_trace


def partition_summary(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for metric in metrics:
        grouped.setdefault(str(metric["logical_id"]), []).append(metric)
    summary: dict[str, dict[str, Any]] = {}
    for logical_id, items in grouped.items():
        durations = [float(item.get("duration_ms") or 0.0) for item in items]
        billed = [float(item.get("billed_duration_ms") or 0.0) for item in items]
        requests = [float(item.get("request_payload_bytes") or 0.0) for item in items]
        writes = [float(item.get("writes_bytes") or 0.0) for item in items]
        costs = [float(item.get("estimated_cost_usd") or 0.0) for item in items]
        summary[logical_id] = {
            "count": len(items),
            "avg_duration_ms": round(statistics.mean(durations), 3) if durations else None,
            "avg_billed_duration_ms": round(statistics.mean(billed), 3) if billed else None,
            "avg_request_payload_bytes": round(statistics.mean(requests), 3) if requests else None,
            "avg_writes_bytes": round(statistics.mean(writes), 3) if writes else None,
            "estimated_cost_usd_total": round(sum(costs), 10),
        }
    return summary


def find_spec(candidate: CandidateConfig, logical_id: str) -> WorkerSpec:
    for spec in candidate.worker_defs:
        if spec.logical_id == logical_id:
            return spec
    raise KeyError(f"candidate {candidate.candidate} does not define {logical_id}")


def execute_router_run(
    candidate: CandidateConfig,
    dataset_profile: dict[str, Any],
    args: argparse.Namespace,
    invocation_label: str,
) -> dict[str, Any]:
    state = build_initial_state(dataset_profile, invocation_label)
    metrics: list[dict[str, Any]] = []
    checkpoint_trace: list[dict[str, Any]] = []
    invoked_partitions: list[str] = []
    started = perf_counter()
    error: str | None = None

    def step(logical_id: str, read_keys: list[str], invoker: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]) -> list[dict[str, Any]]:
        nonlocal error
        spec = find_spec(candidate, logical_id)
        metric, sends, trace = run_partition(spec=spec, state=state, read_keys=read_keys, invoker=invoker)
        metrics.append(metric)
        checkpoint_trace.append(trace)
        invoked_partitions.append(logical_id)
        if metric.get("function_error"):
            error = str(metric["function_error"])
        return sends

    if candidate.candidate in {"RT-B", "RT-C", "RT-L", "RT-R", "RT-U3"}:
        sends = step(
            "p_intake_request_triage_request_fanout",
            ["request_text", "account_tier"],
            invoke_dispatch_partition,
        )
        if error is None and sends:
            specialist_logical_id = str(sends[0]["logical_id"])
            step(
                specialist_logical_id,
                ["case_id", "account_tier", "normalized_text"],
                make_specialist_invoker(specialist_logical_id),
            )
        if error is None:
            step(
                "p_finalize_response",
                ["route_decision", "response_draft", "evidence_refs", "follow_up_plan"],
                invoke_finalize_partition,
            )
    elif candidate.candidate == "RT-U1":
        step("p_intake_request", ["request_text"], invoke_intake_partition)
        sends: list[dict[str, Any]] = []
        if error is None:
            sends = step(
                "p_triage_request",
                ["account_tier", "normalized_text"],
                invoke_triage_partition,
            )
        if error is None and sends:
            specialist_logical_id = str(sends[0]["logical_id"])
            step(
                specialist_logical_id,
                ["case_id", "account_tier", "normalized_text"],
                make_specialist_invoker(specialist_logical_id),
            )
        if error is None:
            step(
                "p_finalize_response",
                ["route_decision", "response_draft", "evidence_refs", "follow_up_plan"],
                invoke_finalize_partition,
            )
    elif candidate.candidate == "RT-U2":
        step(
            "p_intake_request_triage_request",
            ["request_text", "account_tier"],
            invoke_intake_triage_partition,
        )
        if error is None:
            step(
                "p_merged_specialist",
                ["case_id", "account_tier", "normalized_text", "route_decision"],
                invoke_merged_specialist,
            )
        if error is None:
            step(
                "p_finalize_response",
                ["route_decision", "response_draft", "evidence_refs", "follow_up_plan"],
                invoke_finalize_partition,
            )
    else:
        raise ValueError(f"unsupported candidate {candidate.candidate}")

    workflow_elapsed_ms = round((perf_counter() - started) * 1000, 3)
    worker_cost_total = round(sum(float(metric.get("estimated_cost_usd") or 0.0) for metric in metrics), 10)
    coordinator_duration_ms = max(0.0, round(workflow_elapsed_ms - sum(float(metric.get("duration_ms") or 0.0) for metric in metrics), 3))
    coordinator_billed_duration_ms = max(1, int(math.ceil(coordinator_duration_ms))) if coordinator_duration_ms > 0 else 0
    final_state_bytes = json_size_bytes(state)
    result_meta = {
        "expected_route": dataset_profile["expected_route"],
        "observed_route": state.get("route_decision"),
        "route_match": state.get("route_decision") == dataset_profile["expected_route"],
        "final_response_chars": len(str(state.get("final_response") or "")),
        "declared_total_memory_mb": sum(int(spec.memory_mb) for spec in candidate.worker_defs),
        "max_declared_partition_memory_mb": max(int(spec.memory_mb) for spec in candidate.worker_defs),
        "invoked_partition_count": len(metrics),
        "deployed_partition_count": len(candidate.worker_defs),
        "partition_sequence": invoked_partitions,
        "final_state_bytes": final_state_bytes,
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
        "checkpoint_trace": checkpoint_trace,
        "error": error,
        "result_meta": result_meta,
    }


def timed_execution(
    candidate: CandidateConfig,
    dataset_profile: dict[str, Any],
    args: argparse.Namespace,
    invocation_label: str,
) -> dict[str, Any]:
    started = perf_counter()
    payload = execute_router_run(candidate, dataset_profile, args, invocation_label)
    return {
        "payload": payload,
        "client_elapsed_ms": round((perf_counter() - started) * 1000, 3),
    }


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
            executor.submit(
                timed_execution,
                candidate,
                dataset_profile,
                args,
                f"load-b{batch_index:02d}-s{slot_index:02d}",
            ): slot_index
            for slot_index in range(concurrency)
        }
        for future in concurrent.futures.as_completed(futures):
            slot_index = futures[future]
            timed = future.result()
            records.append(
                {
                    "phase": "load_measured",
                    "run_index": batch_index,
                    "slot_index": slot_index,
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


def summarize_phase_entries(
    entries: list[dict[str, Any]],
    *,
    modeled_stepfn_transition_usd: float,
    modeled_checkpoint_read_request_usd: float,
    modeled_checkpoint_write_request_usd: float,
    modeled_checkpoint_read_gb_usd: float,
    modeled_checkpoint_write_gb_usd: float,
    modeled_bedrock_input_1k_token_usd: float,
    modeled_bedrock_output_1k_token_usd: float,
) -> dict[str, Any]:
    payloads = [entry["payload"] for entry in entries if entry.get("payload")]
    workflow_latencies = [
        float(payload["workflow_elapsed_ms"])
        for payload in payloads
        if payload.get("workflow_elapsed_ms") is not None
    ]
    client_latencies = [
        float(entry.get("client_elapsed_ms", entry["payload"].get("workflow_elapsed_ms")))
        for entry in entries
        if entry.get("client_elapsed_ms", entry["payload"].get("workflow_elapsed_ms")) is not None
    ]
    costs = [
        float(payload["estimated_cost_usd"])
        for payload in payloads
        if payload.get("estimated_cost_usd") is not None
    ]
    workflow_failures = sum(1 for payload in payloads if payload.get("error") is not None)
    timeout_runs = sum(1 for payload in payloads if int(payload.get("timeout_count", 0) or 0) > 0)
    route_matches = [
        1.0 if payload.get("result_meta", {}).get("route_match") else 0.0
        for payload in payloads
        if payload.get("result_meta") is not None
    ]
    final_state_bytes = [
        float(payload.get("result_meta", {}).get("final_state_bytes") or 0.0)
        for payload in payloads
        if payload.get("result_meta") is not None
    ]

    worker_metrics = [metric for payload in payloads for metric in payload.get("metrics", [])]
    worker_total = len(worker_metrics)
    worker_errors = sum(1 for metric in worker_metrics if metric.get("function_error"))
    worker_timeouts = sum(1 for metric in worker_metrics if metric.get("timed_out"))
    init_durations = [
        float(metric["init_duration_ms"])
        for metric in worker_metrics
        if metric.get("init_duration_ms") is not None
    ]
    coordinator_metrics = [payload.get("coordinator_metric", {}) for payload in payloads]
    coordinator_durations = [
        float(metric["duration_ms"])
        for metric in coordinator_metrics
        if metric.get("duration_ms") is not None
    ]
    coordinator_billed_durations = [
        float(metric["billed_duration_ms"])
        for metric in coordinator_metrics
        if metric.get("billed_duration_ms") is not None
    ]
    coordinator_costs = [
        float(metric["estimated_cost_usd"])
        for metric in coordinator_metrics
        if metric.get("estimated_cost_usd") is not None
    ]
    client_overheads = [max(0.0, client - workflow) for client, workflow in zip(client_latencies, workflow_latencies)]

    run_init_totals: list[float] = []
    max_partition_cost_shares: list[float] = []
    max_partition_duration_shares: list[float] = []
    partition_cost_maps: list[dict[str, float]] = []
    partition_billed_duration_maps: list[dict[str, float]] = []
    partition_duration_maps: list[dict[str, float]] = []
    partition_request_bytes_maps: list[dict[str, float]] = []
    partition_response_bytes_maps: list[dict[str, float]] = []
    partition_write_bytes_maps: list[dict[str, float]] = []
    partition_send_count_maps: list[dict[str, float]] = []
    partition_send_payload_bytes_maps: list[dict[str, float]] = []
    run_transition_counts: list[float] = []
    run_checkpoint_read_counts: list[float] = []
    run_checkpoint_write_counts: list[float] = []
    run_checkpoint_read_bytes: list[float] = []
    run_checkpoint_write_bytes: list[float] = []
    run_stepfn_costs: list[float] = []
    run_checkpoint_costs: list[float] = []
    run_remote_call_counts: list[float] = []
    run_remote_latency_totals: list[float] = []
    run_remote_input_tokens: list[float] = []
    run_remote_output_tokens: list[float] = []
    run_remote_total_tokens: list[float] = []
    run_bedrock_costs: list[float] = []
    run_total_modeled_costs: list[float] = []
    run_total_lambda_costs: list[float] = []
    run_send_payload_bytes: list[float] = []
    for payload in payloads:
        metrics = list(payload.get("metrics", []))
        run_init_totals.append(round(sum(float(metric.get("init_duration_ms") or 0.0) for metric in metrics), 6))
        cost_share = max_partition_share(metrics, "estimated_cost_usd")
        if cost_share is not None:
            max_partition_cost_shares.append(cost_share)
        duration_share = max_partition_share(metrics, "billed_duration_ms")
        if duration_share is None:
            duration_share = max_partition_share(metrics, "duration_ms")
        if duration_share is not None:
            max_partition_duration_shares.append(duration_share)
        partition_cost_maps.append(aggregate_metric_map(metrics, "estimated_cost_usd"))
        partition_billed_duration_maps.append(aggregate_metric_map(metrics, "billed_duration_ms"))
        partition_duration_maps.append(aggregate_metric_map(metrics, "duration_ms"))
        partition_request_bytes_maps.append(aggregate_metric_map(metrics, "request_payload_bytes"))
        partition_response_bytes_maps.append(aggregate_metric_map(metrics, "response_payload_bytes"))
        partition_write_bytes_maps.append(aggregate_metric_map(metrics, "writes_bytes"))
        partition_send_count_maps.append(aggregate_metric_map(metrics, "send_count"))
        partition_send_payload_bytes_maps.append(aggregate_metric_map(metrics, "send_payload_bytes"))

        transition_count = float(len(metrics))
        checkpoint_read_count = float(sum(1 for metric in metrics if float(metric.get("request_payload_bytes") or 0.0) > 0.0))
        checkpoint_write_count = float(sum(1 for metric in metrics if float(metric.get("writes_bytes") or 0.0) > 0.0))
        checkpoint_read_bytes = float(sum(float(metric.get("request_payload_bytes") or 0.0) for metric in metrics))
        checkpoint_write_bytes = float(sum(float(metric.get("writes_bytes") or 0.0) for metric in metrics))
        worker_cost_total = float(sum(float(metric.get("estimated_cost_usd") or 0.0) for metric in metrics))
        coordinator_cost_total = float(payload.get("coordinator_metric", {}).get("estimated_cost_usd") or 0.0)
        remote_call_count = float(sum(float(metric.get("remote_call_count") or 0.0) for metric in metrics))
        remote_latency_total = float(sum(float(metric.get("remote_latency_ms") or 0.0) for metric in metrics))
        remote_input_token_total = float(sum(float(metric.get("remote_input_tokens") or 0.0) for metric in metrics))
        remote_output_token_total = float(sum(float(metric.get("remote_output_tokens") or 0.0) for metric in metrics))
        remote_total_token_total = float(sum(float(metric.get("remote_total_tokens") or 0.0) for metric in metrics))
        send_payload_bytes = float(sum(float(metric.get("send_payload_bytes") or 0.0) for metric in metrics))
        stepfn_cost = transition_count * modeled_stepfn_transition_usd
        checkpoint_cost = modeled_checkpoint_cost_usd(
            read_count=checkpoint_read_count,
            write_count=checkpoint_write_count,
            read_bytes=checkpoint_read_bytes,
            write_bytes=checkpoint_write_bytes,
            read_request_usd=modeled_checkpoint_read_request_usd,
            write_request_usd=modeled_checkpoint_write_request_usd,
            read_gb_usd=modeled_checkpoint_read_gb_usd,
            write_gb_usd=modeled_checkpoint_write_gb_usd,
        )
        bedrock_cost = (
            (remote_input_token_total / 1000.0) * modeled_bedrock_input_1k_token_usd
            + (remote_output_token_total / 1000.0) * modeled_bedrock_output_1k_token_usd
        )

        run_transition_counts.append(transition_count)
        run_checkpoint_read_counts.append(checkpoint_read_count)
        run_checkpoint_write_counts.append(checkpoint_write_count)
        run_checkpoint_read_bytes.append(checkpoint_read_bytes)
        run_checkpoint_write_bytes.append(checkpoint_write_bytes)
        run_stepfn_costs.append(stepfn_cost)
        run_checkpoint_costs.append(checkpoint_cost)
        run_remote_call_counts.append(remote_call_count)
        run_remote_latency_totals.append(remote_latency_total)
        run_remote_input_tokens.append(remote_input_token_total)
        run_remote_output_tokens.append(remote_output_token_total)
        run_remote_total_tokens.append(remote_total_token_total)
        run_bedrock_costs.append(bedrock_cost)
        run_total_lambda_costs.append(worker_cost_total + coordinator_cost_total)
        run_total_modeled_costs.append(worker_cost_total + coordinator_cost_total + stepfn_cost + checkpoint_cost + bedrock_cost)
        run_send_payload_bytes.append(send_payload_bytes)

    batch_indexes = sorted(
        {
            int(entry["batch_index"])
            for entry in entries
            if entry.get("batch_index") is not None
        }
    )
    batch_makespans = [
        float(entry["batch_makespan_ms"])
        for entry in entries
        if entry.get("slot_index") == 0 and entry.get("batch_makespan_ms") is not None
    ]
    batch_throughputs = [
        float(entry["batch_throughput_rps"])
        for entry in entries
        if entry.get("slot_index") == 0 and entry.get("batch_throughput_rps") is not None
    ]
    batch_concurrency = None
    for entry in entries:
        if entry.get("batch_concurrency") is not None:
            batch_concurrency = int(entry["batch_concurrency"])
            break

    return {
        "run_count": len(entries),
        "batch_count": len(batch_indexes),
        "batch_concurrency": batch_concurrency,
        "workflow_latency_mean_ms": mean_or_none(workflow_latencies, digits=3),
        "workflow_latency_p95_ms": percentile(workflow_latencies, 0.95),
        "client_latency_mean_ms": mean_or_none(client_latencies, digits=3),
        "client_latency_p95_ms": percentile(client_latencies, 0.95),
        "client_latency_cv": coefficient_of_variation(client_latencies),
        "client_overhead_mean_ms": mean_or_none(client_overheads, digits=3),
        "cost_mean_usd": mean_or_none(costs, digits=10),
        "cost_cv": coefficient_of_variation(costs),
        "workflow_failure_rate": round(workflow_failures / len(payloads), 6) if payloads else 0.0,
        "timeout_run_rate": round(timeout_runs / len(payloads), 6) if payloads else 0.0,
        "worker_invocations_mean": round(worker_total / len(payloads), 3) if payloads else None,
        "worker_error_rate": round(worker_errors / worker_total, 6) if worker_total else 0.0,
        "worker_timeout_rate": round(worker_timeouts / worker_total, 6) if worker_total else 0.0,
        "worker_init_rate": round(len(init_durations) / worker_total, 6) if worker_total else 0.0,
        "worker_init_mean_ms": mean_or_none(init_durations, digits=3),
        "worker_init_total_ms_mean": mean_or_none(run_init_totals, digits=3),
        "max_partition_cost_share_mean": mean_or_none(max_partition_cost_shares, digits=6),
        "max_partition_duration_share_mean": mean_or_none(max_partition_duration_shares, digits=6),
        "coordinator_duration_mean_ms": mean_or_none(coordinator_durations, digits=3),
        "coordinator_billed_duration_mean_ms": mean_or_none(coordinator_billed_durations, digits=3),
        "coordinator_init_mean_ms": None,
        "coordinator_cost_mean_usd": mean_or_none(coordinator_costs, digits=10),
        "lambda_total_cost_mean_usd": mean_or_none(run_total_lambda_costs, digits=10),
        "modeled_stepfn_transition_count_mean": mean_or_none(run_transition_counts, digits=3),
        "modeled_checkpoint_read_count_mean": mean_or_none(run_checkpoint_read_counts, digits=3),
        "modeled_checkpoint_write_count_mean": mean_or_none(run_checkpoint_write_counts, digits=3),
        "modeled_checkpoint_read_bytes_mean": mean_or_none(run_checkpoint_read_bytes, digits=3),
        "modeled_checkpoint_write_bytes_mean": mean_or_none(run_checkpoint_write_bytes, digits=3),
        "modeled_stepfn_cost_mean_usd": mean_or_none(run_stepfn_costs, digits=10),
        "modeled_checkpoint_cost_mean_usd": mean_or_none(run_checkpoint_costs, digits=10),
        "remote_call_count_mean": mean_or_none(run_remote_call_counts, digits=3),
        "remote_latency_total_mean_ms": mean_or_none(run_remote_latency_totals, digits=3),
        "remote_input_tokens_mean": mean_or_none(run_remote_input_tokens, digits=3),
        "remote_output_tokens_mean": mean_or_none(run_remote_output_tokens, digits=3),
        "remote_total_tokens_mean": mean_or_none(run_remote_total_tokens, digits=3),
        "modeled_bedrock_cost_mean_usd": mean_or_none(run_bedrock_costs, digits=10),
        "modeled_total_cost_mean_usd": mean_or_none(run_total_modeled_costs, digits=10),
        "partition_cost_mean_usd_json": json_compact(mean_mapping(partition_cost_maps, digits=10)),
        "partition_billed_duration_mean_ms_json": json_compact(mean_mapping(partition_billed_duration_maps, digits=3)),
        "partition_duration_mean_ms_json": json_compact(mean_mapping(partition_duration_maps, digits=3)),
        "partition_request_bytes_mean_json": json_compact(mean_mapping(partition_request_bytes_maps, digits=3)),
        "partition_response_bytes_mean_json": json_compact(mean_mapping(partition_response_bytes_maps, digits=3)),
        "partition_write_bytes_mean_json": json_compact(mean_mapping(partition_write_bytes_maps, digits=3)),
        "partition_send_count_mean_json": json_compact(mean_mapping(partition_send_count_maps, digits=3)),
        "partition_send_payload_bytes_mean_json": json_compact(mean_mapping(partition_send_payload_bytes_maps, digits=3)),
        "batch_makespan_mean_ms": mean_or_none(batch_makespans, digits=3),
        "batch_throughput_mean_rps": mean_or_none(batch_throughputs, digits=6),
        "route_match_rate": mean_or_none(route_matches, digits=6),
        "final_state_bytes_mean": mean_or_none(final_state_bytes, digits=3),
        "send_payload_bytes_mean": mean_or_none(run_send_payload_bytes, digits=3),
    }


def ordered_partition_ids(candidate: CandidateConfig) -> list[str]:
    return [spec.logical_id for spec in candidate.worker_defs]


def aggregate_partition_means(measured: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for payload in measured:
        for logical_id, summary in payload.get("partition_summary", {}).items():
            avg = summary.get("avg_duration_ms")
            if avg is None:
                continue
            values.setdefault(logical_id, []).append(float(avg))
    return {key: round(statistics.mean(items), 3) for key, items in values.items()}


def build_measurements_json(
    detailed_results: list[dict[str, Any]],
    *,
    candidates: list[CandidateConfig],
    args: argparse.Namespace,
) -> dict[str, Any]:
    measurements: list[dict[str, Any]] = []
    baseline_costs: dict[tuple[str, str], float] = {}
    modeled_stepfn_transition_usd = float(args.modeled_stepfn_transition_usd)
    modeled_checkpoint_read_request_usd = float(args.modeled_checkpoint_read_request_usd)
    modeled_checkpoint_write_request_usd = float(args.modeled_checkpoint_write_request_usd)
    modeled_checkpoint_read_gb_usd = float(args.modeled_checkpoint_read_gb_usd)
    modeled_checkpoint_write_gb_usd = float(args.modeled_checkpoint_write_gb_usd)
    modeled_bedrock_input_1k_token_usd = float(args.modeled_bedrock_input_1k_token_usd)
    modeled_bedrock_output_1k_token_usd = float(args.modeled_bedrock_output_1k_token_usd)

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
            latencies = [
                payload["workflow_elapsed_ms"]
                for payload in measured
                if payload.get("workflow_elapsed_ms") is not None
            ]
            costs = [
                payload["estimated_cost_usd"]
                for payload in measured
                if payload.get("estimated_cost_usd") is not None
            ]
            error_total = sum(int(payload.get("error_count", 0) or 0) for payload in measured)
            timeout_total = sum(int(payload.get("timeout_count", 0) or 0) for payload in measured)
            total_invocations = sum(len(payload.get("metrics", [])) for payload in measured)
            partition_means = aggregate_partition_means(measured)
            single_summary = summarize_phase_entries(
                single_entries,
                modeled_stepfn_transition_usd=modeled_stepfn_transition_usd,
                modeled_checkpoint_read_request_usd=modeled_checkpoint_read_request_usd,
                modeled_checkpoint_write_request_usd=modeled_checkpoint_write_request_usd,
                modeled_checkpoint_read_gb_usd=modeled_checkpoint_read_gb_usd,
                modeled_checkpoint_write_gb_usd=modeled_checkpoint_write_gb_usd,
                modeled_bedrock_input_1k_token_usd=modeled_bedrock_input_1k_token_usd,
                modeled_bedrock_output_1k_token_usd=modeled_bedrock_output_1k_token_usd,
            )
            cold_summary = summarize_phase_entries(
                cold_entries,
                modeled_stepfn_transition_usd=modeled_stepfn_transition_usd,
                modeled_checkpoint_read_request_usd=modeled_checkpoint_read_request_usd,
                modeled_checkpoint_write_request_usd=modeled_checkpoint_write_request_usd,
                modeled_checkpoint_read_gb_usd=modeled_checkpoint_read_gb_usd,
                modeled_checkpoint_write_gb_usd=modeled_checkpoint_write_gb_usd,
                modeled_bedrock_input_1k_token_usd=modeled_bedrock_input_1k_token_usd,
                modeled_bedrock_output_1k_token_usd=modeled_bedrock_output_1k_token_usd,
            )
            load_summary = summarize_phase_entries(
                load_entries,
                modeled_stepfn_transition_usd=modeled_stepfn_transition_usd,
                modeled_checkpoint_read_request_usd=modeled_checkpoint_read_request_usd,
                modeled_checkpoint_write_request_usd=modeled_checkpoint_write_request_usd,
                modeled_checkpoint_read_gb_usd=modeled_checkpoint_read_gb_usd,
                modeled_checkpoint_write_gb_usd=modeled_checkpoint_write_gb_usd,
                modeled_bedrock_input_1k_token_usd=modeled_bedrock_input_1k_token_usd,
                modeled_bedrock_output_1k_token_usd=modeled_bedrock_output_1k_token_usd,
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
                "dataset_expected_route": dataset_profile.get("expected_route"),
                "dataset_account_tier": dataset_profile.get("account_tier"),
                "result_completed": completed_map.get(key, True),
                "run_on": candidate.run_on,
                "partition_count": len(candidate.worker_defs),
                "partitioning_vector": candidate.partitioning_vector,
                "memory_vector_mb_full": candidate.memory_vector_mb,
                "timeout_vector_sec_full": [spec.timeout_sec for spec in candidate.worker_defs],
                "concurrency_vector_full": [spec.concurrency_limit for spec in candidate.worker_defs],
                "worker_resource_plan_json": json_compact([worker_spec_payload(spec) for spec in candidate.worker_defs]),
                "what_it_should_show": candidate.what_it_should_show,
                "latency_p95_ms": percentile(latencies, 0.95),
                "cost_usd": round(statistics.mean(costs), 10) if costs else None,
                "error_rate": round(error_total / total_invocations, 6) if total_invocations else 0.0,
                "timeout_rate": round(timeout_total / total_invocations, 6) if total_invocations else 0.0,
                "latency_target_ms": float(args.latency_target_ms),
                "partition_runtime_ms": [partition_means[key] for key in ordered_partition_ids(candidate) if key in partition_means],
                "partition_input_units": [],
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
                "single_workflow_latency_mean_ms": single_summary["workflow_latency_mean_ms"],
                "single_workflow_latency_p95_ms": single_summary["workflow_latency_p95_ms"],
                "single_client_latency_mean_ms": single_summary["client_latency_mean_ms"],
                "single_client_latency_p95_ms": single_summary["client_latency_p95_ms"],
                "single_client_latency_cv": single_summary["client_latency_cv"],
                "single_client_overhead_mean_ms": single_summary["client_overhead_mean_ms"],
                "single_cost_mean_usd": single_summary["cost_mean_usd"],
                "single_cost_cv": single_summary["cost_cv"],
                "single_workflow_failure_rate": single_summary["workflow_failure_rate"],
                "single_timeout_run_rate": single_summary["timeout_run_rate"],
                "single_worker_invocations_mean": single_summary["worker_invocations_mean"],
                "single_worker_error_rate": single_summary["worker_error_rate"],
                "single_worker_timeout_rate": single_summary["worker_timeout_rate"],
                "single_worker_init_rate": single_summary["worker_init_rate"],
                "single_worker_init_mean_ms": single_summary["worker_init_mean_ms"],
                "single_worker_init_total_ms_mean": single_summary["worker_init_total_ms_mean"],
                "single_max_partition_cost_share_mean": single_summary["max_partition_cost_share_mean"],
                "single_max_partition_duration_share_mean": single_summary["max_partition_duration_share_mean"],
                "single_coordinator_duration_mean_ms": single_summary["coordinator_duration_mean_ms"],
                "single_coordinator_billed_duration_mean_ms": single_summary["coordinator_billed_duration_mean_ms"],
                "single_coordinator_init_mean_ms": single_summary["coordinator_init_mean_ms"],
                "single_lambda_coordinator_cost_mean_usd": single_summary["coordinator_cost_mean_usd"],
                "single_lambda_worker_cost_mean_usd": single_summary["cost_mean_usd"],
                "single_lambda_total_cost_mean_usd": single_summary["lambda_total_cost_mean_usd"],
                "single_modeled_stepfn_transition_count_mean": single_summary["modeled_stepfn_transition_count_mean"],
                "single_modeled_stepfn_cost_mean_usd": single_summary["modeled_stepfn_cost_mean_usd"],
                "single_modeled_checkpoint_read_count_mean": single_summary["modeled_checkpoint_read_count_mean"],
                "single_modeled_checkpoint_write_count_mean": single_summary["modeled_checkpoint_write_count_mean"],
                "single_modeled_checkpoint_read_bytes_mean": single_summary["modeled_checkpoint_read_bytes_mean"],
                "single_modeled_checkpoint_write_bytes_mean": single_summary["modeled_checkpoint_write_bytes_mean"],
                "single_modeled_checkpoint_cost_mean_usd": single_summary["modeled_checkpoint_cost_mean_usd"],
                "single_remote_call_count_mean": single_summary["remote_call_count_mean"],
                "single_remote_latency_mean_ms": single_summary["remote_latency_total_mean_ms"],
                "single_remote_latency_total_mean_ms": single_summary["remote_latency_total_mean_ms"],
                "single_remote_input_tokens_mean": single_summary["remote_input_tokens_mean"],
                "single_remote_output_tokens_mean": single_summary["remote_output_tokens_mean"],
                "single_remote_total_tokens_mean": single_summary["remote_total_tokens_mean"],
                "single_modeled_bedrock_cost_mean_usd": single_summary["modeled_bedrock_cost_mean_usd"],
                "single_modeled_total_cost_mean_usd": single_summary["modeled_total_cost_mean_usd"],
                "single_partition_cost_mean_usd_json": single_summary["partition_cost_mean_usd_json"],
                "single_partition_billed_duration_mean_ms_json": single_summary["partition_billed_duration_mean_ms_json"],
                "single_partition_duration_mean_ms_json": single_summary["partition_duration_mean_ms_json"],
                "single_partition_request_bytes_mean_json": single_summary["partition_request_bytes_mean_json"],
                "single_partition_response_bytes_mean_json": single_summary["partition_response_bytes_mean_json"],
                "single_partition_write_bytes_mean_json": single_summary["partition_write_bytes_mean_json"],
                "single_partition_send_count_mean_json": single_summary["partition_send_count_mean_json"],
                "single_partition_send_payload_bytes_mean_json": single_summary["partition_send_payload_bytes_mean_json"],
                "single_result_correct_rate": single_summary["route_match_rate"],
                "single_route_match_rate": single_summary["route_match_rate"],
                "single_final_state_bytes_mean": single_summary["final_state_bytes_mean"],
                "single_send_payload_bytes_mean": single_summary["send_payload_bytes_mean"],
                "cold_probe_run_count": cold_summary["run_count"],
                "cold_client_latency_mean_ms": cold_summary["client_latency_mean_ms"],
                "cold_client_latency_p95_ms": cold_summary["client_latency_p95_ms"],
                "cold_worker_init_rate": cold_summary["worker_init_rate"],
                "cold_worker_init_mean_ms": cold_summary["worker_init_mean_ms"],
                "cold_worker_init_total_ms_mean": cold_summary["worker_init_total_ms_mean"],
                "cold_coordinator_init_mean_ms": cold_summary["coordinator_init_mean_ms"],
                "cold_lambda_coordinator_cost_mean_usd": cold_summary["coordinator_cost_mean_usd"],
                "load_request_count": load_summary["run_count"],
                "load_batch_count": load_summary["batch_count"],
                "load_concurrency": load_concurrency,
                "load_client_latency_mean_ms": load_summary["client_latency_mean_ms"],
                "load_client_latency_p95_ms": load_summary["client_latency_p95_ms"],
                "load_client_latency_cv": load_summary["client_latency_cv"],
                "load_client_overhead_mean_ms": load_summary["client_overhead_mean_ms"],
                "load_batch_makespan_mean_ms": load_summary["batch_makespan_mean_ms"],
                "load_throughput_mean_rps": load_summary["batch_throughput_mean_rps"],
                "load_workflow_failure_rate": load_summary["workflow_failure_rate"],
                "load_timeout_run_rate": load_summary["timeout_run_rate"],
                "load_worker_timeout_rate": load_summary["worker_timeout_rate"],
                "load_coordinator_duration_mean_ms": load_summary["coordinator_duration_mean_ms"],
                "load_lambda_worker_cost_mean_usd": load_summary["cost_mean_usd"],
                "load_lambda_coordinator_cost_mean_usd": load_summary["coordinator_cost_mean_usd"],
                "load_lambda_total_cost_mean_usd": load_summary["lambda_total_cost_mean_usd"],
                "load_modeled_stepfn_transition_count_mean": load_summary["modeled_stepfn_transition_count_mean"],
                "load_modeled_stepfn_cost_mean_usd": load_summary["modeled_stepfn_cost_mean_usd"],
                "load_modeled_checkpoint_read_count_mean": load_summary["modeled_checkpoint_read_count_mean"],
                "load_modeled_checkpoint_write_count_mean": load_summary["modeled_checkpoint_write_count_mean"],
                "load_modeled_checkpoint_read_bytes_mean": load_summary["modeled_checkpoint_read_bytes_mean"],
                "load_modeled_checkpoint_write_bytes_mean": load_summary["modeled_checkpoint_write_bytes_mean"],
                "load_modeled_checkpoint_cost_mean_usd": load_summary["modeled_checkpoint_cost_mean_usd"],
                "load_remote_call_count_mean": load_summary["remote_call_count_mean"],
                "load_remote_latency_mean_ms": load_summary["remote_latency_total_mean_ms"],
                "load_remote_latency_total_mean_ms": load_summary["remote_latency_total_mean_ms"],
                "load_remote_input_tokens_mean": load_summary["remote_input_tokens_mean"],
                "load_remote_output_tokens_mean": load_summary["remote_output_tokens_mean"],
                "load_remote_total_tokens_mean": load_summary["remote_total_tokens_mean"],
                "load_modeled_bedrock_cost_mean_usd": load_summary["modeled_bedrock_cost_mean_usd"],
                "load_modeled_total_cost_mean_usd": load_summary["modeled_total_cost_mean_usd"],
                "load_partition_cost_mean_usd_json": load_summary["partition_cost_mean_usd_json"],
                "load_partition_billed_duration_mean_ms_json": load_summary["partition_billed_duration_mean_ms_json"],
                "load_partition_duration_mean_ms_json": load_summary["partition_duration_mean_ms_json"],
                "load_partition_request_bytes_mean_json": load_summary["partition_request_bytes_mean_json"],
                "load_partition_response_bytes_mean_json": load_summary["partition_response_bytes_mean_json"],
                "load_partition_write_bytes_mean_json": load_summary["partition_write_bytes_mean_json"],
                "load_partition_send_count_mean_json": load_summary["partition_send_count_mean_json"],
                "load_partition_send_payload_bytes_mean_json": load_summary["partition_send_payload_bytes_mean_json"],
                "load_result_correct_rate": load_summary["route_match_rate"],
                "load_route_match_rate": load_summary["route_match_rate"],
                "load_final_state_bytes_mean": load_summary["final_state_bytes_mean"],
                "load_send_payload_bytes_mean": load_summary["send_payload_bytes_mean"],
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
            "stepfn_transition_usd": modeled_stepfn_transition_usd,
            "checkpoint_read_request_usd": modeled_checkpoint_read_request_usd,
            "checkpoint_write_request_usd": modeled_checkpoint_write_request_usd,
            "checkpoint_read_gb_usd": modeled_checkpoint_read_gb_usd,
            "checkpoint_write_gb_usd": modeled_checkpoint_write_gb_usd,
            "bedrock_input_1k_token_usd": modeled_bedrock_input_1k_token_usd,
            "bedrock_output_1k_token_usd": modeled_bedrock_output_1k_token_usd,
        },
        "notes": [
            "The router candidate matrix matches the thesis experiment table for RT-B, RT-C, RT-L, RT-R, RT-U1, RT-U2, and RT-U3.",
            "single_* fields summarize measured single-request runs.",
            "cold_* fields summarize first-touch probes before warmups.",
            "load_* fields summarize measured concurrent batches when load flags are enabled.",
            "cost_total_modeled_usd includes worker cost, modeled coordinator cost, Step Functions transitions, checkpoint transfer, and optional Bedrock token cost.",
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


def build_result_index(detailed_results: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (row["workflow"], row["candidate"], row["dataset_id"]): row
        for row in detailed_results
    }


def ensure_result_entry(
    detailed_results: list[dict[str, Any]],
    *,
    candidate: CandidateConfig,
    dataset_profile: dict[str, Any],
) -> dict[str, Any]:
    result_index = build_result_index(detailed_results)
    key = (candidate.workflow, candidate.candidate, dataset_profile["dataset_id"])
    existing = result_index.get(key)
    if existing is not None:
        return existing
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


def cmd_catalog(args: argparse.Namespace) -> int:
    del args
    print("Candidates")
    for candidate in ROUTER_CANDIDATES:
        print(
            f"- {candidate.candidate}: group={candidate.group}, "
            f"slo_profile={candidate.slo_profile or 'baseline/user'}, "
            f"pi={candidate.partitioning_vector}, m={candidate.memory_vector_mb}"
        )
    print("\nDatasets")
    for dataset_id in sorted(ROUTER_DATASETS):
        dataset = ROUTER_DATASETS[dataset_id]
        print(
            f"- {dataset_id}: route={dataset['expected_route']}, "
            f"tier={dataset['account_tier']}, work_units={dataset['input_work_units']}, "
            f"scenario={dataset['scenario']}"
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
            dataset_profile = dict(ROUTER_DATASETS[dataset_id])
            row = ensure_result_entry(detailed_results, candidate=candidate, dataset_profile=dataset_profile)
            if args.resume and row.get("completed"):
                print(f"skip completed {candidate.candidate}/{dataset_id}")
                continue

            cold_done = count_phase_entries(row["runs"], "cold_probe")
            for run_index in range(cold_done, int(args.cold_runs)):
                timed = timed_execution(candidate, dataset_profile, args, f"cold-{run_index:02d}")
                row["runs"].append(
                    {
                        "phase": "cold_probe",
                        "run_index": run_index,
                        "payload": timed["payload"],
                        "client_elapsed_ms": timed["client_elapsed_ms"],
                    }
                )
                checkpoint_results(args, detailed_results, candidates)
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)

            warmup_done = count_phase_entries(row["runs"], "warmup")
            for run_index in range(warmup_done, int(args.warmup_runs)):
                timed = timed_execution(candidate, dataset_profile, args, f"warmup-{run_index:02d}")
                row["runs"].append(
                    {
                        "phase": "warmup",
                        "run_index": run_index,
                        "payload": timed["payload"],
                        "client_elapsed_ms": timed["client_elapsed_ms"],
                    }
                )
                checkpoint_results(args, detailed_results, candidates)
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)

            measured_done = count_phase_entries(row["runs"], "measured")
            for run_index in range(measured_done, int(args.runs)):
                timed = timed_execution(candidate, dataset_profile, args, f"measured-{run_index:02d}")
                row["runs"].append(
                    {
                        "phase": "measured",
                        "run_index": run_index,
                        "payload": timed["payload"],
                        "client_elapsed_ms": timed["client_elapsed_ms"],
                    }
                )
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
            next_batch_index = (existing_batches[-1] + 1) if existing_batches else 0
            while next_batch_index < int(args.load_batches):
                if int(args.load_concurrency) <= 0:
                    break
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
    parser = argparse.ArgumentParser(description="Run the final service-desk router experiment matrix locally.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("catalog", help="List available router candidates and datasets.")

    run = subparsers.add_parser("run", help="Execute the router candidate matrix and write resumable results.")
    run.add_argument("--candidate", action="append", help="Candidate id to run. Repeat to select multiple candidates.")
    run.add_argument("--dataset", action="append", help="Dataset id to run. Repeat to select multiple datasets.")
    run.add_argument("--cold-runs", type=int, default=1, help="Number of cold probes per candidate/dataset.")
    run.add_argument("--warmup-runs", type=int, default=1, help="Number of warmup runs per candidate/dataset.")
    run.add_argument("--runs", type=int, default=3, help="Number of measured single-request runs per candidate/dataset.")
    run.add_argument("--load-batches", type=int, default=0, help="Number of concurrent load batches per candidate/dataset.")
    run.add_argument("--load-concurrency", type=int, default=0, help="Concurrent requests per load batch.")
    run.add_argument("--sleep-sec", type=float, default=0.0, help="Delay between cold/warmup/measured runs.")
    run.add_argument("--load-sleep-sec", type=float, default=0.0, help="Delay between load batches.")
    run.add_argument("--resume", action="store_true", help="Resume from an existing results JSON file.")
    run.add_argument("--require-bedrock", action="store_true", help="Fail instead of falling back if Bedrock is not configured.")
    run.add_argument("--latency-target-ms", type=float, default=DEFAULT_LATENCY_TARGET_MS, help="Latency target used for summary metadata.")
    run.add_argument("--coordinator-memory-mb", type=int, default=DEFAULT_COORDINATOR_MEMORY_MB, help="Modeled coordinator memory for total-cost summaries.")
    run.add_argument(
        "--modeled-stepfn-transition-usd",
        type=float,
        default=DEFAULT_MODELED_STEPFN_TRANSITION_USD,
        help="Modeled Step Functions transition cost in USD per transition.",
    )
    run.add_argument(
        "--modeled-checkpoint-read-request-usd",
        type=float,
        default=DEFAULT_MODELED_CHECKPOINT_READ_REQUEST_USD,
        help="Modeled checkpoint read request cost in USD per read.",
    )
    run.add_argument(
        "--modeled-checkpoint-write-request-usd",
        type=float,
        default=DEFAULT_MODELED_CHECKPOINT_WRITE_REQUEST_USD,
        help="Modeled checkpoint write request cost in USD per write.",
    )
    run.add_argument(
        "--modeled-checkpoint-read-gb-usd",
        type=float,
        default=DEFAULT_MODELED_CHECKPOINT_READ_GB_USD,
        help="Modeled checkpoint read transfer cost in USD per GiB.",
    )
    run.add_argument(
        "--modeled-checkpoint-write-gb-usd",
        type=float,
        default=DEFAULT_MODELED_CHECKPOINT_WRITE_GB_USD,
        help="Modeled checkpoint write transfer cost in USD per GiB.",
    )
    run.add_argument(
        "--modeled-bedrock-input-1k-token-usd",
        type=float,
        default=DEFAULT_MODELED_BEDROCK_INPUT_1K_TOKEN_USD,
        help="Optional modeled Bedrock input-token cost in USD per 1K tokens.",
    )
    run.add_argument(
        "--modeled-bedrock-output-1k-token-usd",
        type=float,
        default=DEFAULT_MODELED_BEDROCK_OUTPUT_1K_TOKEN_USD,
        help="Optional modeled Bedrock output-token cost in USD per 1K tokens.",
    )
    run.add_argument("--results-json", type=Path, default=DEFAULT_RESULTS_JSON, help="Checkpointable raw results JSON path.")
    run.add_argument("--measurements-json", type=Path, default=DEFAULT_MEASUREMENTS_JSON, help="Summary measurements JSON path.")
    run.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV, help="Summary measurements CSV path.")

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
