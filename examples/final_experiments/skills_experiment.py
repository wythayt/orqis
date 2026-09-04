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

from langchain.tools import ToolRuntime
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from orqis.examples.final_experiments import skills as skills_example
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


DEFAULT_RESULTS_JSON = Path("artifacts/final_experiments/skills_results.json")
DEFAULT_MEASUREMENTS_JSON = Path("artifacts/final_experiments/skills_measurements.json")
DEFAULT_OUTPUT_CSV = Path("artifacts/final_experiments/skills_measurements.csv")
DEFAULT_LATENCY_TARGET_MS = 1800.0
DEFAULT_COORDINATOR_MEMORY_MB = 512
DEFAULT_MAX_LOOP_STEPS = 8
WORKFLOW_ID = "skills"
BASELINE_CANDIDATE = "SK-B"
DATASET_RANGE = "SK1-SK5"
MODEL_LOGICAL_ID = "p_model_fanout"
TOOLS_LOGICAL_ID = "p_tools_fanout"

SKILLS_CANDIDATES = [
    CandidateConfig(
        WORKFLOW_ID,
        "compiler baseline",
        "SK-B",
        ["m", "t"],
        [1024, 1024],
        DATASET_RANGE,
        "Replay-safe two-worker loop without memory search.",
        [
            ws(MODEL_LOGICAL_ID, 1024, timeout_sec=30),
            ws(TOOLS_LOGICAL_ID, 1024, timeout_sec=30),
        ],
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "compiler SLO",
        "SK-C",
        ["m", "t"],
        [512, 512],
        DATASET_RANGE,
        "Same partitioning as SK-B but with the model partition reduced under the cost-oriented SLO profile.",
        [
            ws(MODEL_LOGICAL_ID, 512, timeout_sec=30),
            ws(TOOLS_LOGICAL_ID, 512, timeout_sec=30, concurrency_limit=2),
        ],
        slo_profile="cost_relaxed",
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "compiler SLO",
        "SK-L",
        ["m", "t"],
        [512, 512],
        DATASET_RANGE,
        "Same partitioning and memory outcome as SK-C under the latency-oriented SLO profile.",
        [
            ws(MODEL_LOGICAL_ID, 512, timeout_sec=30),
            ws(TOOLS_LOGICAL_ID, 512, timeout_sec=30, concurrency_limit=2),
        ],
        slo_profile="latency_tight",
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "compiler SLO",
        "SK-R",
        ["m", "t"],
        [1024, 512],
        DATASET_RANGE,
        "Reliability pressure keeps the model partition at the baseline tier while reducing the tool partition.",
        [
            ws(MODEL_LOGICAL_ID, 1024, timeout_sec=30),
            ws(TOOLS_LOGICAL_ID, 512, timeout_sec=30, concurrency_limit=2),
        ],
        slo_profile="reliability_tight",
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "user scenario",
        "SK-U1",
        ["(m+t)"],
        [2048],
        DATASET_RANGE,
        "Fused loop worker that removes the replay-safe model-tools boundary.",
        [ws("p_monolith", 2048, timeout_sec=30)],
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "user scenario",
        "SK-U2",
        ["m", "t"],
        [3008, 3008],
        DATASET_RANGE,
        "Same partitioning as the compiler rows, but clearly oversized manual sizing.",
        [
            ws(MODEL_LOGICAL_ID, 3008, timeout_sec=30),
            ws(TOOLS_LOGICAL_ID, 3008, timeout_sec=30, concurrency_limit=2),
        ],
    ),
    CandidateConfig(
        WORKFLOW_ID,
        "user scenario",
        "SK-U3",
        ["m", "t"],
        [512, 512],
        DATASET_RANGE,
        "Same partitioning as the compiler rows, but assumes tool-side concurrency 1 while keeping the reduced tiers.",
        [
            ws(MODEL_LOGICAL_ID, 512, timeout_sec=30),
            ws(TOOLS_LOGICAL_ID, 512, timeout_sec=30, concurrency_limit=1),
        ],
    ),
]

CANDIDATE_BY_ID = {candidate.candidate: candidate for candidate in SKILLS_CANDIDATES}

SKILLS_DATASETS = {
    "SK1": {
        "dataset_id": "SK1",
        "user_request": "Write a SQL query to find all customers who made orders over $1000 in the last month.",
        "expected_skill": "sales_analytics",
        "scenario": "high_value_orders_last_month",
    },
    "SK2": {
        "dataset_id": "SK2",
        "user_request": "Write SQL to list the top 10 customers by completed-order revenue in the last quarter.",
        "expected_skill": "sales_analytics",
        "scenario": "top_customers_by_revenue",
    },
    "SK3": {
        "dataset_id": "SK3",
        "user_request": "Write a query to find active customers whose lifetime value from completed orders exceeds 5000.",
        "expected_skill": "sales_analytics",
        "scenario": "active_customer_clv_filter",
    },
    "SK4": {
        "dataset_id": "SK4",
        "user_request": "Write SQL to summarize completed-order revenue by sales_region for the last 90 days.",
        "expected_skill": "sales_analytics",
        "scenario": "regional_revenue_summary",
    },
    "SK5": {
        "dataset_id": "SK5",
        "user_request": (
            "Write a SQL query to return customer_id, customer_tier, signup_date, and total_amount for "
            "completed orders above 1000, limited to customers who have been active for at least 90 days."
        ),
        "expected_skill": "sales_analytics",
        "scenario": "high_value_active_customer_orders",
    },
}

for dataset in SKILLS_DATASETS.values():
    dataset["workflow"] = WORKFLOW_ID
    dataset["input_work_units"] = len(str(dataset["user_request"]))

SKILLS_SYSTEM_PROMPT = (
    "You are a SQL query assistant that helps users write queries against business databases.\n\n"
    "## Available Skills\n\n"
    + "\n".join(f"- {skill['name']}: {skill['description']}" for skill in skills_example.SKILLS)
    + "\n\nUse the load_skill tool when you need detailed information about handling a specific type of request."
)


def filter_candidates(candidate_ids: list[str] | None) -> list[CandidateConfig]:
    if not candidate_ids:
        return list(SKILLS_CANDIDATES)
    selected = set(candidate_ids)
    filtered = [candidate for candidate in SKILLS_CANDIDATES if candidate.candidate in selected]
    found = {candidate.candidate for candidate in filtered}
    missing = sorted(selected - found)
    if missing:
        available = ", ".join(sorted(CANDIDATE_BY_ID))
        raise ValueError(f"unknown candidate id(s): {', '.join(missing)}; available: {available}")
    return filtered


def filter_dataset_ids(dataset_ids: list[str] | None) -> list[str]:
    if not dataset_ids:
        return sorted(SKILLS_DATASETS)
    selected = [dataset_id for dataset_id in dataset_ids]
    missing = [dataset_id for dataset_id in selected if dataset_id not in SKILLS_DATASETS]
    if missing:
        available = ", ".join(sorted(SKILLS_DATASETS))
        raise ValueError(f"unknown dataset id(s): {', '.join(missing)}; available: {available}")
    return selected


def serialize_message(message: BaseMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": getattr(message, "type", message.__class__.__name__).lower(),
        "content": message.content,
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = tool_calls
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        payload["tool_call_id"] = tool_call_id
    return payload


def serialize_value(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return serialize_message(value)
    if isinstance(value, list):
        return [serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): serialize_value(item) for key, item in value.items()}
    return value


def json_size_bytes(value: Any) -> int:
    return len(json_compact(serialize_value(value)).encode("utf-8"))


def message_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part).strip()
    return str(content or "").strip()


def final_ai_response_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if not isinstance(message, AIMessage):
            continue
        if getattr(message, "tool_calls", None):
            continue
        text = message_text_content(message.content)
        if text:
            return text
    return ""


def state_slice(state: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: state.get(key) for key in keys if key in state}


def build_initial_state(dataset_profile: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            SystemMessage(content=SKILLS_SYSTEM_PROMPT),
            HumanMessage(content=str(dataset_profile["user_request"])),
        ],
        "skills_loaded": [],
        "final_response": "",
    }


def merge_state(state: dict[str, Any], writes: dict[str, Any]) -> None:
    for key, value in writes.items():
        if key == "messages":
            state.setdefault("messages", [])
            state["messages"].extend(list(value or []))
        elif key == "skills_loaded":
            existing = list(state.get("skills_loaded") or [])
            for item in list(value or []):
                if item not in existing:
                    existing.append(item)
            state["skills_loaded"] = existing
        else:
            state[key] = value


def merge_preview(state: dict[str, Any], writes: dict[str, Any]) -> dict[str, Any]:
    preview = {"messages": list(state.get("messages") or [])}
    for key, value in state.items():
        if key != "messages":
            preview[key] = value
    merge_state(preview, writes)
    return preview


def accumulate_writes(target: dict[str, Any], writes: dict[str, Any]) -> None:
    for key, value in writes.items():
        if key == "messages":
            target.setdefault("messages", [])
            target["messages"].extend(list(value or []))
        elif key == "skills_loaded":
            existing = list(target.get("skills_loaded") or [])
            for item in list(value or []):
                if item not in existing:
                    existing.append(item)
            target["skills_loaded"] = existing
        else:
            target[key] = value


def build_tool_runtime(state: dict[str, Any], tool_call_id: str | None):
    return ToolRuntime(
        state=state,
        context={},
        config={},
        stream_writer=lambda *_args, **_kwargs: None,
        tool_call_id=tool_call_id,
        store=None,
        tools=[skills_example.load_skill, skills_example.write_sql_query],
    )


def infer_loaded_skill(state: dict[str, Any]) -> str | None:
    loaded_skills = list(state.get("skills_loaded") or [])
    if len(loaded_skills) == 1:
        return str(loaded_skills[0])
    for message in reversed(list(state.get("messages") or [])):
        if not isinstance(message, ToolMessage):
            continue
        content = str(message.content or "")
        if content.startswith("Loaded skill: "):
            return content.split("Loaded skill: ", 1)[1].splitlines()[0].strip()
    return None


def model_invoker_factory(model) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    def _invoke(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        messages = list(request_payload.get("messages") or [])
        response = model.invoke(messages)
        writes: dict[str, Any] = {"messages": [response]}
        sends: list[dict[str, Any]] = []
        if getattr(response, "tool_calls", None):
            sends.append({"node": "tools", "logical_id": TOOLS_LOGICAL_ID})
        else:
            writes["final_response"] = message_text_content(response.content)
        return {"writes": writes, "sends": sends}

    return _invoke


def apply_command_update(local_state: dict[str, Any], writes_total: dict[str, Any], update: dict[str, Any]) -> None:
    merge_state(local_state, update)
    accumulate_writes(writes_total, update)


def invoke_tools_partition(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
    local_state = {
        "messages": list(request_payload.get("messages") or []),
        "skills_loaded": list(request_payload.get("skills_loaded") or []),
        "final_response": request_payload.get("final_response", ""),
    }
    writes_total: dict[str, Any] = {}
    messages = list(local_state["messages"])
    if not messages or not isinstance(messages[-1], AIMessage):
        raise RuntimeError("tools partition expected the last message to be an AIMessage with tool calls")
    ai_message = messages[-1]
    tool_calls = list(getattr(ai_message, "tool_calls", None) or [])
    for tool_call in tool_calls:
        tool_name = str(tool_call.get("name", ""))
        tool_id = str(tool_call.get("id", ""))
        tool_args = dict(tool_call.get("args") or {})
        runtime = build_tool_runtime(local_state, tool_id)
        if tool_name == "load_skill":
            result = skills_example.load_skill.func(skill_name=str(tool_args["skill_name"]), runtime=runtime)
            if isinstance(result, Command):
                apply_command_update(local_state, writes_total, dict(result.update or {}))
            else:
                tool_message = ToolMessage(content=str(result), tool_call_id=tool_id)
                apply_command_update(local_state, writes_total, {"messages": [tool_message]})
        elif tool_name == "write_sql_query":
            vertical = tool_args.get("vertical") or infer_loaded_skill(local_state)
            if vertical is None:
                raise KeyError("vertical")
            result = skills_example.write_sql_query.func(
                query=str(tool_args["query"]),
                vertical=str(vertical),
                runtime=runtime,
            )
            tool_message = ToolMessage(content=str(result), tool_call_id=tool_id)
            apply_command_update(local_state, writes_total, {"messages": [tool_message]})
        else:
            tool_message = ToolMessage(content=f"Unknown tool: {tool_name}", tool_call_id=tool_id)
            apply_command_update(local_state, writes_total, {"messages": [tool_message]})
    return {"writes": writes_total, "sends": [{"node": "model", "logical_id": MODEL_LOGICAL_ID}]}


def monolith_invoker_factory(model, max_loop_steps: int) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    def _invoke(request_payload: dict[str, Any], _: dict[str, Any]) -> dict[str, Any]:
        local_state = {
            "messages": list(request_payload.get("messages") or []),
            "skills_loaded": list(request_payload.get("skills_loaded") or []),
            "final_response": request_payload.get("final_response", ""),
        }
        writes_total: dict[str, Any] = {}
        for _ in range(max_loop_steps):
            result = model_invoker_factory(model)(local_state, local_state)
            apply_command_update(local_state, writes_total, dict(result["writes"]))
            if not result["sends"]:
                break
            tools_result = invoke_tools_partition(local_state, local_state)
            apply_command_update(local_state, writes_total, dict(tools_result["writes"]))
        return {"writes": writes_total, "sends": []}

    return _invoke


def execute_partition(
    *,
    spec,
    base_state: dict[str, Any],
    read_keys: list[str],
    invoker: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    request_payload = state_slice(base_state, read_keys)
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


def execute_workflow_run(
    candidate: CandidateConfig,
    dataset_profile: dict[str, Any],
    args: argparse.Namespace,
    invocation_label: str,
) -> dict[str, Any]:
    del invocation_label
    state = build_initial_state(dataset_profile)
    model = skills_example.build_agent_model().bind_tools([skills_example.load_skill, skills_example.write_sql_query])
    metrics: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    invoked_partitions: list[str] = []
    started = perf_counter()
    error: str | None = None

    def run_step(logical_id: str, invoker: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]):
        spec = find_spec(candidate, logical_id)
        metric, writes, sends, trace = execute_partition(
            spec=spec,
            base_state=state,
            read_keys=["messages", "skills_loaded", "final_response"],
            invoker=invoker,
        )
        metrics.append(metric)
        traces.append(trace)
        invoked_partitions.append(logical_id)
        if not metric.get("function_error"):
            merge_state(state, writes)
        return metric, sends

    if candidate.candidate == "SK-U1":
        metric, _ = run_step("p_monolith", monolith_invoker_factory(model, int(args.max_loop_steps)))
        if metric.get("function_error"):
            error = str(metric["function_error"])
    else:
        step_count = 0
        next_partition = MODEL_LOGICAL_ID
        while step_count < int(args.max_loop_steps):
            if next_partition == MODEL_LOGICAL_ID:
                metric, sends = run_step(MODEL_LOGICAL_ID, model_invoker_factory(model))
            else:
                metric, sends = run_step(TOOLS_LOGICAL_ID, invoke_tools_partition)
            if metric.get("function_error"):
                error = str(metric["function_error"])
                break
            if not sends:
                break
            next_partition = str(sends[0]["logical_id"])
            step_count += 1
        if step_count >= int(args.max_loop_steps):
            error = error or f"LoopLimitError: exceeded max loop steps of {args.max_loop_steps}"

    workflow_elapsed_ms = round((perf_counter() - started) * 1000, 3)
    worker_cost_total = round(sum(float(metric.get("estimated_cost_usd") or 0.0) for metric in metrics), 10)
    coordinator_duration_ms = max(0.0, round(workflow_elapsed_ms - sum(float(metric.get("duration_ms") or 0.0) for metric in metrics), 3))
    coordinator_billed_duration_ms = max(1, int(math.ceil(coordinator_duration_ms))) if coordinator_duration_ms > 0 else 0

    messages = list(state.get("messages") or [])
    tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
    ai_messages = [message for message in messages if isinstance(message, AIMessage)]
    sql_vertical_hits = [
        str(message.content).split("SQL Query for ", 1)[1].split(":", 1)[0]
        for message in tool_messages
        if isinstance(message.content, str) and message.content.startswith("SQL Query for ")
    ]
    final_response = str(state.get("final_response") or "").strip() or final_ai_response_text(messages)
    expected_skill = str(dataset_profile["expected_skill"])
    result_correct = (
        expected_skill in list(state.get("skills_loaded") or [])
        and expected_skill in sql_vertical_hits
        and bool(final_response)
    )
    result_meta = {
        "route_match": result_correct,
        "model_turn_count": len(ai_messages),
        "tool_turn_count": len(tool_messages),
        "message_count": len(messages),
        "loaded_skill_count": len(list(state.get("skills_loaded") or [])),
        "loaded_skills": list(state.get("skills_loaded") or []),
        "sql_verticals": sql_vertical_hits,
        "final_response_chars": len(final_response),
        "final_response_preview": final_response[:200],
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
            model_turn_mean = (
                round(statistics.mean(payload["result_meta"]["model_turn_count"] for payload in measured), 3)
                if measured
                else None
            )
            tool_turn_mean = (
                round(statistics.mean(payload["result_meta"]["tool_turn_count"] for payload in measured), 3)
                if measured
                else None
            )
            message_count_mean = (
                round(statistics.mean(payload["result_meta"]["message_count"] for payload in measured), 3)
                if measured
                else None
            )
            loaded_skill_mean = (
                round(statistics.mean(payload["result_meta"]["loaded_skill_count"] for payload in measured), 3)
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
                "dataset_expected_skill": dataset_profile.get("expected_skill"),
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
                "single_model_turn_count_mean": model_turn_mean,
                "single_tool_turn_count_mean": tool_turn_mean,
                "single_message_count_mean": message_count_mean,
                "single_loaded_skill_count_mean": loaded_skill_mean,
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
            "The skills candidate matrix matches the thesis experiment table for SK-B, SK-C, SK-L, SK-R, SK-U1, SK-U2, and SK-U3.",
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
        "max_loop_steps": int(args.max_loop_steps),
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
    for candidate in SKILLS_CANDIDATES:
        print(
            f"- {candidate.candidate}: group={candidate.group}, "
            f"slo_profile={candidate.slo_profile or 'baseline/user'}, "
            f"pi={candidate.partitioning_vector}, m={candidate.memory_vector_mb}"
        )
    print("\nDatasets")
    for dataset_id in sorted(SKILLS_DATASETS):
        dataset = SKILLS_DATASETS[dataset_id]
        print(
            f"- {dataset_id}: expected_skill={dataset['expected_skill']}, "
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
            dataset_profile = dict(SKILLS_DATASETS[dataset_id])
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
    parser = argparse.ArgumentParser(description="Run the final skills experiment matrix locally.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("catalog", help="List available skills candidates and datasets.")

    run = subparsers.add_parser("run", help="Execute the skills candidate matrix and write resumable results.")
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
    run.add_argument("--max-loop-steps", type=int, default=DEFAULT_MAX_LOOP_STEPS)
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
