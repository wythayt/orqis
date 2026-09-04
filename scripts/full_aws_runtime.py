from __future__ import annotations

import base64
import concurrent.futures
import json
import math
import os
import pickle
import re
import time
import traceback
from collections import defaultdict
from time import perf_counter
from typing import Any, Callable


AWS_LAMBDA_GB_SECOND_USD = 0.0000166667
AWS_LAMBDA_REQUEST_USD = 0.20 / 1_000_000
REPORT_PATTERNS = {
    "duration_ms": re.compile(r"Duration: ([0-9.]+) ms"),
    "billed_duration_ms": re.compile(r"Billed Duration: ([0-9]+) ms"),
    "max_memory_used_mb": re.compile(r"Max Memory Used: ([0-9]+) MB"),
    "init_duration_ms": re.compile(r"Init Duration: ([0-9.]+) ms"),
}
_LAMBDA_CLIENT = None
_DDB_CLIENT = None
_S3_CLIENT = None
_SKILLS_MODEL = None


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    role = os.environ.get("ORQIS_LAMBDA_ROLE", "worker")
    if role == "coordinator":
        return coordinator_handler(event, context)
    if role == "worker":
        return worker_handler(event, context)
    if role == "barrier":
        return barrier_handler(event, context)
    raise ValueError(f"unknown ORQIS_LAMBDA_ROLE: {role}")


def worker_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    started = perf_counter()
    workflow = os.environ.get("ORQIS_WORKFLOW") or str(event["workflow"])
    candidate_id = os.environ.get("ORQIS_CANDIDATE") or str(event["candidate"])
    logical_id = str(event.get("logical_id") or os.environ.get("ORQIS_PARTITION_ID"))
    read_keys = list(event.get("read_keys") or [])
    task_input = dict(event.get("task_input") or {})
    checkpoint_metric = None
    if "state_blob" in event:
        state = decode_blob(str(event["state_blob"]))
    elif "checkpoint_ref" in event:
        state, checkpoint_metric = read_checkpoint(dict(event["checkpoint_ref"]))
    else:
        raise KeyError("worker input must include state_blob or checkpoint_ref")
    request_payload = {key: state.get(key) for key in read_keys if key in state}
    request_payload.update(task_input)
    telemetry_events: list[dict[str, Any]] = []
    writes_payload: dict[str, Any] = {}
    send_descriptors: list[dict[str, Any]] = []
    function_error = None
    error_trace = None
    try:
        invoker = select_partition_invoker(workflow, candidate_id, logical_id)
        with router_module().bedrock_telemetry_session(telemetry_events.append):
            result = invoker(request_payload, state)
        writes_payload = dict(result.get("writes") or {})
        send_descriptors = list(result.get("sends") or [])
    except Exception as exc:
        function_error = f"{type(exc).__name__}: {exc}"
        error_trace = traceback.format_exc(limit=10)
        writes_payload = {}
        send_descriptors = []

    writes_blob = encode_blob(writes_payload)
    sends_blob = encode_blob(send_descriptors)
    remote_metrics = router_module().aggregate_telemetry(telemetry_events)
    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    metric = {
        "workflow": workflow,
        "candidate": candidate_id,
        "logical_id": logical_id,
        "function_name": getattr(context, "function_name", os.environ.get("AWS_LAMBDA_FUNCTION_NAME")),
        "configured_memory_mb": int(os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "0") or 0),
        "handler_elapsed_ms": elapsed_ms,
        "request_pickle_bytes": len(encode_bytes(request_payload)),
        "state_pickle_bytes": len(encode_bytes(state)),
        "writes_pickle_bytes": len(encode_bytes(writes_payload)),
        "send_count": len(send_descriptors),
        "send_payload_pickle_bytes": len(encode_bytes(send_descriptors)),
        "function_error": function_error,
        "timed_out": False,
        "remote_call_count": remote_metrics["remote_call_count"],
        "remote_latency_ms": remote_metrics["remote_latency_ms"],
        "remote_input_tokens": remote_metrics["remote_input_tokens"],
        "remote_output_tokens": remote_metrics["remote_output_tokens"],
        "remote_total_tokens": remote_metrics["remote_total_tokens"],
        "error_trace": error_trace,
        "aws_request_id": getattr(context, "aws_request_id", None),
    }
    if checkpoint_metric is not None:
        metric["checkpoint_read_latency_ms"] = checkpoint_metric.get("latency_ms")
        metric["checkpoint_read_bytes"] = checkpoint_metric.get("bytes")
    return {
        "ok": function_error is None,
        "writes_blob": writes_blob,
        "sends_blob": sends_blob,
        "metric": metric,
        "checkpoint_metric": checkpoint_metric,
        "error": function_error,
    }


def barrier_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    started = perf_counter()
    candidate = json.loads(os.environ["ORQIS_CANDIDATE_JSON"])
    workflow = str(candidate["workflow"])
    candidate_id = str(candidate["candidate"])
    action = str(event.get("action") or "")
    if action == "init":
        return barrier_init(event, context, started, workflow, candidate_id)
    if action == "apply":
        return barrier_apply(event, context, started, workflow, candidate_id)
    if action == "finish":
        return barrier_finish(event, context, started, candidate, workflow, candidate_id)
    raise ValueError(f"unsupported barrier action: {action}")


def barrier_init(
    event: dict[str, Any],
    context: Any,
    started: float,
    workflow: str,
    candidate_id: str,
) -> dict[str, Any]:
    dataset_profile = dict(event["dataset_profile"])
    invocation_label = str(event.get("invocation_label") or getattr(context, "aws_request_id", "run"))
    run_id = str(event.get("run_id") or f"{workflow}:{candidate_id}:{invocation_label}")
    state = build_initial_state(workflow, dataset_profile, invocation_label)
    checkpoint_ref = write_checkpoint(run_id, 0, "initial", state)
    barrier_metrics = [
        build_barrier_metric(context, "init", started),
    ]
    return {
        "workflow": workflow,
        "candidate": candidate_id,
        "dataset_id": dataset_profile["dataset_id"],
        "dataset_profile": dataset_profile,
        "run_id": run_id,
        "invocation_label": invocation_label,
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_trace": [checkpoint_ref["metric"]],
        "metrics": [],
        "barrier_metrics": barrier_metrics,
        "invoked_partitions": [],
        "last_sends": [],
        "step_count": 0,
        "barrier_apply_count": 0,
        "error": None,
        "start_time_ms": int(time.time() * 1000),
    }


def barrier_apply(
    event: dict[str, Any],
    context: Any,
    started: float,
    workflow: str,
    candidate_id: str,
) -> dict[str, Any]:
    checkpoint_trace = list(event.get("checkpoint_trace") or [])
    metrics = list(event.get("metrics") or [])
    barrier_metrics = list(event.get("barrier_metrics") or [])
    invoked_partitions = list(event.get("invoked_partitions") or [])
    error = event.get("error")
    apply_step_index = int(event.get("barrier_apply_count") or 0) + 1
    state, read_metric = read_checkpoint(dict(event["checkpoint_ref"]))
    checkpoint_trace.append(read_metric)
    worker_results = collect_worker_results(event)
    last_sends: list[dict[str, Any]] = []
    for result in worker_results:
        result = unwrap_lambda_payload(result)
        checkpoint_metric = result.get("checkpoint_metric")
        if checkpoint_metric:
            checkpoint_trace.append(checkpoint_metric)
        metric = dict(result.get("metric") or {})
        if metric:
            metric["apply_step_index"] = apply_step_index
            metrics.append(metric)
            invoked_partitions.append(str(metric.get("logical_id") or "unknown"))
        sends = decode_blob(result.get("sends_blob", encode_blob([])))
        if metric.get("function_error") or not result.get("ok", False):
            error = error or str(metric.get("function_error") or result.get("error") or "worker failed")
            last_sends = []
            continue
        writes = decode_blob(result.get("writes_blob", encode_blob({})))
        merge_state(workflow, state, writes)
        last_sends = list(sends)
    label = str(event.get("label") or (metrics[-1].get("logical_id") if metrics else "apply"))
    checkpoint_ref = write_checkpoint(str(event["run_id"]), len(metrics), label, state)
    checkpoint_trace.append(checkpoint_ref["metric"])
    barrier_metrics.append(build_barrier_metric(context, f"apply:{label}", started))
    return {
        "workflow": workflow,
        "candidate": candidate_id,
        "dataset_id": event["dataset_id"],
        "dataset_profile": event["dataset_profile"],
        "run_id": event["run_id"],
        "invocation_label": event["invocation_label"],
        "checkpoint_ref": checkpoint_ref,
        "checkpoint_trace": checkpoint_trace,
        "metrics": metrics,
        "barrier_metrics": barrier_metrics,
        "invoked_partitions": invoked_partitions,
        "last_sends": last_sends,
        "step_count": int(event.get("step_count") or 0) + len(worker_results),
        "barrier_apply_count": apply_step_index,
        "error": error,
        "start_time_ms": event.get("start_time_ms"),
    }


def barrier_finish(
    event: dict[str, Any],
    context: Any,
    started: float,
    candidate: dict[str, Any],
    workflow: str,
    candidate_id: str,
) -> dict[str, Any]:
    checkpoint_trace = list(event.get("checkpoint_trace") or [])
    state, read_metric = read_checkpoint(dict(event["checkpoint_ref"]))
    checkpoint_trace.append(read_metric)
    barrier_metrics = list(event.get("barrier_metrics") or [])
    barrier_metrics.append(build_barrier_metric(context, "finish", started))
    metrics = list(event.get("metrics") or [])
    start_time_ms = int(event.get("start_time_ms") or int(time.time() * 1000))
    elapsed_ms = max(0, int(time.time() * 1000) - start_time_ms)
    result_meta = build_result_meta(workflow, dict(event["dataset_profile"]), state, candidate, list(event.get("invoked_partitions") or []))
    return {
        "workflow": workflow,
        "candidate": candidate_id,
        "dataset_id": event["dataset_id"],
        "workflow_elapsed_ms": elapsed_ms,
        "orchestration_mode": "stepfunctions",
        "error": event.get("error"),
        "timeout_count": sum(1 for metric in metrics if metric.get("timed_out")),
        "error_count": sum(1 for metric in metrics if metric.get("function_error")),
        "estimated_cost_usd": round(sum(float(metric.get("estimated_cost_usd") or 0.0) for metric in metrics), 10),
        "metrics": metrics,
        "barrier_metrics": barrier_metrics,
        "partition_summary": partition_summary(metrics),
        "checkpoint_trace": checkpoint_trace,
        "result_meta": result_meta,
    }


def collect_worker_results(event: dict[str, Any]) -> list[dict[str, Any]]:
    if "worker_results" in event:
        return [unwrap_lambda_payload(item) for item in list(event.get("worker_results") or [])]
    if "worker_result" in event:
        return [unwrap_lambda_payload(dict(event["worker_result"]))]
    return []


def unwrap_lambda_payload(result: Any) -> dict[str, Any]:
    if isinstance(result, dict) and "Payload" in result and isinstance(result["Payload"], dict):
        return dict(result["Payload"])
    if isinstance(result, dict):
        return dict(result)
    return {"ok": False, "error": f"unexpected worker result: {result!r}"}


def build_barrier_metric(context: Any, action: str, started: float) -> dict[str, Any]:
    return {
        "action": action,
        "function_name": getattr(context, "function_name", os.environ.get("AWS_LAMBDA_FUNCTION_NAME")),
        "aws_request_id": getattr(context, "aws_request_id", None),
        "configured_memory_mb": int(os.environ.get("AWS_LAMBDA_FUNCTION_MEMORY_SIZE", "0") or 0),
        "handler_elapsed_ms": round((perf_counter() - started) * 1000, 3),
    }


def coordinator_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    started = perf_counter()
    candidate = json.loads(os.environ["ORQIS_CANDIDATE_JSON"])
    worker_map = json.loads(os.environ["ORQIS_WORKER_MAP_JSON"])
    workflow = str(candidate["workflow"])
    candidate_id = str(candidate["candidate"])
    dataset_profile = dict(event["dataset_profile"])
    invocation_label = str(event.get("invocation_label") or getattr(context, "aws_request_id", "run"))
    run_id = str(event.get("run_id") or f"{workflow}:{candidate_id}:{invocation_label}")
    max_loop_steps = int(os.environ.get("ORQIS_MAX_LOOP_STEPS", "8"))

    state = build_initial_state(workflow, dataset_profile, invocation_label)
    metrics: list[dict[str, Any]] = []
    checkpoint_trace: list[dict[str, Any]] = []
    invoked_partitions: list[str] = []
    error: str | None = None

    checkpoint_ref = write_checkpoint(run_id, 0, "initial", state)
    checkpoint_trace.append(checkpoint_ref["metric"])
    apply_step_index = 0

    def run_step(logical_id: str, read_keys: list[str], *, task_input: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
        nonlocal checkpoint_ref, state, error, apply_step_index
        apply_step_index += 1
        loaded_state, read_metric = read_checkpoint(checkpoint_ref)
        checkpoint_trace.append(read_metric)
        response = invoke_worker(
            worker_map,
            workflow=workflow,
            candidate_id=candidate_id,
            logical_id=logical_id,
            state=loaded_state,
            read_keys=read_keys,
            task_input=task_input or {},
        )
        worker_metric = response["metric"]
        worker_metric["apply_step_index"] = apply_step_index
        metrics.append(worker_metric)
        invoked_partitions.append(logical_id)
        writes = decode_blob(response.get("writes_blob", encode_blob({})))
        sends = decode_blob(response.get("sends_blob", encode_blob([])))
        if worker_metric.get("function_error") or not response.get("ok", False):
            error = str(worker_metric.get("function_error") or response.get("error") or "worker failed")
            writes = {}
            sends = []
        else:
            merge_state(workflow, loaded_state, writes)
            state = loaded_state
        checkpoint_ref = write_checkpoint(run_id, len(metrics), logical_id, state)
        checkpoint_trace.append(checkpoint_ref["metric"])
        return worker_metric, writes, sends

    def run_subagent_branches(sends: list[dict[str, Any]]) -> None:
        nonlocal checkpoint_ref, state, error, apply_step_index
        if not sends:
            return
        apply_step_index += 1
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sends)) as executor:
            futures = []
            for index, send in enumerate(sends):
                snapshot, read_metric = read_checkpoint(checkpoint_ref)
                checkpoint_trace.append(read_metric)
                futures.append(
                    (
                        index,
                        executor.submit(
                            invoke_worker,
                            worker_map,
                            workflow=workflow,
                            candidate_id=candidate_id,
                            logical_id=str(send["logical_id"]),
                            state=snapshot,
                            read_keys=["incident_id", "severity", "evidence_scope"],
                            task_input=dict(send.get("task_input") or {}),
                        ),
                    )
                )
            for index, future in futures:
                results.append((index, future.result()))
        for _index, response in sorted(results, key=lambda item: item[0]):
            metric = response["metric"]
            metric["apply_step_index"] = apply_step_index
            metrics.append(metric)
            invoked_partitions.append(str(metric["logical_id"]))
            if metric.get("function_error") or not response.get("ok", False):
                error = error or str(metric.get("function_error") or response.get("error") or "subagent worker failed")
                continue
            writes = decode_blob(response.get("writes_blob", encode_blob({})))
            merge_state(workflow, state, writes)
        checkpoint_ref = write_checkpoint(run_id, len(metrics), "subagent_branches", state)
        checkpoint_trace.append(checkpoint_ref["metric"])

    try:
        if workflow == router_module().WORKFLOW_ID:
            execute_router(candidate_id, run_step)
        elif workflow == skills_module().WORKFLOW_ID:
            execute_skills(candidate_id, max_loop_steps, run_step)
        elif workflow == subagent_module().WORKFLOW_ID:
            execute_subagents(candidate_id, run_step, run_subagent_branches)
        else:
            raise ValueError(f"unsupported workflow: {workflow}")
    except Exception as exc:
        error = error or f"{type(exc).__name__}: {exc}"

    elapsed_ms = round((perf_counter() - started) * 1000, 3)
    worker_cost_total = round(sum(float(metric.get("estimated_cost_usd") or 0.0) for metric in metrics), 10)
    result_meta = build_result_meta(workflow, dataset_profile, state, candidate, invoked_partitions)
    return {
        "workflow": workflow,
        "candidate": candidate_id,
        "dataset_id": dataset_profile["dataset_id"],
        "workflow_elapsed_ms": elapsed_ms,
        "error": error,
        "timeout_count": sum(1 for metric in metrics if metric.get("timed_out")),
        "error_count": sum(1 for metric in metrics if metric.get("function_error")),
        "estimated_cost_usd": worker_cost_total,
        "metrics": metrics,
        "partition_summary": partition_summary(metrics),
        "coordinator_metric": {
            "handler_elapsed_ms": elapsed_ms,
            "aws_request_id": getattr(context, "aws_request_id", None),
        },
        "checkpoint_trace": checkpoint_trace,
        "result_meta": result_meta,
    }


def execute_router(candidate_id: str, run_step: Callable[..., tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]) -> None:
    if candidate_id in {"RT-B", "RT-C", "RT-L", "RT-R", "RT-U3"}:
        _metric, _writes, sends = run_step("p_intake_request_triage_request_fanout", ["request_text", "account_tier"])
        if sends:
            run_step(str(sends[0]["logical_id"]), ["case_id", "account_tier", "normalized_text"])
        run_step("p_finalize_response", ["route_decision", "response_draft", "evidence_refs", "follow_up_plan"])
        return
    if candidate_id == "RT-U1":
        run_step("p_intake_request", ["request_text"])
        _metric, _writes, sends = run_step("p_triage_request", ["account_tier", "normalized_text"])
        if sends:
            run_step(str(sends[0]["logical_id"]), ["case_id", "account_tier", "normalized_text"])
        run_step("p_finalize_response", ["route_decision", "response_draft", "evidence_refs", "follow_up_plan"])
        return
    if candidate_id == "RT-U2":
        run_step("p_intake_request_triage_request", ["request_text", "account_tier"])
        run_step("p_merged_specialist", ["case_id", "account_tier", "normalized_text", "route_decision"])
        run_step("p_finalize_response", ["route_decision", "response_draft", "evidence_refs", "follow_up_plan"])
        return
    raise ValueError(f"unsupported router candidate: {candidate_id}")


def execute_skills(candidate_id: str, max_loop_steps: int, run_step: Callable[..., tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]]) -> None:
    if candidate_id == "SK-U1":
        run_step("p_monolith", ["messages", "skills_loaded", "final_response"])
        return
    next_partition = skills_module().MODEL_LOGICAL_ID
    for _ in range(max_loop_steps):
        _metric, _writes, sends = run_step(next_partition, ["messages", "skills_loaded", "final_response"])
        if not sends:
            return
        next_partition = str(sends[0]["logical_id"])
    raise RuntimeError(f"LoopLimitError: exceeded max loop steps of {max_loop_steps}")


def execute_subagents(
    candidate_id: str,
    run_step: Callable[..., tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]],
    run_branches: Callable[[list[dict[str, Any]]], None],
) -> None:
    if candidate_id in {"IR-B", "IR-C", "IR-L", "IR-R", "IR-U3"}:
        _metric, _writes, sends = run_step("p_ingest_alert_plan_response_fanout", ["incident_id", "severity", "alert_summary"])
        run_branches(sends)
        run_step("p_synthesize_recommendation", ["findings", "containment_actions", "communication_drafts"])
        run_step("p_finalize_incident", ["executive_summary", "final_recommendation", "communication_drafts"])
        return
    if candidate_id == "IR-U1":
        run_step("p_ingest_alert", ["severity", "alert_summary"])
        _metric, _writes, sends = run_step("p_plan_response_fanout", ["incident_id", "severity", "alert_summary", "evidence_scope"])
        run_branches(sends)
        run_step("p_synthesize_recommendation", ["findings", "containment_actions", "communication_drafts"])
        run_step("p_finalize_incident", ["executive_summary", "final_recommendation", "communication_drafts"])
        return
    if candidate_id == "IR-U2":
        run_step("p_monolith", ["incident_id", "severity", "alert_summary"])
        return
    raise ValueError(f"unsupported subagent candidate: {candidate_id}")


def select_partition_invoker(workflow: str, candidate_id: str, logical_id: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    if workflow == router_module().WORKFLOW_ID:
        return select_router_invoker(logical_id)
    if workflow == skills_module().WORKFLOW_ID:
        return select_skills_invoker(candidate_id, logical_id)
    if workflow == subagent_module().WORKFLOW_ID:
        return select_subagent_invoker(logical_id)
    raise ValueError(f"unsupported workflow: {workflow}")


def select_router_invoker(logical_id: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    rt = router_module()
    if logical_id == "p_intake_request_triage_request_fanout":
        return rt.invoke_dispatch_partition
    if logical_id == "p_intake_request":
        return rt.invoke_intake_partition
    if logical_id == "p_triage_request":
        return rt.invoke_triage_partition
    if logical_id == "p_intake_request_triage_request":
        return rt.invoke_intake_triage_partition
    if logical_id in rt.LOGICAL_ID_TO_SPECIALIST_FN:
        return rt.make_specialist_invoker(logical_id)
    if logical_id == "p_merged_specialist":
        return rt.invoke_merged_specialist
    if logical_id == "p_finalize_response":
        return rt.invoke_finalize_partition
    raise ValueError(f"unsupported router partition: {logical_id}")


def select_skills_invoker(candidate_id: str, logical_id: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    sk = skills_module()
    model = skills_model()
    if logical_id == sk.MODEL_LOGICAL_ID:
        return sk.model_invoker_factory(model)
    if logical_id == sk.TOOLS_LOGICAL_ID:
        return sk.invoke_tools_partition
    if logical_id == "p_monolith":
        return sk.monolith_invoker_factory(model, int(os.environ.get("ORQIS_MAX_LOOP_STEPS", "8")))
    raise ValueError(f"unsupported skills partition for {candidate_id}: {logical_id}")


def select_subagent_invoker(logical_id: str) -> Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]:
    ir = subagent_module()
    if logical_id == "p_ingest_alert_plan_response_fanout":
        return ir.invoke_dispatch_partition
    if logical_id == "p_ingest_alert":
        return ir.invoke_ingest_partition
    if logical_id == "p_plan_response_fanout":
        return ir.invoke_plan_partition
    if logical_id in ir.SUBAGENT_FUNCS:
        return ir.make_subagent_invoker(logical_id)
    if logical_id == "p_synthesize_recommendation":
        return ir.invoke_synthesis_partition
    if logical_id == "p_finalize_incident":
        return ir.invoke_finalize_partition
    if logical_id == "p_monolith":
        return ir.invoke_monolith
    raise ValueError(f"unsupported subagent partition: {logical_id}")


def build_initial_state(workflow: str, dataset_profile: dict[str, Any], invocation_label: str) -> dict[str, Any]:
    if workflow == router_module().WORKFLOW_ID:
        return router_module().build_initial_state(dataset_profile, invocation_label)
    if workflow == skills_module().WORKFLOW_ID:
        return skills_module().build_initial_state(dataset_profile)
    if workflow == subagent_module().WORKFLOW_ID:
        return subagent_module().build_initial_state(dataset_profile, invocation_label)
    raise ValueError(f"unsupported workflow: {workflow}")


def merge_state(workflow: str, state: dict[str, Any], writes: dict[str, Any]) -> None:
    if workflow == router_module().WORKFLOW_ID:
        state.update(writes)
        return
    if workflow == skills_module().WORKFLOW_ID:
        skills_module().merge_state(state, writes)
        return
    if workflow == subagent_module().WORKFLOW_ID:
        subagent_module().merge_state(state, writes)
        return
    raise ValueError(f"unsupported workflow: {workflow}")


def build_result_meta(
    workflow: str,
    dataset_profile: dict[str, Any],
    state: dict[str, Any],
    candidate: dict[str, Any],
    invoked_partitions: list[str],
) -> dict[str, Any]:
    worker_defs = list(candidate.get("worker_defs") or [])
    base = {
        "declared_total_memory_mb": sum(int(spec.get("memory_mb") or 0) for spec in worker_defs),
        "max_declared_partition_memory_mb": max([int(spec.get("memory_mb") or 0) for spec in worker_defs] or [0]),
        "invoked_partition_count": len(invoked_partitions),
        "deployed_partition_count": len(worker_defs),
        "partition_sequence": list(invoked_partitions),
        "final_state_pickle_bytes": len(encode_bytes(state)),
    }
    if workflow == router_module().WORKFLOW_ID:
        base.update(
            {
                "expected_route": dataset_profile.get("expected_route"),
                "observed_route": state.get("route_decision"),
                "route_match": state.get("route_decision") == dataset_profile.get("expected_route"),
                "final_response_chars": len(str(state.get("final_response") or "")),
            }
        )
    elif workflow == skills_module().WORKFLOW_ID:
        final_response = str(state.get("final_response") or "")
        base.update(
            {
                "final_response_chars": len(final_response),
                "loaded_skill_count": len(list(state.get("skills_loaded") or [])),
                "message_count": len(list(state.get("messages") or [])),
                "task_success": bool(final_response),
            }
        )
    elif workflow == subagent_module().WORKFLOW_ID:
        base.update(
            {
                "final_recommendation_chars": len(str(state.get("final_recommendation") or "")),
                "executive_summary_chars": len(str(state.get("executive_summary") or "")),
                "finding_count": len(list(state.get("findings") or [])),
                "containment_action_count": len(list(state.get("containment_actions") or [])),
                "communication_draft_count": len(list(state.get("communication_drafts") or [])),
                "task_success": bool(state.get("final_recommendation")),
            }
        )
    return make_json_safe(base)


def invoke_worker(
    worker_map: dict[str, Any],
    *,
    workflow: str,
    candidate_id: str,
    logical_id: str,
    state: dict[str, Any],
    read_keys: list[str],
    task_input: dict[str, Any],
) -> dict[str, Any]:
    function = worker_map[logical_id]
    payload = {
        "workflow": workflow,
        "candidate": candidate_id,
        "logical_id": logical_id,
        "read_keys": read_keys,
        "task_input": task_input,
        "state_blob": encode_blob(state),
    }
    started = perf_counter()
    response = lambda_client().invoke(
        FunctionName=function["function_name"],
        InvocationType="RequestResponse",
        LogType="Tail",
        Payload=json.dumps(payload).encode("utf-8"),
    )
    client_elapsed_ms = round((perf_counter() - started) * 1000, 3)
    raw_payload = response["Payload"].read().decode("utf-8") or "{}"
    body = json.loads(raw_payload)
    log_text = decode_log_result(response.get("LogResult"))
    report = parse_report_log(log_text)
    metric = dict(body.get("metric") or {})
    function_error = response.get("FunctionError") or metric.get("function_error")
    timed_out = "Task timed out" in log_text
    metric.update(
        {
            "logical_id": logical_id,
            "function_name": function["function_name"],
            "configured_memory_mb": function.get("memory_mb"),
            "configured_timeout_sec": function.get("timeout_sec"),
            "configured_concurrency_limit": function.get("concurrency_limit"),
            "status_code": int(response.get("StatusCode", 0) or 0),
            "function_error": function_error,
            "duration_ms": report.get("duration_ms", metric.get("handler_elapsed_ms")),
            "billed_duration_ms": report.get("billed_duration_ms"),
            "init_duration_ms": report.get("init_duration_ms"),
            "max_memory_used_mb": report.get("max_memory_used_mb"),
            "estimated_cost_usd": estimate_lambda_cost_usd(function.get("memory_mb"), report.get("billed_duration_ms")),
            "timed_out": bool(timed_out),
            "client_elapsed_ms": client_elapsed_ms,
            "invoke_payload_bytes": len(json.dumps(payload).encode("utf-8")),
            "response_payload_bytes": len(raw_payload.encode("utf-8")),
        }
    )
    body["metric"] = metric
    if function_error and not body.get("error"):
        body["error"] = raw_payload[:2048]
    return body


def checkpoint_enabled() -> bool:
    return bool(os.environ.get("ORQIS_CHECKPOINT_TABLE"))


def write_checkpoint(run_id: str, step: int, label: str, state: dict[str, Any]) -> dict[str, Any]:
    checkpoint_id = f"{step:04d}#{label}"
    blob = encode_bytes(state)
    metric = {
        "operation": "write",
        "run_id": run_id,
        "checkpoint_id": checkpoint_id,
        "bytes": len(blob),
        "backend": "none",
        "latency_ms": 0.0,
    }
    if not checkpoint_enabled():
        return {"run_id": run_id, "checkpoint_id": checkpoint_id, "state_blob": encode_blob(state), "metric": metric}

    table_name = os.environ["ORQIS_CHECKPOINT_TABLE"]
    bucket = os.environ.get("ORQIS_CHECKPOINT_BUCKET")
    inline_limit = int(os.environ.get("ORQIS_STATE_INLINE_LIMIT_BYTES", "300000"))
    item = {
        "run_id": {"S": run_id},
        "checkpoint_id": {"S": checkpoint_id},
        "label": {"S": label},
        "step": {"N": str(step)},
        "state_bytes": {"N": str(len(blob))},
        "created_at_ms": {"N": str(int(time.time() * 1000))},
    }
    started = perf_counter()
    if bucket and len(blob) > inline_limit:
        key = f"{run_id.replace(':', '/')}/{checkpoint_id}.pkl"
        s3_client().put_object(Bucket=bucket, Key=key, Body=blob)
        item["state_s3_key"] = {"S": key}
        metric["backend"] = "s3+dynamodb"
        metric["s3_key"] = key
    else:
        item["state_b64"] = {"S": base64.b64encode(blob).decode("ascii")}
        metric["backend"] = "dynamodb"
    ddb_client().put_item(TableName=table_name, Item=item)
    metric["latency_ms"] = round((perf_counter() - started) * 1000, 3)
    return {"run_id": run_id, "checkpoint_id": checkpoint_id, "metric": metric}


def read_checkpoint(ref: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metric = {
        "operation": "read",
        "run_id": ref["run_id"],
        "checkpoint_id": ref["checkpoint_id"],
        "bytes": 0,
        "backend": "none",
        "latency_ms": 0.0,
    }
    if not checkpoint_enabled():
        state = decode_blob(str(ref["state_blob"]))
        metric["bytes"] = len(encode_bytes(state))
        return state, metric

    started = perf_counter()
    response = ddb_client().get_item(
        TableName=os.environ["ORQIS_CHECKPOINT_TABLE"],
        Key={
            "run_id": {"S": str(ref["run_id"])},
            "checkpoint_id": {"S": str(ref["checkpoint_id"])},
        },
        ConsistentRead=True,
    )
    item = response.get("Item")
    if not item:
        raise KeyError(f"missing checkpoint {ref['run_id']} {ref['checkpoint_id']}")
    if "state_s3_key" in item:
        key = item["state_s3_key"]["S"]
        blob = s3_client().get_object(Bucket=os.environ["ORQIS_CHECKPOINT_BUCKET"], Key=key)["Body"].read()
        metric["backend"] = "s3+dynamodb"
        metric["s3_key"] = key
    else:
        blob = base64.b64decode(item["state_b64"]["S"].encode("ascii"))
        metric["backend"] = "dynamodb"
    metric["bytes"] = len(blob)
    metric["latency_ms"] = round((perf_counter() - started) * 1000, 3)
    return pickle.loads(blob), metric


def partition_summary(metrics: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        grouped[str(metric.get("logical_id", ""))].append(metric)
    summary = {}
    for logical_id, items in sorted(grouped.items()):
        durations = [float(item.get("duration_ms") or 0.0) for item in items]
        billed = [float(item.get("billed_duration_ms") or 0.0) for item in items]
        costs = [float(item.get("estimated_cost_usd") or 0.0) for item in items]
        summary[logical_id] = {
            "count": len(items),
            "avg_duration_ms": round(sum(durations) / len(durations), 3) if durations else None,
            "avg_billed_duration_ms": round(sum(billed) / len(billed), 3) if billed else None,
            "max_memory_used_mb": max((int(item.get("max_memory_used_mb") or 0) for item in items), default=None),
            "estimated_cost_usd_total": round(sum(costs), 10),
        }
    return summary


def encode_bytes(value: Any) -> bytes:
    return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)


def encode_blob(value: Any) -> str:
    return base64.b64encode(encode_bytes(value)).decode("ascii")


def decode_blob(value: str) -> Any:
    return pickle.loads(base64.b64decode(value.encode("ascii")))


def decode_log_result(value: str | None) -> str:
    if not value:
        return ""
    try:
        return base64.b64decode(value).decode("utf-8", errors="replace")
    except Exception:
        return ""


def parse_report_log(log_text: str) -> dict[str, float | int]:
    report: dict[str, float | int] = {}
    for key, pattern in REPORT_PATTERNS.items():
        match = pattern.search(log_text)
        if not match:
            continue
        value = match.group(1)
        report[key] = float(value) if "." in value else int(value)
    return report


def estimate_lambda_cost_usd(memory_mb: int | float | None, billed_duration_ms: int | float | None) -> float | None:
    if memory_mb is None or billed_duration_ms is None:
        return None
    billed_seconds = float(billed_duration_ms) / 1000.0
    memory_gb = float(memory_mb) / 1024.0
    return round((memory_gb * billed_seconds * AWS_LAMBDA_GB_SECOND_USD) + AWS_LAMBDA_REQUEST_USD, 10)


def make_json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    return str(value)


def lambda_client():
    global _LAMBDA_CLIENT
    if _LAMBDA_CLIENT is None:
        import boto3
        from botocore.config import Config

        _LAMBDA_CLIENT = boto3.client("lambda", config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))
    return _LAMBDA_CLIENT


def ddb_client():
    global _DDB_CLIENT
    if _DDB_CLIENT is None:
        import boto3
        from botocore.config import Config

        _DDB_CLIENT = boto3.client("dynamodb", config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))
    return _DDB_CLIENT


def s3_client():
    global _S3_CLIENT
    if _S3_CLIENT is None:
        import boto3
        from botocore.config import Config

        _S3_CLIENT = boto3.client("s3", config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))
    return _S3_CLIENT


def router_module():
    from orqis.examples.final_experiments import router_experiment

    return router_experiment


def skills_module():
    from orqis.examples.final_experiments import skills_experiment

    return skills_experiment


def subagent_module():
    from orqis.examples.final_experiments import subagents_experiment

    return subagents_experiment


def skills_model():
    global _SKILLS_MODEL
    if _SKILLS_MODEL is None:
        sk = skills_module()
        _SKILLS_MODEL = sk.skills_example.build_agent_model().bind_tools(
            [sk.skills_example.load_skill, sk.skills_example.write_sql_query]
        )
    return _SKILLS_MODEL
