from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from orqis.compiler.ir import GraphIR0, StepTraceIR, TaskTraceIR


def build_runtime_trace(
    compiled_graph: CompiledStateGraph,
    lgir0: GraphIR0,
    sample_input: dict[str, Any] | None,
) -> list[StepTraceIR]:
    if sample_input is None:
        return []
    events = list(compiled_graph.stream(copy.deepcopy(sample_input), stream_mode="debug"))
    steps: dict[int, dict[str, Any]] = defaultdict(lambda: {"tasks": [], "results": {}})
    for event in events:
        step = int(event["step"])
        payload = event["payload"]
        if event["type"] == "task":
            steps[step]["tasks"].append(payload)
        elif event["type"] == "task_result":
            steps[step]["results"][payload["id"]] = payload
    state = copy.deepcopy(sample_input)
    reducer_map = {
        key: state_key.reducer
        for key, state_key in lgir0.state_schema.keys.items()
        if state_key.reducer is not None
    }
    traces: list[StepTraceIR] = []
    for step in sorted(steps):
        grouped_writes: dict[str, list[Any]] = defaultdict(list)
        task_traces: list[TaskTraceIR] = []
        for task in steps[step]["tasks"]:
            result = steps[step]["results"].get(task["id"], {})
            result_payload = result.get("result", {}) or {}
            for key, value in result_payload.items():
                grouped_writes[key].append(value)
            task_kind = "PUSH" if "__pregel_push" in task.get("triggers", ()) else "PULL"
            task_traces.append(
                TaskTraceIR(
                    task_id=task["id"],
                    task_kind=task_kind,
                    node_id=task["name"],
                    triggers=list(task.get("triggers", ())),
                    input_slice=_payload_as_dict(task.get("input", {})),
                    result=_payload_as_dict(result_payload),
                )
            )
        for key, values in grouped_writes.items():
            reducer = reducer_map.get(key)
            if reducer and reducer.reducer_id == "operator.add":
                current = state.get(key, [])
                merged = list(current)
                for value in values:
                    merged = merged + list(value)
                state[key] = merged
            else:
                state[key] = values[-1]
        notes = []
        if any(task.task_kind == "PUSH" for task in task_traces):
            notes.append("dynamic Send fanout executed in this step")
        traces.append(
            StepTraceIR(
                step=step,
                tasks=task_traces,
                grouped_writes=dict(grouped_writes),
                state_after_step=copy.deepcopy(state),
                notes=notes,
            )
        )
    return traces


def _payload_as_dict(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    # wrap non-mapping payloads so the trace format stays uniform across agent graphs and state graphs.
    return {"payload": payload}
