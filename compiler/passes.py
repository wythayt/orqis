from __future__ import annotations

from importlib import import_module
from typing import Any

from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from orqis.compiler.analysis_builder import build_analysis
from orqis.compiler.introspect import GraphIntrospector
from orqis.compiler.lgir2_builder import build_lgir2
from orqis.compiler.runtime_trace_builder import build_runtime_trace
from orqis.compiler.srv_plan_builder import build_srv_plan
from orqis.compiler.ir import CompilationBundle, PassTraceIR


def ensure_compiled_graph(graph_or_factory: Any) -> CompiledStateGraph:
    graph = graph_or_factory() if callable(graph_or_factory) else graph_or_factory
    if isinstance(graph, CompiledStateGraph):
        return graph
    if isinstance(graph, StateGraph):
        return graph.compile(name=getattr(graph, "name", None))
    raise TypeError("Expected a compiled graph, a StateGraph, or a zero-arg factory.")


def load_object(spec: str) -> Any:
    module_name, attr_name = spec.split(":", 1)
    module = import_module(module_name)
    return getattr(module, attr_name)


def compile_graph(
    graph_or_factory: Any,
    *,
    graph_id: str | None = None,
    sample_input: dict[str, Any] | None = None,
) -> CompilationBundle:
    compiled_graph = ensure_compiled_graph(graph_or_factory)

    introspector = GraphIntrospector(compiled_graph, graph_id=graph_id)
    lgir0, lgir1 = introspector.extract()
    pass_trace = [
        PassTraceIR(
            pass_name="extract-lgir0",
            description="Extract graph-level LangGraph IR.",
            highlights=[
                f"{len(lgir0.nodes)} nodes",
                f"{len(lgir0.edges)} edges",
                f"{len(lgir0.state_schema.keys)} state keys",
            ],
        ),
        PassTraceIR(
            pass_name="extract-lgir1",
            description="Lower compiled graph to explicit Pregel channels, triggers, and writers.",
            highlights=[
                f"{len(lgir1.channels)} channels",
                f"{len(lgir1.nodes)} executable nodes",
                f"{len(lgir1.trigger_to_nodes)} trigger mappings",
            ],
        ),
    ]

    analysis = build_analysis(compiled_graph, lgir0, lgir1)
    pass_trace.append(
        PassTraceIR(
            pass_name="static-analysis",
            description="Infer read/write sets, reducers, fanout, loops, side effects, and cache safety.",
            highlights=[
                f"{len(analysis.read_write)} node analyses",
                f"{len(analysis.fanout_regions)} fanout regions",
                f"{len([loop for loop in analysis.loops if loop.requires_loop_capable_orchestrator])} looping SCCs",
            ],
        )
    )

    lgir2 = build_lgir2(lgir0, analysis)
    pass_trace.append(
        PassTraceIR(
            pass_name="partition",
            description="Greedy legality-checked fusion into executable partitions.",
            highlights=[
                f"{len(lgir2.partitions)} partitions",
                f"{len([d for d in lgir2.fusion_decisions if d.accepted])} accepted fusions",
            ],
        )
    )

    runtime_trace = build_runtime_trace(compiled_graph, lgir0, sample_input) if sample_input is not None else []
    if runtime_trace:
        pass_trace.append(
            PassTraceIR(
                pass_name="runtime-trace",
                description="Capture a baseline Pregel execution trace from the real compiled graph.",
                highlights=[
                    f"{len(runtime_trace)} supersteps",
                    f"{sum(len(step.tasks) for step in runtime_trace)} tasks",
                ],
            )
        )

    srv_plan = build_srv_plan(lgir0, analysis, lgir2, runtime_trace=runtime_trace)
    pass_trace.append(
        PassTraceIR(
            pass_name="srv-plan",
            description="Lower partition IR to an AWS-oriented serverless deployment plan with memory optimization.",
            highlights=[
                srv_plan.orchestration.get("mode", "unknown"),
                f"{len(srv_plan.compute.get('workers', {}))} workers",
                f"resource optimizer total_compute_mb={srv_plan.compute.get('resource_summary', {}).get('total_compute_mb')}",
            ],
        )
    )

    next_steps = [
        "Replace AST-only write inference with explicit per-node read/write metadata or decorators.",
        "Integrate a real checkpointer so the compiler can emit concrete checkpoint and pending-write schemas from live runs.",
        "Use LangGraph internal task metadata to capture exact task_path ordering for non-commutative reducers.",
        "Replace static resource estimates with production latency, cost, and peak-memory measurements.",
        "Extend lowering for subgraphs, Command routes, loops, and deployable AWS IaC output.",
    ]

    return CompilationBundle(
        graph_id=lgir0.graph_id,
        pass_trace=pass_trace,
        lgir0=lgir0,
        lgir1=lgir1,
        analysis=analysis,
        lgir2=lgir2,
        srv_plan=srv_plan,
        runtime_trace=runtime_trace,
        next_steps=next_steps,
    )
