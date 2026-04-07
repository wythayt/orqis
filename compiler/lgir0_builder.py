from __future__ import annotations

from typing import Any

from langgraph.graph.state import CompiledStateGraph

from orqis.compiler.builder_common import (
    channel_kind,
    extract_cache_policy,
    extract_reducer,
    extract_resources,
    extract_retry_policy,
    extract_side_effects,
    infer_branch_return_kind,
    normalize_sequence,
)
from orqis.compiler.ir import EdgeIR0, GraphIR0, NodeIR0, StateKeyIR, StateSchemaIR
from orqis.compiler.utils import callable_ref, sanitize_identifier, type_name, typed_dict_annotations, typed_dict_keys, unwrap_callable


def build_lgir0(compiled_graph: CompiledStateGraph, graph_id: str | None = None) -> GraphIR0:
    builder = compiled_graph.builder
    state_annotations = typed_dict_annotations(builder.state_schema)
    nodes = {
        node_id: _extract_node_ir0(node_id, spec)
        for node_id, spec in sorted(builder.nodes.items())
    }
    edges = _extract_edges(builder)
    entrypoints = sorted(edge.target for edge in edges if edge.source == "__start__" and edge.target)
    finishpoints = sorted(edge.source for edge in edges if edge.target == "__end__" and edge.source)
    raw_graph_id = graph_id or compiled_graph.name or "langgraph"
    return GraphIR0(
        ir_version="lgir-0.1",
        graph_id=sanitize_identifier(raw_graph_id),
        entrypoints=entrypoints,
        finishpoints=finishpoints,
        context_schema=type_name(builder.context_schema),
        state_schema=_extract_state_schema(compiled_graph, state_annotations),
        nodes=nodes,
        edges=edges,
        subgraphs={},
        options={"durability": None, "recursion_limit": None},
    )


def _extract_state_schema(
    compiled_graph: CompiledStateGraph,
    state_annotations: dict[str, Any],
) -> StateSchemaIR:
    keys: dict[str, StateKeyIR] = {}
    for key in normalize_sequence(compiled_graph.output_channels):
        channel = compiled_graph.channels[key]
        annotation = state_annotations.get(key, getattr(channel, "typ", Any))
        reducer = extract_reducer(annotation, channel)
        ordering_sensitive = True
        if reducer is not None:
            ordering_sensitive = not reducer.commutative
            if reducer.reducer_id == "operator.add" and "list" in type_name(annotation):
                ordering_sensitive = True
        keys[key] = StateKeyIR(
            key=key,
            value_type=type_name(annotation),
            channel_kind=channel_kind(channel),
            reducer=reducer,
            ordering_sensitive=ordering_sensitive,
            dedupe=getattr(channel, "dedupe", None),
        )
    return StateSchemaIR(keys=keys)


def _extract_node_ir0(node_id: str, spec: Any) -> NodeIR0:
    runnable = unwrap_callable(spec.runnable)
    kind = "Python"
    if isinstance(runnable, CompiledStateGraph):
        kind = "Subgraph"
    elif not callable(runnable):
        kind = "Runnable"
    metadata = dict(spec.metadata or {})
    return NodeIR0(
        node_id=node_id,
        kind=kind,
        callable_ref=callable_ref(spec.runnable),
        input_schema_type=type_name(spec.input_schema),
        input_schema_keys=typed_dict_keys(spec.input_schema),
        defer=bool(spec.defer),
        retry_policy=extract_retry_policy(spec.retry_policy),
        cache_policy=extract_cache_policy(spec.cache_policy),
        side_effects=extract_side_effects(metadata),
        resources=extract_resources(metadata),
        metadata=metadata,
        destinations_decl=sorted(getattr(spec, "ends", ()) or ()),
    )


def _extract_edges(builder: Any) -> list[EdgeIR0]:
    edges: list[EdgeIR0] = []
    for source, target in sorted(builder.edges):
        edges.append(EdgeIR0(kind="Static", source=source, target=target))
    for waiting in sorted(getattr(builder, "waiting_edges", ()) or ()):
        sources, target = waiting
        edges.append(EdgeIR0(kind="Join", sources=sorted(sources), target=target))
    for source, branches in sorted(builder.branches.items()):
        for _branch_name, branch_spec in sorted(branches.items()):
            edges.append(
                EdgeIR0(
                    kind="Conditional",
                    source=source,
                    router_ref=callable_ref(branch_spec.path),
                    returns=infer_branch_return_kind(branch_spec.path),
                    path_map=dict(branch_spec.ends or {}),
                )
            )
    return edges
