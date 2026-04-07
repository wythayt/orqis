from __future__ import annotations

from typing import Any

from langgraph.graph.state import CompiledStateGraph
from langgraph.pregel._write import ChannelWrite

from orqis.compiler.builder_common import (
    channel_kind,
    extract_cache_policy,
    extract_reducer,
    extract_retry_policy,
    normalize_sequence,
)
from orqis.compiler.ir import ChannelIR1, GraphIR0, NodeIR1, PregelIR1, WriterIR1


def build_lgir1(compiled_graph: CompiledStateGraph, lgir0: GraphIR0) -> PregelIR1:
    conditional_by_source = {
        edge.source: edge
        for edge in lgir0.edges
        if edge.kind == "Conditional" and edge.source is not None
    }
    channels = {
        name: _extract_channel_ir(name, channel)
        for name, channel in sorted(compiled_graph.channels.items())
    }
    nodes: dict[str, NodeIR1] = {}
    for node_id, pregel_node in sorted(compiled_graph.nodes.items()):
        if node_id == "__start__":
            continue
        reads = normalize_sequence(pregel_node.channels)
        triggers = normalize_sequence(pregel_node.triggers)
        writer_spec = WriterIR1()
        for writer in pregel_node.writers:
            static_writes = ChannelWrite.get_static_writes(writer) or []
            for channel_name, _value, label in static_writes:
                writer_spec.declared_channels.append(channel_name)
                if label:
                    writer_spec.route_targets.append(label)
            if static_writes:
                writer_spec.notes.append(
                    f"static writer declarations extracted from {type(writer).__name__}"
                )
        if node_id in conditional_by_source and conditional_by_source[node_id].returns == "SendList":
            writer_spec.may_emit_send = True
        node_spec = compiled_graph.builder.nodes[node_id]
        nodes[node_id] = NodeIR1(
            node_id=node_id,
            reads=sorted(reads),
            triggers=sorted(triggers),
            writer_spec=writer_spec,
            retry_policy=extract_retry_policy(node_spec.retry_policy),
            cache_policy=extract_cache_policy(node_spec.cache_policy),
            defer=bool(node_spec.defer),
            subgraph_ref=None,
        )
    trigger_to_nodes = {
        channel: sorted(nodes)
        for channel, nodes in sorted(compiled_graph.trigger_to_nodes.items())
    }
    reserved_channels = {}
    if "__pregel_tasks" in compiled_graph.channels:
        reserved_channels["TASKS"] = {
            "physical_channel": "__pregel_tasks",
            "kind": "Topic",
            "element_type": "Send",
            "accumulate": False,
        }
    return PregelIR1(
        ir_version="lgir-1.0",
        graph_id=lgir0.graph_id,
        channels=channels,
        nodes=nodes,
        trigger_to_nodes=trigger_to_nodes,
        reserved_channels=reserved_channels,
    )


def _extract_channel_ir(name: str, channel: Any) -> ChannelIR1:
    reducer = None
    if hasattr(channel, "operator"):
        reducer = extract_reducer(getattr(channel, "typ", Any), channel)
    return ChannelIR1(
        name=name,
        kind=channel_kind(channel),
        reducer=reducer,
        available_predicate=f"{type(channel).__name__}.is_available",
        reserved=name.startswith("__"),
    )
