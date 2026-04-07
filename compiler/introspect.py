from __future__ import annotations

from langgraph.graph.state import CompiledStateGraph

from orqis.compiler.lgir0_builder import build_lgir0
from orqis.compiler.lgir1_builder import build_lgir1


class GraphIntrospector:
    # compatibility wrapper around the stage-specific lgir builders

    def __init__(self, compiled_graph: CompiledStateGraph, graph_id: str | None = None):
        if not isinstance(compiled_graph, CompiledStateGraph):
            raise TypeError("GraphIntrospector expects a CompiledStateGraph.")
        self.compiled_graph = compiled_graph
        self.graph_id = graph_id

    def extract(self):
        lgir0 = build_lgir0(self.compiled_graph, graph_id=self.graph_id)
        lgir1 = build_lgir1(self.compiled_graph, lgir0)
        return lgir0, lgir1
