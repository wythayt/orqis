from __future__ import annotations

import ast
from collections import defaultdict
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from orqis.compiler.ir import (
    AnalysisBundle,
    CacheAnalysisIR,
    FanoutRegionIR,
    GraphIR0,
    ReadWriteIR,
    ReducerAnalysisIR,
    SideEffectIR,
)
from orqis.compiler.utils import callable_ref, safe_getsource, typed_dict_keys, unwrap_callable


class CallableAstVisitor(ast.NodeVisitor):
    EFFECT_DOMAINS = {
        "open": "filesystem",
        "requests": "network",
        "httpx": "network",
        "aiohttp": "network",
        "urllib": "network",
        "boto3": "network",
        "openai": "llm",
        "anthropic": "llm",
        "cohere": "llm",
        "groq": "llm",
        "sqlite3": "db",
        "psycopg": "db",
        "redis": "db",
    }

    def __init__(self, state_param: str | None):
        self.state_param = state_param
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self.send_targets: set[str] = set()
        self.send_payload_keys: set[str] = set()
        self.command_gotos: set[str] = set()
        self.command_parent = False
        self.effect_domains: set[str] = set()

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == self.state_param:
            key = self._extract_string(node.slice)
            if key:
                self.reads.add(key)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = self._dotted_name(node.func)
        if dotted.endswith(".get") and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == self.state_param and node.args:
                key = self._extract_string(node.args[0])
                if key:
                    self.reads.add(key)
        base = dotted.split(".")[0]
        if base in self.EFFECT_DOMAINS:
            self.effect_domains.add(self.EFFECT_DOMAINS[base])
        if dotted.endswith("Send"):
            if node.args:
                target = self._extract_string(node.args[0])
                if target:
                    self.send_targets.add(target)
            if len(node.args) > 1 and isinstance(node.args[1], ast.Dict):
                for key_node in node.args[1].keys:
                    key = self._extract_string(key_node)
                    if key:
                        self.send_payload_keys.add(key)
        if dotted.endswith("Command"):
            for keyword in node.keywords:
                if keyword.arg == "goto":
                    goto = self._extract_string(keyword.value)
                    if goto:
                        self.command_gotos.add(goto)
                if keyword.arg == "graph":
                    graph_target = self._extract_string(keyword.value)
                    if graph_target == "PARENT":
                        self.command_parent = True
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if isinstance(node.value, ast.Dict):
            for key_node in node.value.keys:
                key = self._extract_string(key_node)
                if key:
                    self.writes.add(key)
        self.generic_visit(node)

    @staticmethod
    def _extract_string(node: ast.AST | None) -> str | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    @classmethod
    def _dotted_name(cls, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            prefix = cls._dotted_name(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        return ""


def analyze_callable(fn: Any) -> dict[str, Any]:
    source = safe_getsource(fn)
    if not source:
        return {
            "reads": [],
            "writes": [],
            "send_targets": [],
            "send_payload_keys": [],
            "command_gotos": [],
            "command_parent": False,
            "effect_domains": [],
            "source_available": False,
        }
    parsed = ast.parse(source)
    first_param = None
    function_nodes = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if function_nodes and function_nodes[0].args.args:
        first_param = function_nodes[0].args.args[0].arg
    visitor = CallableAstVisitor(first_param)
    visitor.visit(parsed)
    return {
        "reads": sorted(visitor.reads),
        "writes": sorted(visitor.writes),
        "send_targets": sorted(visitor.send_targets),
        "send_payload_keys": sorted(visitor.send_payload_keys),
        "command_gotos": sorted(visitor.command_gotos),
        "command_parent": visitor.command_parent,
        "effect_domains": sorted(visitor.effect_domains),
        "source_available": True,
    }


def build_analysis(
    compiled_graph: CompiledStateGraph,
    lgir0: GraphIR0,
    lgir1: Any,
) -> AnalysisBundle:
    analysis = AnalysisBundle()
    global_state_keys = set(lgir0.state_schema.keys)
    builder = compiled_graph.builder

    for node_id, node_ir in lgir0.nodes.items():
        spec = builder.nodes[node_id]
        fn = unwrap_callable(spec.runnable)
        fn_analysis = analyze_callable(fn)
        declared_reads = node_ir.input_schema_keys
        read_set = sorted(set(declared_reads or fn_analysis["reads"] or lgir1.nodes[node_id].reads))
        checkpoint_reads = sorted(key for key in read_set if key in global_state_keys)
        task_input_keys = sorted(key for key in read_set if key not in global_state_keys)
        write_set = sorted(set(fn_analysis["writes"]))
        if not write_set:
            possible_static_writes = [
                key
                for key in lgir1.nodes[node_id].writer_spec.declared_channels
                if key in global_state_keys
            ]
            write_set = sorted(set(possible_static_writes))
        inferred_from = []
        exactness = "best_effort"
        if declared_reads:
            inferred_from.append("input_schema")
            exactness = "declared-upper-bound"
        if fn_analysis["source_available"]:
            inferred_from.append("ast")
        if not inferred_from:
            inferred_from.append("pregel-node")
        side_effects = node_ir.side_effects or infer_side_effects(fn_analysis)
        analysis.side_effects[node_id] = side_effects
        lgir1.nodes[node_id].writer_spec.state_write_keys = list(write_set)
        analysis.read_write[node_id] = ReadWriteIR(
            subject_id=node_id,
            subject_kind="node",
            read_set=read_set,
            checkpoint_read_set=checkpoint_reads,
            task_input_keys=task_input_keys,
            write_set=write_set,
            inferred_from=inferred_from,
            exactness=exactness,
            notes=[],
        )

    route_meta = {}
    for source, branches in sorted(builder.branches.items()):
        for branch_name, branch_spec in sorted(branches.items()):
            route_id = f"route:{source}:{branch_name}"
            route_fn = unwrap_callable(branch_spec.path)
            fn_analysis = analyze_callable(route_fn)
            declared_reads = typed_dict_keys(branch_spec.input_schema)
            if fn_analysis["reads"] and declared_reads and set(fn_analysis["reads"]).issubset(set(declared_reads)):
                read_set = sorted(set(fn_analysis["reads"]))
                exactness = "best_effort"
                inferred_from = ["ast", "input_schema-upper-bound"]
            else:
                read_set = sorted(set(declared_reads or fn_analysis["reads"]))
                exactness = "declared-upper-bound" if declared_reads else "best_effort"
                inferred_from = ["input_schema" if declared_reads else "ast"]
            send_targets = sorted(set(fn_analysis["send_targets"] or list((branch_spec.ends or {}).values())))
            checkpoint_reads = sorted(key for key in read_set if key in global_state_keys)
            analysis.route_read_write[route_id] = ReadWriteIR(
                subject_id=route_id,
                subject_kind="route",
                read_set=read_set,
                checkpoint_read_set=checkpoint_reads,
                task_input_keys=[],
                write_set=[],
                send_targets=send_targets,
                send_payload_keys=fn_analysis["send_payload_keys"],
                inferred_from=inferred_from,
                exactness=exactness,
                notes=["conditional route attached to source node"],
            )
            route_meta[(source, callable_ref(branch_spec.path))] = route_id

    adjacency = build_node_adjacency(lgir0)
    sccs = strongly_connected_components(sorted(lgir0.nodes), adjacency)
    for index, members in enumerate(sccs):
        requires_loop = len(members) > 1 or any(node in adjacency[node] for node in members)
        from orqis.compiler.ir import LoopIR

        analysis.loops.append(
            LoopIR(
                component_id=f"scc_{index}",
                members=sorted(members),
                kind="loop" if requires_loop else "acyclic",
                requires_loop_capable_orchestrator=requires_loop,
            )
        )

    incoming = defaultdict(set)
    for src, dests in adjacency.items():
        for dst in dests:
            incoming[dst].add(src)

    for edge in lgir0.edges:
        if edge.kind != "Conditional" or edge.returns != "SendList" or not edge.source:
            continue
        route_id = route_meta.get((edge.source, edge.router_ref))
        route_rw = analysis.route_read_write.get(route_id or "")
        map_nodes = sorted(set(edge.path_map.values()))
        reduce_join = sorted(
            {
                dst
                for map_node in map_nodes
                for dst in adjacency.get(map_node, set())
                if lgir0.nodes[dst].defer or len(incoming.get(dst, set())) > 1 or dst not in map_nodes
            }
        )
        analysis.fanout_regions.append(
            FanoutRegionIR(
                fanout_source=edge.source,
                router_id=edge.router_ref or "unknown",
                map_nodes=map_nodes,
                reduce_join=reduce_join,
                send_payload_keys=route_rw.send_payload_keys if route_rw else [],
                notes=["dynamic Send fanout region"],
            )
        )

    writers_by_key = defaultdict(set)
    parallel_writers_by_key = defaultdict(set)
    for node_id, rw in analysis.read_write.items():
        for key in rw.write_set:
            writers_by_key[key].add(node_id)
    for fanout in analysis.fanout_regions:
        for map_node in fanout.map_nodes:
            for key in analysis.read_write.get(map_node, ReadWriteIR("", "")).write_set:
                parallel_writers_by_key[key].add(map_node)
    for key, state_key in sorted(lgir0.state_schema.keys.items()):
        reducer_notes = []
        if key in parallel_writers_by_key and state_key.reducer is None:
            reducer_notes.append("parallel fanout writes detected without reducer")
        if state_key.reducer and state_key.reducer.reducer_id == "operator.add" and state_key.ordering_sensitive:
            reducer_notes.append("list concatenation remains ordering-sensitive even though it is associative")
        analysis.reducers.append(
            ReducerAnalysisIR(
                key=key,
                channel_kind=state_key.channel_kind,
                reducer=state_key.reducer,
                writers=sorted(writers_by_key.get(key, set())),
                parallel_writers=sorted(parallel_writers_by_key.get(key, set())),
                ordering_sensitive=state_key.ordering_sensitive,
                safe_parallel_merge=not parallel_writers_by_key.get(key) or state_key.reducer is not None,
                notes=reducer_notes,
            )
        )

    for node_id, node_ir in lgir0.nodes.items():
        has_cache = node_ir.cache_policy is not None
        side_effects = analysis.side_effects[node_id]
        if not has_cache:
            analysis.cache_analysis.append(
                CacheAnalysisIR(
                    node_id=node_id,
                    has_cache_policy=False,
                    safe_to_cache=side_effects.purity == "Pure",
                    recommended_boundary=False,
                    reason="no explicit cache policy",
                )
            )
            continue
        safe_to_cache = side_effects.purity == "Pure" or (
            side_effects.purity == "Idempotent" and bool(side_effects.idempotency_key_strategy)
        )
        reason = "pure node cache is safe" if safe_to_cache else "effectful node cache is unsafe unless explicitly deduplicated"
        analysis.cache_analysis.append(
            CacheAnalysisIR(
                node_id=node_id,
                has_cache_policy=True,
                safe_to_cache=safe_to_cache,
                recommended_boundary=safe_to_cache,
                reason=reason,
            )
        )

    analysis.notes.append("Read/write inference uses explicit input_schema first, then Python AST heuristics.")
    if any(not loop.requires_loop_capable_orchestrator for loop in analysis.loops):
        analysis.notes.append("Acyclic components can be scheduled as straight supersteps.")
    if any(loop.requires_loop_capable_orchestrator for loop in analysis.loops):
        analysis.warnings.append("Looping SCCs need a replay-safe coordinator.")
    if any(reducer.parallel_writers and not reducer.safe_parallel_merge for reducer in analysis.reducers):
        analysis.warnings.append("At least one state key is written in parallel without a reducer.")
    return analysis


def infer_side_effects(fn_analysis: dict[str, Any]) -> SideEffectIR:
    domains = list(fn_analysis["effect_domains"])
    if not domains:
        return SideEffectIR(purity="Pure", effect_domains=[])
    purity = "Effectful"
    idempotency = "task_id" if "llm" in domains or "network" in domains else None
    return SideEffectIR(purity=purity, effect_domains=domains, idempotency_key_strategy=idempotency)


def build_node_adjacency(lgir0: GraphIR0) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in lgir0.nodes}
    for edge in lgir0.edges:
        if edge.kind == "Static" and edge.source in adjacency and edge.target in adjacency:
            adjacency[edge.source].add(edge.target)
        elif edge.kind == "Join" and edge.target in adjacency:
            for source in edge.sources:
                if source in adjacency:
                    adjacency[source].add(edge.target)
        elif edge.kind == "Conditional" and edge.source in adjacency:
            for target in edge.path_map.values():
                if target in adjacency:
                    adjacency[edge.source].add(target)
        elif edge.kind == "CommandRoute" and edge.source in adjacency:
            for target in edge.may_goto:
                if target in adjacency:
                    adjacency[edge.source].add(target)
    return adjacency


def strongly_connected_components(nodes: list[str], adjacency: dict[str, set[str]]) -> list[list[str]]:
    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    on_stack: set[str] = set()
    components: list[list[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for neighbor in adjacency.get(node, set()):
            if neighbor not in indices:
                visit(neighbor)
                lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
            elif neighbor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[neighbor])
        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                popped = stack.pop()
                on_stack.remove(popped)
                component.append(popped)
                if popped == node:
                    break
            components.append(component)

    for node in nodes:
        if node not in indices:
            visit(node)
    return components
