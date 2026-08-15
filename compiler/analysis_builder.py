from __future__ import annotations

import ast
from collections import Counter, defaultdict
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from orqis.compiler.capability_builder import (
    apply_capability_analysis,
    capability_cache_hint,
    executor_resource_profile,
    resolve_node_capability_view,
)
from orqis.compiler.ir import (
    AnalysisBundle,
    CacheAnalysisIR,
    FanoutRegionIR,
    GraphIR0,
    LoopIR,
    ReadWriteIR,
    ReducerAnalysisIR,
    ResourceIR,
    SideEffectIR,
    StepTraceIR,
    WorkProfileIR,
)
from orqis.compiler.utils import callable_ref, payload_units, safe_getsource, typed_dict_keys, unwrap_callable


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
    STRING_METHODS = {"join", "split", "format", "replace", "strip", "lower", "upper"}
    COLLECTION_METHODS = {"append", "extend", "update"}

    def __init__(self, state_param: str | None):
        self.state_param = state_param
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        self.send_targets: set[str] = set()
        self.send_payload_keys: set[str] = set()
        self.command_gotos: set[str] = set()
        self.command_parent = False
        self.effect_domains: set[str] = set()
        self.call_names: Counter[str] = Counter()
        self.op_counts: Counter[str] = Counter()

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == self.state_param:
            key = self._extract_string(node.slice)
            if key:
                self.reads.add(key)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        dotted = self._dotted_name(node.func)
        if dotted:
            self.call_names[dotted] += 1
            self.op_counts["calls"] += 1
        if dotted.endswith(".get") and isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and node.func.value.id == self.state_param and node.args:
                key = self._extract_string(node.args[0])
                if key:
                    self.reads.add(key)
        base = dotted.split(".")[0]
        tail = dotted.split(".")[-1]
        if base in self.EFFECT_DOMAINS:
            self.effect_domains.add(self.EFFECT_DOMAINS[base])
        if tail in self.STRING_METHODS:
            self.op_counts["string_ops"] += 1
        if tail in self.COLLECTION_METHODS:
            self.op_counts["collection_ops"] += 1
        if dotted.endswith("Send"):
            self.op_counts["sends"] += 1
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
            self.op_counts["commands"] += 1
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

    def visit_For(self, node: ast.For) -> None:
        self.op_counts["loops"] += 1
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.op_counts["loops"] += 1
        self.generic_visit(node)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.op_counts["comprehensions"] += 1
        self.generic_visit(node)

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.op_counts["comprehensions"] += 1
        self.generic_visit(node)

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.op_counts["comprehensions"] += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.op_counts["comprehensions"] += 1
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        self.op_counts["branches"] += 1
        self.generic_visit(node)

    def visit_IfExp(self, node: ast.IfExp) -> None:
        self.op_counts["branches"] += 1
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        self.op_counts["compares"] += 1
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
            self.op_counts["arithmetics"] += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.op_counts["arithmetics"] += 1
        self.generic_visit(node)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self.op_counts["string_ops"] += 1
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
            "call_names": [],
            "op_counts": {},
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
        "call_names": sorted(visitor.call_names),
        "op_counts": dict(visitor.op_counts),
        "source_available": True,
    }


def merge_side_effects(side_effects: list[SideEffectIR]) -> SideEffectIR:
    if any(effect.purity == "Effectful" for effect in side_effects):
        purity = "Effectful"
    elif any(effect.purity == "Idempotent" for effect in side_effects):
        purity = "Idempotent"
    else:
        purity = "Pure"
    domains = sorted({domain for effect in side_effects for domain in effect.effect_domains})
    strategy = next((effect.idempotency_key_strategy for effect in side_effects if effect.idempotency_key_strategy), None)
    return SideEffectIR(purity=purity, effect_domains=domains, idempotency_key_strategy=strategy)


def merge_resources(resources: list[ResourceIR | None]) -> ResourceIR | None:
    concrete_resources = [resource for resource in resources if resource is not None]
    if not concrete_resources:
        return None
    return ResourceIR(
        cpu_class=next((resource.cpu_class for resource in concrete_resources if resource.cpu_class), None),
        memory_mb=max((resource.memory_mb or 0) for resource in concrete_resources) or None,
        timeout_sec=max((resource.timeout_sec or 0) for resource in concrete_resources) or None,
        concurrency_limit=min(
            [resource.concurrency_limit for resource in concrete_resources if resource.concurrency_limit is not None],
            default=None,
        ),
        batchable=all(resource.batchable for resource in concrete_resources if resource.batchable is not None)
        if any(resource.batchable is not None for resource in concrete_resources)
        else None,
    )


def effective_node_side_effects(lgir0: GraphIR0, node_id: str, base: SideEffectIR) -> SideEffectIR:
    view = resolve_node_capability_view(lgir0, node_id)
    capability_effects = [
        lgir0.tools[tool_id].side_effects
        for tool_id in view.tool_ids
        if tool_id in lgir0.tools and lgir0.tools[tool_id].side_effects is not None
    ]
    if not capability_effects:
        return base
    return merge_side_effects([base, *capability_effects])


def effective_node_resources(lgir0: GraphIR0, node_id: str) -> ResourceIR | None:
    view = resolve_node_capability_view(lgir0, node_id)
    resources = [lgir0.nodes[node_id].resources]
    resources.extend(
        lgir0.tools[tool_id].resources
        for tool_id in view.tool_ids
        if tool_id in lgir0.tools and lgir0.tools[tool_id].resources is not None
    )
    resources.append(executor_resource_profile(lgir0, view.executor_id))
    return merge_resources(resources)


def build_analysis(
    compiled_graph: CompiledStateGraph,
    lgir0: GraphIR0,
    lgir1: Any,
    *,
    runtime_trace: list[StepTraceIR] | None = None,
) -> AnalysisBundle:
    analysis = AnalysisBundle()
    global_state_keys = set(lgir0.state_schema.keys)
    builder = compiled_graph.builder
    fn_analyses: dict[str, dict[str, Any]] = {}

    for node_id, node_ir in lgir0.nodes.items():
        spec = builder.nodes[node_id]
        fn = unwrap_callable(spec.runnable)
        fn_analysis = analyze_callable(fn)
        fn_analyses[node_id] = fn_analysis
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
        side_effects = effective_node_side_effects(
            lgir0,
            node_id,
            node_ir.side_effects or infer_side_effects(fn_analysis),
        )
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
        component = set(members)
        requires_loop = len(members) > 1 or any(node in adjacency[node] for node in members)
        entry_nodes = sorted(
            {
                edge.target
                for edge in lgir0.edges
                if edge.kind == "Static"
                and edge.target in component
                and edge.source not in component
            }
            | {
                target
                for edge in lgir0.edges
                if edge.kind == "Conditional"
                and edge.source not in component
                for target in edge.path_map.values()
                if target in component
            }
        )
        exit_nodes = sorted(
            {
                edge.source
                for edge in lgir0.edges
                if edge.kind == "Static"
                and edge.source in component
                and edge.target not in component
            }
            | {
                edge.source
                for edge in lgir0.edges
                if edge.kind == "Conditional"
                and edge.source in component
                and any(target not in component for target in edge.path_map.values())
            }
        )
        cycle_edges = sorted(
            {
                f"{edge.source}->{edge.target}"
                for edge in lgir0.edges
                if edge.kind == "Static"
                and edge.source in component
                and edge.target in component
            }
            | {
                f"{edge.source}->{target}"
                for edge in lgir0.edges
                if edge.kind == "Conditional"
                and edge.source in component
                for target in edge.path_map.values()
                if target in component
            }
        )
        termination_style = "none"
        scheduler_hint = "single_pass"
        notes: list[str] = []
        if requires_loop:
            conditional_exits = [
                edge
                for edge in lgir0.edges
                if edge.kind == "Conditional"
                and edge.source in component
                and any(target not in component for target in edge.path_map.values())
            ]
            if conditional_exits:
                termination_style = "conditional_route"
                scheduler_hint = "iterate_until_router_exit"
                notes.append("loop exits through a conditional route to a node outside the scc")
            elif exit_nodes:
                termination_style = "static_exit"
                scheduler_hint = "iterate_until_external_edge_is_enabled"
                notes.append("loop has an external exit edge but no explicit conditional route was identified")
            else:
                termination_style = "quiescence"
                scheduler_hint = "iterate_until_no_updates"
                notes.append("loop has no explicit exit edge, so runtime must rely on quiescence or an interrupt")
        analysis.loops.append(
            LoopIR(
                component_id=f"scc_{index}",
                members=sorted(members),
                entry_nodes=entry_nodes,
                exit_nodes=exit_nodes,
                cycle_edges=cycle_edges,
                kind="loop" if requires_loop else "acyclic",
                termination_style=termination_style,
                scheduler_hint=scheduler_hint,
                requires_loop_capable_orchestrator=requires_loop,
                notes=notes,
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
            analysis.cache_analysis[-1] = capability_cache_hint(lgir0, node_id, analysis.cache_analysis[-1])
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
        analysis.cache_analysis[-1] = capability_cache_hint(lgir0, node_id, analysis.cache_analysis[-1])

    apply_capability_analysis(analysis, lgir0)
    observations = build_runtime_observations(lgir0, runtime_trace or [])
    analysis.work_profiles = build_work_profiles(lgir0, analysis, fn_analyses, observations)

    analysis.notes.append("Read/write inference uses explicit input_schema first, then Python AST heuristics.")
    analysis.notes.append("Earned partitioning uses runtime-informed work profiles rather than only graph shape.")
    if any(not loop.requires_loop_capable_orchestrator for loop in analysis.loops):
        analysis.notes.append("Acyclic components can be scheduled as straight supersteps.")
    if any(loop.requires_loop_capable_orchestrator for loop in analysis.loops):
        analysis.warnings.append("Looping SCCs need a replay-safe coordinator.")
    if any(reducer.parallel_writers and not reducer.safe_parallel_merge for reducer in analysis.reducers):
        analysis.warnings.append("At least one state key is written in parallel without a reducer.")
    return analysis


def build_runtime_observations(
    lgir0: GraphIR0,
    runtime_trace: list[StepTraceIR],
) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {
        node_id: {
            "observed_invocations": 1,
            "observed_peak_concurrency": 1,
            "avg_input_units": 0.0,
            "max_input_units": 0.0,
            "avg_result_units": 0.0,
            "max_result_units": 0.0,
        }
        for node_id in lgir0.nodes
    }
    if not runtime_trace:
        return observations

    inputs: dict[str, list[float]] = defaultdict(list)
    results: dict[str, list[float]] = defaultdict(list)
    per_step_counts: dict[str, Counter[int]] = defaultdict(Counter)
    total_invocations: Counter[str] = Counter()

    for step in runtime_trace:
        step_counts: Counter[str] = Counter()
        for task in step.tasks:
            node_id = task.node_id
            if node_id not in observations:
                continue
            total_invocations[node_id] += 1
            step_counts[node_id] += 1
            inputs[node_id].append(payload_units(task.input_slice))
            results[node_id].append(payload_units(task.result))
        for node_id, count in step_counts.items():
            per_step_counts[node_id][step.step] = count

    for node_id in observations:
        invocations = max(1, total_invocations[node_id])
        peak = max(per_step_counts[node_id].values(), default=1)
        observations[node_id] = {
            "observed_invocations": invocations,
            "observed_peak_concurrency": peak,
            "avg_input_units": average(inputs[node_id]),
            "max_input_units": max(inputs[node_id], default=0.0),
            "avg_result_units": average(results[node_id]),
            "max_result_units": max(results[node_id], default=0.0),
        }
    return observations


def build_work_profiles(
    lgir0: GraphIR0,
    analysis: AnalysisBundle,
    fn_analyses: dict[str, dict[str, Any]],
    observations: dict[str, dict[str, Any]],
) -> dict[str, WorkProfileIR]:
    fanout_source_nodes = {region.fanout_source for region in analysis.fanout_regions}
    fanout_map_nodes = {node for region in analysis.fanout_regions for node in region.map_nodes}
    fanout_join_nodes = {node for region in analysis.fanout_regions for node in region.reduce_join}
    loop_members = {
        member
        for loop in analysis.loops
        if loop.requires_loop_capable_orchestrator
        for member in loop.members
    }
    profiles: dict[str, WorkProfileIR] = {}

    for node_id, node_ir in lgir0.nodes.items():
        fn_analysis = fn_analyses[node_id]
        rw = analysis.read_write[node_id]
        side_effects = analysis.side_effects[node_id]
        observation = observations.get(node_id, {})
        fanout_role = "none"
        if node_id in fanout_source_nodes:
            fanout_role = "fanout_source"
        elif node_id in fanout_map_nodes:
            fanout_role = "fanout_map"
        elif node_id in fanout_join_nodes:
            fanout_role = "fanout_join"
        loop_member = node_id in loop_members
        avg_input_units = float(observation.get("avg_input_units", 0.0) or 0.0)
        max_input_units = float(observation.get("max_input_units", 0.0) or 0.0)
        avg_result_units = float(observation.get("avg_result_units", 0.0) or 0.0)
        max_result_units = float(observation.get("max_result_units", 0.0) or 0.0)
        payload_expansion_ratio = round((avg_result_units + 1.0) / max(avg_input_units + 1.0, 1.0), 3)
        body_kind = classify_body_kind(
            node_ir=node_ir,
            fn_analysis=fn_analysis,
            side_effects=side_effects,
            write_set=rw.write_set,
            fanout_role=fanout_role,
            loop_member=loop_member,
            payload_expansion_ratio=payload_expansion_ratio,
        )
        dominant_ops = dominant_operations(fn_analysis)
        static_work_score = round(
            0.8
            + 0.45 * len(rw.read_set)
            + 0.55 * len(rw.write_set)
            + 0.35 * fn_count(fn_analysis, "calls")
            + 0.9 * fn_count(fn_analysis, "comprehensions")
            + 1.1 * fn_count(fn_analysis, "loops")
            + 0.4 * fn_count(fn_analysis, "branches")
            + 0.25 * fn_count(fn_analysis, "compares")
            + 0.3 * fn_count(fn_analysis, "arithmetics")
            + 0.2 * fn_count(fn_analysis, "string_ops")
            + 0.2 * fn_count(fn_analysis, "collection_ops")
            + (2.8 if side_effects.purity == "Effectful" else 0.0)
            + (0.6 if node_ir.defer else 0.0),
            3,
        )
        payload_work_score = round(
            min(4.0, (avg_input_units + avg_result_units) / 6.0)
            + min(2.0, max_result_units / 16.0),
            3,
        )
        intrinsic_work_score = round(static_work_score + payload_work_score, 3)
        orchestration_overhead_score = round(
            1.6
            + 0.25 * (len(rw.read_set) + len(rw.write_set))
            + (0.8 if fanout_role != "none" else 0.0)
            + (0.9 if loop_member else 0.0)
            + (0.4 if node_ir.defer else 0.0)
            + (0.3 if fn_count(fn_analysis, "sends") else 0.0),
            3,
        )
        ratio = round(intrinsic_work_score / max(orchestration_overhead_score, 0.1), 3)
        profile = assess_work_profile(
            node_id=node_id,
            body_kind=body_kind,
            dominant_ops=dominant_ops,
            static_work_score=static_work_score,
            payload_work_score=payload_work_score,
            intrinsic_work_score=intrinsic_work_score,
            orchestration_overhead_score=orchestration_overhead_score,
            work_to_overhead_ratio=ratio,
            observed_invocations=int(observation.get("observed_invocations", 1) or 1),
            observed_peak_concurrency=int(observation.get("observed_peak_concurrency", 1) or 1),
            avg_input_units=avg_input_units,
            max_input_units=max_input_units,
            avg_result_units=avg_result_units,
            max_result_units=max_result_units,
            payload_expansion_ratio=payload_expansion_ratio,
            fanout_role=fanout_role,
            loop_member=loop_member,
            effect_domains=list(side_effects.effect_domains),
            resources=effective_node_resources(lgir0, node_id),
            defer=node_ir.defer,
        )
        profiles[node_id] = profile
    return profiles


def classify_body_kind(
    *,
    node_ir,
    fn_analysis: dict[str, Any],
    side_effects: SideEffectIR,
    write_set: list[str],
    fanout_role: str,
    loop_member: bool,
    payload_expansion_ratio: float,
) -> str:
    if side_effects.effect_domains:
        return "remote_map_worker" if fanout_role == "fanout_map" else "remote_effect"
    if fanout_role == "fanout_source" or fn_count(fn_analysis, "sends") or fn_count(fn_analysis, "commands"):
        return "dispatch_controller"
    if node_ir.defer or fanout_role == "fanout_join":
        return "aggregation"
    if loop_member and fn_count(fn_analysis, "compares") and len(write_set) <= 1:
        return "control_predicate"
    if fn_count(fn_analysis, "loops") or fn_count(fn_analysis, "comprehensions"):
        if payload_expansion_ratio >= 1.15 or any("split" in name for name in fn_analysis.get("call_names", [])):
            return "batch_transform"
        return "iterative_transform"
    if (
        fn_count(fn_analysis, "arithmetics") <= 1
        and fn_count(fn_analysis, "compares") == 0
        and fn_count(fn_analysis, "calls") <= 1
        and len(write_set) <= 2
    ):
        return "micro_transform"
    return "state_transform"


def dominant_operations(fn_analysis: dict[str, Any]) -> list[str]:
    operations: list[str] = []
    if fn_count(fn_analysis, "sends"):
        operations.append("send")
    if fn_count(fn_analysis, "loops") or fn_count(fn_analysis, "comprehensions"):
        operations.append("iterate")
    if fn_count(fn_analysis, "compares"):
        operations.append("compare")
    if fn_count(fn_analysis, "arithmetics"):
        operations.append("arithmetic")
    if fn_count(fn_analysis, "string_ops"):
        operations.append("string")
    if fn_count(fn_analysis, "collection_ops"):
        operations.append("collection")
    if fn_count(fn_analysis, "calls"):
        operations.append("calls")
    return operations or ["state"]


def assess_work_profile(
    *,
    node_id: str,
    body_kind: str,
    dominant_ops: list[str],
    static_work_score: float,
    payload_work_score: float,
    intrinsic_work_score: float,
    orchestration_overhead_score: float,
    work_to_overhead_ratio: float,
    observed_invocations: int,
    observed_peak_concurrency: int,
    avg_input_units: float,
    max_input_units: float,
    avg_result_units: float,
    max_result_units: float,
    payload_expansion_ratio: float,
    fanout_role: str,
    loop_member: bool,
    effect_domains: list[str],
    resources,
    defer: bool,
) -> WorkProfileIR:
    earned_reasons: list[str] = []
    unearned_reasons: list[str] = []
    theoretical_basis: list[str] = []
    notes: list[str] = []
    earned_score = intrinsic_work_score - orchestration_overhead_score

    if effect_domains:
        earned_score += 2.4
        earned_reasons.append("remote or effectful work benefits from retry, timeout, and idempotency isolation")
        theoretical_basis.append(
            "Effectful or remote work earns a boundary because retries and failure containment act on the task itself."
        )
    if fanout_role == "fanout_map" and observed_peak_concurrency > 1:
        earned_score += 1.4
        earned_reasons.append("fanout map stage exposes real parallel slack across items")
        theoretical_basis.append(
            "Fanout item workers earn isolation when concurrency can convert one upstream step into many useful parallel tasks."
        )
    if resources is not None and resources.memory_mb and resources.memory_mb > 512:
        earned_score += 0.8
        earned_reasons.append("resource footprint is materially larger than a helper-stage baseline")
        theoretical_basis.append(
            "A boundary is justified when it changes the resource model, such as requiring a meaningfully different memory tier."
        )
    if defer or body_kind == "aggregation":
        earned_score += 0.9
        earned_reasons.append("deferred aggregation acts as a barriered fan-in stage")
        theoretical_basis.append(
            "Fan-in and deferred aggregate stages often need a barrier because they observe many upstream writes at once."
        )
    if fanout_role == "fanout_source":
        earned_score += 0.5
        notes.append("controller cost must be amortized by downstream parallel work, not by its own local body")

    if observed_invocations > 1 and body_kind in {"control_predicate", "micro_transform", "dispatch_controller"}:
        earned_score -= 1.8
        unearned_reasons.append("repeated control or micro-body work pays orchestration tax on every repetition")
        theoretical_basis.append(
            "If local work stays O(1) while invocations grow with loop iterations or item count, orchestration eventually dominates."
        )
    if loop_member and body_kind in {"control_predicate", "micro_transform"}:
        earned_score -= 0.8
        unearned_reasons.append("loop-local bookkeeping should stay close to the loop body unless it changes resources or effects")
    if work_to_overhead_ratio < 1.0:
        earned_score -= 1.2
        unearned_reasons.append("intrinsic work is smaller than the invoke and barrier overhead")
        theoretical_basis.append(
            "A partition has not earned isolation when the useful work inside it is smaller than the cost of invoking and coordinating it."
        )
    if fanout_role == "fanout_map" and observed_invocations > 1 and work_to_overhead_ratio < 1.75:
        unearned_reasons.append("one item per invocation is too fine-grained for the observed per-item work")
        theoretical_basis.append(
            "Per-item fanout needs enough work per item to amortize the cost of one invocation and one orchestration step."
        )
    if observed_peak_concurrency <= 1 and not effect_domains and body_kind in {"micro_transform", "state_transform"}:
        earned_score -= 0.6
        unearned_reasons.append("no exploitable parallel slack is visible for this helper stage")

    structural_justification = bool(effect_domains) or defer or fanout_role == "fanout_map" or (
        resources is not None and bool(resources.memory_mb and resources.memory_mb > 512)
    )
    if not structural_justification and not loop_member:
        unearned_reasons.append("pure linear helper does not change resources, failure domains, or available parallelism")
        theoretical_basis.append(
            "A pure linear helper should be fused unless it changes the resource model, creates exploitable parallelism, or protects side effects."
        )
    earned_partition = earned_score >= 1.0 and structural_justification
    granularity_hint = "keep" if earned_partition else "fuse_linear"
    recommended_batch_size = None
    if fanout_role == "fanout_map" and observed_invocations > 1 and work_to_overhead_ratio < 1.75:
        granularity_hint = "batch_map"
        recommended_batch_size = max(2, min(16, int(round(orchestration_overhead_score / max(intrinsic_work_score, 0.5))) + 1))
    elif not earned_partition and loop_member:
        granularity_hint = "fuse_loop"
    elif not earned_partition and fanout_role == "fanout_source":
        granularity_hint = "fuse_linear"
    elif earned_partition and observed_invocations <= 2 and not effect_domains and work_to_overhead_ratio < 1.2:
        granularity_hint = "consider_monolith_fallback"

    return WorkProfileIR(
        subject_id=node_id,
        subject_kind="node",
        body_kind=body_kind,
        dominant_operations=dominant_ops,
        static_work_score=static_work_score,
        payload_work_score=payload_work_score,
        intrinsic_work_score=intrinsic_work_score,
        orchestration_overhead_score=orchestration_overhead_score,
        work_to_overhead_ratio=work_to_overhead_ratio,
        observed_invocations=observed_invocations,
        observed_peak_concurrency=observed_peak_concurrency,
        avg_input_units=round(avg_input_units, 3),
        max_input_units=round(max_input_units, 3),
        avg_result_units=round(avg_result_units, 3),
        max_result_units=round(max_result_units, 3),
        payload_expansion_ratio=payload_expansion_ratio,
        fanout_role=fanout_role,
        loop_member=loop_member,
        effect_domains=effect_domains,
        earned_partition=earned_partition,
        earned_score=round(earned_score, 3),
        granularity_hint=granularity_hint,
        recommended_batch_size=recommended_batch_size,
        earned_reasons=dedupe_text(earned_reasons),
        unearned_reasons=dedupe_text(unearned_reasons),
        theoretical_basis=dedupe_text(theoretical_basis),
        notes=notes,
    )


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


def fn_count(fn_analysis: dict[str, Any], key: str) -> int:
    return int(fn_analysis.get("op_counts", {}).get(key, 0) or 0)


def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 3)


def dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
