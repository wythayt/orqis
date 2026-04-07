from __future__ import annotations

from collections import defaultdict

from orqis.compiler.analysis_builder import build_node_adjacency
from orqis.compiler.ir import (
    AnalysisBundle,
    FusionDecisionIR,
    GraphIR0,
    GraphIR2,
    PartitionIR2,
    ReadWriteIR,
    ResourceIR,
    SideEffectIR,
)
from orqis.compiler.utils import sanitize_identifier


def build_lgir2(lgir0: GraphIR0, analysis: AnalysisBundle) -> GraphIR2:
    adjacency = build_node_adjacency(lgir0)
    reverse = defaultdict(set)
    for src, dests in adjacency.items():
        for dst in dests:
            reverse[dst].add(src)
    scc_membership = {}
    for loop in analysis.loops:
        for member in loop.members:
            scc_membership[member] = loop.component_id

    partitions: dict[str, list[str]] = {node_id: [node_id] for node_id in lgir0.nodes}
    node_to_partition = {node_id: node_id for node_id in lgir0.nodes}
    decisions: list[FusionDecisionIR] = []
    static_edges = [
        (edge.source, edge.target)
        for edge in lgir0.edges
        if edge.kind == "Static" and edge.source in lgir0.nodes and edge.target in lgir0.nodes
    ]
    for source, target in static_edges:
        part_a = node_to_partition[source]
        part_b = node_to_partition[target]
        if part_a == part_b:
            continue
        decision = evaluate_merge(
            lgir0,
            analysis,
            adjacency,
            reverse,
            partitions[part_a],
            partitions[part_b],
            source,
            target,
            scc_membership,
        )
        decisions.append(decision)
        if not decision.accepted:
            continue
        merged_members = partitions[part_a] + partitions[part_b]
        for member in merged_members:
            node_to_partition[member] = part_a
        partitions[part_a] = merged_members
        del partitions[part_b]

    partition_objects: dict[str, PartitionIR2] = {}
    for members in partitions.values():
        partition = build_partition_object(lgir0, analysis, members, adjacency)
        partition_objects[partition.partition_id] = partition
        for member in members:
            node_to_partition[member] = partition.partition_id

    partition_edges = defaultdict(set)
    for src, dests in adjacency.items():
        src_partition = node_to_partition[src]
        for dst in dests:
            dst_partition = node_to_partition[dst]
            if src_partition != dst_partition:
                partition_edges[src_partition].add(dst_partition)

    partitioned_task_model = {
        "pull_targets": {node: node_to_partition[node] for node in lgir0.nodes},
        "push_targets": {node: node_to_partition[node] for node in lgir0.nodes},
    }
    return GraphIR2(
        ir_version="lgir-2.0",
        graph_id=lgir0.graph_id,
        partitions=partition_objects,
        partitioned_task_model=partitioned_task_model,
        edges={key: sorted(value) for key, value in partition_edges.items()},
        fusion_decisions=decisions,
    )


def evaluate_merge(
    lgir0: GraphIR0,
    analysis: AnalysisBundle,
    adjacency: dict[str, set[str]],
    reverse: dict[str, set[str]],
    members_a: list[str],
    members_b: list[str],
    source: str,
    target: str,
    scc_membership: dict[str, str],
) -> FusionDecisionIR:
    side_effects = analysis.side_effects
    rules = {
        "F1_linear": len(adjacency[source]) == 1 and reverse[target] == {source},
        "F2_barrier_simulatable": True,
        "F3_side_effect_safe": all(side_effects[node].purity == "Pure" for node in members_a + members_b),
        "F4_cache_compatible": lgir0.nodes[source].cache_policy == lgir0.nodes[target].cache_policy,
        "fanout_boundary": not any(
            edge.kind == "Conditional" and edge.source == source and edge.returns == "SendList"
            for edge in lgir0.edges
        ),
        "defer_boundary": not (lgir0.nodes[source].defer or lgir0.nodes[target].defer),
        "loop_boundary": scc_membership.get(source) != scc_membership.get(target)
        or source == target == "",
    }
    if scc_membership.get(source) == scc_membership.get(target):
        loop_component = next(
            (loop for loop in analysis.loops if loop.component_id == scc_membership.get(source)),
            None,
        )
        if loop_component and len(loop_component.members) > 1:
            rules["loop_boundary"] = False
    cost_before = estimate_partition_cost(lgir0, analysis, members_a) + estimate_partition_cost(lgir0, analysis, members_b)
    merged_cost = estimate_partition_cost(lgir0, analysis, members_a + members_b)
    accepted = all(rules.values()) and merged_cost < cost_before
    notes = []
    if rules["F2_barrier_simulatable"]:
        notes.append("merged worker can use local apply between members to simulate the hidden Pregel barrier")
    if not rules["F3_side_effect_safe"]:
        notes.append("effectful nodes are kept isolated to avoid duplicated side effects on retry")
    if not rules["fanout_boundary"]:
        notes.append("do not fuse across Send fanout boundaries")
    return FusionDecisionIR(
        edge=f"{source}->{target}",
        source_partition=list(members_a),
        target_partition=list(members_b),
        accepted=accepted,
        rule_results=rules,
        estimated_cost_before=round(cost_before, 3),
        estimated_cost_after=round(merged_cost, 3),
        notes=notes,
    )


def estimate_partition_cost(lgir0: GraphIR0, analysis: AnalysisBundle, members: list[str]) -> float:
    checkpoint_reads, _task_inputs, write_set, emits_send = compute_partition_sets(lgir0, analysis, members)
    invocation = 1.0
    ckpt_read = 0.2 * len(checkpoint_reads)
    ckpt_write = 0.25 * len(write_set)
    queue = 0.1 if emits_send else 0.0
    return invocation + ckpt_read + ckpt_write + queue


def build_partition_object(
    lgir0: GraphIR0,
    analysis: AnalysisBundle,
    members: list[str],
    adjacency: dict[str, set[str]],
) -> PartitionIR2:
    attached_routes = [
        route_id
        for route_id in analysis.route_read_write
        if route_id.split(":")[1] in members
    ]
    checkpoint_reads, task_inputs, write_set, emits_send = compute_partition_sets(lgir0, analysis, members)
    read_set = sorted(
        {
            key
            for member in members
            for key in analysis.read_write[member].read_set
        }
        | {
            key
            for route_id in attached_routes
            for key in analysis.route_read_write[route_id].read_set
        }
    )
    requires_barrier = emits_send or any(
        successor not in members for member in members for successor in adjacency.get(member, set())
    )
    side_effects = combine_side_effects([analysis.side_effects[member] for member in members])
    resources = combine_resources([lgir0.nodes[member].resources for member in members if lgir0.nodes[member].resources])
    retry_policy = next((lgir0.nodes[member].retry_policy for member in members if lgir0.nodes[member].retry_policy), None)
    cache_policy = next((lgir0.nodes[member].cache_policy for member in members if lgir0.nodes[member].cache_policy), None)
    partition_name = "_".join(members)
    if emits_send:
        partition_name += "_fanout"
    partition_id = f"p_{sanitize_identifier(partition_name)}"
    return PartitionIR2(
        partition_id=partition_id,
        members=list(members),
        attached_routes=attached_routes,
        retry_policy=retry_policy,
        cache_policy=cache_policy,
        resources=resources,
        side_effects=side_effects,
        read_set=read_set,
        checkpoint_read_set=checkpoint_reads,
        task_input_keys=task_inputs,
        write_set=write_set,
        emits_send=emits_send,
        requires_barrier_after=requires_barrier,
        estimated_cost=round(estimate_partition_cost(lgir0, analysis, members), 3),
    )


def compute_partition_sets(
    lgir0: GraphIR0,
    analysis: AnalysisBundle,
    members: list[str],
) -> tuple[list[str], list[str], list[str], bool]:
    produced_locally: set[str] = set()
    checkpoint_reads: set[str] = set()
    task_inputs: set[str] = set()
    write_set: set[str] = set()
    emits_send = False
    for member in members:
        rw = analysis.read_write[member]
        for key in rw.checkpoint_read_set:
            if key not in produced_locally:
                checkpoint_reads.add(key)
        task_inputs.update(rw.task_input_keys)
        write_set.update(rw.write_set)
        produced_locally.update(rw.write_set)
        for route_id, route_rw in analysis.route_read_write.items():
            if route_id.split(":")[1] != member:
                continue
            for key in route_rw.checkpoint_read_set:
                if key not in produced_locally:
                    checkpoint_reads.add(key)
            task_inputs.update(route_rw.task_input_keys)
            emits_send = emits_send or bool(route_rw.send_targets)
    return sorted(checkpoint_reads), sorted(task_inputs), sorted(write_set), emits_send


def combine_side_effects(side_effects: list[SideEffectIR]) -> SideEffectIR:
    if any(effect.purity == "Effectful" for effect in side_effects):
        purity = "Effectful"
    elif any(effect.purity == "Idempotent" for effect in side_effects):
        purity = "Idempotent"
    else:
        purity = "Pure"
    domains = sorted({domain for effect in side_effects for domain in effect.effect_domains})
    strategy = next((effect.idempotency_key_strategy for effect in side_effects if effect.idempotency_key_strategy), None)
    return SideEffectIR(purity=purity, effect_domains=domains, idempotency_key_strategy=strategy)


def combine_resources(resources: list[ResourceIR]) -> ResourceIR | None:
    if not resources:
        return None
    return ResourceIR(
        cpu_class=next((resource.cpu_class for resource in resources if resource.cpu_class), None),
        memory_mb=max((resource.memory_mb or 0) for resource in resources) or None,
        timeout_sec=max((resource.timeout_sec or 0) for resource in resources) or None,
        concurrency_limit=min(
            [resource.concurrency_limit for resource in resources if resource.concurrency_limit is not None],
            default=None,
        ),
        batchable=all(resource.batchable for resource in resources if resource.batchable is not None) if resources else None,
    )
