from __future__ import annotations

from collections import defaultdict

from orqis.compiler.analysis_builder import build_node_adjacency
from orqis.compiler.capability_builder import executor_resource_profile, partition_capability_view, resolve_node_capability_view
from orqis.compiler.ir import (
    AnalysisBundle,
    FusionDecisionIR,
    GraphIR0,
    GraphIR2,
    PartitionIR2,
    ReadWriteIR,
    ResourceIR,
    SideEffectIR,
    WorkProfileIR,
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
        if not loop.requires_loop_capable_orchestrator:
            continue
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
        loop_component = next((scc_membership.get(member) for member in members if member in scc_membership), None)
        partition = build_partition_object(lgir0, analysis, members, adjacency, loop_component=loop_component)
        partition_objects[partition.partition_id] = partition
        for member in members:
            node_to_partition[member] = partition.partition_id

    partition_objects, node_to_partition, fallback_decisions = apply_fanout_monolith_fallbacks(
        lgir0,
        analysis,
        partition_objects,
        node_to_partition,
        adjacency,
        scc_membership,
    )
    decisions.extend(fallback_decisions)

    partition_work_profiles = build_partition_work_profiles(partition_objects, analysis)

    loop_clusters: dict[str, set[str]] = defaultdict(set)
    for node_id, partition_id in node_to_partition.items():
        loop_component = scc_membership.get(node_id)
        if loop_component:
            loop_clusters[loop_component].add(partition_id)

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
        partition_work_profiles=partition_work_profiles,
        partitioned_task_model=partitioned_task_model,
        loop_clusters={component_id: sorted(partitions) for component_id, partitions in loop_clusters.items()},
        edges={key: sorted(value) for key, value in partition_edges.items()},
        fusion_decisions=decisions,
    )


def apply_fanout_monolith_fallbacks(
    lgir0: GraphIR0,
    analysis: AnalysisBundle,
    partition_objects: dict[str, PartitionIR2],
    node_to_partition: dict[str, str],
    adjacency: dict[str, set[str]],
    scc_membership: dict[str, str],
) -> tuple[dict[str, PartitionIR2], dict[str, str], list[FusionDecisionIR]]:
    decisions: list[FusionDecisionIR] = []
    node_order = topological_node_order(adjacency, list(lgir0.nodes))

    for region in analysis.fanout_regions:
        involved_nodes = ordered_nodes(
            [region.fanout_source, *region.map_nodes, *region.reduce_join],
            node_order,
        )
        if not should_collapse_fanout_region(lgir0, analysis, region, involved_nodes):
            continue

        partition_ids = ordered_partition_ids(
            [node_to_partition.get(node_id) for node_id in involved_nodes],
            partition_objects,
        )
        if len(partition_ids) <= 1:
            continue

        merged_members = ordered_nodes(
            [
                member
                for partition_id in partition_ids
                for member in partition_objects[partition_id].members
            ],
            node_order,
        )
        loop_component = next((scc_membership.get(member) for member in merged_members if member in scc_membership), None)
        merged_partition = build_partition_object(
            lgir0,
            analysis,
            merged_members,
            adjacency,
            loop_component=loop_component,
        )
        before_cost = round(
            sum(
                partition_objects[partition_id].estimated_cost
                or estimate_partition_cost(lgir0, analysis, partition_objects[partition_id].members)
                for partition_id in partition_ids
            ),
            3,
        )
        after_cost = round(
            merged_partition.estimated_cost
            or estimate_partition_cost(lgir0, analysis, merged_members),
            3,
        )
        notes = [
            "collapse low-value fanout into one worker when per-item local work does not amortize orchestration",
            "this is a monolith fallback for pure local map work, not a generic ban on fanout",
        ]
        map_batch_sizes = [
            analysis.work_profiles[node_id].recommended_batch_size
            for node_id in region.map_nodes
            if node_id in analysis.work_profiles and analysis.work_profiles[node_id].recommended_batch_size
        ]
        if map_batch_sizes:
            notes.append(
                f"map stage requested batching at {max(map_batch_sizes)} items per invocation, so the compiler collapses the toy fanout boundary entirely"
            )
        decisions.append(
            FusionDecisionIR(
                edge=f"fanout-region:{region.fanout_source}->{','.join(region.map_nodes)}->{','.join(region.reduce_join)}",
                source_partition=list(involved_nodes),
                target_partition=list(merged_members),
                accepted=True,
                rule_results={
                    "pure_local_map": True,
                    "batch_hint_present": True,
                    "no_remote_effects": True,
                    "monolith_fallback": True,
                },
                estimated_cost_before=before_cost,
                estimated_cost_after=after_cost,
                notes=notes,
            )
        )
        for partition_id in partition_ids:
            partition_objects.pop(partition_id, None)
        partition_objects[merged_partition.partition_id] = merged_partition
        for member in merged_members:
            node_to_partition[member] = merged_partition.partition_id
    return partition_objects, node_to_partition, decisions


def should_collapse_fanout_region(
    lgir0: GraphIR0,
    analysis: AnalysisBundle,
    region,
    involved_nodes: list[str],
) -> bool:
    if not region.map_nodes:
        return False
    map_profiles = [analysis.work_profiles.get(node_id) for node_id in region.map_nodes]
    map_profiles = [profile for profile in map_profiles if profile is not None]
    if not map_profiles:
        return False
    if not any(profile.granularity_hint == "batch_map" for profile in map_profiles):
        return False
    if any(profile.effect_domains for profile in map_profiles):
        return False
    if any(analysis.side_effects.get(node_id) and analysis.side_effects[node_id].purity != "Pure" for node_id in involved_nodes):
        return False
    if any(
        (lgir0.nodes[node_id].resources and (lgir0.nodes[node_id].resources.memory_mb or 0) > 512)
        for node_id in region.map_nodes
    ):
        return False
    return True


def ordered_nodes(nodes: list[str], node_order: dict[str, int]) -> list[str]:
    seen: set[str] = set()
    ordered = []
    for node_id in sorted(nodes, key=lambda item: node_order.get(item, 10**9)):
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node_id)
    return ordered


def ordered_partition_ids(partition_ids: list[str | None], partitions: dict[str, PartitionIR2]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for partition_id in partition_ids:
        if partition_id is None or partition_id in seen or partition_id not in partitions:
            continue
        seen.add(partition_id)
        ordered.append(partition_id)
    return ordered


def topological_node_order(adjacency: dict[str, set[str]], declared_nodes: list[str]) -> dict[str, int]:
    reverse = defaultdict(set)
    indegree = {node_id: 0 for node_id in declared_nodes}
    for src, dests in adjacency.items():
        for dst in dests:
            if dst not in indegree:
                indegree[dst] = 0
            reverse[dst].add(src)
            indegree[dst] += 1
    ready = [node_id for node_id in declared_nodes if indegree.get(node_id, 0) == 0]
    ordered: list[str] = []
    seen: set[str] = set()
    while ready:
        node_id = ready.pop(0)
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered.append(node_id)
        for successor in sorted(adjacency.get(node_id, set()), key=declared_nodes.index):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                ready.append(successor)
    for node_id in declared_nodes:
        if node_id not in seen:
            ordered.append(node_id)
    return {node_id: index for index, node_id in enumerate(ordered)}


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
    capability_a = partition_capability_view(lgir0, members_a)
    capability_b = partition_capability_view(lgir0, members_b)
    source_capability = resolve_node_capability_view(lgir0, source)
    target_capability = resolve_node_capability_view(lgir0, target)
    source_profile = analysis.work_profiles.get(source)
    target_profile = analysis.work_profiles.get(target)
    same_loop_component = (
        scc_membership.get(source) is not None
        and scc_membership.get(source) == scc_membership.get(target)
    )
    earned_boundary = boundary_earns_isolation(source_profile, target_profile)
    rules = {
        "F1_linear": len(adjacency[source]) == 1 and reverse[target] == {source},
        "F2_barrier_simulatable": True,
        "F3_side_effect_safe": all(side_effects[node].purity == "Pure" for node in members_a + members_b),
        "F4_cache_compatible": lgir0.nodes[source].cache_policy == lgir0.nodes[target].cache_policy,
        "executor_boundary": capability_a.executor_id == capability_b.executor_id,
        "approval_boundary": not (source_capability.requires_interrupt or target_capability.requires_interrupt),
        "subagent_boundary": not (source_capability.subagent_ids or target_capability.subagent_ids),
        "tool_scope_compatible": (
            capability_a.tool_binding_ids == capability_b.tool_binding_ids
            and capability_a.skill_ids == capability_b.skill_ids
        ),
        "asset_boundary": merge_safe_for_assets(source_capability, target_capability),
        "fanout_boundary": not any(
            edge.kind == "Conditional" and edge.source == source and edge.returns == "SendList"
            for edge in lgir0.edges
        ),
        "defer_boundary": not (lgir0.nodes[source].defer or lgir0.nodes[target].defer),
        "loop_boundary": not same_loop_component
        or allow_loop_local_fusion(source_profile, target_profile),
        "earned_boundary": not earned_boundary,
    }
    cost_before = estimate_partition_cost(lgir0, analysis, members_a) + estimate_partition_cost(lgir0, analysis, members_b)
    merged_cost = estimate_partition_cost(lgir0, analysis, members_a + members_b)
    accepted = all(rules.values()) and merged_cost < cost_before
    notes = []
    if rules["F2_barrier_simulatable"]:
        notes.append("merged worker can use local apply between members to simulate the hidden Pregel barrier")
    if not rules["F3_side_effect_safe"]:
        notes.append("effectful nodes are kept isolated to avoid duplicated side effects on retry")
    if not rules["executor_boundary"]:
        notes.append("do not fuse across executor backend boundaries")
    if not rules["approval_boundary"]:
        notes.append("do not fuse across approval or interrupt boundaries")
    if not rules["subagent_boundary"]:
        notes.append("subagent delegation stays isolated from surrounding nodes")
    if not rules["tool_scope_compatible"]:
        notes.append("tool visibility changes introduce a capability boundary")
    if not rules["asset_boundary"]:
        notes.append("externalized asset staging stays at a separate partition boundary")
    if not rules["fanout_boundary"]:
        notes.append("do not fuse across Send fanout boundaries")
    if not rules["loop_boundary"]:
        notes.append("loop boundary is kept only when the repeated stage earns standalone isolation")
    if not rules["earned_boundary"]:
        notes.append("boundary is preserved because the downstream work already earns isolation through effects, resources, or parallel slack")
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
    capability = partition_capability_view(lgir0, members)
    invocation = 1.0
    ckpt_read = 0.2 * len(checkpoint_reads)
    ckpt_write = 0.25 * len(write_set)
    queue = 0.1 if emits_send else 0.0
    tooling = 0.2 * len(capability.tool_ids)
    assets = 0.15 * len(capability.externalized_assets)
    interrupts = 0.5 if capability.requires_interrupt else 0.0
    subagents = 0.3 * len(capability.subagent_ids)
    repeated_micro = 0.0
    for member in members:
        profile = analysis.work_profiles.get(member)
        if profile is None:
            continue
        if profile.granularity_hint in {"fuse_linear", "fuse_loop"} and profile.observed_invocations > 1:
            repeated_micro += 0.35
        if profile.work_to_overhead_ratio < 1.0:
            repeated_micro += 0.15
    return invocation + ckpt_read + ckpt_write + queue + tooling + assets + interrupts + subagents + repeated_micro


def build_partition_object(
    lgir0: GraphIR0,
    analysis: AnalysisBundle,
    members: list[str],
    adjacency: dict[str, set[str]],
    *,
    loop_component: str | None = None,
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
    requires_barrier = loop_component is not None or emits_send or any(
        successor not in members for member in members for successor in adjacency.get(member, set())
    )
    capability = partition_capability_view(lgir0, members)
    side_effects = combine_side_effects([analysis.side_effects[member] for member in members])
    # combine node hints, tool hints, and executor defaults so sizing sees the full workload picture.
    explicit_resources = [lgir0.nodes[member].resources for member in members if lgir0.nodes[member].resources] + [
        lgir0.tools[tool_id].resources
        for tool_id in capability.tool_ids
        if lgir0.tools.get(tool_id) and lgir0.tools[tool_id].resources
    ]
    resources = combine_resources(explicit_resources)
    if resources is None:
        resources = combine_resources([executor_resource_profile(lgir0, capability.executor_id)])
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
        loop_component=loop_component,
        retry_policy=retry_policy,
        cache_policy=cache_policy,
        resources=resources,
        side_effects=side_effects,
        read_set=read_set,
        checkpoint_read_set=checkpoint_reads,
        task_input_keys=task_inputs,
        write_set=write_set,
        emits_send=emits_send,
        asset_ids=capability.asset_ids,
        tool_binding_ids=capability.tool_binding_ids,
        subagent_ids=capability.subagent_ids,
        executor_id=capability.executor_id,
        requires_barrier_after=requires_barrier,
        estimated_cost=round(estimate_partition_cost(lgir0, analysis, members), 3),
    )


def compute_partition_sets(
    lgir0: GraphIR0,
    analysis: AnalysisBundle,
    members: list[str],
) -> tuple[list[str], list[str], list[str], bool]:
    member_set = set(members)
    internal_task_payload_keys: set[str] = set()
    for route_id, route_rw in analysis.route_read_write.items():
        route_source = route_id.split(":")[1]
        if route_source not in member_set:
            continue
        if any(target in member_set for target in route_rw.send_targets):
            internal_task_payload_keys.update(route_rw.send_payload_keys)
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
        task_inputs.update(key for key in rw.task_input_keys if key not in internal_task_payload_keys)
        write_set.update(rw.write_set)
        produced_locally.update(rw.write_set)
        for route_id, route_rw in analysis.route_read_write.items():
            if route_id.split(":")[1] != member:
                continue
            for key in route_rw.checkpoint_read_set:
                if key not in produced_locally:
                    checkpoint_reads.add(key)
            task_inputs.update(route_rw.task_input_keys)
            emits_send = emits_send or any(target not in member_set for target in route_rw.send_targets)
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
        if concrete_resources
        else None,
    )


def merge_safe_for_assets(source_capability, target_capability) -> bool:
    if not source_capability.externalized_assets:
        return True
    return set(source_capability.externalized_assets).issubset(set(target_capability.asset_ids))


def boundary_earns_isolation(source_profile: WorkProfileIR | None, target_profile: WorkProfileIR | None) -> bool:
    profiles = [profile for profile in (source_profile, target_profile) if profile is not None]
    if not profiles:
        return False
    if any(profile.granularity_hint == "batch_map" for profile in profiles):
        return True
    if any(profile.earned_partition and profile.body_kind in {"remote_effect", "remote_map_worker", "aggregation"} for profile in profiles):
        return True
    if any(profile.effect_domains for profile in profiles):
        return True
    return any(profile.earned_partition and profile.granularity_hint not in {"fuse_linear", "fuse_loop"} for profile in profiles)


def allow_loop_local_fusion(source_profile: WorkProfileIR | None, target_profile: WorkProfileIR | None) -> bool:
    profiles = [profile for profile in (source_profile, target_profile) if profile is not None]
    if not profiles:
        return False
    if any(profile.effect_domains for profile in profiles):
        return False
    if any(profile.granularity_hint == "batch_map" for profile in profiles):
        return False
    if all(profile.body_kind in {"control_predicate", "micro_transform", "state_transform"} for profile in profiles):
        return True
    if all(profile.granularity_hint in {"fuse_linear", "fuse_loop"} for profile in profiles):
        return True
    return False


def build_partition_work_profiles(
    partitions: dict[str, PartitionIR2],
    analysis: AnalysisBundle,
) -> dict[str, WorkProfileIR]:
    profiles: dict[str, WorkProfileIR] = {}
    for partition_id, partition in partitions.items():
        member_profiles = [analysis.work_profiles[member] for member in partition.members if member in analysis.work_profiles]
        if not member_profiles:
            continue
        body_kind = classify_partition_body(partition, member_profiles)
        dominant_ops = sorted({op for profile in member_profiles for op in profile.dominant_operations}) or ["state"]
        static_work_score = round(sum(profile.static_work_score for profile in member_profiles), 3)
        payload_work_score = round(sum(profile.payload_work_score for profile in member_profiles), 3)
        intrinsic_work_score = round(sum(profile.intrinsic_work_score for profile in member_profiles), 3)
        orchestration_overhead_score = round(
            1.8
            + 0.4 * len(partition.members)
            + 0.2 * len(partition.checkpoint_read_set)
            + 0.25 * len(partition.write_set)
            + (0.8 if partition.emits_send else 0.0)
            + (0.9 if partition.loop_component is not None else 0.0)
            + (0.4 if partition.requires_barrier_after else 0.0),
            3,
        )
        ratio = round(intrinsic_work_score / max(orchestration_overhead_score, 0.1), 3)
        observed_invocations = max(profile.observed_invocations for profile in member_profiles)
        observed_peak_concurrency = max(profile.observed_peak_concurrency for profile in member_profiles)
        avg_input_units = round(sum(profile.avg_input_units for profile in member_profiles), 3)
        max_input_units = round(max(profile.max_input_units for profile in member_profiles), 3)
        avg_result_units = round(sum(profile.avg_result_units for profile in member_profiles), 3)
        max_result_units = round(max(profile.max_result_units for profile in member_profiles), 3)
        payload_expansion_ratio = round((avg_result_units + 1.0) / max(avg_input_units + 1.0, 1.0), 3)
        effect_domains = sorted({domain for profile in member_profiles for domain in profile.effect_domains})
        earned_reasons = dedupe_text([reason for profile in member_profiles for reason in profile.earned_reasons])
        unearned_reasons = dedupe_text([reason for profile in member_profiles for reason in profile.unearned_reasons])
        theoretical_basis = dedupe_text([reason for profile in member_profiles for reason in profile.theoretical_basis])
        notes = dedupe_text([note for profile in member_profiles for note in profile.notes])
        earned_score = intrinsic_work_score - orchestration_overhead_score
        if partition.emits_send:
            earned_score += 0.8
            earned_reasons.append("controller and fanout setup are only justified if they feed useful parallel work")
        if partition.loop_component is not None and all(
            profile.body_kind in {"control_predicate", "micro_transform", "state_transform"} for profile in member_profiles
        ):
            earned_score -= 1.4
            unearned_reasons.append("loop bookkeeping remains too small to justify multiple partitions")
        structural_justification = bool(effect_domains) or partition.emits_send or any(
            profile.body_kind == "aggregation" or profile.earned_partition for profile in member_profiles
        )
        if not structural_justification:
            unearned_reasons.append("pure linear partition does not create a new failure, resource, or parallelism boundary")
            theoretical_basis.append(
                "Fusing helpers is preferable when a partition does not expose new parallel slack, side-effect isolation, or resource separation."
            )
        earned_partition = (earned_score >= 1.0 and structural_justification) or any(
            profile.earned_partition and profile.granularity_hint not in {"fuse_linear", "fuse_loop"}
            for profile in member_profiles
        )
        granularity_hint = "keep" if earned_partition else "fuse_linear"
        recommended_batch_size = None
        if any(profile.granularity_hint == "batch_map" for profile in member_profiles):
            granularity_hint = "batch_map"
            recommended_batch_size = max(
                profile.recommended_batch_size or 2
                for profile in member_profiles
                if profile.granularity_hint == "batch_map"
            )
        elif not earned_partition and partition.loop_component is not None:
            granularity_hint = "fuse_loop"
        profiles[partition_id] = WorkProfileIR(
            subject_id=partition_id,
            subject_kind="partition",
            body_kind=body_kind,
            dominant_operations=dominant_ops,
            static_work_score=static_work_score,
            payload_work_score=payload_work_score,
            intrinsic_work_score=intrinsic_work_score,
            orchestration_overhead_score=orchestration_overhead_score,
            work_to_overhead_ratio=ratio,
            observed_invocations=observed_invocations,
            observed_peak_concurrency=observed_peak_concurrency,
            avg_input_units=avg_input_units,
            max_input_units=max_input_units,
            avg_result_units=avg_result_units,
            max_result_units=max_result_units,
            payload_expansion_ratio=payload_expansion_ratio,
            fanout_role="fanout_map" if any(profile.fanout_role == "fanout_map" for profile in member_profiles) else "none",
            loop_member=partition.loop_component is not None,
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
    return profiles


def classify_partition_body(partition: PartitionIR2, member_profiles: list[WorkProfileIR]) -> str:
    if partition.emits_send:
        return "dispatch_pipeline"
    if any(profile.body_kind == "aggregation" for profile in member_profiles):
        return "aggregation"
    if any(profile.body_kind in {"remote_effect", "remote_map_worker"} for profile in member_profiles):
        return "remote_worker"
    if partition.loop_component is not None and all(
        profile.body_kind in {"control_predicate", "micro_transform", "state_transform"} for profile in member_profiles
    ):
        return "loop_micro_body"
    if len(partition.members) > 1:
        return "fused_pipeline"
    return member_profiles[0].body_kind


def dedupe_text(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
