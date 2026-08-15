from __future__ import annotations

from collections import defaultdict

from orqis.compiler.capability_builder import (
    effective_subagent_tool_binding_ids,
    partition_capability_view,
    resolve_node_capability_view,
    resolve_subagent_capability_view,
)
from orqis.compiler.ir import AnalysisBundle, GraphIR0, GraphIR2, ServerlessPlanIR, StepTraceIR
from orqis.compiler.resource_optimizer import describe_slo_profile, optimize_partition_resources
from orqis.compiler.utils import topological_layers


def build_srv_plan(
    lgir0: GraphIR0,
    analysis: AnalysisBundle,
    lgir2: GraphIR2,
    runtime_trace: list[StepTraceIR] | None = None,
    *,
    resource_policy: str = "om2",
    slo_profile: str = "prototype",
) -> ServerlessPlanIR:
    resource_optimizations = optimize_partition_resources(
        lgir0,
        analysis,
        lgir2,
        runtime_trace,
        policy_id=resource_policy,
        slo_profile_id=slo_profile,
    )
    workers = {}
    for partition in lgir2.partitions.values():
        resource_optimization = resource_optimizations[partition.partition_id]
        work_profile = lgir2.partition_work_profiles.get(partition.partition_id)
        memory = resource_optimization.selected_memory_mb
        timeout = resource_optimization.selected_timeout_sec
        notes = [
            "reads only checkpoint_read_set plus task_input_keys",
            "applies local writes between fused members before moving to the next member",
            resource_optimization.reason,
        ]
        if work_profile is not None:
            notes.append(
                f"earned partitioning: body={work_profile.body_kind}, ratio={work_profile.work_to_overhead_ratio}, hint={work_profile.granularity_hint}"
            )
            if work_profile.recommended_batch_size:
                notes.append(f"recommended fanout batch size is {work_profile.recommended_batch_size} items per invocation")
        if partition.loop_component is not None:
            notes.append(f"participates in loop component {partition.loop_component} and must checkpoint between iterations")
        notes.extend(resource_optimization.notes)
        workers[partition.partition_id] = {
            "lambda_name": f"{lgir0.graph_id}-{partition.partition_id}",
            "memory_mb": memory,
            "timeout_sec": timeout,
            "concurrency_limit": resource_optimization.selected_concurrency_limit,
            "total_compute_mb": resource_optimization.total_compute_mb,
            "resource_optimization": resource_optimization,
            "notes": notes,
        }
    has_fanout = any(region.map_nodes for region in analysis.fanout_regions)
    has_loops = any(loop.requires_loop_capable_orchestrator for loop in analysis.loops)
    has_interrupts = any(item.requires_interrupt for item in analysis.capability_analysis)
    has_subagents = any(item.subject_kind == "subagent" for item in analysis.capability_analysis)
    mode = "StepFunctions" if has_fanout or has_loops or has_interrupts or has_subagents else "EventDriven"
    partition_edges = {src: set(dsts) for src, dsts in lgir2.edges.items()}
    layers = topological_layers(sorted(lgir2.partitions), partition_edges)
    loop_plans = build_loop_plans(analysis, lgir2)
    asset_plan = build_asset_plan(lgir0, lgir2)
    tooling_plan = build_tooling_plan(lgir0)
    agent_runtime = build_agent_runtime_plan(lgir0, analysis, lgir2)
    distributed_map_partitions = [
        partition_id
        for partition_id, partition in lgir2.partitions.items()
        if any(node in region.map_nodes for region in analysis.fanout_regions for node in partition.members)
    ]
    interrupt_partitions = [
        partition_id
        for partition_id, partition in lgir2.partitions.items()
        if partition_capability_view(lgir0, partition.members).requires_interrupt
    ]
    subagent_partitions = [
        partition_id
        for partition_id, partition in lgir2.partitions.items()
        if partition.subagent_ids
    ]
    planner_outline = []
    if loop_plans:
        for loop_plan in loop_plans:
            planner_outline.append(
                f"Loop {loop_plan['component_id']}: iterate partitions {', '.join(loop_plan['partitions'])} using {loop_plan['termination_style']}"
            )
            planner_outline.append(
                f"Loop {loop_plan['component_id']}: checkpoint after each iteration and re-plan until the loop exits"
            )
    if agent_runtime.get("delegation_nodes"):
        delegated_nodes = ", ".join(sorted(agent_runtime["delegation_nodes"]))
        planner_outline.append(f"Delegate: invoke isolated subagent workers from nodes {delegated_nodes}")
    if agent_runtime.get("router_nodes"):
        router_nodes = ", ".join(sorted(agent_runtime["router_nodes"]))
        planner_outline.append(f"Route: evaluate router nodes {router_nodes} before scheduling specialist work")
    for index, layer in enumerate(layers):
        planner_outline.append(f"Plan{index}: compute tasks for partitions {', '.join(layer)}")
        if any(lgir2.partitions[partition_id].emits_send for partition_id in layer):
            planner_outline.append(f"Exec{index}: run {', '.join(layer)} and stage Send tasks for the next superstep")
        elif any(
            any(node in region.map_nodes for region in analysis.fanout_regions for node in lgir2.partitions[partition_id].members)
            for partition_id in layer
        ):
            planner_outline.append(f"Map{index}: distributed map over {', '.join(layer)}")
        else:
            planner_outline.append(f"Exec{index}: run {', '.join(layer)}")
        planner_outline.append(f"Apply{index}: apply writes deterministically and checkpoint")
    return ServerlessPlanIR(
        plan_version="srv-aws-1.0",
        graph_id=lgir0.graph_id,
        persistence={
            "checkpoint_store": "DynamoDB+S3",
            "tables": {
                "checkpoints": f"{lgir0.graph_id}_checkpoints",
                "pending_writes": f"{lgir0.graph_id}_pending_writes",
                "task_ledger": f"{lgir0.graph_id}_task_ledger",
                "cache": f"{lgir0.graph_id}_cache",
                "skill_catalog": f"{lgir0.graph_id}_skill_catalog",
            },
            "buckets": {
                "state_blobs": f"{lgir0.graph_id}-state-blobs",
                "task_manifests": f"{lgir0.graph_id}-task-manifests",
                "assets": f"{lgir0.graph_id}-assets",
            },
        },
        messaging={
            "task_queue": f"{lgir0.graph_id}_task_queue",
            "result_queue": f"{lgir0.graph_id}_result_queue",
        },
        compute={
            "workers": workers,
            "coordinator": {"lambda_name": f"{lgir0.graph_id}-coordinator"},
            "support_workers": {
                "skill_loader": f"{lgir0.graph_id}-skill-loader",
                "prompt_assembler": f"{lgir0.graph_id}-prompt-assembler",
            },
            "resource_summary": {
                "policy_id": resource_policy,
                "slo_profile_id": slo_profile,
                "slo_profile": describe_slo_profile(slo_profile),
                "strategy": resource_strategy_summary(resource_policy),
                "candidate_memory_mb": [128, 256, 512, 1024, 1536, 2048, 3008],
                "total_compute_mb": sum(
                    optimization.total_compute_mb for optimization in resource_optimizations.values()
                ),
            },
        },
        assets=asset_plan,
        tooling=tooling_plan,
        agent_runtime=agent_runtime,
        orchestration={
            "mode": mode,
            "routing_nodes": sorted(agent_runtime.get("router_nodes", {})),
            "distributed_map_partitions": distributed_map_partitions,
            "loop_components": [loop_plan["component_id"] for loop_plan in loop_plans],
            "interrupt_partitions": interrupt_partitions,
            "subagent_partitions": subagent_partitions,
            "external_asset_partitions": [
                partition_id
                for partition_id, partition in lgir2.partitions.items()
                if partition_capability_view(lgir0, partition.members).externalized_assets
            ],
            "batching_candidates": {
                partition_id: profile.recommended_batch_size
                for partition_id, profile in lgir2.partition_work_profiles.items()
                if profile.granularity_hint == "batch_map" and profile.recommended_batch_size
            },
        },
        security={
            "roles": {
                "coordinator": f"{lgir0.graph_id}-coordinator-role",
                "worker": f"{lgir0.graph_id}-worker-role",
                "support": f"{lgir0.graph_id}-support-role",
            },
            "kms_keys": [f"{lgir0.graph_id}-kms-placeholder"],
        },
        task_contract_fields=[
            "graph_id",
            "run_id",
            "thread_id",
            "checkpoint_id",
            "step",
            "task_id",
            "task_path",
            "partition_id",
            "task_kind",
            "input_slice",
            "attempt",
        ],
        checkpoint_schema={
            "pk": "thread_id",
            "sk": "checkpoint_id",
            "attributes": [
                "step",
                "checkpoint_ns",
                "parent_checkpoint_id",
                "channel_versions",
                "state_inline|state_s3_key",
                "updated_channels",
            ],
        },
        pending_writes_schema={
            "pk": "thread_id",
            "sk": "checkpoint_id#task_id",
            "attributes": [
                "task_path_order_key",
                "writes|writes_s3_key",
                "status",
                "node_id",
            ],
        },
        loop_plans=loop_plans,
        planner_outline=planner_outline,
    )


def resource_strategy_summary(resource_policy: str) -> str:
    if resource_policy == "baseline":
        return "earned-partitioning plus direct passthrough of partition resource metadata"
    if resource_policy == "om":
        return "earned-partitioning plus fixed-weight Lambda memory candidate evaluation"
    return "earned-partitioning plus profile-aware Lambda memory evaluation with latency guardrails for hot repeated work"


def build_loop_plans(analysis: AnalysisBundle, lgir2: GraphIR2) -> list[dict[str, object]]:
    loop_plans: list[dict[str, object]] = []
    loop_partition_members = defaultdict(list)
    for partition_id, partition in lgir2.partitions.items():
        if partition.loop_component is not None:
            loop_partition_members[partition.loop_component].append(partition_id)
    for loop in analysis.loops:
        if not loop.requires_loop_capable_orchestrator:
            continue
        partitions = sorted(set(loop_partition_members.get(loop.component_id, [])) or set(lgir2.loop_clusters.get(loop.component_id, [])))
        loop_plans.append(
            {
                "component_id": loop.component_id,
                "partitions": partitions,
                "member_nodes": loop.members,
                "entry_nodes": loop.entry_nodes,
                "exit_nodes": loop.exit_nodes,
                "cycle_edges": loop.cycle_edges,
                "termination_style": loop.termination_style,
                "scheduler_hint": loop.scheduler_hint,
                "requires_quiescence_check": loop.termination_style == "quiescence",
                "notes": loop.notes,
            }
        )
    return loop_plans


def build_asset_plan(lgir0: GraphIR0, lgir2: GraphIR2) -> dict[str, object]:
    asset_consumers: dict[str, list[str]] = defaultdict(list)
    for partition_id, partition in lgir2.partitions.items():
        for asset_id in partition.asset_ids:
            asset_consumers[asset_id].append(partition_id)
    for skill_id, skill in lgir0.skills.items():
        if skill.prompt_asset_id:
            asset_consumers[skill.prompt_asset_id].append(f"skill:{skill_id}")
        for asset_id in skill.asset_ids:
            asset_consumers[asset_id].append(f"skill:{skill_id}")
    for agent_id, subagent in lgir0.subagents.items():
        if subagent.system_prompt_asset_id:
            asset_consumers[subagent.system_prompt_asset_id].append(f"subagent:{agent_id}")

    items = {}
    needs_efs = False
    for asset_id, asset in lgir0.assets.items():
        storage = choose_asset_storage(asset)
        needs_efs = needs_efs or storage == "efs"
        items[asset_id] = {
            "version": asset.version,
            "kind": asset.kind,
            "packaging": asset.packaging,
            "storage": storage,
            "load_strategy": asset.load_strategy,
            "uri": asset.uri,
            "size_bytes": asset.size_bytes,
            "mutable": asset.mutable,
            "consumers": sorted(set(asset_consumers.get(asset_id, []))),
        }
    return {
        "catalog_table": f"{lgir0.graph_id}_skill_catalog",
        "bucket": f"{lgir0.graph_id}-assets",
        "efs_access_point": f"{lgir0.graph_id}-assets-ap" if needs_efs else None,
        "items": items,
    }


def build_tooling_plan(lgir0: GraphIR0) -> dict[str, object]:
    tools = {}
    bindings = {}
    backends: dict[str, list[str]] = defaultdict(list)
    for tool_id, tool in lgir0.tools.items():
        executor = lgir0.executors.get(tool.executor_id or "lambda_default")
        backend = executor.backend if executor is not None else "lambda"
        backends[backend].append(tool_id)
        tools[tool_id] = {
            "tool_kind": tool.tool_kind,
            "executor_id": tool.executor_id,
            "backend": backend,
            "callable_ref": tool.callable_ref,
            "required_asset_ids": tool.required_asset_ids,
            "side_effects": tool.side_effects,
            "worker_name": tool_worker_name(lgir0.graph_id, tool_id, backend),
            "supports_streaming": bool(executor and executor.supports_streaming),
        }
    for binding_id, binding in lgir0.tool_bindings.items():
        bindings[binding_id] = {
            "tool_id": binding.tool_id,
            "scope_kind": binding.scope_kind,
            "scope_ref": binding.scope_ref,
            "visibility": binding.visibility,
            "requires_skill_id": binding.requires_skill_id,
            "approval": binding.approval_policy,
            "state_updates": binding.state_updates,
        }
    return {
        "tools": tools,
        "bindings": bindings,
        "backends": {backend: sorted(tool_ids) for backend, tool_ids in backends.items()},
    }


def build_agent_runtime_plan(lgir0: GraphIR0, analysis: AnalysisBundle, lgir2: GraphIR2) -> dict[str, object]:
    memory_namespaces = sorted(
        {
            namespace
            for policy in lgir0.memory_policies.values()
            for namespace in policy.long_term_namespaces
        }
    )
    active_skill_nodes = {
        item.subject_id: item.loaded_skills
        for item in analysis.capability_analysis
        if item.subject_kind == "node" and item.loaded_skills
    }
    router_nodes = build_router_nodes(analysis)
    delegation_nodes = {}
    for node_id, node in lgir0.nodes.items():
        if not node.subagent_id:
            continue
        capability = resolve_node_capability_view(lgir0, node_id)
        delegation_nodes[node_id] = {
            "subagent_id": node.subagent_id,
            "executor_id": capability.executor_id,
            "invocation_model": "isolated_worker",
            "tool_bindings": capability.tool_binding_ids,
            "reachable_tools": capability.tool_ids,
            "skills": capability.skill_ids,
            "externalized_assets": capability.externalized_assets,
            "requires_interrupt": capability.requires_interrupt,
            "memory_policies": capability.memory_policy_ids,
            "notes": capability.notes,
        }
    subagents = {}
    for agent_id, subagent in lgir0.subagents.items():
        capability = resolve_subagent_capability_view(lgir0, agent_id)
        subagents[agent_id] = {
            "graph_ref": subagent.graph_ref,
            "callable_ref": subagent.callable_ref,
            "executor_id": capability.executor_id,
            "skills": capability.skill_ids,
            "tool_bindings": effective_subagent_tool_binding_ids(lgir0, subagent),
            "reachable_tools": capability.tool_ids,
            "store_namespaces": subagent.store_namespaces,
            "system_prompt_asset_id": subagent.system_prompt_asset_id,
            "inheritance": subagent.inheritance,
            "memory_policy_id": subagent.memory_policy_id,
            "externalized_assets": capability.externalized_assets,
            "requires_interrupt": capability.requires_interrupt,
            "delegated_from_nodes": sorted(
                node_id for node_id, node in lgir0.nodes.items() if node.subagent_id == agent_id
            ),
        }
    return {
        "thread_state_contract": [
            "active_skills",
            "loaded_asset_refs",
            "tool_scope",
            "subagent_stack",
            "pending_approvals",
        ],
        "skill_loader": {
            "lambda_name": f"{lgir0.graph_id}-skill-loader",
            "catalog_table": f"{lgir0.graph_id}_skill_catalog",
            "asset_bucket": f"{lgir0.graph_id}-assets",
            "returns": ["skill_id", "asset_refs", "tool_bindings"],
        },
        "prompt_assembler": {
            "lambda_name": f"{lgir0.graph_id}-prompt-assembler",
            "inputs": ["system_prompt", "active_skills", "loaded_asset_refs", "messages"],
            "externalizes_large_assets": True,
        },
        "memory": {
            "checkpoint_keys": sorted(lgir0.state_schema.keys),
            "long_term_namespaces": memory_namespaces,
        },
        "active_skill_nodes": active_skill_nodes,
        "router_nodes": router_nodes,
        "delegation_nodes": delegation_nodes,
        "subagents": subagents,
        "interrupt_router": {
            "enabled": any(item.requires_interrupt for item in analysis.capability_analysis),
            "resume_table": f"{lgir0.graph_id}_pending_writes",
        },
        "streaming_channels": [
            partition_id
            for partition_id, partition in lgir2.partitions.items()
            if partition.executor_id and lgir0.executors.get(partition.executor_id)
            and lgir0.executors[partition.executor_id].supports_streaming
        ],
        "partitions_with_capabilities": {
            partition_id: {
                "skills": partition_capability_view(lgir0, partition.members).skill_ids,
                "tools": partition.tool_binding_ids,
                "subagents": partition.subagent_ids,
                "executor": partition.executor_id,
            }
            for partition_id, partition in lgir2.partitions.items()
            if partition.tool_binding_ids or partition.subagent_ids or partition.asset_ids
        },
    }


def choose_asset_storage(asset) -> str:
    if asset.packaging == "efs":
        return "efs"
    if asset.packaging in {"s3", "blob"}:
        return "s3"
    if asset.size_bytes is not None and asset.size_bytes > 512 * 1024 * 1024:
        return "efs"
    if asset.size_bytes is not None and asset.size_bytes > 16 * 1024:
        return "s3"
    return "inline"


def tool_worker_name(graph_id: str, tool_id: str, backend: str) -> str:
    if backend == "fargate":
        return f"{graph_id}-{tool_id}-task"
    return f"{graph_id}-{tool_id}-worker"


def build_router_nodes(analysis: AnalysisBundle) -> dict[str, dict[str, object]]:
    router_nodes: dict[str, dict[str, object]] = {}
    for route_id, route in analysis.route_read_write.items():
        _, source, branch_name = route_id.split(":", 2)
        router_nodes[source] = {
            "branch_count": router_nodes.get(source, {}).get("branch_count", 0) + 1,
            "branch_names": sorted(
                {
                    *router_nodes.get(source, {}).get("branch_names", []),
                    branch_name,
                }
            ),
            "send_targets": sorted(
                {
                    *router_nodes.get(source, {}).get("send_targets", []),
                    *route.send_targets,
                }
            ),
            "fanout": bool(route.send_payload_keys),
            "payload_keys": sorted(
                {
                    *router_nodes.get(source, {}).get("payload_keys", []),
                    *route.send_payload_keys,
                }
            ),
        }
    return router_nodes
