from __future__ import annotations

from collections import defaultdict

from orqis.compiler.ir import AnalysisBundle, GraphIR0, GraphIR2, ServerlessPlanIR, StepTraceIR
from orqis.compiler.resource_optimizer import optimize_partition_resources
from orqis.compiler.utils import topological_layers


def build_srv_plan(
    lgir0: GraphIR0,
    analysis: AnalysisBundle,
    lgir2: GraphIR2,
    runtime_trace: list[StepTraceIR] | None = None,
) -> ServerlessPlanIR:
    resource_optimizations = optimize_partition_resources(lgir0, analysis, lgir2, runtime_trace)
    workers = {}
    for partition in lgir2.partitions.values():
        resource_optimization = resource_optimizations[partition.partition_id]
        memory = resource_optimization.selected_memory_mb
        timeout = resource_optimization.selected_timeout_sec
        notes = [
            "reads only checkpoint_read_set plus task_input_keys",
            "applies local writes between fused members before moving to the next member",
            resource_optimization.reason,
        ]
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
    mode = "StepFunctions" if has_fanout or has_loops else "EventDriven"
    partition_edges = {src: set(dsts) for src, dsts in lgir2.edges.items()}
    layers = topological_layers(sorted(lgir2.partitions), partition_edges)
    loop_plans = build_loop_plans(analysis, lgir2)
    planner_outline = []
    if loop_plans:
        for loop_plan in loop_plans:
            planner_outline.append(
                f"Loop {loop_plan['component_id']}: iterate partitions {', '.join(loop_plan['partitions'])} using {loop_plan['termination_style']}"
            )
            planner_outline.append(
                f"Loop {loop_plan['component_id']}: checkpoint after each iteration and re-plan until the loop exits"
            )
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
            },
            "buckets": {
                "state_blobs": f"{lgir0.graph_id}-state-blobs",
                "task_manifests": f"{lgir0.graph_id}-task-manifests",
            },
        },
        messaging={
            "task_queue": f"{lgir0.graph_id}_task_queue",
            "result_queue": f"{lgir0.graph_id}_result_queue",
        },
        compute={
            "workers": workers,
            "coordinator": {"lambda_name": f"{lgir0.graph_id}-coordinator"},
            "resource_summary": {
                "strategy": "evaluate Lambda memory as a per-partition hyperparameter",
                "candidate_memory_mb": [128, 256, 512, 1024, 1536, 2048, 3008],
                "total_compute_mb": sum(
                    optimization.total_compute_mb for optimization in resource_optimizations.values()
                ),
            },
        },
        orchestration={
            "mode": mode,
            "distributed_map_partitions": [
                partition_id
                for partition_id, partition in lgir2.partitions.items()
                if any(node in region.map_nodes for region in analysis.fanout_regions for node in partition.members)
            ],
            "loop_components": [loop_plan["component_id"] for loop_plan in loop_plans],
        },
        security={
            "roles": {
                "coordinator": f"{lgir0.graph_id}-coordinator-role",
                "worker": f"{lgir0.graph_id}-worker-role",
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
