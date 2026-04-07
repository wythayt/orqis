from __future__ import annotations

from orqis.compiler.ir import AnalysisBundle, GraphIR0, GraphIR2, ResourceIR, ServerlessPlanIR
from orqis.compiler.utils import topological_layers


def build_srv_plan(lgir0: GraphIR0, analysis: AnalysisBundle, lgir2: GraphIR2) -> ServerlessPlanIR:
    workers = {}
    for partition in lgir2.partitions.values():
        resources = partition.resources or ResourceIR()
        memory = resources.memory_mb or (1536 if partition.side_effects and partition.side_effects.purity == "Effectful" else 512)
        timeout = resources.timeout_sec or (90 if partition.emits_send or (partition.side_effects and partition.side_effects.purity == "Effectful") else 30)
        workers[partition.partition_id] = {
            "lambda_name": f"{lgir0.graph_id}-{partition.partition_id}",
            "memory_mb": memory,
            "timeout_sec": timeout,
            "concurrency_limit": resources.concurrency_limit,
            "notes": [
                "reads only checkpoint_read_set plus task_input_keys",
                "applies local writes between fused members before moving to the next member",
            ],
        }
    has_fanout = any(region.map_nodes for region in analysis.fanout_regions)
    has_loops = any(loop.requires_loop_capable_orchestrator for loop in analysis.loops)
    mode = "StepFunctions" if has_fanout or has_loops else "EventDriven"
    partition_edges = {src: set(dsts) for src, dsts in lgir2.edges.items()}
    layers = topological_layers(sorted(lgir2.partitions), partition_edges)
    planner_outline = []
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
        },
        orchestration={
            "mode": mode,
            "distributed_map_partitions": [
                partition_id
                for partition_id, partition in lgir2.partitions.items()
                if any(node in region.map_nodes for region in analysis.fanout_regions for node in partition.members)
            ],
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
        planner_outline=planner_outline,
    )
